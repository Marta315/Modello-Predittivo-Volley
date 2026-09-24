# One_hot encode di una specifica partita per poter testare il modello (funziona come 1_final_dataset_onehot.py)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from datavolley import read_dv
import os

folder = "&RS24B_RIT03_TRE-FUT"
os.makedirs(folder, exist_ok=True)

file_path = f"../Dati/experiments/{folder}.dvw"
match = read_dv.DataVolley(file_path)
match_df = match.get_plays()

# colonne utili
df_touches = match_df[['match_id', 'team', 'player_number', 'player_name', 'skill', 'evaluation_code', 'home_setter_position', 'visiting_setter_position', 'attack_code', 'set_code', 'set_type', 'start_zone', 'end_zone', 'set_number', 'rally_number', 'home_team_score', 'visiting_team_score', 'point_won_by', 'serving_team', 'home_team', 'visiting_team']]

# eliminiamo le righe di setup (teniamo solo quelle con informazioni riguardo skill e evaluation code)
encoded_df = df_touches.dropna(subset=['skill', 'evaluation_code']).copy()

# salviamo i dati della partita
csv_path = os.path.join(folder, "data.csv")
encoded_df.to_csv(csv_path, index=False)

# uniamo le colonne skill e evaluation code in un'unica colonna + one-hot
encoded_df['skill_eval'] = encoded_df['skill'].astype(str) + '_' + encoded_df['evaluation_code'].astype(str)
encoded_df = pd.get_dummies(encoded_df, columns=['skill_eval'], dtype=int)
encoded_df = encoded_df.drop(columns=['skill_eval_Reception_!'])

# aggiungiamo le informazioni riguardo il tocco precedente + one-hot
encoded_df['prev_touch'] = encoded_df['skill'].shift(1)
encoded_df['prev_touch_eval'] = encoded_df['evaluation_code'].shift(1)
is_serve = encoded_df['skill'] == 'Serve'
encoded_df.loc[is_serve, 'prev_touch'] = 0
encoded_df.loc[is_serve, 'prev_touch_eval'] = 0
encoded_df['prev_skill_eval'] = encoded_df['prev_touch'].astype(str) + '_' + encoded_df['prev_touch_eval'].astype(str)
encoded_df.loc[is_serve, 'prev_skill_eval'] = 0
encoded_df = pd.get_dummies(encoded_df, columns=['prev_skill_eval'], dtype=int)
encoded_df = encoded_df.drop(columns=['prev_skill_eval_0'])

# aggiungiamo le informazioni riguardo il numero di tocco + one-hot
match_changed = encoded_df['match_id'] != encoded_df['match_id'].shift(1)
team_changed = encoded_df['team'] != encoded_df['team'].shift(1)
rally_changed = encoded_df['rally_number'] != encoded_df['rally_number'].shift(1)
encoded_df['custom_possession_id'] = (match_changed | team_changed | rally_changed).cumsum()

zero_touch_skills = ['Serve', 'Block']
first_touch_skills = ['Dig', 'Freeball', 'Reception']
second_touch_skills = ['Set']
third_touch_skills = ['Attack']

# pulizia dei tocchi anomali duplicati (teniamo solo l'ultima azione)
same_possession = encoded_df['custom_possession_id'] == encoded_df['custom_possession_id'].shift(-1)
consecutive_first_touch = encoded_df['skill'].isin(first_touch_skills) & encoded_df['skill'].shift(-1).isin(first_touch_skills)
consecutive_set = (encoded_df['skill'] == 'Set') & (encoded_df['skill'].shift(-1) == 'Set')
to_drop = (consecutive_first_touch | consecutive_set) & same_possession
encoded_df = encoded_df[~to_drop].copy()

encoded_df.loc[encoded_df['skill'].isin(zero_touch_skills), 'touch_number'] = 0
encoded_df.loc[encoded_df['skill'].isin(first_touch_skills), 'touch_number'] = 1
encoded_df.loc[encoded_df['skill'].isin(second_touch_skills), 'touch_number'] = 2
encoded_df.loc[encoded_df['skill'].isin(third_touch_skills), 'touch_number'] = 3

encoded_df['is_serve'] = (encoded_df['skill'] == 'Serve').astype(int)
encoded_df['is_block'] = (encoded_df['skill'] == 'Block').astype(int)

encoded_df = pd.get_dummies(encoded_df, columns=['touch_number'], dtype=int)
encoded_df = encoded_df.drop(columns=['touch_number_0.0'])

print(f"Total touches: {len(encoded_df)}")

# aggiungiamo la variabile binaria action_by_serving_team variable (vale 1 se l'azione la sta compiendo la squadra in battuta, 0 altrimenti)
encoded_df['action_by_serving_team'] = (encoded_df['serving_team'] == encoded_df['team']).astype(int)

# aggiungiamo le informazioni riguardo al punteggio tenendo conto che nei dati scout sono salvati i punteggi di fine rally
# isoliamo il risultato post rally per ogni rally e per ogni set
rally_summary = encoded_df.groupby(['set_number', 'rally_number'], as_index=False)[
	['home_team_score', 'visiting_team_score']
].last()
# shiftiamo il risultato di 1 all'interno di ogni set e fissiamo il risultato a inizio set 0-0
rally_summary['actual_start_home_score'] = (
	rally_summary.groupby('set_number')['home_team_score']
	.shift(1)
	.fillna(0)
	.astype(int)
)
rally_summary['actual_start_visiting_score'] = (
	rally_summary.groupby('set_number')['visiting_team_score']
	.shift(1)
	.fillna(0)
	.astype(int)
)
# mettiamo il punteggio di pre rally ad ogni riga
encoded_df = encoded_df.merge(
	rally_summary[['set_number', 'rally_number', 'actual_start_home_score', 'actual_start_visiting_score']],
	on=['set_number', 'rally_number'],
	how='left',
)
# calcoliamo il punteggio in relazione alla squadra in battuta e al servizio
encoded_df['serving_team_score'] = np.where(
	encoded_df['home_team'] == encoded_df['serving_team'],
	encoded_df['actual_start_home_score'],
	encoded_df['actual_start_visiting_score'],
)
encoded_df['receiving_team_score'] = np.where(
	encoded_df['home_team'] == encoded_df['serving_team'],
	encoded_df['actual_start_visiting_score'],
	encoded_df['actual_start_home_score'],
)
# calcoliamo la differenza punti (sempre rispetto alla squadra in battuta)
encoded_df['score_difference'] = (encoded_df['serving_team_score'] - encoded_df['receiving_team_score'])

# aggingiamo la variabile bianria sui critical moments (vale 1 se il punteggio è >20 (o >10 nel 5 set) e la differenza <= 2, 0 altrimenti)
encoded_df['set_number'] = encoded_df['set_number'].astype(int)
critical_threshold = np.where(
	encoded_df['set_number'] == 5, 
	10, 
	20
)
encoded_df['critical_moments'] = (
	((encoded_df['serving_team_score'] >= critical_threshold) | (encoded_df['receiving_team_score'] >= critical_threshold)) & 
	(abs(encoded_df['score_difference']) <= 2)
).astype(int)

# informazioni sulla posizione di entrambi i palleggiatori + one-hot
encoded_df['serving_setter_pos'] = np.where(
	encoded_df['home_team'] == encoded_df['serving_team'],
	encoded_df['home_setter_position'],
	encoded_df['visiting_setter_position']
)
encoded_df['receiving_setter_pos'] = np.where( 
	encoded_df['home_team'] == encoded_df['serving_team'],
	encoded_df['visiting_setter_position'],
	encoded_df['home_setter_position']
)
encoded_df = pd.get_dummies(encoded_df, columns=['serving_setter_pos', 'receiving_setter_pos'], dtype=int)
encoded_df = encoded_df.drop(columns=['serving_setter_pos_1', 'receiving_setter_pos_1'])

# set_code and set_type + one-hot
encoded_df['original_set_code'] = encoded_df['set_code']
encoded_df['original_set_type'] = encoded_df['set_type']
is_set = encoded_df['skill'] == 'Set'
encoded_df.loc[is_set & encoded_df['set_code'].isna(), 'set_code'] = 'Unknown_Code'
encoded_df.loc[is_set & encoded_df['set_type'].isna(), 'set_type'] = 'Unknown_Type'
encoded_df['set_code'] = encoded_df['set_code'].fillna('not_a_set')
encoded_df['set_type'] = encoded_df['set_type'].fillna('not_a_set')
encoded_df = pd.get_dummies(encoded_df, columns=['set_code', 'set_type'], dtype=int)
encoded_df = encoded_df.drop(columns=['set_code_not_a_set', 'set_type_not_a_set'])

# attack code + one_hot
encoded_df['original_attack_code'] = encoded_df['attack_code']
is_attack = encoded_df['skill'] == 'Attack'
encoded_df.loc[is_attack & encoded_df['attack_code'].isna(), 'attack_code'] = 'Unknown_Attack'
encoded_df['attack_code'] = encoded_df['attack_code'].fillna('not_an_attack')
encoded_df = pd.get_dummies(encoded_df, columns=['attack_code'], dtype=int)
encoded_df = encoded_df.drop(columns=['attack_code_not_an_attack'])

# start and end zone + one-hot
encoded_df['start_zone'] = encoded_df['start_zone'].fillna('no_zone')
encoded_df['end_zone'] = encoded_df['end_zone'].fillna('no_zone')
encoded_df = pd.get_dummies(encoded_df, columns=['start_zone', 'end_zone'], dtype=int)
encoded_df = encoded_df.drop(columns=['start_zone_no_zone', 'end_zone_no_zone'])

# normalizziamo le info relative al punteggio e il set number
divisors = np.where(
	encoded_df['set_number'] == 5, 
	15.0, 
	25.0
)
encoded_df['score_difference'] = encoded_df['score_difference'] / divisors
encoded_df['serving_team_score'] = encoded_df['serving_team_score'] / divisors
encoded_df['receiving_team_score'] = encoded_df['receiving_team_score'] / divisors
encoded_df['set_number'] = (encoded_df['set_number'] - 1) / 4

# ghost row per calcolare la pre-serve win probability
ghost_rows = encoded_df[encoded_df['skill'] == 'Serve'].copy()
action_keywords = ['skill', 'evaluation', 'skill_eval', 'prev_', 'is_serve', 'is_block', 'set_code_', 'set_type_', 'start_zone_', 'end_zone_', 'attack_code_', 'touch_number_', 'original_']
action_columns = [col for col in ghost_rows.columns if any(keyword in col for keyword in action_keywords)]
ghost_rows[action_columns] = 0
ghost_rows = ghost_rows.copy()
ghost_rows['touch_sequence'] = 0
encoded_df['touch_sequence'] = 1
df = pd.concat([ghost_rows, encoded_df])
df = df.sort_values(by=['set_number', 'rally_number', 'touch_sequence']).reset_index(drop=True)

csv_path = os.path.join(folder, "encoded_data.csv")
df.to_csv(csv_path, index=False)
