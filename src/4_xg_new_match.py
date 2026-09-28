# Applicazione del modello delle random forest ad una specifica partita (la quale deve già essere codificata usando il codice one_hot_one.py)

import pandas as pd
import joblib
import os

folder = "&CI24_FIN_TRE-MAG_25-02-09"
file_path = os.path.join(folder, "encoded_data.csv")
new_game_encoded = pd.read_csv(file_path)

# carichiamo la lista delle colonne usate nel training
expected_columns = joblib.load('src/training_columns.pkl')

# facciamo in modo che il nuovo DataFrame abbia le stesse colonne del training (quelle diverse vengono eliminate)
X = new_game_encoded.reindex(columns=expected_columns, fill_value=0)

model = joblib.load('src/volleyball_xg_model.pkl')

# Probabilità
y_pred_probs = model.predict_proba(X)
wpa_df = new_game_encoded.copy()
wpa_df['Serving_Team_Win_Prob'] = y_pred_probs[:, 1]

wpa_df['Probability_Shift'] = wpa_df.groupby(['rally_number', 'set_number'])['Serving_Team_Win_Prob'].diff()

# eliminiamo le righe del preservizio (consideriamo solo i tocchi effettivi)
clean_wpa_df = wpa_df[wpa_df['touch_sequence'] == 1].copy()

output_columns = ['set_number', 'rally_number', 'actual_start_home_score', 'actual_start_visiting_score', 'critical_moments', 'home_team', 'visiting_team', 'serving_team', 'team', 'player_number', 'player_name', 'skill', 'evaluation_code', 'home_setter_position', 'visiting_setter_position', 'original_set_type', 'original_set_code', 'original_attack_code', 'Serving_Team_Win_Prob', 'Probability_Shift']

clean_wpa_df = clean_wpa_df[output_columns]

csv_path = os.path.join(folder, "xg_win_prob.csv")
clean_wpa_df.to_csv(csv_path, index=False)
