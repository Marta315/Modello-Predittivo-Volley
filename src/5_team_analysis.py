# I grafici relativi ad una determinata squadra di una determinata partita

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
from adjustText import adjust_text

focus_team = "ITAS TRENTINO"
folder = "&RS24B_RIT03_TRE-FUT"
file_path = os.path.join(folder, "xg_win_prob.csv")
df = pd.read_csv(file_path)

# ====================
# FUNZIONI DI SUPPORTO
# ====================

# per fissare gli estremi degli assi
def apply_axis_padding(ax, axis='both', padding=0.15, min_pad=0.05):
    axes_to_pad = ['x', 'y'] if axis == 'both' else [axis]
    if 'x' in axes_to_pad:
        c_min, c_max = ax.get_xlim()
        pad = max((c_max - c_min) * padding, min_pad)
        ax.set_xlim(c_min - pad, c_max + pad)
    if 'y' in axes_to_pad:
        c_min, c_max = ax.get_ylim()
        pad = max((c_max - c_min) * padding, min_pad)
        ax.set_ylim(c_min - pad, c_max + pad)

# per salvare e chiudere i grafici
def save_and_close_plot(fig, base_folder, sub_folder, filename):
    if sub_folder:
        target_folder = os.path.join(base_folder, sub_folder)
        os.makedirs(target_folder, exist_ok=True)
    else:
        target_folder = base_folder
    file_path = os.path.join(target_folder, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

# per denormalizzare il numero di set
def get_display_set(s_num):
    return int(float(s_num) * 4 + 1)

# per aggiungere le labels testuali
def annotate_and_adjust(ax, df, x_col, y_col, label_col, count_col=None):
    texts = []
    x_coords = df[x_col].tolist()
    y_coords = df[y_col].tolist()
    for _, row in df.iterrows():
        display_text = f"{row[label_col]} (n = {int(row[count_col])})" if count_col else str(row[label_col])
        texts.append(
            ax.text(
                row[x_col], 
                row[y_col], 
                display_text,
                fontsize=10, 
                fontweight='bold',
                zorder=6,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round, pad=0.2')
            )
        )
    adjust_text(texts, x=x_coords, y=y_coords, ax=ax, force_points=0.2, arrowprops=dict(arrowstyle='-', color='gray', lw=1))

# per definire la laber relativa ai dati scout nell'heatmap
def get_grade_dist(series):
	series = series.dropna()
	if series.empty:
		return ""
	counts = series.value_counts()
	order = ['#', '+', '!', '-', '/', '=']
	# aggiungiamo il simbolo solo se è effettivamente successo (count > 0)
	dist = [f"{counts[s]}{s}" for s in order if s in counts and counts[s] > 0]
	return " ".join(dist)

# per la divisione in quadranti
def add_quadrant_labels(ax, tr, tl, br, bl, fontsize=14, alpha=0.2):
    kwargs = {'transform': ax.transAxes, 'fontsize': fontsize, 'fontweight': 'bold', 'alpha': alpha, 'zorder': 1}
    ax.text(0.98, 0.98, tr, color='green', ha='right', va='top', **kwargs)
    ax.text(0.02, 0.98, tl, color='darkorange', ha='left', va='top', **kwargs)
    ax.text(0.98, 0.02, br, color='darkorange', ha='right', va='bottom', **kwargs)
    ax.text(0.02, 0.02, bl, color='red', ha='left', va='bottom', **kwargs)

# ================
# DATA PREPARATION
# ================

# isoliamo il contributo che ogni giocatore apporta alla sua squadra
df['Player_WPA'] = np.where(
	df['team'] == df['serving_team'], 
	df['Probability_Shift'], 
	-df['Probability_Shift'] 
)

# definiamo i buoni tocchi (la probabiità di vittoria della propria squadra è aumentata) e i cattivi tocchi (probabilità diminuita)
df['Is_Good_Touch'] = df['Player_WPA'] >= 0
df['Is_Bad_Touch'] = df['Player_WPA'] < 0

# la report card è fatta per giocatore e calcola il contributo totale (Total_WPA), il numero totale di tocchi (Total_Touches), il numero totale di tocchi buoni (Good_Touaches) e il numero toatle di tocchi cattivi (Bad_Touches)
report_card = df.groupby(['team', 'player_number', 'player_name']).agg(
	Total_WPA=('Player_WPA', 'sum'),
	Total_Touches=('Player_WPA', 'count'),
	Good_Touches=('Is_Good_Touch', 'sum'),
	Bad_Touches=('Is_Bad_Touch', 'sum')
).reset_index()

# calcoliamo la percentuale di tocchi buoni e cattivi 
report_card['Good_Touch_Pct'] = (report_card['Good_Touches'] / report_card['Total_Touches']) * 100
report_card['Bad_Touch_Pct'] = (report_card['Bad_Touches'] / report_card['Total_Touches']) * 100

# lista delle skills
skills = ['Serve', 'Reception', 'Set', 'Attack', 'Block', 'Dig', 'Freeball']

# skill_wpa è una tabella divisa per giocatore e per fondamentale, le informazioni sono il contributo totale (sum) e i tocchi totali (count)
skill_wpa = df.pivot_table(
	index=['team', 'player_number', 'player_name'], 
	columns='skill', 
	values='Player_WPA', 
	aggfunc=['sum', 'count'],
	fill_value=0
).reset_index()

# organizziamo meglio la tabella skill_wpa lasciando invariate le colonne riguardanti le info sul giocatore e rinominando le colonne 'skill'_WPA e 'skill'_Touches che contengono rispettivamente informazioni di sum e count
new_columns = []
for col in skill_wpa.columns:
	if col[0] in ['team', 'player_number', 'player_name']:
		new_columns.append(col[0])
	elif col[0] == 'sum':
		new_columns.append(f"{col[1]}_WPA")
	elif col[0] == 'count':
		new_columns.append(f"{col[1]}_Touches")
skill_wpa.columns = new_columns

# il full report conterrà sia le informazioni contenute in report_card (informazioni generali per giocatori) che in skill_wpa (informazioni divise per giocatore e fondamentale)
full_report = pd.merge(report_card, skill_wpa, on=['team', 'player_number', 'player_name'])

# riordiniamo le colonne, al momento ho tutte le skill_wpa e poi tutte le skill_touches, cosa voglio è per fondamentale le informazioni riguardo la wpa e il numero di tocchi vicine
final_column_order = ['team', 'player_number', 'player_name', 'Total_WPA', 'Total_Touches', 'Good_Touches', 'Bad_Touches', 'Good_Touch_Pct', 'Bad_Touch_Pct']
for skill in skills:
	final_column_order.append(f"{skill}_WPA")
	final_column_order.append(f"{skill}_Touches")
full_report = full_report[final_column_order]

# ordiamo la tabella prima in ordine alfabetico per squadra e poi in ordine decrescente per Total_WPA (quindi divisi per squadra e dal migliore al peggiore)
qualified_report = full_report.sort_values(
	by=['team', 'Total_WPA'], 
	ascending=[True, False] 
).reset_index(drop=True)

# arrotondiamo i valori a 4 cifre decimali
float_cols = qualified_report.select_dtypes(include=['float64']).columns
qualified_report[float_cols] = qualified_report[float_cols].round(4)

csv_path = os.path.join(folder, "player_wpa_report.csv")
qualified_report.to_csv(csv_path, index=False)

# salviamo la rotazione della focus_team
df['current_rotation'] = np.where(
	df['home_team'] == focus_team,
	df['home_setter_position'],
	df['visiting_setter_position']
)

# salviamo i nomi delle squadre per i titoli dei grafici
home_team = df['home_team'].iloc[0]
away_team = df['visiting_team'].iloc[0]

# calcoliamo la Win Probability della focus_team
df['Focus_Team_Win_Prob'] = np.where(
	df['serving_team'] == focus_team,
	df['Serving_Team_Win_Prob'],
	1 - df['Serving_Team_Win_Prob']
)

# fissiamo il tema dei grafici
sns.set_theme(style="darkgrid")

# mappatura valutazioni scout
scout_mapping = {
	'=': -1.0,
	'/': -0.6,
	'-': -0.2,
	'!': 0.2,
	'+': 0.6,
	'#': 1.0
}

# ==============================================================
# GRAPH 1: come la probabilità della focus_team cambia nel tempo
# ==============================================================

# salviamo quanti set sono stati fatti
unique_sets = sorted(df['set_number'].dropna().unique())
num_sets = len(unique_sets)

# basiamo l'altezza del grafico in base al numero di set
fig, axes = plt.subplots(nrows=num_sets, ncols=1, figsize=(30, 5 * num_sets), sharey=True)

# disegnamo il grafico per ogni set
for i, s_num in enumerate(unique_sets):
	ax = axes[i]

	# filtriamo i dati per set
	set_df = df[df['set_number'] == s_num].copy()
	
	# isoliamo le battute per creare le "ghost rows"
	serves_df = set_df[set_df['skill'] == 'Serve'].copy()
	
	# calcoliamo l'impatto della focus team per sottrarlo e trovare la probabilità pre-battuta
	focus_wpa = np.where(serves_df['team'] == focus_team, serves_df['Player_WPA'], -serves_df['Player_WPA'])
	serves_df['Focus_Team_Win_Prob'] = serves_df['Focus_Team_Win_Prob'] - focus_wpa
	
	# assegniamo un indice frazionario per posizionare la ghost row appena prima della vera battuta
	serves_df['sort_idx'] = serves_df.index - 0.5
	set_df['sort_idx'] = set_df.index
	
	# uniamo, ordiniamo e resettiamo l'indice
	set_df = pd.concat([set_df, serves_df]).sort_values('sort_idx').reset_index(drop=True)

	set_df['set_touch_index'] = range(len(set_df))

	# disegniamo la curva iterando sui rally
	for r_id, rally_data in set_df.groupby('rally_number'):
		# estraiamo il punteggio all'inizio del rally
		first_row = rally_data.iloc[0]
		is_home = (focus_team == first_row['home_team'])
		
		focus_score = first_row['actual_start_home_score'] if is_home else first_row['actual_start_visiting_score']
		opp_score = first_row['actual_start_visiting_score'] if is_home else first_row['actual_start_home_score']
		
		# determiniamo il colore: rosso se sta perdendo, blu se sta vincendo, grigio se sta pareggiando
		if focus_score < opp_score:
			line_color = 'red'
		elif focus_score == opp_score:
			line_color = 'gray'
		else:
			line_color = 'blue'
		
		ax.plot(rally_data['set_touch_index'], rally_data['Focus_Team_Win_Prob'], color=line_color, linewidth=2)

	# disegniamo la baseline a 50%
	ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.7)  

	# denormalizziamo il numero di set
	display_set = get_display_set(s_num)

	ax.set_title(f"Set {display_set}", fontsize=14, fontweight='bold')
	ax.set_xlabel(f"Touches in Set {display_set}", fontsize=10)

	# posizioniamo i pallini a fine rally (verde se la squadra vince il rally, rosso se lo perde)
	last_touches = set_df.drop_duplicates(subset=['rally_number'], keep='last').copy()
	# estraiamo i punteggi di partenza di entrambe le squadre
	focus_start = np.where(
		last_touches['home_team'] == focus_team, 
		last_touches['actual_start_home_score'], 
		last_touches['actual_start_visiting_score']
	)
	opp_start = np.where(
		last_touches['home_team'] == focus_team, 
		last_touches['actual_start_visiting_score'], 
		last_touches['actual_start_home_score']
	)
	# verifichiamo se il punteggio della focus_team è aumentato nel rally successivo (shift)
	focus_won = pd.Series(focus_start).shift(-1) > pd.Series(focus_start)
	# gestiamo l'ultimo punto del set: chi era in vantaggio prima dell'ultima battuta ha matematicamente vinto il set
	focus_won.iloc[-1] = (focus_start[-1] > opp_start[-1])
	# assegniamo i colori
	point_colors = np.where(focus_won, 'green', 'red')
	
	ax.scatter(
		last_touches['set_touch_index'], 
		last_touches['Focus_Team_Win_Prob'], 
		color=point_colors, 
		edgecolor='black', 
		s=50, 
		zorder=5
	)
	
	# etichette asse X posizionate a inizio rally
	first_touches = set_df.drop_duplicates(subset=['rally_number'], keep='first').copy()
	first_touches['tick_label'] = first_touches['actual_start_home_score'].astype(str) + "-" + first_touches['actual_start_visiting_score'].astype(str)
	# aggiungiamo il punteggio finale considerando che chi era in vantaggio al penultimo punto ha vinto il set
	last_home_start = first_touches['actual_start_home_score'].iloc[-1]
	last_visit_start = first_touches['actual_start_visiting_score'].iloc[-1]
	if last_home_start > last_visit_start:
		final_score_label = f"{last_home_start + 1}-{last_visit_start}"
	else:
		final_score_label = f"{last_home_start}-{last_visit_start + 1}"
	# recuperiamo la coordinata X esatta dell'ultimo tocco del set
	final_touch_index = last_touches['set_touch_index'].iloc[-1]
	# combiniamo gli indici e le etichette normali con l'indice e l'etichetta finale
	x_ticks = first_touches['set_touch_index'].tolist() + [final_touch_index]
	x_labels = first_touches['tick_label'].tolist() + [final_score_label]

	ax.set_xticks(x_ticks)
	ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
	ax.set_ylabel(f"Win Probability", fontsize=10)
	ax.set_ylim(-0.05, 1.05)
	ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
	ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])

	# aggiorniamo la legenda per riflettere i nuovi colori
	legend_elements = [
		mlines.Line2D([0], [0], color='blue', lw=2, label=f'{focus_team} Winning'),
		mlines.Line2D([0], [0], color='red', lw=2, label=f'{focus_team} Losing'),
		mlines.Line2D([0], [0], color='gray', lw=2, label=f'Tied Score'),
		mlines.Line2D([0], [0], marker='o', color='w', label='Point Won', 
			markerfacecolor='green', markeredgecolor='black', markersize=8),
		mlines.Line2D([0], [0], marker='o', color='w', label='Point Lost', 
			markerfacecolor='red', markeredgecolor='black', markersize=8)
	]
	ax.legend(handles=legend_elements, loc='upper left', fontsize=10, facecolor='white', framealpha=0.9)

# titolo dell'immagine
fig.suptitle(f"{home_team} vs {away_team}", fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
save_and_close_plot(fig, folder, "", "match_prob_by_set.png")

# =========================================================================================
# GRAPH 2 & 3: Impatto di un singolo giocatore (Bar Chart) & Pattern Heatmap durante i set
# =========================================================================================

# filtriamo per la focus_team
team_report = qualified_report[qualified_report['team'] == focus_team].reset_index(drop=True)
n_players = team_report.shape[0]

for k in range(n_players):
	mvp_row = team_report.iloc[k]
	target_player = mvp_row['player_name']

	# isoliamo i dati dello specifico giocatore
	player_df = df[(df['player_name'] == target_player)].copy()
	unique_sets = sorted(player_df['set_number'].dropna().unique())
	num_sets = len(unique_sets)
    
	# inizializziamo entrambe le figure (Graph 2 e Graph 3)
	fig1, axes1 = plt.subplots(nrows=num_sets, ncols=1, figsize=(15, 5 * num_sets), sharey=True, squeeze=False)
	
	n_rows = (num_sets + 1) // 2
	fig2, axes2 = plt.subplots(nrows=n_rows, ncols=2, figsize=(34, 12 * n_rows), squeeze=False)
	
	player_df['eval_numeric'] = player_df['evaluation_code'].map(scout_mapping)

	# ciclo singolo per i set che gestisce entrambi i grafici
	for i, s_num in enumerate(unique_sets):
		set_df = player_df[player_df['set_number'] == s_num].copy()
		display_set = get_display_set(s_num)
		
		# ------------------------
		# GRAPH 2: Grafico a barre
		# ------------------------
		ax1 = axes1[i, 0]
		
		set_df['player_touch_sequence'] = range(1, len(set_df) + 1) 
		
		set_df['score_label'] = np.where(
			focus_team == set_df['home_team'],
			set_df['actual_start_home_score'].astype(str) + "-" + set_df['actual_start_visiting_score'].astype(str),
			set_df['actual_start_visiting_score'].astype(str) + "-" + set_df['actual_start_home_score'].astype(str)
		)

		set_df['start_score_diff'] = np.where(
			focus_team == set_df['home_team'],
			set_df['actual_start_home_score'] - set_df['actual_start_visiting_score'],
			set_df['actual_start_visiting_score'] - set_df['actual_start_home_score']
		)

		conditions = [
			set_df['start_score_diff'] > 0,   
			set_df['start_score_diff'] == 0,  
			set_df['start_score_diff'] < 0    
		]
		choices = ['blue', 'grey', 'red']
		set_df['bar_color'] = np.select(conditions, choices, default='grey')

		ax1.bar(
			set_df['player_touch_sequence'], 
			set_df['Player_WPA'], 
			color=set_df['bar_color'], 
			edgecolor='black', 
			linewidth=0.5,
			zorder=3
		)
		
		ax1_twin = ax1.twinx()
		ax1_twin.grid(False)
		ax1_twin.set_ylim(-1.2, 1.2)
		ax1_twin.set_yticks(list(scout_mapping.values()))
		ax1_twin.set_yticklabels(list(scout_mapping.keys()), fontsize=12, fontweight='bold')
		ax1_twin.set_ylabel("Scout Grade", fontsize=10, rotation=270, labelpad=15)
		ax1_twin.scatter(
			set_df['player_touch_sequence'],
			set_df['eval_numeric'],
			marker='o', color='gold', edgecolor='black', s=60, zorder=5,
		)
		
		skill_abbr = {'Serve': 'Srv', 'Reception': 'Rec', 'Attack': 'Att', 'Block': 'Blk', 'Dig': 'Dig', 'Set': 'Set', 'Freeball': 'Frb'}
		for idx, row in set_df.iterrows():
			skill_text = skill_abbr.get(row['skill'], str(row['skill'])[:3])
			wpa_val = row['Player_WPA']
			y_offset = 0.015 if wpa_val >= 0 else -0.015
			v_align = 'bottom' if wpa_val >= 0 else 'top'
			
			ax1_twin.text(
				row['player_touch_sequence'],
				wpa_val + y_offset,
				skill_text,
				ha='center',
				va=v_align,
				fontsize=8,
				rotation=90,
				color='black',
				zorder=6,
				bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
				transform=ax1.transData
			)

		global_ymin = player_df['Player_WPA'].min()
		global_ymax = player_df['Player_WPA'].max()
		ax1.set_ylim(global_ymin - 0.15, global_ymax + 0.15)
		
		ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=4)
		
		legend_elements1 = [
			mpatches.Patch(facecolor='blue', edgecolor='black', label='Advantage (Winning)'),
			mpatches.Patch(facecolor='grey', edgecolor='black', label='Tie Score'),
			mpatches.Patch(facecolor='red', edgecolor='black', label='Disadvantage (Losing)'),
			mlines.Line2D([0], [0], marker='o', color='w', label='Scout Grade', 
				markerfacecolor='gold', markeredgecolor='black', markersize=8)
		]
		ax1.legend(handles=legend_elements1, loc='upper right', fontsize=10, facecolor='white', framealpha=0.9)

		ax1.set_title(f"Set {display_set}", fontsize=14, fontweight='bold')
		ax1.set_xlabel("Score at the start of the rally", fontsize=10)
		ax1.set_ylabel("Touch Impact (WPA)", fontsize=10)

		touches = set_df['player_touch_sequence'].values 
		scores = set_df['score_label'].values 
		rally_centers = []
		rally_scores = []

		start_x = touches[0] - 0.5
		current_score = scores[0]
		for j in range(1, len(touches)):
			if scores[j] != current_score:
				end_x = touches[j] - 0.5
				rally_centers.append((start_x + end_x) / 2)
				rally_scores.append(current_score)
				ax1.axvline(x=end_x, color='gray', linestyle=':', alpha=0.7, zorder=1)
				start_x = end_x
				current_score = scores[j]

		end_x = touches[-1] + 0.5
		rally_centers.append((start_x + end_x) / 2)
		rally_scores.append(current_score)
		ax1.set_xticks(rally_centers)
		ax1.set_xticklabels(rally_scores, rotation=45, ha='center', fontsize=9)

		total_set_touches = len(set_df)
		good_set_touches = (set_df['Player_WPA'] > 0).sum()
		neutral_set_touches = (set_df['Player_WPA'] == 0).sum()
		bad_set_touches = (set_df['Player_WPA'] < 0).sum()

		good_pct = (good_set_touches / total_set_touches) * 100
		neutral_pct = (neutral_set_touches / total_set_touches) * 100
		bad_pct = (bad_set_touches / total_set_touches) * 100
		final_wpa = set_df['Player_WPA'].sum() 

		display_text = f"Set WPA: {final_wpa:+.4f}\nGood Touches: {good_pct:.1f}%, {good_set_touches}/{total_set_touches}\nNeutral Touches: {neutral_pct:.1f}%, {neutral_set_touches}/{total_set_touches}\nBad Touches: {bad_pct:.1f}%, {bad_set_touches}/{total_set_touches}"

		ax1.text(0.02, 0.95, display_text, 
			transform=ax1.transAxes, fontsize=12, fontweight='bold',
			bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
			verticalalignment='top'
		)

		ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
		ax1.grid(False, axis='x')
		ax1.set_xlim(touches[0] - 0.5, touches[-1] + 0.5)

		# ------------------------
		# GRAPH 3: Pattern Heatmap
		# ------------------------
		r_idx = i // 2
		c_idx = i % 2
		ax2 = axes2[r_idx, c_idx]
		
		set_df['good_evals'] = set_df['evaluation_code'].where(player_df['Is_Good_Touch'])
		set_df['bad_evals'] = set_df['evaluation_code'].where(player_df['Is_Bad_Touch'])
		
		pattern_mean = set_df.pivot_table(index='skill', columns='current_rotation', values='Player_WPA', aggfunc='mean', fill_value=np.nan, dropna=False)
		pattern_sum = set_df.pivot_table(index='skill', columns='current_rotation', values='Player_WPA', aggfunc='sum', fill_value=np.nan, dropna=False)
		pattern_std = set_df.pivot_table(index='skill', columns='current_rotation', values='Player_WPA', aggfunc='std', fill_value=np.nan, dropna=False)		
		pattern_count = set_df.pivot_table(index='skill', columns='current_rotation', values='Player_WPA', aggfunc='count', fill_value=0, dropna=False)
		pattern_good_pct = set_df.pivot_table(index='skill', columns='current_rotation', values='Is_Good_Touch', aggfunc='mean', fill_value=np.nan, dropna=False) * 100  
		pattern_good_count = set_df.pivot_table(index='skill', columns='current_rotation', values='Is_Good_Touch', aggfunc='sum', fill_value=np.nan, dropna=False) 
		pattern_good_grades = set_df.pivot_table(index='skill', columns='current_rotation', values='good_evals', aggfunc=get_grade_dist).reindex(index=pattern_mean.index, columns=pattern_mean.columns, fill_value="")
		pattern_bad_pct = set_df.pivot_table(index='skill', columns='current_rotation', values='Is_Bad_Touch', aggfunc='mean', fill_value=np.nan, dropna=False) * 100  
		pattern_bad_count = set_df.pivot_table(index='skill', columns='current_rotation', values='Is_Bad_Touch', aggfunc='sum', fill_value=np.nan, dropna=False)
		pattern_bad_grades = set_df.pivot_table(index='skill', columns='current_rotation', values='bad_evals', aggfunc=get_grade_dist).reindex(index=pattern_mean.index, columns=pattern_mean.columns, fill_value="")

		if not pattern_mean.empty:
			annot_labels = np.empty_like(pattern_mean, dtype=object) 
			for r in range(pattern_mean.shape[0]):
				for c in range(pattern_mean.shape[1]):
					mean_val = pattern_mean.iloc[r, c]
					tot_val = pattern_sum.iloc[r, c]
					std_val = pattern_std.iloc[r, c]
					g_grd_str = pattern_good_grades.iloc[r, c]
					b_grd_str = pattern_bad_grades.iloc[r, c]
					count_val = pattern_count.iloc[r, c]
					good_pct = pattern_good_pct.iloc[r, c]
					good_n = pattern_good_count.iloc[r, c]
					bad_pct = pattern_bad_pct.iloc[r, c]
					bad_n = pattern_bad_count.iloc[r, c]

					if pd.isna(std_val) and not pd.isna(mean_val):
						std_val = 0.0
					if pd.isna(mean_val):
						annot_labels[r, c] = ""
					else:
						g_grd_text = f" {g_grd_str}" if g_grd_str else ""
						b_grd_text = f" {b_grd_str}" if b_grd_str else ""
						annot_labels[r, c] = f"tot WPA={tot_val:+.3f} \n avg={mean_val:+.3f} (±{std_val:.3f}) \n\n gt={good_pct:.1f}% ({int(good_n)}/{int(count_val)}) \n {g_grd_text}\n\n bt={bad_pct:.1f}% ({int(bad_n)}/{int(count_val)}) \n {b_grd_text}"

			sns.heatmap(
				pattern_sum, 
				annot=annot_labels, 
				fmt="",             
				cmap="RdYlGn",       
				center=0,            
				linewidths=1, 
				linecolor='black',
				ax=ax2
			)
	    
		ax2.set_title(f"Set {display_set}", fontsize=16, fontweight='bold')
		ax2.set_xlabel("Team Setter Position", fontsize=12)
		ax2.set_ylabel("Skill", fontsize=12)

	fig1.suptitle(f"{target_player} ({focus_team}) \n {home_team} vs {away_team}", fontsize=20, fontweight='bold', y=1.02)
	fig1.tight_layout()
	save_and_close_plot(fig1, folder, "player_impact", f"{target_player}.png")
	
	# puliamo i subplot vuoti per il Graph 3
	for j in range(num_sets, n_rows * 2):
		r_empty = j // 2
		c_empty = j % 2
		fig2.delaxes(axes2[r_empty, c_empty])

	fig2.suptitle(f"{target_player}", fontsize=20, fontweight='bold', y=1.02)
	fig2.tight_layout()
	save_and_close_plot(fig2, folder, "heatmap", f"{target_player}.png")

# =========================================================================================
# GRAPH 4 & 5: Dashboard della squadra (per set) & Efficienza vs WPA Impact (per Match)
# =========================================================================================

for current_skill in skills:
	# --------------------------
	# GRAPH 4: Dashboard per Set
	# --------------------------
	for s_num in unique_sets:
		set_df = df[(df['set_number'] == s_num) & (df['team'] == focus_team) & (df['skill'] == current_skill)].copy()

		if set_df.empty or len(set_df) < 3:
			continue

		# vettorializzazione del calcolo metriche tramite pandas groupby
		is_critical = set_df['critical_moments'].astype(bool)
		set_df['HL_WPA_val'] = np.where(is_critical, set_df['Player_WPA'], 0.0)
		set_df['Norm_WPA_val'] = np.where(~is_critical, set_df['Player_WPA'], 0.0)
		set_df['Is_HL_Touch'] = np.where(is_critical, 1, 0)
		set_df['Is_Norm_Touch'] = np.where(~is_critical, 1, 0)
		
		metrics_df = set_df.groupby('player_name').agg(
			Touches=('Player_WPA', 'count'),
			Total_WPA=('Player_WPA', 'sum'),
			Volatility=('Player_WPA', 'std'),
			HL_WPA=('HL_WPA_val', 'sum'),
			HL_Touches=('Is_HL_Touch', 'sum'),
			Norm_WPA=('Norm_WPA_val', 'sum'),
			Norm_Touches=('Is_Norm_Touch', 'sum')
		).reset_index()
		
		# eliminiamo anomalie di chi ha giocato 1 solo pallone
		metrics_df = metrics_df[metrics_df['Touches'] >= 2].copy()
		if metrics_df.empty or len(metrics_df) < 2:
			continue

		metrics_df['WPA_per_Touch'] = metrics_df['Total_WPA'] / metrics_df['Touches']
		metrics_df['Volatility'] = metrics_df['Volatility'].fillna(0.0)
		metrics_df['HL_WPA_Touch'] = np.where(metrics_df['HL_Touches'] > 0, metrics_df['HL_WPA'] / metrics_df['HL_Touches'], 0.0)
		metrics_df['Norm_WPA_Touch'] = np.where(metrics_df['Norm_Touches'] > 0, metrics_df['Norm_WPA'] / metrics_df['Norm_Touches'], 0.0)

		metrics_df = metrics_df.sort_values(by = 'Total_WPA', ascending = True).reset_index(drop=True)
		display_set = get_display_set(s_num)
		
		fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (24, 7))

		# PANEL 1: WPA/T vs TOTAL WPA
		ax1 = axes[0]
		y_pos = np.arange(len(metrics_df))

		ax1.barh(
			y_pos,
			metrics_df['Total_WPA'],
			color = 'royalblue',
			alpha = 0.3,
			edgecolor = 'blue',
			height = 0.6,
			label = 'Total WPA',
		)

		colors = np.where(metrics_df['WPA_per_Touch'] >= 0, 'forestgreen', 'crimson')
		ax1.barh(
			y_pos,
			metrics_df['WPA_per_Touch'],
			color = colors,
			alpha = 0.9,
			height = 0.3,
			label = 'WPA / Touch',
		)

		for idx_num, (original_index, row) in enumerate(metrics_df.iterrows()):
			wpa_t = row['WPA_per_Touch']
			touches = int(row['Touches'])

			if wpa_t >= 0:
				align = 'left'
				text_string = f"   n = {touches}"
			else:
				align = 'right'
				text_string = f"n = {touches}   "

			ax1.text(
				wpa_t, 
				idx_num, 
				text_string, 
				va = 'center', 
				ha = align, 
				fontsize = 9, 
				fontweight = 'bold', 
				color = 'black',
				zorder = 5
			)
		
		apply_axis_padding(ax1, 'x')

		ax1.set_yticks(y_pos)
		ax1.set_yticklabels(metrics_df['player_name'], fontsize = 10, fontweight = 'bold')
		ax1.axvline(0, color = 'black', linewidth = 1, linestyle = '--', zorder = 3)
		ax1.set_title('WPA/Touch vs Total WPA', fontsize = 12, fontweight = 'bold')
		ax1.set_xlabel('WPA', fontsize = 10)
		ax1.legend(loc = 'lower right', fontsize = 8, framealpha = 0.9)
		ax1.grid(True, linestyle = ':', alpha = 0.5)

		# PANEL 2: Total WPA vs Volatility
		ax2 = axes[1]
		ax2.scatter(
			metrics_df['Volatility'],
			metrics_df['Total_WPA'],
			color = 'darkorange',
			edgecolor = 'black',
			s = 100,
			zorder = 5,
		)
		
		annotate_and_adjust(ax2, metrics_df, 'Volatility', 'Total_WPA', 'player_name', 'Touches')

		med_vol = metrics_df['Volatility'].mean()
		ax2.axvline(med_vol, color = 'red', linestyle = '--', alpha = 0.5, zorder = 3)
		ax2.axhline(0.0, color = 'red', linestyle = '--', linewidth = 0.5, zorder = 3)

		add_quadrant_labels(
			ax2, 
			'High Vol, High WPA', 'Low Vol, High WPA', 
			'High Vol, Low WPA', 'Low Vol, Low WPA', 
			fontsize=9, alpha=0.3
		)
		
		apply_axis_padding(ax2, 'both')

		ax2.set_title('WPA Volatility vs Total WPA', fontsize = 12, fontweight = 'bold')
		ax2.set_xlabel('WPA Volatility', fontsize = 10)
		ax2.set_ylabel('Total WPA', fontsize = 10)
		ax2.grid(True, linestyle = ':', alpha = 0.5)

		# PANEL 3: NORMAL vs HIGH-LEVERAGE
		ax3 = axes[2]
		
		ax3.barh(
			y_pos,
			metrics_df['HL_WPA'], 
			color='crimson',
			alpha=0.3,
			edgecolor='darkred',
			height=0.6,
			zorder=1
		)			
		ax3.barh(
			y_pos,
			metrics_df['Norm_WPA'], 
			color='royalblue',
			alpha=0.3,
			edgecolor='blue',
			height=0.6,
			zorder=1
		)
		
		for i, row in metrics_df.iterrows():
			y = y_pos[metrics_df.index.get_loc(i)]
			n_val = row['Norm_WPA_Touch']
			n_count = int(row['Norm_Touches'])
			h_val = row['HL_WPA_Touch']
			h_count = int(row['HL_Touches'])

			if n_count > 0 and h_count > 0:
				ax3.plot([n_val, h_val], [y, y], color = 'gray', linestyle = '-', linewidth = 1.5)
			elif h_count == 0: 
				ax3.plot([n_val, 0], [y, y], color = 'gray', linestyle = '-', linewidth = 1.5)
			elif n_count == 0:
				ax3.plot([0, h_val], [y, y], color = 'gray', linestyle = '-', linewidth = 1.5)
			
			if n_count > 0: 
				ax3.scatter(n_val, y, color = 'blue', edgecolor = 'black', s = 70, zorder = 5)
			if h_count > 0:
				ax3.scatter(h_val, y, color = 'red', edgecolor = 'black', s = 90, zorder = 5)
		
			if n_val <= h_val:
				n_align, n_text = 'right', f"n = {n_count}   "
				h_align, h_text = 'left', f"   n = {h_count}"
			else:
				n_align, n_text = 'left', f"   n = {n_count}"
				h_align, h_text = 'right', f"n = {h_count}   "

			if n_count > 0:
				ax3.text(n_val, y, n_text, va = 'center', ha = n_align, fontsize = 9, fontweight = 'bold', color = 'black', zorder = 6)
			if h_count > 0:
				ax3.text(h_val, y, h_text, va = 'center', ha = h_align, fontsize = 9, fontweight = 'bold', color = 'black', zorder = 6)
				
		apply_axis_padding(ax3, 'x')
		
		ax3.set_yticks(y_pos)
		ax3.set_yticklabels(metrics_df['player_name'], fontsize = 10, fontweight = 'bold')
		ax3.axvline(0, color = 'black', linewidth = 1, linestyle = '--')
		ax3.set_title('Normal vs High Leverage', fontsize = 12, fontweight = 'bold')
		ax3.set_xlabel('WPA', fontsize=10)

		legend_elements = [
			mlines.Line2D([0], [0], marker = 'o', color = 'w', label = 'Normal Situations WPA / Touch', markerfacecolor = 'blue', markeredgecolor = 'black', markersize = 8),
			mlines.Line2D([0], [0], marker = 'o', color = 'w', label = 'High Leverage WPA / Touch', markerfacecolor = 'red', markeredgecolor = 'black', markersize = 8),
			mpatches.Patch(facecolor='royalblue', edgecolor='blue', alpha=0.3, label='Normal Situtations Total WPA'),
			mpatches.Patch(facecolor='crimson', edgecolor='darkred', alpha=0.3, label='High Leverage Total WPA')
		]
		ax3.legend(handles = legend_elements, loc = 'lower right', fontsize = 8, framealpha = 0.9)
		ax3.grid(True, linestyle = ':', alpha = 0.5)

		fig.suptitle(f"{focus_team} — Set {display_set} \n {current_skill} \n {home_team} vs {away_team}", fontsize = 16, fontweight = 'bold', y = 1.03)
		plt.tight_layout()

		save_and_close_plot(fig, folder, "set_dashboards", f"Set_{display_set}_{current_skill}.png")

	# ----------------------------------------
	# GRAPH 5: Efficienza vs WPA (Total Match)
	# ----------------------------------------
	team_skill_df = df[(df['team'] == focus_team) & (df['skill'] == current_skill)].copy()
	
	player_comparison = team_skill_df.groupby('player_name').agg(
		Total_Touches = ('evaluation_code', 'count'),
		Code_Hash = ('evaluation_code', lambda x: (x == '#').sum()), 
		Code_Plus = ('evaluation_code', lambda x: (x == '+').sum()), 
		Code_Esclamation = ('evaluation_code', lambda x: (x == '!').sum()), 
		Code_Minus = ('evaluation_code', lambda x: (x == '-').sum()), 
		Code_Slash = ('evaluation_code', lambda x: (x == '/').sum()),
		Code_Equals = ('evaluation_code', lambda x: (x == '=').sum()),
		Total_WPA=('Player_WPA', 'sum')
	).reset_index()
		
	# calcolo vettorializzato dell'efficienza
	numerator = player_comparison['Code_Hash'] + player_comparison['Code_Plus'] - player_comparison['Code_Equals']
	if current_skill == 'Serve':
		numerator += player_comparison['Code_Slash']
	else:
		numerator -= player_comparison['Code_Slash']
	
	player_comparison['Efficiency'] = (numerator / player_comparison['Total_Touches']).fillna(0.0)
	
	fig5, ax5 = plt.subplots(figsize = (12, 8))

	ax5.scatter(
		player_comparison['Efficiency'], 
		player_comparison['Total_WPA'], 
		color = 'blue', 
		s = 120, 
		edgecolor='black', 
		zorder = 5
	)
	
	annotate_and_adjust(ax5, player_comparison, 'Efficiency', 'Total_WPA', 'player_name', 'Total_Touches')

	baseline_eff = player_comparison['Efficiency'].mean()
	ax5.axvline(x = baseline_eff, color = 'red', linestyle = '--', alpha = 0.5, zorder = 3)
	ax5.axhline(y = 0, color = 'red', linestyle = '--', alpha = 0.5, zorder = 3)

	add_quadrant_labels(
		ax5, 
		'High Eff, High WPA', 'Low Eff, High WPA', 
		'High Eff, Low WPA', 'Low Eff, Low WPA', 
		fontsize=14, alpha=0.2
	)

	ax5.set_title(f"{current_skill} \n Efficiency vs WPA Impact \n {focus_team}", fontsize = 16, fontweight = 'bold', pad = 15)
	ax5.set_xlabel(f"Efficiency", fontsize = 12)
	ax5.set_ylabel(f"WPA", fontsize = 12)

	ax5.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
	ax5.grid(True, linestyle = ':', alpha = 0.6, zorder = 0)
	
	apply_axis_padding(ax5, 'both')

	plt.tight_layout()
	save_and_close_plot(fig5, folder, "Efficiency_vs_WPA", f"{current_skill}.png")
