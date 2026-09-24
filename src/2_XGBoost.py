# Addestramento del modello XGBoost

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import log_loss

print("Loading dataset into memory...")
df = pd.read_csv("encoded_final_dataset.csv")
print(f"Dataset loaded! Total touches: {len(df)}")

hidden_cols = ['match_id', 'skill', 'evaluation_code', 'prev_touch', 'prev_touch_eval', 'team', 'player_number', 'player_name', 'rally_number', 'custom_possession_id', 'home_team_score', 'visiting_team_score', 'point_won_by', 'serving_team', 'home_team', 'visiting_team', 'home_setter_position', 'visiting_setter_position', 'attack_code', 'set_code', 'set_type', 'original_attack_code', 'original_set_code', 'original_set_type', 'start_zone', 'end_zone', 'touch_sequence', 'actual_start_home_score', 'actual_start_visiting_score']

# identifichiamo le colonne che verranno usaete nel training
training_features = [col for col in df.columns if col not in hidden_cols + ['target']]
joblib.dump(training_features, 'training_columns.pkl')

# separiamo input e target 
y = df['target']
X = df[training_features]

train_size = int(len(df) * 0.80)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]
print(f"Training on {len(X_train)} rows. Validating on {len(X_test)} rows.")

print("Starting the XGBoost Engine")

model = XGBClassifier(
	n_estimators=500,        # n_estimators: numero di alberi
	max_depth=6,             # max_depth: profondità massima degli alberi (evita l'overfitting)
	learning_rate=0.05,      # step cauti del 5% steps per evitare over-correcting
	subsample=0.8,           # guarda solo all'80% dei dati per ogni albero (aggiunge altra randomicità)
	n_jobs=-1,               # n_jobs=-1: fa si che il pc usi il 100% della cpu
	random_state=42		 # random_state=42: assicura che si ottengano gi stessi risultati ogni volta
)
model.fit(X_train, y_train)

print("\n--- RESULTS ---")
y_pred_probs = model.predict_proba(X_test)
final_log_loss = log_loss(y_test, y_pred_probs)
print("Training Accuracy:", round(model.score(X_train, y_train), 4))
print("Validation Accuracy:", round(model.score(X_test, y_test), 4))
print("Log Loss:", round(final_log_loss, 4))

# Pesi 
feature_names = X.columns
importances = model.feature_importances_
importance_df = pd.DataFrame({
	'Feature': feature_names,
	'Importance': importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv("xg_feature_importances.csv", index=False)

# Probabilità
wpa_df = df.iloc[train_size:].copy()
wpa_df['Serving_Team_Win_Prob'] = y_pred_probs[:, 1]

wpa_df['Probability_Shift'] = wpa_df.groupby(['match_id', 'rally_number'])['Serving_Team_Win_Prob'].diff()
wpa_df['Probability_Shift'] = wpa_df['Probability_Shift'].fillna(wpa_df['Serving_Team_Win_Prob'])

output_columns = ['match_id', 'set_number', 'rally_number', 'home_team', 'serving_team', 'point_won_by', 'home_team_score', 'visiting_team_score', 'team', 'player_number', 'skill', 'evaluation_code', 'target', 'Serving_Team_Win_Prob', 'Probability_Shift']

clean_wpa_df = wpa_df[output_columns]
clean_wpa_df.to_csv("xg_win_prob.csv", index=False)

joblib.dump(model, 'volleyball_xg_model.pkl')
