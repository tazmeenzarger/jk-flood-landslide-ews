# src/train_temporal_model.py — Phase 4.4: Temporal EWS Model Training
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

OUT_DIR = r"data\processed\evaluations_temporal"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading temporal EWS matrix...")
df = pd.read_csv(r"data\processed\temporal_ews_matrix.csv")
df['date'] = pd.to_datetime(df['date'])

# 1. TIME-SERIES AWARE SPLIT (NO DATA LEAKAGE!)
# Train: 2000 to 2018 (18 years of history)
# Test: 2019 to 2023 (5 years of recent history to validate)
train_mask = df['date'] < '2019-01-01'
test_mask = df['date'] >= '2019-01-01'

X_train = df.loc[train_mask].drop(columns=['date', 'is_flood'])
y_train = df.loc[train_mask]['is_flood']

X_test = df.loc[test_mask].drop(columns=['date', 'is_flood'])
y_test = df.loc[test_mask]['is_flood']

print(f"Train set: {len(X_train)} days | Flood days: {y_train.sum()}")
print(f"Test set:  {len(X_test)} days | Flood days: {y_test.sum()}")

# 2. Define Models
# scale_pos_weight=8 tells XGBoost to pay 8x more attention to the rare flood days
models = {
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, scale_pos_weight=8, eval_metric='logloss', random_state=42, use_label_encoder=False)
}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Safe Day', 'Flood Alert Day']))
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"False Alarm Rate (Predicted Flood, but Safe): {cm[0,1] / (cm[0,0] + cm[0,1]):.2%}")
    print(f"Missed Flood Rate (Predicted Safe, but Flooded): {cm[1,0] / (cm[1,0] + cm[1,1]):.2%}")
    
    joblib.dump(model, os.path.join(OUT_DIR, f"temporal_{name}.joblib"))

print("\nTemporal Model Training Complete!")