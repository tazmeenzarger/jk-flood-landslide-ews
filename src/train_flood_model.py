# src/train_flood_model.py — Phase 3B.4: Flood Model Training & Feature Importance
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import joblib

DATA_PATH = r"data\processed\features_flood_final.csv"
OUT_DIR = r"data\processed\evaluations_flood"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading flood matrix...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["is_flood"])
y = df["is_flood"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight=3, eval_metric='logloss', random_state=42, use_label_encoder=False)
}

plt.figure(figsize=(10, 8))
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Flood']))
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    joblib.dump(model, os.path.join(OUT_DIR, f"flood_{name}.joblib"))

joblib.dump(scaler, os.path.join(OUT_DIR, "flood_scaler.joblib"))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Flood Model ROC Curves')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "flood_roc_curves.png"))

# Feature Importance for RF
rf_model = models["Random_Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Flood Random Forest)")
plt.bar(range(len(X.columns)), importances[indices], align="center")
plt.xticks(range(len(X.columns)), [X.columns[i] for i in indices], rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "flood_feature_importance.png"))

print("\n=== TOP 5 FLOOD DRIVERS ===")
for i in range(5):
    idx = indices[i]
    print(f"{i+1}. {X.columns[idx]} : {importances[idx]:.3f}")

print("\nFlood Model Training Complete!")