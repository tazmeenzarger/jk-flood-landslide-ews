# src/train_models.py — Phase 3.4: Model Training & Evaluation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (classification_report, roc_auc_score, 
                             confusion_matrix, roc_curve)
import joblib

SPLITS_DIR = r"data\processed\ml_splits"
OUT_DIR = r"data\processed\evaluations"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load the data
print("Loading splits...")
X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
X_test = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_test = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

with open(os.path.join(SPLITS_DIR, "feature_names.txt")) as f:
    feature_names = [line.strip() for line in f]

# 2. Define the models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, 
                             scale_pos_weight=3, eval_metric='logloss', random_state=42, use_label_encoder=False)
}

results = []
plt.figure(figsize=(10, 8))

print("\n=== TRAINING AND EVALUATING MODELS ===\n")

for name, model in models.items():
    print(f"--- Training {name} ---")
    model.fit(X_train, y_train)
    
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probability of class 1 (Landslide)
    
    # Calculate Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Background', 'Landslide']))
    
    results.append({"Model": name, "ROC_AUC": roc_auc})
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Background', 'Landslide'], 
                yticklabels=['Background', 'Landslide'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{name} Confusion Matrix')
    plt_cm.tight_layout()
    plt_cm.savefig(os.path.join(OUT_DIR, f"cm_{name}.png"))
    plt.close(plt_cm)
    
    # Save the model for the Dashboard
    joblib.dump(model, os.path.join(OUT_DIR, f"{name}.joblib"))

# 3. Finalize ROC Curve Plot
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Background falsely flagged)')
plt.ylabel('True Positive Rate (Landslides successfully warned)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "roc_curves.png"))
plt.close()

# 4. Summary Table
df_results = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print("\n=== FINAL MODEL RANKING ===")
print(df_results.round(3).to_string(index=False))

print(f"\nSaved models and plots to: {OUT_DIR}")