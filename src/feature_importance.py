# src/feature_importance.py — Phase 3.5: Model Interpretability
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

SPLITS_DIR = r"data\processed\ml_splits"
EVAL_DIR = r"data\processed\evaluations"

# 1. Load the winning model and feature names
model_path = os.path.join(EVAL_DIR, "Random_Forest.joblib")
names_path = os.path.join(SPLITS_DIR, "feature_names.txt")

print(f"Loading model: {model_path}")
model = joblib.load(model_path)

with open(names_path) as f:
    feature_names = [line.strip() for line in f]

# 2. Extract Importances
importances = model.feature_importances_

# 3. Sort and Plot
indices = np.argsort(importances)[::-1] # Sort descending

plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(len(feature_names)), importances[indices], align="center")
plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha="right")
plt.xlim([-1, len(feature_names)])
plt.ylabel("Gini Importance")
plt.tight_layout()

out_path = os.path.join(EVAL_DIR, "feature_importance.png")
plt.savefig(out_path)
print(f"Saved plot to: {out_path}")

# Print the top 5 features to the console
print("\n=== TOP 5 DRIVERS ===")
for i in range(5):
    idx = indices[i]
    print(f"{i+1}. {feature_names[idx]} : {importances[idx]:.3f}")