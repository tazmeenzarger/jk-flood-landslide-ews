# src/split_and_scale.py — Phase 3.3: Train/Test Splitting & Scaling
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = r"data\processed\features_final.csv"
OUT_DIR = r"data\processed\ml_splits"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading final feature matrix...")
df = pd.read_csv(DATA_PATH)

# 1. Separate Features (X) and Target (y)
X = df.drop(columns=["is_landslide"])
y = df["is_landslide"]

print(f"Features (X): {X.shape[1]} columns")
print(f"Target (y):   {y.shape[0]} rows")

# 2. Stratified Train/Test Split (80% Train, 20% Test)
# stratify=y ensures the 25% landslide ratio is preserved in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== SPLIT STATISTICS ===")
print(f"Train Set: {len(X_train)} rows | Landslides: {y_train.sum()} ({(y_train.mean()*100):.1f}%)")
print(f"Test Set:  {len(X_test)} rows  | Landslides: {y_test.sum()} ({(y_test.mean()*100):.1f}%)")

# 3. Scale the Features
# Fit the scaler ONLY on the training data to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Save Everything
# Save arrays for fast loading in Python
np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train_scaled)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.values)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test_scaled)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test.values)

# Save the scaler itself (needed for the Dashboard later)
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

# Save the feature names (needed to map dashboard inputs to model inputs)
with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
    f.write("\n".join(X.columns))

print(f"\n=== SPLIT & SCALE COMPLETE ===")
print(f"Saved splits and scaler to: {OUT_DIR}")