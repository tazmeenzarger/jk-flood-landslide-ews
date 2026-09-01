# src/train_ls_temporal_model.py — Landslide temporal model (final)
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score

df = pd.read_csv(r"data\processed\temporal_ls_matrix.csv")
df['date'] = pd.to_datetime(df['date'])
feature_cols = [c for c in df.columns if c not in ('date', 'is_ls')]

train = df[df['date'] < '2016-01-01']
val   = df[(df['date'] >= '2016-01-01') & (df['date'] < '2019-01-01')]
test  = df[df['date'] >= '2019-01-01']
Xtr, ytr = train[feature_cols], train['is_ls']
Xva, yva = val[feature_cols], val['is_ls']
Xte, yte = test[feature_cols], test['is_ls']
print(f"Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)} | Test alert days {yte.sum()}")

spw = round((ytr == 0).sum() / max((ytr == 1).sum(), 1), 2)
print(f"Class balance -> scale_pos_weight: {spw}")
model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                      scale_pos_weight=spw, eval_metric='logloss', random_state=42)
model.fit(Xtr, ytr)

probs_te = model.predict_proba(Xte)[:, 1]
print(f"ROC-AUC: {roc_auc_score(yte, probs_te):.3f}")

probs_va = model.predict_proba(Xva)[:, 1]
best_thr, best_f2 = 0.5, 0.0
for thr in np.arange(0.10, 0.60, 0.01):
    f2 = fbeta_score(yva, (probs_va >= thr).astype(int), beta=2)
    if f2 > best_f2:
        best_f2, best_thr = f2, thr
print(f"Optimal threshold (F2 on val): {best_thr:.2f}")

pred = (probs_te >= best_thr).astype(int)
cm = confusion_matrix(yte, pred)
print(cm)
print(f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.2%} | FAR: {cm[0,1]/(cm[0,0]+cm[0,1]):.2%}")

joblib.dump(model, r"data\processed\evaluations_temporal\temporal_ls_XGBoost.joblib")
joblib.dump({'threshold': float(best_thr), 'features': feature_cols},
            r"data\processed\evaluations_temporal\temporal_ls_config.joblib")
print("Saved landslide temporal model + config.")