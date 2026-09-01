# src/train_temporal_model_v2.py — Phase 4.5: Tuned Temporal Model + Event Evaluation
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score

df = pd.read_csv(r"data\processed\temporal_ews_matrix_v2.csv")
df['date'] = pd.to_datetime(df['date'])
feature_cols = [c for c in df.columns if c not in ('date', 'is_flood')]

train = df[df['date'] < '2016-01-01']
val   = df[(df['date'] >= '2016-01-01') & (df['date'] < '2019-01-01')]
test  = df[df['date'] >= '2019-01-01']

Xtr, ytr = train[feature_cols], train['is_flood']
Xva, yva = val[feature_cols], val['is_flood']
Xte, yte = test[feature_cols], test['is_flood']
print(f"Train {len(Xtr)} | Val {len(Xva)} | Test {len(Xte)} days")

# 1. Train XGBoost (primary) and RF (comparison)
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                    scale_pos_weight=8, eval_metric='logloss', random_state=42)
xgb.fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(Xtr, ytr)

probs_te_x = xgb.predict_proba(Xte)[:, 1]
probs_te_r = rf.predict_proba(Xte)[:, 1]
print(f"\nROC-AUC (XGBoost): {roc_auc_score(yte, probs_te_x):.3f}")
print(f"ROC-AUC (RF):      {roc_auc_score(yte, probs_te_r):.3f}")

# 2. Threshold tuning on VALIDATION only (F2 = recall-weighted)
probs_va = xgb.predict_proba(Xva)[:, 1]
best_thr, best_f2 = 0.5, 0.0
for thr in np.arange(0.10, 0.60, 0.01):
    f2 = fbeta_score(yva, (probs_va >= thr).astype(int), beta=2)
    if f2 > best_f2:
        best_f2, best_thr = f2, thr
print(f"\nOptimal decision threshold (F2 on val): {best_thr:.2f}")

# 3. Final test metrics at tuned threshold
pred = (probs_te_x >= best_thr).astype(int)
cm = confusion_matrix(yte, pred)
print("Confusion Matrix (test):")
print(cm)
print(f"Recall (floods caught): {cm[1,1]/(cm[1,0]+cm[1,1]):.2%}")
print(f"False Alarm Rate:       {cm[0,1]/(cm[0,0]+cm[0,1]):.2%}")

# 4. EVENT-BASED evaluation (the operational truth)
te = test.copy(); te['pred'] = pred
inv = pd.read_csv(r"data\processed\j_k_flood_inventory.csv")
inv['Began'] = pd.to_datetime(inv['Began'])
events = inv[inv['Began'] >= '2019-01-01']

detected, leads = 0, []
for _, ev in events.iterrows():
    w = te[(te['date'] >= ev['Began'] - pd.Timedelta(days=7)) & (te['date'] <= ev['Began'])]
    first_alert = w.loc[w['pred'] == 1, 'date']
    if len(first_alert) > 0:
        detected += 1
        leads.append((ev['Began'] - first_alert.iloc[0]).days)

print(f"\nEvent detection: {detected}/{len(events)} test events")
if leads:
    print(f"Mean lead time: {np.mean(leads):.1f} days")

# Alert episodes per year (operational false-alarm burden)
alert_days = te.loc[te['pred'] == 1, 'date']
runs, prev = 0, None
for d in alert_days:
    if prev is None or (d - prev).days > 1:
        runs += 1
    prev = d
years = (te['date'].max() - te['date'].min()).days / 365.25
print(f"Alert episodes per year: {runs/years:.1f}")

# 5. Save model + config for the dashboard
joblib.dump(xgb, r"data\processed\evaluations_temporal\temporal_XGBoost_v2.joblib")
joblib.dump({'threshold': float(best_thr), 'features': feature_cols},
            r"data\processed\evaluations_temporal\temporal_config_v2.joblib")
print("\nSaved v2 model + config.")