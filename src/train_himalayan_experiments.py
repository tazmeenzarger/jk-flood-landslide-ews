# src/train_himalayan_experiments.py — Phase 8.3: The 3 Experiments
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, fbeta_score
import warnings
import joblib
warnings.filterwarnings('ignore')

df = pd.read_csv(r"data\processed\himalayan_flood_matrix.csv")
df['date'] = pd.to_datetime(df['date'])
feature_cols = [c for c in df.columns if c not in ('date', 'region', 'is_flood')]

jk_train = df[(df['region'] == 'jk') & (df['date'] < '2016-01-01')]
jk_val   = df[(df['region'] == 'jk') & (df['date'] >= '2016-01-01') & (df['date'] < '2019-01-01')]
jk_test  = df[(df['region'] == 'jk') & (df['date'] >= '2019-01-01')]

other_train = df[(df['region'] != 'jk') & (df['date'] < '2016-01-01')]
other_val   = df[(df['region'] != 'jk') & (df['date'] >= '2016-01-01') & (df['date'] < '2019-01-01')]

X_te, y_te = jk_test[feature_cols], jk_test['is_flood']

def run_exp(name, X_tr, y_tr, X_va, y_va, save_name=None):
    print(f"\n=== Experiment {name} ===")
    if y_tr.sum() == 0:
        print("No positive samples in training set. Skipping.")
        return
    spw = round((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 2)
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          scale_pos_weight=spw, eval_metric='logloss', random_state=42)
    model.fit(X_tr, y_tr)
    
    probs_te = model.predict_proba(X_te)[:, 1]
    if len(np.unique(y_te)) > 1:
        print(f"Test ROC-AUC: {roc_auc_score(y_te, probs_te):.3f}")
    else:
        print("Test set has only one class. AUC undefined.")
        
    probs_va = model.predict_proba(X_va)[:, 1]
    best_thr, best_f2 = 0.5, 0.0
    if len(np.unique(y_va)) > 1:
        for thr in np.arange(0.10, 0.60, 0.01):
            f2 = fbeta_score(y_va, (probs_va >= thr).astype(int), beta=2)
            if f2 > best_f2: best_f2, best_thr = f2, thr
    print(f"Optimal threshold: {best_thr:.2f}")
    
    pred = (probs_te >= best_thr).astype(int)
    cm = confusion_matrix(y_te, pred)
    if cm[1,0] + cm[1,1] > 0: print(f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.2%}")
    if cm[0,0] + cm[0,1] > 0: print(f"FAR: {cm[0,1]/(cm[0,0]+cm[0,1]):.2%}")
    if save_name:
        joblib.dump(model, rf"data\processed\evaluations_temporal\{save_name}.joblib")
        joblib.dump({'threshold': float(best_thr), 'features': feature_cols},
                    rf"data\processed\evaluations_temporal\{save_name}_config.joblib")
        print(f"Saved {save_name}.joblib + config.")

run_exp("(c) J&K Only", jk_train[feature_cols], jk_train['is_flood'], jk_val[feature_cols], jk_val['is_flood'])
run_exp("(a) Pure Transfer", other_train[feature_cols], other_train['is_flood'], other_val[feature_cols], other_val['is_flood'])

pooled_train = pd.concat([jk_train, other_train])
pooled_val = pd.concat([jk_val, other_val])
run_exp("(b) Pooled", pooled_train[feature_cols], pooled_train['is_flood'], pooled_val[feature_cols], pooled_val['is_flood'],
        save_name="temporal_flood_pooled")