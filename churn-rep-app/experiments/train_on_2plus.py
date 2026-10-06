"""
EXPERIMENT -- Does adding 2-order accounts to TRAINING help?

Builds its own snapshots with a minimum of 2 orders (the pipeline's saved datasets are not
touched), trains on 2+ vs 3+, and tests on the same June 2018 accounts:
the 3+ accounts the pipeline scores today, split by size, plus the 2-order accounts.

Run from the project folder:   python experiments/train_on_2plus.py
"""
import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from step1_load_data import load_tables
from step4_build_datasets import snapshot
from step5_models import MODELS, features_for

config.MIN_ORDERS = 2            # only inside this experiment
t = load_tables()
train = pd.concat([snapshot(t, c)[0] for c in config.TRAIN_CUTOFFS])
test, groups = snapshot(t, config.TEST_CUTOFF)
features = list(groups)

name = "Logistic regression"
cols = features_for(name, features)
rows = []
for k in [2, 3]:
    tr = train[train.n_orders >= k]
    m = MODELS[name]["make"]().fit(tr[cols], tr.churn)
    p = pd.Series(m.predict_proba(test[cols])[:, 1], index=test.index)
    row = {"trained on": f"{k}+ orders" + (" (current)" if k == 3 else ""), "training rows": len(tr)}
    for g, mask in [("2 orders", test.n_orders == 2), ("3-4", test.n_orders.between(3, 4)),
                    ("5-9", test.n_orders.between(5, 9)), ("10+", test.n_orders >= 10),
                    ("all 3+", test.n_orders >= 3)]:
        row[f"AUC {g}"] = roc_auc_score(test.churn[mask], p[mask])
    rows.append(row)

print(f"test accounts: 2 orders = {(test.n_orders == 2).sum():,}, 3+ = {(test.n_orders >= 3).sum():,}")
print(f"actual churn: 2 orders = {test.churn[test.n_orders == 2].mean():.1%}, 3+ = {test.churn[test.n_orders >= 3].mean():.1%}\n")
pd.set_option("display.width", 160)
print(pd.DataFrame(rows).round(3).to_string(index=False))
