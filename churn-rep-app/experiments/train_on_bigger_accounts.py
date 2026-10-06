"""
EXPERIMENT -- Train only on bigger accounts (k+ orders), then predict the smaller ones too?

Idea: churn labels of big accounts are "cleaner" (their silence means something), so a model
that learns from them might rank the small, noisy accounts better.

Run from the project folder:   python experiments/train_on_10plus.py

Compares, on the June 2018 test period:
    A  trained on 10+ order accounts only
    B  trained on all 3+ order accounts (current pipeline)
and reports, separately for 3-9 and 10+ accounts:
    AUC                ranking quality (what the worklist needs)
    predicted / actual average churn rate (calibration: are the probabilities right?)
"""
import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step4_build_datasets import load_datasets
from step5_models import MODELS, features_for

BIG = 10
b = load_datasets()
train, test = b["train"], b["test"]
groups = {"3-9 orders": test.n_orders < BIG, "10+ orders": test.n_orders >= BIG, "all": test.n_orders > 0}

print(f"training rows: 10+ only = {(train.n_orders >= BIG).sum():,} | all 3+ = {len(train):,}")
print("test rows:", {g: int(m.sum()) for g, m in groups.items()}, "\n")

rows = []
name = "Logistic regression"
cols = features_for(name, b["features"])
for k in [3, 4, 5, 6, 7, 8, 10]:
    tr = train[train.n_orders >= k]
    m = MODELS[name]["make"]().fit(tr[cols], tr.churn)
    p = pd.Series(m.predict_proba(test[cols])[:, 1], index=test.index)
    row = {"trained on": f"{k}+ orders" + (" (current)" if k == 3 else ""), "training rows": len(tr)}
    for g, mask in [("3-4", test.n_orders <= 4), ("5-9", test.n_orders.between(5, 9)),
                    ("10+", test.n_orders >= 10), ("all", test.n_orders > 0)]:
        row[f"AUC {g}"] = roc_auc_score(test.churn[mask], p[mask])
    row["pred churn 3-9"] = p[test.n_orders < 10].mean()
    rows.append(row)
res = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(f"model: {name} | actual churn 3-9: {test.churn[test.n_orders < 10].mean():.3f}\n")
print(res.round(3).to_string(index=False))
