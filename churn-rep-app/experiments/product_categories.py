"""
EXPERIMENT -- Does knowing WHICH product categories an account buys help predict churn?

Adds, per account and cutoff, the share of its spend in each of the TOP_N biggest categories
(known before the cutoff), then compares test AUC with and without them.
Run from the project folder:   python experiments/product_categories.py
"""
import os, sys
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step1_load_data import load_tables
from step4_build_datasets import load_datasets
from step5_models import MODELS, features_for

TOP_N = 20
t, b = load_tables(), load_datasets()
items = t["items"]
top = items.groupby("product_category").price.sum().nlargest(TOP_N).index

def category_shares(df):
    parts = []
    for c in df.cutoff.unique():
        it = items[(items.order_date < c) & items.account_id.isin(df[df.cutoff == c].index)]
        sh = it.pivot_table(index="account_id", columns="product_category", values="price", aggfunc="sum", fill_value=0)
        sh = sh.div(sh.sum(axis=1), axis=0).reindex(columns=top, fill_value=0)
        sh.columns = ["cat_" + x for x in sh.columns]
        parts.append(sh.reindex(df[df.cutoff == c].index).fillna(0).assign(cutoff=c))
    e = pd.concat(parts).reset_index()
    return df.reset_index().merge(e, on=["account_id", "cutoff"], how="left").set_index("account_id")

train, test = category_shares(b["train"]), category_shares(b["test"])
cat_cols = [c for c in train.columns if c.startswith("cat_")]
rows = []
for name in ["Logistic regression", "Logistic regression (L1)", "Gradient boosting"]:
    base = features_for(name, b["features"])
    auc = lambda cols: roc_auc_score(test.churn, MODELS[name]["make"]().fit(train[cols], train.churn).predict_proba(test[cols])[:, 1])
    a0, a1 = auc(base), auc(base + cat_cols)
    rows.append({"model": name, "AUC_without": a0, "AUC_with_categories": a1, "change": a1 - a0})
print(f"top {TOP_N} categories by spend:", ", ".join(top), "\n")
print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")
m = MODELS["Logistic regression (L1)"]["make"]().fit(train[b["features"] + cat_cols], train.churn)
w = pd.Series(m[-1].coef_[0], index=b["features"] + cat_cols)[cat_cols]
print("L1 weights on the category shares (0 = dropped):")
print(w[w.abs() > 1e-6].sort_values().round(3).to_string() or "  all dropped")
