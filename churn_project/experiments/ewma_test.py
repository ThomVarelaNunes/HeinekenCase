"""
EXPERIMENT -- Does EWMA (exponentially weighted moving average) help?

Run from the project folder:   python experiments/ewma_test.py

EWMA here is computed in continuous time, per order: an order `age` days before the cutoff
gets weight exp(-ln2 * age / half_life). Dividing by the total weight the account COULD have
had since its first order turns this into a recency-weighted spend rate, so young accounts
are not penalised for having less history.

Test 1 -- EWMA as model inputs
    ewma_spend_30d / ewma_spend_180d  : spend per year, weighted with 30- and 180-day half-lives
    ewma_orders_30d / ewma_orders_180d: orders per year, same weighting
    ewma_spend_trend, ewma_orders_trend: short / long (below 1 = slowing down)
    -> compare test AUC with and without them, for every model.

Test 2 -- EWMA as the "value" in the ranking
    Which estimate best predicts what an account spends in the NEXT 90 days
    (among accounts that did not churn, i.e. what it is worth if we keep it)?
    flat 365-day revenue (current)  vs  EWMA revenue with different half-lives.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from step1_load_data import load_tables
from step4_build_datasets import load_datasets
from step5_models import MODELS, features_for

LN2 = np.log(2)


def ewma_rate(items, cutoff, accounts, half_life, what="spend"):
    """Recency-weighted rate per YEAR (spend or orders) for each account, as of cutoff."""
    it = items[items.order_date < cutoff]
    lam = LN2 / half_life
    if what == "spend":
        rows = it.groupby(["account_id", "order_id", "order_date"]).price.sum().reset_index(name="x")
    else:
        rows = it[["account_id", "order_id", "order_date"]].drop_duplicates().assign(x=1.0)
    age = (cutoff - rows.order_date).dt.days.to_numpy()
    rows["w"] = rows.x * np.exp(-lam * age)
    num = rows.groupby("account_id").w.sum()
    tenure = (cutoff - it.groupby("account_id").order_date.min()).dt.days.clip(lower=30)
    possible_weight = (1 - np.exp(-lam * tenure)) / lam            # integral of the weights since first order
    return (num / possible_weight * 365).reindex(accounts)


def ewma_features(items, cutoff, accounts):
    f = pd.DataFrame(index=accounts)
    for what in ["spend", "orders"]:
        f[f"ewma_{what}_30d"] = ewma_rate(items, cutoff, accounts, 30, what)
        f[f"ewma_{what}_180d"] = ewma_rate(items, cutoff, accounts, 180, what)
        f[f"ewma_{what}_trend"] = (f[f"ewma_{what}_30d"] / f[f"ewma_{what}_180d"].replace(0, np.nan)).clip(0, 5)
    return f.fillna(0)


def add_ewma(df, items):
    parts = [ewma_features(items, c, df[df.cutoff == c].index).assign(cutoff=c) for c in df.cutoff.unique()]
    e = pd.concat(parts)
    return df.join(e.drop(columns="cutoff"), how="left") if df.index.is_unique else \
        df.reset_index().merge(e.reset_index(), on=["account_id", "cutoff"], how="left").set_index("account_id")


def auc(model_name, train, test, cols):
    m = MODELS[model_name]["make"]().fit(train[cols], train.churn)
    return roc_auc_score(test.churn, m.predict_proba(test[cols])[:, 1])


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    t = load_tables()
    b = load_datasets()
    items = t["items"]
    train = add_ewma(b["train"], items)
    test = add_ewma(b["test"], items)
    ewma_cols = [c for c in train.columns if c.startswith("ewma_")]

    # ---------------- Test 1: EWMA as model inputs
    print("TEST 1 -- test AUC with and without EWMA features\n")
    rows = []
    for name in ["Logistic regression", "Logistic regression (L1)", "Gradient boosting"]:
        base_cols = features_for(name, b["features"])
        a0 = auc(name, train, test, base_cols)
        a1 = auc(name, train, test, base_cols + ewma_cols)
        rows.append({"model": name, "AUC_without": a0, "AUC_with_ewma": a1, "change": a1 - a0})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\nEach EWMA feature on its own vs the closest existing feature (test AUC):")
    single = []
    for c in ewma_cols + ["trend_vs_usual", "orders_last_90d", "annual_value"]:
        a = roc_auc_score(test.churn, test[c].fillna(0))
        single.append({"feature": c, "AUC_alone": max(a, 1 - a)})
    print(pd.DataFrame(single).round(3).to_string(index=False))

    # ---------------- Test 2: EWMA as the value estimate
    print("\nTEST 2 -- which value estimate best predicts next-90-day spend of accounts we keep?\n")
    c = config.TEST_CUTOFF
    nxt = items[(items.order_date >= c) & (items.order_date < c + pd.Timedelta(days=config.CHURN_DAYS))]
    kept = test[test.churn == 0].copy()
    kept["spend_next_90d"] = nxt.groupby("account_id").price.sum().reindex(kept.index).fillna(0)
    candidates = {"flat 365-day revenue (current)": kept.annual_value}
    for h in [30, 60, 90, 180]:
        candidates[f"EWMA revenue, half-life {h}d"] = ewma_rate(items, c, kept.index, h, "spend")
    truth = kept.spend_next_90d
    rows = []
    for name, v in candidates.items():
        pred_90 = v * config.CHURN_DAYS / 365
        rows.append({"value estimate": name,
                     "rank_correlation": spearmanr(v, truth).statistic,
                     "median_abs_error": (pred_90 - truth).abs().median(),
                     "mean_abs_error": (pred_90 - truth).abs().mean()})
    print(f"{len(kept):,} retained accounts, median next-90-day spend {truth.median():.0f}\n")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
