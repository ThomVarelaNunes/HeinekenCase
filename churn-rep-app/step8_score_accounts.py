"""
STEP 8 -- Score every account as of "today" (1 Sep 2018) with the chosen model.

  1. Retrain config.CHOSEN_MODEL on ALL labelled snapshots (training + test cutoffs):
     we have tested it, now let it learn from the most recent months too.
  2. Score every ACTIVE account today: churn_risk = P(no order in the next CHURN_DAYS).
  3. value_at_risk  = churn_risk x annual_value          (expected revenue lost, in money)
     priority_score = churn_risk^w x annual_value^(1-w)  (w = config.RISK_WEIGHT; the ranking)
     segment        = risk x value matrix                 (decides the channel: visit, call, AI agent...)
  4. Reasons: for each actionable feature, ask "how much lower would this account's risk be
     if this one thing were typical?" The drop, in percentage points, is the reason's size.
     (This works for ANY model, so it keeps working if you swap the model.)
     Note: the sizes don't add up exactly to the total risk -- each is measured on its own.

Output: outputs/scored_accounts.csv (all accounts, with status), sorted by priority.
This file is the input for the action/AI layer and the rep app.
Run:   python step8_score_accounts.py
"""
import os
import numpy as np
import pandas as pd

import config
from step1_load_data import load_tables
from step2_churn_definition import account_status
from step3_features import build_features
from step4_build_datasets import load_datasets
from step5_models import MODELS, features_for

# Features a rep can do something about, with the sentence shown in the app.
# DECISION: volume-type features (n_orders, n_buyers, tenure...) are left out on purpose:
# "this account is small" is true but not something a rep can fix.
REASON_TEXT = {
    "overdue_ratio":       "Silent for longer than its usual ordering rhythm",
    "days_since_last":     "Long time since the last order",
    "orders_last_90d":     "Fewer orders in the last 90 days",
    "trend_vs_usual":      "Ordering below its usual pace",
    "freight_share":       "Delivery cost is a large share of what it spends",
    "last_order_late":     "Last order arrived late (or still hasn't arrived)",
    "late_share":          "Deliveries are often late",
    "late_last_90d":       "Late deliveries in the last 90 days",
    "max_days_late_90d":   "A recent delivery was very late",
    "low_reviews_90d":     "Left a 1-2 star review recently",
    "last_review":         "Last review was negative",
    "avg_review":          "Reviews are low on average",
    "failed_orders_90d":   "Had a cancelled / unavailable order recently",
    "dropped_categories":  "Stopped buying a product line it used to buy",
    "dropped_spend":       "Stopped buying a product line it used to buy",
    "categories_last_90d": "Buying fewer product lines than before",
    "usual_gap_days":      "Orders only every few months",
}

# FACT CHECK: a reason is only shown when the account's OWN numbers back it up.
# Without this, overlapping inputs can produce reasons that contradict the data
# (e.g. "ordering below its usual pace" for an account ordering above it).
# Each check gets the account's features (x) and the typical values (typ).
FACT_CHECK = {
    "overdue_ratio":       lambda x, typ: x.overdue_ratio > 1,
    "days_since_last":     lambda x, typ: x.days_since_last > x.usual_gap_days,
    "orders_last_90d":     lambda x, typ: x.orders_last_90d < x.usual_orders_per_90d,
    "trend_vs_usual":      lambda x, typ: x.trend_vs_usual < 0.8,
    "freight_share":       lambda x, typ: x.freight_share > typ["freight_share"],
    "last_order_late":     lambda x, typ: x.last_order_late == 1,
    "late_share":          lambda x, typ: x.late_share > max(typ["late_share"], 0),
    "late_last_90d":       lambda x, typ: x.late_last_90d > 0,
    "max_days_late_90d":   lambda x, typ: x.max_days_late_90d > 0,
    "low_reviews_90d":     lambda x, typ: x.low_reviews_90d > 0,
    "last_review":         lambda x, typ: x.has_review == 1 and x.last_review <= 2,
    "avg_review":          lambda x, typ: x.has_review == 1 and x.avg_review < 3.5,
    "failed_orders_90d":   lambda x, typ: x.failed_orders_90d > 0,
    "dropped_categories":  lambda x, typ: x.dropped_categories > 0,
    "dropped_spend":       lambda x, typ: x.dropped_spend > 0,
    "categories_last_90d": lambda x, typ: x.dropped_categories > 0,
    "usual_gap_days":      lambda x, typ: x.usual_gap_days > config.CHURN_DAYS,
}
N_REASONS = 3              # how many reasons to keep per account
MIN_REASON_POINTS = 1.0    # ignore reasons that move risk by less than 1 percentage point


def what_if_reasons(model, X, cols, typical):
    """Top reasons per account, sized so that they ADD UP.

    1. typical_risk = the model's risk for a typical account (every input at its median).
    2. For EVERY input, measure how much it pushes this account's risk up, on the model's
       log-odds scale (where effects add up), by setting that one input to typical.
    3. The account's extra risk above typical (risk - typical_risk) is split over the inputs
       that push it up, in proportion. So all shares together = the extra risk.
    4. Only actionable inputs that pass the fact check are shown; size and other
       non-fixable inputs keep their share but are not listed.
    """
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    base = model.predict_proba(X[cols])[:, 1]
    typical_risk = float(model.predict_proba(typical[cols].to_frame().T.astype(float))[0, 1])

    push = {}
    for f in cols:
        X2 = X[cols].copy()
        X2[f] = typical[f]
        push[f] = logit(base) - logit(model.predict_proba(X2)[:, 1])   # + = raises risk
    push = pd.DataFrame(push, index=X.index).clip(lower=0)
    share = push.div(push.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    extra = np.clip(base - typical_risk, 0, None) * 100                  # percentage points
    points = share.mul(extra, axis=0)

    # keep actionable inputs only, and only when the account's own numbers back them up
    points = points[[f for f in REASON_TEXT if f in points.columns]].copy()
    for f in points.columns:
        ok = X.apply(lambda x: bool(FACT_CHECK[f](x, typical)), axis=1)
        points.loc[~ok, f] = np.nan
    # inputs that share one sentence are added together
    by_text = {}
    for f in points.columns:
        by_text.setdefault(REASON_TEXT[f], []).append(f)
    points = pd.DataFrame({t: points[fs].sum(axis=1, min_count=1) for t, fs in by_text.items()})

    def top_reasons(row):
        row = row.dropna()
        row = row[row >= MIN_REASON_POINTS].sort_values(ascending=False)[:N_REASONS]
        texts, sizes = list(row.index), [round(v, 1) for v in row.values]
        pad = N_REASONS - len(texts)
        return texts + [""] * pad + sizes + [np.nan] * pad

    reasons = points.apply(top_reasons, axis=1, result_type="expand")
    reasons.columns = [f"reason_{i}" for i in range(1, N_REASONS + 1)] + \
                      [f"reason_{i}_points" for i in range(1, N_REASONS + 1)]
    reasons["typical_risk"] = typical_risk
    return base, reasons


def priority_score(risk, annual_value, w=config.RISK_WEIGHT):
    """w = 0.5 gives the same order as value at risk; higher w favours risk over revenue."""
    return risk ** w * annual_value.clip(lower=1) ** (1 - w)


def segment(risk, annual_value, high_value_cut):
    high_r = risk >= config.HIGH_RISK
    high_v = annual_value >= high_value_cut
    return np.select(
        [high_r & high_v, high_r & ~high_v, ~high_r & high_v],
        ["1 Save now (high risk, high value)",      # -> rep visit / call
         "2 Rescue cheaply (high risk, low value)",  # -> AI voice agent / automated offer
         "3 Protect (low risk, high value)"],        # -> keep the relationship warm
        "4 Monitor (low risk, low value)")           # -> automated reorder reminders


def score(tables=None):
    tables = tables or load_tables()
    b = load_datasets()
    name = config.CHOSEN_MODEL
    cols = features_for(name, b["features"])

    # 1. retrain on everything we have labels for
    all_labelled = pd.concat([b["train"], b["test"]])
    model = MODELS[name]["make"]().fit(all_labelled[cols], all_labelled.churn)

    # 2. today's features for every account
    status = account_status(tables["orders"], config.DATA_END)
    X, _ = build_features(tables, config.DATA_END, status.index)
    out = status[["status", "n_orders", "days_since_last"]].join(
        X[["annual_value", "usual_gap_days", "orders_last_90d", "usual_orders_per_90d"]])

    active = status.index[status.status == "active"]
    typical = all_labelled[cols].median()
    typical["has_review"] = 1
    risk, reasons = what_if_reasons(model, X.loc[active], cols, typical)

    # 3. risk, money, ranking, segment
    a = out.loc[active].copy()
    a["churn_risk"] = risk
    a["value_at_risk"] = a.churn_risk * a.annual_value
    a["priority_score"] = priority_score(a.churn_risk, a.annual_value)
    a["priority_rank"] = a.priority_score.rank(ascending=False, method="first")
    a["segment"] = segment(a.churn_risk, a.annual_value, a.annual_value.quantile(config.HIGH_VALUE_QUANTILE))
    out = out.join(a[["churn_risk", "value_at_risk", "priority_score", "priority_rank", "segment"]]).join(reasons)

    # churned accounts: a separate win-back list, ranked by what they used to spend
    churned = out.status == "churned"
    out.loc[churned, "winback_rank"] = out.loc[churned, "annual_value"].rank(ascending=False, method="first")

    order = {"active": 0, "churned": 1, "too few orders": 2}
    out = out.sort_values(["status", "priority_rank", "winback_rank"],
                          key=lambda s: s.map(order) if s.name == "status" else s)
    return out.rename_axis("account_id").reset_index(), model


if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 50)
    out, _ = score()
    path = os.path.join(config.OUTPUT_DIR, "scored_accounts.csv")
    out.to_csv(path, index=False)
    print(f"Model: {config.CHOSEN_MODEL} | RISK_WEIGHT = {config.RISK_WEIGHT} | scored on {config.DATA_END.date()}\n")
    print(out.status.value_counts().to_string(), "\n")
    act = out[out.status == "active"]
    print("Top 10 active accounts:")
    print(act.head(10)[["priority_rank", "account_id", "churn_risk", "annual_value", "value_at_risk",
                        "reason_1", "reason_1_points", "reason_2", "reason_2_points"]].round(2).to_string(index=False))
    print("\nSegments:")
    print(act.groupby("segment").agg(accounts=("account_id", "size"), avg_risk=("churn_risk", "mean"),
                                     value_at_risk=("value_at_risk", "sum")).round(2).to_string())
    print(f"\nsaved {path}")
