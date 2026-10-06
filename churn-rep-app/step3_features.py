"""
STEP 3 -- Features: what we know about each account on a given cutoff date.

Features are grouped by the BEHAVIOUR they measure. Each group is one small function,
so you can read, test, add or drop a whole idea at once:

    rhythm       how often and how recently it orders, and whether that is slowing
    value        how much it spends
    assortment   how many product lines it buys, and which ones it stopped buying
    experience   late deliveries, cancelled orders, bad reviews
    cost         delivery cost and how it pays
    buyers       how many people in the account order, and whether they come back

THE ONE RULE: a feature may only use information known BEFORE the cutoff.
(e.g. a late delivery only counts once it has been delivered, or once its promised
date has passed.) Breaking this rule makes the model look better than it is.

To add a feature: add a column inside the right group function.
To add a group:   write a function with the same signature and add it to FEATURE_GROUPS.

Run on its own to see the features for a handful of accounts:
    python step3_features.py
"""
import numpy as np
import pandas as pd

RECENT_DAYS = 90     # what "recently" means in the features below
YEAR_DAYS = 365


def _before(df, cutoff, days=None):
    """Rows dated before the cutoff (optionally only the last `days` days)."""
    m = df.order_date < cutoff
    if days is not None:
        m &= df.order_date >= cutoff - pd.Timedelta(days=days)
    return df[m]


# --------------------------------------------------------------------------- groups
def rhythm(t, cutoff):
    o = _before(t["orders"], cutoff)
    g = o.groupby("account_id")
    f = pd.DataFrame({
        "n_orders": g.order_id.nunique(),
        "n_order_days": g.order_date.nunique(),
        "active_months": g.order_date.agg(lambda d: d.dt.to_period("M").nunique()),
        "tenure_days": (cutoff - g.order_date.min()).dt.days,
        "days_since_last": (cutoff - g.order_date.max()).dt.days,
    })
    # usual gap between order days = (last - first) / (number of gaps)
    span = (g.order_date.max() - g.order_date.min()).dt.days
    f["usual_gap_days"] = (span / (f.n_order_days - 1).replace(0, np.nan)).clip(lower=7).fillna(30)
    # how long it has been quiet compared with its own usual gap (1 = normal, 3 = three gaps late)
    f["overdue_ratio"] = f.days_since_last / f.usual_gap_days

    recent = _before(t["orders"], cutoff, RECENT_DAYS).groupby("account_id").order_id.nunique()
    year = _before(t["orders"], cutoff, YEAR_DAYS).groupby("account_id").order_id.nunique()
    f["orders_last_90d"] = recent.reindex(f.index).fillna(0)
    observed_days = f.tenure_days.clip(lower=30, upper=YEAR_DAYS)
    f["usual_orders_per_90d"] = year.reindex(f.index).fillna(0) / observed_days * RECENT_DAYS
    # last 90 days vs its usual pace (1 = normal, 0.5 = half as much)
    f["trend_vs_usual"] = (f.orders_last_90d / f.usual_orders_per_90d.replace(0, np.nan)).fillna(1).clip(0, 3)
    return f.drop(columns="n_order_days")


def value(t, cutoff):
    it = _before(t["items"], cutoff)
    year = _before(t["items"], cutoff, YEAR_DAYS)
    first = t["orders"][t["orders"].order_date < cutoff].groupby("account_id").order_date.min()
    observed_days = (cutoff - first).dt.days.clip(lower=30, upper=YEAR_DAYS)
    f = pd.DataFrame({"spend_total": it.groupby("account_id").price.sum()})
    f["spend_last_365d"] = year.groupby("account_id").price.sum().reindex(f.index).fillna(0)
    f["annual_value"] = f.spend_last_365d / observed_days.reindex(f.index) * YEAR_DAYS  # spend per year
    f["avg_order_value"] = f.spend_total / it.groupby("account_id").order_id.nunique()
    return f


def assortment(t, cutoff):
    it = _before(t["items"], cutoff)
    recent = _before(t["items"], cutoff, RECENT_DAYS)
    earlier = it[(it.order_date >= cutoff - pd.Timedelta(days=YEAR_DAYS)) &
                 (it.order_date < cutoff - pd.Timedelta(days=RECENT_DAYS))]
    f = pd.DataFrame({"n_categories": it.groupby("account_id").product_category.nunique()})
    f["categories_last_90d"] = recent.groupby("account_id").product_category.nunique().reindex(f.index).fillna(0)

    # "dropped" = a category bought in 2+ orders earlier in the year, but not in the last 90 days
    regular = earlier.groupby(["account_id", "product_category"]).agg(
        n=("order_id", "nunique"), spend=("price", "sum")).reset_index()
    regular = regular[regular.n >= 2]
    still_buying = set(zip(recent.account_id, recent.product_category))
    dropped = regular[[(a, c) not in still_buying for a, c in zip(regular.account_id, regular.product_category)]]
    f["dropped_categories"] = dropped.groupby("account_id").size().reindex(f.index).fillna(0)
    f["dropped_spend"] = dropped.groupby("account_id").spend.sum().reindex(f.index).fillna(0)
    return f


def experience(t, cutoff):
    o = t["orders"][t["orders"].order_date < cutoff].copy()
    delivered_by_cutoff = o.order_delivered_customer_date < cutoff
    o["late"] = delivered_by_cutoff & (o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date)
    o["overdue_not_arrived"] = (~delivered_by_cutoff) & (o.order_estimated_delivery_date < cutoff) \
        & ~o.order_status.isin(["canceled", "unavailable"])
    o["days_late"] = np.where(o.late, (o.order_delivered_customer_date.dt.normalize()
                                       - o.order_estimated_delivery_date).dt.days, 0)
    o["failed"] = o.order_status.isin(["canceled", "unavailable"])
    o["recent"] = o.order_date >= cutoff - pd.Timedelta(days=RECENT_DAYS)
    o["delivered"] = delivered_by_cutoff

    g = o.groupby("account_id")
    f = pd.DataFrame({"late_share": g.late.sum() / g.delivered.sum().replace(0, np.nan)})
    f["late_last_90d"] = o[o.recent].groupby("account_id").late.sum().reindex(f.index).fillna(0)
    f["max_days_late_90d"] = o[o.recent].groupby("account_id").days_late.max().reindex(f.index).fillna(0)
    f["failed_orders_90d"] = o[o.recent].groupby("account_id").failed.sum().reindex(f.index).fillna(0)
    last = o.sort_values("order_purchase_timestamp").groupby("account_id").tail(1).set_index("account_id")
    f["last_order_late"] = (last.late | last.overdue_not_arrived).astype(int).reindex(f.index)

    # reviews only count once they were written before the cutoff
    r = t["reviews"]
    r = r[(r.order_date < cutoff) & (r.review_answer_timestamp < cutoff)]
    gr = r.groupby("account_id")
    f["avg_review"] = gr.review_score.mean()
    f["last_review"] = r.sort_values("review_answer_timestamp").groupby("account_id").review_score.last()
    rr = r[r.order_date >= cutoff - pd.Timedelta(days=RECENT_DAYS)]
    f["low_reviews_90d"] = (rr.review_score <= 2).groupby(rr.account_id).sum().reindex(f.index).fillna(0)
    f["has_review"] = f.avg_review.notna().astype(int)
    f["avg_review"] = f.avg_review.fillna(4.0)    # DECISION: no review -> treat as roughly neutral-good
    f["last_review"] = f.last_review.fillna(4.0)
    f["late_share"] = f.late_share.fillna(0)
    return f


def cost(t, cutoff):
    it = _before(t["items"], cutoff)
    g = it.groupby("account_id")
    f = pd.DataFrame({"freight_share": g.freight_value.sum() / (g.price.sum() + g.freight_value.sum())})
    p = _before(t["payments"], cutoff)
    gp = p.groupby("account_id")
    f["avg_installments"] = gp.payment_installments.mean().reindex(f.index)
    f["boleto_share"] = (p.payment_type == "boleto").groupby(p.account_id).mean().reindex(f.index)
    return f.fillna(0)


def buyers(t, cutoff):
    o = t["orders"][t["orders"].order_date < cutoff]
    per_buyer = o.groupby(["account_id", "customer_unique_id"]).order_id.nunique()
    f = pd.DataFrame({"n_buyers": per_buyer.groupby("account_id").size()})
    f["repeat_buyers"] = (per_buyer >= 2).groupby("account_id").sum()
    f["repeat_buyer_share"] = f.repeat_buyers / f.n_buyers
    return f


FEATURE_GROUPS = {
    "rhythm": rhythm,
    "value": value,
    "assortment": assortment,
    "experience": experience,
    "cost": cost,
    "buyers": buyers,
}


def build_features(tables, cutoff, accounts, groups=None):
    """All features for `accounts` on `cutoff`.

    Returns (X, column_groups): the feature table, and a dict {column: group name}
    so later steps can report results per behaviour.
    `groups` lets you build only some of FEATURE_GROUPS.
    """
    groups = groups or list(FEATURE_GROUPS)
    parts, column_groups = [], {}
    for g in groups:
        part = FEATURE_GROUPS[g](tables, cutoff)
        parts.append(part)
        column_groups.update({c: g for c in part.columns})
    X = pd.concat(parts, axis=1).reindex(accounts)
    return X, column_groups


if __name__ == "__main__":
    import config
    from step1_load_data import load_tables
    from step2_churn_definition import account_status
    t = load_tables()
    status = account_status(t["orders"], config.TEST_CUTOFF)
    active = status.index[status.status == "active"]
    X, _ = build_features(t, config.TEST_CUTOFF, active)
    print(f"{X.shape[1]} features for {len(X):,} active accounts on {config.TEST_CUTOFF.date()}\n")
    for g, fn in FEATURE_GROUPS.items():
        print(f"{g:<11} {', '.join(fn(t, config.TEST_CUTOFF).columns)}")
    print("\nFirst 3 accounts:\n", X.head(3).T.round(2).to_string())
