"""
STEP 10 -- Package everything the rep app shows into one JSON file.

For the top active accounts (by priority) and the top churned accounts (win-back list):
    scores        churn risk, yearly revenue, value at risk, priority, segment
    reasons       top 3 reasons with their size in percentage points (from step 8)
    history       spend per month, Jan 2017 - Aug 2018 (for the chart)
    profile       top product lines, a line it stopped buying, delivery and review facts
    complaint     its most recent 1-2 star review (Portuguese) and a rough theme
    action        next best action, offer and channel, chosen by simple rules below

Output: outputs/app_data.json  (also usable as input for Lovable / Streamlit)
Run:   python step10_export_app_data.py
"""
import json
import os
import re
import numpy as np
import pandas as pd

import config
from step1_load_data import load_tables
from step3_features import build_features

N_ACTIVE = 300      # how many active accounts the app gets (by priority)
N_WINBACK = 100     # how many churned accounts for the win-back list

# --- Next best action per main reason (DECISION: edit freely) -----------------
ACTIONS = {
    "Fewer orders in the last 90 days": ("Reorder check-in: send the usual order pre-filled and ask what changed",
                                         "Usual basket with 5% off if reordered this week"),
    "Ordering below its usual pace": ("Reorder check-in: send the usual order pre-filled and ask what changed",
                                      "Usual basket with 5% off if reordered this week"),
    "Silent for longer than its usual ordering rhythm": ("Call now: it is overdue on its normal ordering rhythm",
                                                         "Usual basket with 5% off if reordered this week"),
    "Long time since the last order": ("Call now: it is overdue on its normal ordering rhythm",
                                       "Usual basket with 5% off if reordered this week"),
    "Delivery cost is a large share of what it spends": ("Propose fewer, bigger deliveries to cut delivery cost",
                                                         "Free delivery above a basket threshold"),
    "Last order arrived late (or still hasn't arrived)": ("Service-recovery call within 48 hours: apologise and confirm the fix",
                                                          "Free delivery and a guaranteed date on the next order"),
    "Deliveries are often late": ("Service-recovery call: apologise and agree a reliable delivery slot",
                                  "Free delivery and a guaranteed date on the next order"),
    "Late deliveries in the last 90 days": ("Service-recovery call: apologise and agree a reliable delivery slot",
                                            "Free delivery and a guaranteed date on the next order"),
    "A recent delivery was very late": ("Service-recovery call within 48 hours: apologise and confirm the fix",
                                        "Free delivery and a guaranteed date on the next order"),
    "Left a 1-2 star review recently": ("Complaint follow-up: listen first, resolve the issue, then talk orders",
                                        "Credit on the next order"),
    "Last review was negative": ("Complaint follow-up: listen first, resolve the issue, then talk orders",
                                 "Credit on the next order"),
    "Reviews are low on average": ("Quality check-in: ask what keeps going wrong",
                                   "Credit on the next order"),
    "Had a cancelled / unavailable order recently": ("Apologise for the cancelled order and confirm stock",
                                                     "Priority stock on the next order"),
    "Stopped buying a product line it used to buy": ("Win back the dropped product line",
                                                     "Dropped line bundled with the usual order, 10% off"),
    "Buying fewer product lines than before": ("Cross-sell: suggest a line similar accounts buy",
                                               "Trial pack of a new product line"),
}
DEFAULT_ACTION = ("Relationship check-in: confirm everything is going well", "Usual basket reminder")
WINBACK_ACTION = ("Win-back call: ask why they stopped, fix it, reopen with an offer",
                  "Come-back bundle: usual lines with 10% off the first order back")

CHANNEL = {
    "1 Save now (high risk, high value)": "Rep visit",
    "2 Rescue cheaply (high risk, low value)": "Phone call (rescue cheaply)",
    "3 Protect (low risk, high value)": "Phone call (keep the relationship warm)",
    "4 Monitor (low risk, low value)": "Automated reorder reminder",
}

# Rough complaint theme from Portuguese keywords (a language model can do this better)
THEMES = [
    ("missing item", r"falt|incomplet|apenas um|so veio|só veio|só recebi|so recebi|um dos|metade"),
    ("wrong item", r"errad|diferente|outro produto|outra marca|outra cor|outro modelo|trocad"),
    ("damaged / poor quality", r"defeit|quebr|danific|qualidade|estragad|rasgad|não funciona|nao funciona"),
    ("late / not delivered", r"entreg|cheg|prazo|atras|receb|demor|aguard|esper"),
]


def theme(text):
    if not isinstance(text, str) or not text.strip():
        return "no comment written"
    for name, pat in THEMES:
        if re.search(pat, text.lower()):
            return name
    return "other"


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def export(n_active=N_ACTIVE, n_winback=N_WINBACK, filename="app_data.json"):
    """n_active=None exports every active account; n_winback=0 leaves churned accounts out."""
    t = load_tables()
    scored = pd.read_csv(os.path.join(config.OUTPUT_DIR, "scored_accounts.csv"), dtype={"account_id": str})
    act = scored[scored.status == "active"]
    active = act.nsmallest(n_active, "priority_rank") if n_active else act
    winback = scored[scored.status == "churned"].nsmallest(n_winback, "winback_rank")
    pick = pd.concat([active, winback]).set_index("account_id")
    ids = pick.index

    X, _ = build_features(t, config.DATA_END, ids)
    items = t["items"][t["items"].account_id.isin(ids)]
    orders = t["orders"][t["orders"].account_id.isin(ids)]

    months = pd.period_range("2017-01", "2018-08", freq="M")
    spend_m = (items.assign(m=items.order_date.dt.to_period("M"))
                    .groupby(["account_id", "m"]).price.sum().unstack(fill_value=0)
                    .reindex(columns=months, fill_value=0))
    cat_spend = items.groupby(["account_id", "product_category"]).price.sum().reset_index()
    top_cats = cat_spend.sort_values("price", ascending=False).groupby("account_id").product_category.apply(lambda s: list(s.head(3)))

    # dropped line: bought in 2+ orders between 12 and 3 months ago, nothing in the last 3 months
    cut = config.DATA_END
    earlier = items[(items.order_date >= cut - pd.Timedelta(days=365)) & (items.order_date < cut - pd.Timedelta(days=90))]
    recent = set(zip(items[items.order_date >= cut - pd.Timedelta(days=90)].account_id,
                     items[items.order_date >= cut - pd.Timedelta(days=90)].product_category))
    reg = earlier.groupby(["account_id", "product_category"]).agg(n=("order_id", "nunique"), s=("price", "sum")).reset_index()
    still_buying = np.array([(a, c) in recent for a, c in zip(reg.account_id, reg.product_category)], dtype=bool)
    reg = reg[(reg.n >= 2).to_numpy() & ~still_buying]
    dropped = reg.sort_values("s", ascending=False).groupby("account_id").product_category.first()

    rv = t["reviews"][t["reviews"].account_id.isin(ids)]
    low = rv[rv.review_score <= 2].sort_values("review_answer_timestamp").groupby("account_id").tail(1).set_index("account_id")
    city = orders.groupby("account_id")[["customer_state"]].first()
    geo = pd.read_csv(config.DATA_DIR + "geolocation.csv", dtype={"account_id": str}).set_index("account_id")

    out = []
    for a, r in pick.iterrows():
        is_active = r.status == "active"
        main = r.reason_1 if isinstance(r.reason_1, str) and r.reason_1 else None
        action, offer = (ACTIONS.get(main, DEFAULT_ACTION) if is_active else WINBACK_ACTION)
        if True:
            channel = CHANNEL.get(r.segment, "Phone call") if is_active else (
                "Rep phone call" if r.winback_rank <= 30 else "AI voice agent call (hands over to rep if unhappy)")
        complaint = None
        if a in low.index:
            c = low.loc[a]
            complaint = {"score": int(c.review_score),
                         "days_ago": int((cut - c.review_answer_timestamp).days),
                         "title": c.review_comment_title if isinstance(c.review_comment_title, str) else None,
                         "text": c.review_comment_message if isinstance(c.review_comment_message, str) else None,
                         "theme": theme(" ".join(x for x in [c.review_comment_title, c.review_comment_message] if isinstance(x, str)))}
        x = X.loc[a]
        out.append({
            "id": a,
            "status": r.status,
            "city": geo.city.get(a, None) if a in geo.index else None,
            "state": geo.state.get(a, None) if a in geo.index else (city.customer_state.get(a) if a in city.index else None),
            "lat": clean(geo.lat.get(a)) if a in geo.index else None,
            "lng": clean(geo.lng.get(a)) if a in geo.index else None,
            "rank": clean(r.priority_rank if is_active else r.winback_rank),
            "segment": r.segment if is_active else "Win back (silent 90+ days)",
            "risk": clean(r.churn_risk),
            "annual_value": clean(r.annual_value),
            "value_at_risk": clean(r.value_at_risk),
            "reasons": [{"text": r[f"reason_{i}"], "points": clean(r[f"reason_{i}_points"])}
                        for i in (1, 2, 3) if isinstance(r[f"reason_{i}"], str) and r[f"reason_{i}"]],
            "n_orders": int(r.n_orders),
            "days_since_last": int(r.days_since_last),
            "usual_gap_days": clean(x.usual_gap_days),
            "orders_last_90d": clean(x.orders_last_90d),
            "usual_orders_per_90d": clean(x.usual_orders_per_90d),
            "late_share": clean(x.late_share),
            "last_order_late": clean(x.last_order_late),
            "freight_share": clean(x.freight_share),
            "avg_review": clean(x.avg_review) if x.has_review else None,
            "n_buyers": clean(x.n_buyers),
            "top_lines": top_cats.get(a, []),
            "dropped_line": dropped.get(a, None),
            "monthly_spend": [round(float(v)) for v in spend_m.loc[a]] if a in spend_m.index else [0] * len(months),
            "complaint": complaint,
            "action": action,
            "offer": offer,
            "channel": channel,
        })

    meta = {"today": "2018-08-31", "months": [str(m) for m in months],
            "churn_days": config.CHURN_DAYS, "min_orders": config.MIN_ORDERS,
            "typical_risk": round(float(scored.typical_risk.dropna().iloc[0]), 4),
            "model": config.CHOSEN_MODEL, "risk_weight": config.RISK_WEIGHT,
            "totals": {
                "active": int((scored.status == "active").sum()),
                "churned": int((scored.status == "churned").sum()),
                "too_few": int((scored.status == "too few orders").sum()),
                "value_at_risk_all_active": round(float(scored.value_at_risk.sum()), 0),
                "segments": scored[scored.status == "active"].segment.value_counts().to_dict()}}
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "accounts": out}, fh, ensure_ascii=False)
    return path, out


if __name__ == "__main__":
    path, out = export()
    print(f"{len(out)} accounts -> {path} ({os.path.getsize(path) / 1024:.0f} KB)")
    print(json.dumps(out[0], ensure_ascii=False, indent=1)[:1500])
