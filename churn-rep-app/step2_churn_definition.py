"""
STEP 2 -- The churn definition, written as code (rhythm-based).

Every account gets its OWN churn line:
    churn line = max(CHURN_DAYS, RHYTHM_MULT x its usual gap), capped at MAX_CHURN_DAYS
An account that orders every 20 days is churned after 90 days of silence;
one that orders every 70 days after 140; one that orders every 247 days after 180 (the cap).

On any date we put every account in one of three groups:

    "too few orders"   fewer than MIN_ORDERS orders so far  -> we don't judge these
    "churned"          established, silent for at least its own churn line
    "active"           established and still inside its churn line

The PREDICTION question for the model:
    "Of the accounts that are ACTIVE at a cutoff, which ones will have CROSSED their own
     churn line CHURN_DAYS from now (no order in between)?"  -> label 1 / 0
The window stays CHURN_DAYS long, so every label can be checked within the data.
The usual gap is measured with what was known at the cutoff and kept fixed.

Run on its own to see the groups today and the churn rate at every cutoff:
    python step2_churn_definition.py
"""
import numpy as np
import pandas as pd
import config

RHYTHM_MULT = 2
MAX_CHURN_DAYS = 180   # = 2 x ~90-day cycle of slow accounts; nobody stays "active" after 6 months of silence


def account_status(orders, date):
    """Group every account that ordered before `date` into too few / churned / active."""
    past = orders[orders.order_date < date]
    g = past.groupby("account_id")
    s = pd.DataFrame({"n_orders": g.order_id.nunique(), "last_order": g.order_date.max(),
                      "first_order": g.order_date.min(), "n_days": g.order_date.nunique()})
    s["days_since_last"] = (date - s.last_order).dt.days
    span = (s.last_order - s.first_order).dt.days
    s["usual_gap"] = (span / (s.n_days - 1).replace(0, np.nan)).clip(lower=7)
    s["churn_line"] = np.maximum(config.CHURN_DAYS, (RHYTHM_MULT * s.usual_gap).fillna(0)).clip(upper=MAX_CHURN_DAYS).round()
    s["status"] = "active"
    s.loc[s.days_since_last >= s.churn_line, "status"] = "churned"
    s.loc[s.n_orders < config.MIN_ORDERS, "status"] = "too few orders"
    return s.drop(columns=["first_order", "n_days"])


def churn_label(orders, cutoff):
    """For accounts ACTIVE at `cutoff`: 1 if, CHURN_DAYS later, they have crossed their churn line."""
    status = account_status(orders, cutoff)
    act = status[status.status == "active"]

    window_end = cutoff + pd.Timedelta(days=config.CHURN_DAYS)
    in_window = orders[(orders.order_date >= cutoff) & (orders.order_date < window_end)]
    ordered_again = set(in_window.account_id)

    silence_at_end = act.days_since_last + config.CHURN_DAYS          # if no order in the window
    crossed = (silence_at_end >= act.churn_line) & ~act.index.isin(ordered_again)
    return crossed.astype(int).rename("churn")


if __name__ == "__main__":
    from step1_load_data import load_tables
    orders = load_tables()["orders"]

    print(f"Definition: {config.MIN_ORDERS}+ orders; churned after max({config.CHURN_DAYS} days, "
          f"{RHYTHM_MULT} x own usual gap), at most {MAX_CHURN_DAYS} days, without an order\n")
    st = account_status(orders, config.DATA_END)
    print(f"Status on {config.DATA_END.date()}:")
    print(st.status.value_counts().to_string(), "\n")
    est = st[st.status != "too few orders"]
    print(f"Churn line among established accounts: median {est.churn_line.median():.0f} days, "
          f"{(est.churn_line > config.CHURN_DAYS).mean():.0%} have a line longer than {config.CHURN_DAYS} days\n")

    print("Churn rate among active accounts at each cutoff:")
    for c in config.TRAIN_CUTOFFS + [config.TEST_CUTOFF]:
        y = churn_label(orders, c)
        print(f"  {c.date()}  active accounts: {len(y):>5,}   crossed their line within {config.CHURN_DAYS} days: {y.mean():.1%}")
