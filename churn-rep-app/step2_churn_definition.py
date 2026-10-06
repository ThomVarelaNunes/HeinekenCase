"""
STEP 2 -- The churn definition, written as code.

On any date we put every account in one of three groups:

    "too few orders"   fewer than MIN_ORDERS orders so far  -> we don't judge these
    "churned"          established, but no order for CHURN_DAYS or more
    "active"           established and ordered within the last CHURN_DAYS

The PREDICTION question for the model is then:
    "Of the accounts that are ACTIVE at a cutoff, which ones will place NO order
     in the next CHURN_DAYS?"  -> label = 1 (they churn), 0 (they don't)

Run on its own to see the groups today and the churn rate at every cutoff:
    python step2_churn_definition.py
"""
import pandas as pd
import config


def account_status(orders, date):
    """Group every account that ordered before `date` into too few / churned / active."""
    past = orders[orders.order_date < date]
    s = past.groupby("account_id").agg(n_orders=("order_id", "nunique"),
                                        last_order=("order_date", "max"))
    s["days_since_last"] = (date - s.last_order).dt.days
    s["status"] = "active"
    s.loc[s.days_since_last >= config.CHURN_DAYS, "status"] = "churned"
    s.loc[s.n_orders < config.MIN_ORDERS, "status"] = "too few orders"
    return s


def churn_label(orders, cutoff):
    """For accounts ACTIVE at `cutoff`: 1 if they order nothing in the next CHURN_DAYS, else 0."""
    status = account_status(orders, cutoff)
    active = status.index[status.status == "active"]

    window_end = cutoff + pd.Timedelta(days=config.CHURN_DAYS)
    in_window = orders[(orders.order_date >= cutoff) & (orders.order_date < window_end)]
    ordered_again = set(in_window.account_id)

    label = pd.Series([0 if a in ordered_again else 1 for a in active], index=active, name="churn")
    return label


if __name__ == "__main__":
    from step1_load_data import load_tables
    orders = load_tables()["orders"]

    print(f"Definition: {config.MIN_ORDERS}+ orders, churned after {config.CHURN_DAYS} days without an order\n")
    print(f"Status on {config.DATA_END.date()}:")
    print(account_status(orders, config.DATA_END).status.value_counts().to_string(), "\n")

    print("Churn rate among active accounts at each cutoff:")
    for c in config.TRAIN_CUTOFFS + [config.TEST_CUTOFF]:
        y = churn_label(orders, c)
        print(f"  {c.date()}  active accounts: {len(y):>5,}   churned in next {config.CHURN_DAYS} days: {y.mean():.1%}")
