"""
STEP 9 -- Which ranking should reps work from? Test it on the held-out period.

Reps can only handle the top of the list. So for each ranking we ask: if reps had
worked the top 10% of the list in June 2018, what would they have reached?

    churners_reached      how many of the accounts that really churned are in that top 10%
    revenue_reached       share of all revenue LOST to churn that sits in that top 10%
    avg_risk_in_list      average predicted risk of the accounts in the list
    median_value_in_list  median yearly revenue of the accounts in the list

The rankings compared:  priority = risk^w x revenue^(1-w)  for several w
    w = 0     revenue only
    w = 0.5   value at risk (risk x revenue)
    w = 1     risk only
Pick w in config.RISK_WEIGHT.
Run:   python step9_compare_rankings.py
"""
import os
import pandas as pd

import config
from step4_build_datasets import load_datasets
from step6_compare_models import train_and_test
from step8_score_accounts import priority_score

WEIGHTS = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 1.0]


def compare_rankings(bundle=None):
    b = bundle or load_datasets()
    _, risk = train_and_test(config.CHOSEN_MODEL, b)       # trained on past cutoffs, scored on test
    t = b["test"].assign(risk=risk)
    lost_revenue = t.annual_value * t.churn                  # revenue of accounts that really churned
    k = int(len(t) * config.TOP_SHARE)

    rows = []
    for w in WEIGHTS:
        top = priority_score(t.risk, t.annual_value, w).sort_values(ascending=False).index[:k]
        label = {0.0: "revenue only", 0.5: "value at risk", 1.0: "risk only"}.get(w, "")
        rows.append({
            "RISK_WEIGHT": w, "meaning": label,
            "churners_reached": int(t.loc[top, "churn"].sum()),
            "share_of_churners": t.loc[top, "churn"].sum() / t.churn.sum(),
            "revenue_reached": lost_revenue[top].sum() / lost_revenue.sum(),
            "avg_risk_in_list": t.loc[top, "risk"].mean(),
            "median_value_in_list": t.loc[top, "annual_value"].median(),
        })
    return pd.DataFrame(rows), k


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    res, k = compare_rankings()
    print(f"Top {k:,} accounts ({config.TOP_SHARE:.0%}) of the June 2018 test list, model: {config.CHOSEN_MODEL}\n")
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(config.OUTPUT_DIR, "ranking_comparison.csv"), index=False)
