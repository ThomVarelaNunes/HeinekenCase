# Churn-risk pipeline (HEINEKEN × AISO challenge)

A step-by-step pipeline: define churn → build features → compare models → explain → score accounts.
Each step is one file that does one thing, explains itself at the top, and can be run on its own.

## How to run

1. Put the challenge CSVs in a folder called `data/` next to these files (or change `DATA_DIR` in `config.py`).
2. `pip install pandas numpy scipy scikit-learn`
3. `python run_all.py` (about 30 seconds), or run the steps one by one:

```
python step1_load_data.py         # sanity check: rows, dates, accounts
python step2_churn_definition.py  # active / churned / too few orders, churn rate per cutoff
python step4_build_datasets.py    # build training + test examples
python step6_compare_models.py    # compare all models on the test period
python step7_feature_analysis.py  # which features and behaviours matter
python step8_score_accounts.py    # score every account today, with reasons
python step9_compare_rankings.py  # which ranking reaches the most churners / lost revenue
```

In Google Colab: upload the folder, then `!python run_all.py`.

## The flow

```
config.py ─ all decisions (churn = 3+ orders then 90 days silent, cutoffs, chosen model)
   │
step1  load the CSVs into 4 tables (orders, items, reviews, payments)
step2  churn definition: who is active / churned / too small on any date + the label
step3  features, grouped by behaviour (rhythm, value, assortment, experience, cost, buyers)
step4  snapshots: features at past cutoffs + what happened in the next 90 days
          train = Sep 2017, Dec 2017, Mar 2018      test = Jun 2018 (later, never seen)
step5  the models (a plug-in list: add your own here)
step6  train every model, test on the later period, compare
step7  feature analysis: each feature alone, each behaviour group, L1 selection
step8  retrain the chosen model, score today: risk, yearly revenue, value at risk,
          priority, segment, top 3 reasons with their size in percentage points
step9  compare rankings (revenue only ... value at risk ... risk only) on the test period
```

## What the output contains (`outputs/scored_accounts.csv`)

| Column | Meaning |
|---|---|
| `churn_risk` | probability of no order in the next 90 days |
| `annual_value` | yearly revenue (last 365 days; scaled up for accounts younger than a year) |
| `value_at_risk` | churn_risk × annual_value: expected revenue lost if nothing is done |
| `priority_score`, `priority_rank` | the ranking: risk^w × revenue^(1−w), w = `config.RISK_WEIGHT` |
| `segment` | risk × value matrix: Save now / Rescue cheaply / Protect / Monitor |
| `reason_1..3`, `reason_1..3_points` | biggest actionable reasons, and how many percentage points of risk each adds |
| `winback_rank` | for churned accounts: win-back order by what they used to spend |

## Where you make decisions

| What | Where |
|---|---|
| Churn definition (90 days, 3+ orders) | `config.py` |
| Which past dates to train/test on | `config.py` |
| Which model is used for scoring | `config.CHOSEN_MODEL` (pick after step 6) |
| Add / change a model | `step5_models.py` → write a function, add it to `MODELS` |
| Which features a model uses | `"features"` in its `MODELS` entry |
| Add / change a feature | the matching group function in `step3_features.py` |
| Which features can be "reasons", and their wording | `REASON_TEXT` in `step8_score_accounts.py` |
| Ranking: revenue vs risk balance | `config.RISK_WEIGHT` (check step 9 first) |
| What counts as high risk / high value | `config.HIGH_RISK`, `config.HIGH_VALUE_QUANTILE` |

## Adding a different logistic regression (example)

In `step5_models.py`:

```python
def logistic_regression_balanced():
    return make_pipeline(PREPARE, StandardScaler(),
                         LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))

MODELS["LR balanced"] = {"make": logistic_regression_balanced, "features": None}
```

Run `python step6_compare_models.py` and it appears in the table. To use only some features:
`"features": ["overdue_ratio", "freight_share", "last_order_late", "n_orders"]`.

## The models included

- **Baseline: days since last order**: no learning; the simplest possible rule.
- **BG/NBD**: the classic "Buy Till You Die" model for customers without contracts. Uses only order rhythm. Written out in `bgnbd.py`.
- **Logistic regression**: weighted sum of all features; easy to explain.
- **Logistic regression (L1)**: same, but switches off weak features automatically.
- **Gradient boosting**: many small decision trees; flexible, harder to explain.

## Honest testing

Features only use information known before each cutoff (a late delivery counts once it arrived or its
promised date passed; a review counts once written). The test cutoff (June 2018) is later than every
training outcome, so the test score shows how the model would have done on months it never saw.
