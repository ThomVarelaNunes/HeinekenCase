# Churn risk and rep app (HEINEKEN × AISO challenge)

Predicts which accounts are likely to stop ordering, explains why, ranks them for sales reps,
and turns that into two app prototypes.

| | What | File |
|---|---|---|
| Pipeline | churn definition → features → model comparison → scoring with reasons | `step1` … `step11`, `run_all.py` |
| Mobile app | **Fieldline**: today's stores in route order, next stop with navigation, all accounts | `app/fieldline.html` (also `docs/index.html`) |
| Web app | **Tapline**: desktop worklist with account overview and call brief | `app/tapline.html` (also `docs/web.html`) |
| Experiments | tests that did not make it into the pipeline (EWMA, training on bigger accounts, 2+ orders, product categories) | `experiments/` |

**Key decisions**
- **Churn:** an account with 3+ orders is churned once it goes **2× its own usual ordering gap without an order,
  never less than 90 and never more than 180 days**. An account that orders every 20 days churns after 90 days of
  silence; one that orders every 70 days after 140; one that orders every 247 days after 180.
- **Prediction:** the chance that an active account crosses its own churn line within the next 90 days.
- **Model:** L1 logistic regression (test AUC 0.80 on June 2018; 0.71 on the accounts that could actually cross
  their line, equal to "days since last order", with gradient boosting at 0.72). Chosen over plain logistic
  regression because it switches off duplicate inputs, which keeps the per-account reasons clean.
- **Reasons:** each account's risk above a typical account is split over its causes; a reason is only shown when
  the account's own numbers back it up.
- **Value at risk** = churn risk × yearly revenue. **Priority** = risk^0.6 × revenue^0.4.
- **We also tested** a fixed 90-day definition. It called slow but
  healthy accounts "high risk" just for ordering rarely, so we moved to the rhythm-based line.

**Live demo with GitHub Pages:** Settings → Pages → Source: "Deploy from a branch", branch `main`,
folder `/docs`. The mobile app is then at `https://<user>.github.io/<repo>/` and the web app at
`.../web.html`. Outside Claude, the call brief uses its built-in template instead of AI.

## The pipeline

A step-by-step pipeline: define churn → build features → compare models → explain → score accounts.
Each step is one file that does one thing, explains itself at the top, and can be run on its own.

## How to run

1. Put the challenge CSVs in `data/` (see `data/README.md`; or change `DATA_DIR` in `config.py`).
2. `pip install -r requirements.txt`
3. `python run_all.py` (about a minute; rebuilds both apps too), or run the steps one by one:

```
python step1_load_data.py         # sanity check: rows, dates, accounts
python step2_churn_definition.py  # active / churned / too few orders, churn rate per cutoff
python step4_build_datasets.py    # build training + test examples
python step6_compare_models.py    # compare all models on the test period
python step7_feature_analysis.py  # which features and behaviours matter
python step8_score_accounts.py    # score every account today, with reasons
python step9_compare_rankings.py  # which ranking reaches the most churners / lost revenue
python step10_export_app_data.py  # package the top accounts for the rep app
python app/build_app.py           # build app/tapline.html (one self-contained file)
python step11_build_route.py      # mock route planner output for one Sao Paulo rep
python app/build_mobile.py        # build app/fieldline.html, the mobile rep app
```

In Google Colab: upload the folder, then `!python run_all.py`.

## The flow

```
config.py ─ all decisions (3+ orders, 90-day window, cutoffs, chosen model); step 2 holds the 2x-gap rule
   │
step1  load the CSVs into 4 tables (orders, items, reviews, payments)
step2  churn definition: own churn line = 2x usual gap (90-180 days); who is active / churned / too small + the label
step3  features, grouped by behaviour (rhythm, value, assortment, experience, cost, buyers)
step4  snapshots: features at past cutoffs + whether the account crossed its line in the next 90 days
          train = Sep 2017, Dec 2017, Mar 2018      test = Jun 2018 (later, never seen)
step5  the models (a plug-in list: add your own here)
step6  train every model, test on the later period, compare
step7  feature analysis: each feature alone, each behaviour group, L1 selection
step8  retrain the chosen model, score today: risk, yearly revenue, value at risk,
          priority, segment, top 3 reasons with their size in percentage points
step9  compare rankings (revenue only ... value at risk ... risk only) on the test period
step10 export the top 300 active accounts (+ 100 churned for the web app's win-back tab), with history, complaint and next best action
app    the rep app: worklist, account docket, AI call brief (template fallback outside Claude)
step11 mock route: today's 6 Save-now stores near the depot in driving order (no clock times); calls to Rescue-cheaply + Protect accounts; all active accounts
mobile Fieldline: Today (map + stores in order) · Next stop (navigate, why, brief, checklist, outcome) · Accounts (all active, 6 sort orders)
```

## What the output contains (`outputs/scored_accounts.csv`)

| Column | Meaning |
|---|---|
| `churn_risk` | probability the account crosses its own churn line within the next 90 days |
| `annual_value` | yearly revenue (last 365 days; scaled up for accounts younger than a year) |
| `value_at_risk` | churn_risk × annual_value: expected revenue lost if nothing is done |
| `priority_score`, `priority_rank` | the ranking: risk^w × revenue^(1−w), w = `config.RISK_WEIGHT` |
| `segment` | risk × value matrix: Save now / Rescue cheaply / Protect / Monitor |
| `reason_1..3`, `reason_1..3_points` | biggest actionable reasons, and how many percentage points of risk each adds |
| `winback_rank` | for churned accounts: win-back order by what they used to spend |

## Where you make decisions

| What | Where |
|---|---|
| Churn definition (3+ orders; 2x gap, 90-180 days) | `config.py` (MIN_ORDERS, CHURN_DAYS), `step2_churn_definition.py` (RHYTHM_MULT, MAX_CHURN_DAYS) |
| Which past dates to train/test on | `config.py` |
| Which model is used for scoring | `config.CHOSEN_MODEL` (pick after step 6) |
| Add / change a model | `step5_models.py` → write a function, add it to `MODELS` |
| Which features a model uses | `"features"` in its `MODELS` entry |
| Add / change a feature | the matching group function in `step3_features.py` |
| Which features can be "reasons", and their wording | `REASON_TEXT` in `step8_score_accounts.py` |
| Ranking: revenue vs risk balance | `config.RISK_WEIGHT` (check step 9 first) |
| What counts as high risk / high value | `config.HIGH_RISK`, `config.HIGH_VALUE_QUANTILE` |
| Next best action and offer per reason, channels | `ACTIONS`, `CHANNEL` in `step10_export_app_data.py` |

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
