"""
config.py -- every decision in one place.

Change a value here and re-run; no other file should need editing for these choices.
"""
import pandas as pd

# --- Where things are -------------------------------------------------------
DATA_DIR = "data/"        # folder with the challenge CSVs (orders.csv, order_items.csv, ...)
OUTPUT_DIR = "outputs/"   # everything the scripts produce goes here

# --- "Today" ------------------------------------------------------------------
# The data runs through 31 Aug 2018, so the first day *after* the data is 1 Sep 2018.
DATA_END = pd.Timestamp("2018-09-01")

# --- Churn definition (DECISION) ---------------------------------------------
MIN_ORDERS = 3      # an account needs this many orders before we call it "established"
CHURN_DAYS = 90     # an established account is churned after this many days without an order

# --- Backtesting set-up (DECISION) -------------------------------------------
# A "cutoff" is a pretend "today" in the past. At each cutoff we build features from
# what was known before it, and check what happened in the CHURN_DAYS after it.
#
# Test cutoff: the latest cutoff whose 90-day outcome window still fits in the data.
TEST_CUTOFF = DATA_END - pd.Timedelta(days=CHURN_DAYS)          # 3 June 2018
# Training cutoffs: their outcome windows must END before the test cutoff starts,
# otherwise the model would learn from the same months it is tested on.
TRAIN_CUTOFFS = [pd.Timestamp(d) for d in ["2017-09-01", "2017-12-01", "2018-03-01"]]

for c in TRAIN_CUTOFFS:
    assert c + pd.Timedelta(days=CHURN_DAYS) <= TEST_CUTOFF, f"cutoff {c.date()} overlaps the test window"

# --- Model choice (DECISION, fill in after running step6) ---------------------
# Must be one of the names in step5_models.MODELS.
CHOSEN_MODEL = "Logistic regression"

# --- Prioritisation (DECISION) ------------------------------------------------
TOP_SHARE = 0.10    # used in evaluation: "how many churners are in the riskiest 10%?"

# priority_score = churn_risk ** RISK_WEIGHT  x  annual_value ** (1 - RISK_WEIGHT)
#   RISK_WEIGHT = 0    -> rank by revenue only
#   RISK_WEIGHT = 0.5  -> same order as value at risk (risk x revenue)
#   RISK_WEIGHT = 1    -> rank by churn risk only
# step9_compare_rankings.py shows what each choice would have caught in the test period.
RISK_WEIGHT = 0.6

# Segments shown in the app (risk x value matrix)
HIGH_RISK = 0.40    # churn_risk at or above this = "high risk" (about the riskiest 15% today)
HIGH_VALUE_QUANTILE = 0.50  # annual_value above the median of active accounts = "high value"

RANDOM_STATE = 42
