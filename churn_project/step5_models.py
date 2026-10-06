"""
STEP 5 -- The models to compare. This is the file to edit when you want to try a new one.

Every entry in MODELS has:
    "make":     a function that returns a fresh, untrained model with .fit() and .predict_proba()
    "features": which feature columns it may use (None = all of them)

ADDING YOUR OWN MODEL (e.g. a different logistic regression):
    1. write a function that returns the model (copy one of the examples below)
    2. add it to MODELS with a name
    3. run step6_compare_models.py -- it appears in the comparison table automatically

All models get the same preparation (`prepare`): skewed numbers are log-transformed
and gaps filled, so differences in the results come from the model, not the plumbing.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

import config
from bgnbd import BGNBD

# Counts and money are very skewed (a few huge accounts); log1p makes them behave.
SKEWED = ["n_orders", "active_months", "tenure_days", "days_since_last", "usual_gap_days",
          "orders_last_90d", "usual_orders_per_90d", "spend_total", "spend_last_365d", "annual_value",
          "avg_order_value", "n_categories", "categories_last_90d", "dropped_categories", "dropped_spend",
          "max_days_late_90d", "n_buyers", "repeat_buyers", "overdue_ratio"]


def prepare(X):
    X = X.copy()
    for c in SKEWED:
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    return X.fillna(0)


PREPARE = FunctionTransformer(prepare)


# --------------------------------------------------------------------------- model makers
class OneColumnScore(BaseEstimator, ClassifierMixin):
    """Baseline 'model': rank accounts by one column (e.g. days since last order). No learning."""
    def __init__(self, column="days_since_last"):
        self.column = column

    def fit(self, X, y=None):
        self.max_ = X[self.column].max(); self.classes_ = np.array([0, 1]); return self

    def predict_proba(self, X):
        p = (X[self.column] / self.max_).clip(0, 1).to_numpy()
        return np.column_stack([1 - p, p])


def baseline_days_since_last():
    return OneColumnScore("days_since_last")


def bgnbd():
    return BGNBD(horizon_days=config.CHURN_DAYS)


def logistic_regression():
    # C = how strongly to hold the weights back (smaller = simpler model). TODO: try 0.01 - 10
    return make_pipeline(PREPARE, StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def logistic_regression_l1():
    # L1 pushes weak features' weights to exactly zero = automatic feature selection.
    # Step 7 prints which features it kept.
    return make_pipeline(PREPARE, StandardScaler(),
                         LogisticRegression(l1_ratio=1, solver="saga", C=0.05, max_iter=5000))


def gradient_boosting():
    # Many small decision trees, each fixing the previous ones' mistakes.
    # Kept deliberately small/cautious because the data is small and noisy.
    # TODO (optional): monotonic constraints, e.g. more usual orders may never raise risk:
    #   monotonic_cst={"usual_orders_per_90d": -1, "overdue_ratio": 1}
    return make_pipeline(PREPARE, HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=100,
        l2_regularization=1.0, random_state=config.RANDOM_STATE))


# def my_model():
#     """TEMPLATE -- copy this, change the inside, and add it to MODELS below."""
#     return make_pipeline(PREPARE, StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced"))


RHYTHM_ONLY = ["n_orders", "tenure_days", "days_since_last"]

MODELS = {
    "Baseline: days since last order": {"make": baseline_days_since_last, "features": ["days_since_last"]},
    "BG/NBD (order rhythm only)":      {"make": bgnbd,                    "features": RHYTHM_ONLY},
    "Logistic regression":             {"make": logistic_regression,      "features": None},
    "Logistic regression (L1)":        {"make": logistic_regression_l1,   "features": None},
    "Gradient boosting":               {"make": gradient_boosting,        "features": None},
    # "My model":                      {"make": my_model,                 "features": None},
}


def features_for(name, all_features):
    """The feature columns a given model uses."""
    return MODELS[name]["features"] or all_features
