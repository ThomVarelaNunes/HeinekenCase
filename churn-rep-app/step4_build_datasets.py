"""
STEP 4 -- Turn the history into training and test examples.

For every cutoff:
    1. find the accounts that are ACTIVE on that date              (step 2)
    2. describe them with features known before that date           (step 3)
    3. attach the label: did they churn in the next CHURN_DAYS?     (step 2)

Training cutoffs and the test cutoff come from config.py. This is a split by TIME,
not a random split: the test cutoff is LATER than every training outcome, so the
test score shows how well the model would have worked on months it had never seen.
Many accounts appear in both (with older data in training); no information from
after the test cutoff is used anywhere. That is the honest way to test a churn model.

The result is saved to outputs/datasets.pkl so later steps don't rebuild it.
Run:   python step4_build_datasets.py
"""
import os
import pickle
import pandas as pd

import config
from step1_load_data import load_tables
from step2_churn_definition import churn_label
from step3_features import build_features

DATASET_FILE = os.path.join(config.OUTPUT_DIR, "datasets.pkl")


def snapshot(tables, cutoff):
    """One row per account active on `cutoff`: features + churn label."""
    y = churn_label(tables["orders"], cutoff)
    X, column_groups = build_features(tables, cutoff, y.index)
    df = X.assign(churn=y, cutoff=cutoff)
    return df, column_groups


def build_datasets(tables=None):
    tables = tables or load_tables()
    train_parts = []
    for c in config.TRAIN_CUTOFFS:
        df, groups = snapshot(tables, c)
        train_parts.append(df)
    test, groups = snapshot(tables, config.TEST_CUTOFF)
    train = pd.concat(train_parts)

    bundle = {
        "train": train,
        "test": test,
        "features": list(groups),          # all feature column names
        "column_groups": groups,           # {feature: behaviour group}
        "config": {"MIN_ORDERS": config.MIN_ORDERS, "CHURN_DAYS": config.CHURN_DAYS},
    }
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(DATASET_FILE, "wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


def load_datasets():
    """Load the saved datasets; rebuild them if the churn definition in config.py changed."""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "rb") as fh:
            bundle = pickle.load(fh)
        if bundle["config"] == {"MIN_ORDERS": config.MIN_ORDERS, "CHURN_DAYS": config.CHURN_DAYS}:
            return bundle
        print("config changed -> rebuilding datasets")
    return build_datasets()


if __name__ == "__main__":
    b = build_datasets()
    for name in ["train", "test"]:
        df = b[name]
        print(f"{name:<5}: {len(df):>6,} rows, churn rate {df.churn.mean():.1%}, "
              f"cutoffs {sorted(d.date().isoformat() for d in df.cutoff.unique())}")
    print(f"{len(b['features'])} features, saved to {DATASET_FILE}")
