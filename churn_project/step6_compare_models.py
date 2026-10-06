"""
STEP 6 -- Train every model in step5_models.MODELS on the training cutoffs,
and test it on the later test cutoff.

What the columns in the results table mean:
    AUC                 0.5 = random guessing, 1.0 = perfect ranking. "If I pick one churner and one
                        non-churner at random, how often does the model rank the churner higher?"
    avg_precision       like AUC but focused on the top of the list (where reps work)
    churn_rate_top10    of the 10% of accounts the model calls riskiest, how many really churned
    lift_top10          churn_rate_top10 / overall churn rate (2.0 = twice as good as random)
    churners_caught     share of ALL churners that sit in that riskiest 10%

Pick the model you want in config.CHOSEN_MODEL. If two are close, prefer the simpler one.
Run:   python step6_compare_models.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from step4_build_datasets import load_datasets
from step5_models import MODELS, features_for


def evaluate(y_true, p, top_share=config.TOP_SHARE):
    y_true = np.asarray(y_true)
    k = max(1, int(len(y_true) * top_share))
    top = np.argsort(-p)[:k]
    return {
        "AUC": roc_auc_score(y_true, p),
        "avg_precision": average_precision_score(y_true, p),
        "churn_rate_top10": y_true[top].mean(),
        "lift_top10": y_true[top].mean() / y_true.mean(),
        "churners_caught": y_true[top].sum() / y_true.sum(),
    }


def train_and_test(name, bundle):
    cols = features_for(name, bundle["features"])
    model = MODELS[name]["make"]()
    model.fit(bundle["train"][cols], bundle["train"].churn)
    p = model.predict_proba(bundle["test"][cols])[:, 1]
    return model, p


def compare(bundle=None):
    bundle = bundle or load_datasets()
    rows = []
    for name in MODELS:
        _, p = train_and_test(name, bundle)
        rows.append({"model": name, **evaluate(bundle["test"].churn, p)})
    return pd.DataFrame(rows).sort_values("AUC", ascending=False)


if __name__ == "__main__":
    b = load_datasets()
    print(f"train: {len(b['train']):,} rows | test: {len(b['test']):,} rows "
          f"(cutoff {config.TEST_CUTOFF.date()}, churn rate {b['test'].churn.mean():.1%})\n")
    res = compare(b)
    print(res.round(3).to_string(index=False))
    res.to_csv(os.path.join(config.OUTPUT_DIR, "model_comparison.csv"), index=False)
    print(f"\nCurrently chosen in config.py: {config.CHOSEN_MODEL!r}")
