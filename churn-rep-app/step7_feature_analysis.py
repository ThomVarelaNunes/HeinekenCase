"""
STEP 7 -- Which inputs actually matter? (the "why these features?" table for the jury)

Three views, all measured on the TEST period:

  A. Each feature on its own        how well does this ONE feature rank churners? (AUC)
                                    and in which direction (higher value = more or less churn)?
  B. Each behaviour group           AUC of the chosen model using ONLY that group (how much it knows),
                                    and WITHOUT that group (how much it adds that the others don't).
                                    Groups overlap a lot (busy accounts score high on orders, spend,
                                    buyers AND categories), so "without" changes are often small.
  C. L1 logistic regression         which features survive automatic selection, and their weights

Run:   python step7_feature_analysis.py
Then decide: keep the features/groups that help; drop the ones that don't
(set "features" for your model in step5_models.MODELS).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from step4_build_datasets import load_datasets
from step5_models import MODELS, logistic_regression_l1


def single_feature_report(b):
    rows = []
    y = b["test"].churn
    for f in b["features"]:
        x = b["test"][f].fillna(0)
        auc = roc_auc_score(y, x) if x.nunique() > 1 else 0.5
        rows.append({"feature": f, "group": b["column_groups"][f],
                     "AUC_alone": max(auc, 1 - auc),
                     "direction": "higher = MORE churn" if auc >= 0.5 else "higher = LESS churn"})
    return pd.DataFrame(rows).sort_values("AUC_alone", ascending=False)


def group_ablation(b, model_name=config.CHOSEN_MODEL):
    make = MODELS[model_name]["make"]
    feats = b["features"]
    ytr, yte = b["train"].churn, b["test"].churn

    def auc_with(cols):
        m = make().fit(b["train"][cols], ytr)
        return roc_auc_score(yte, m.predict_proba(b["test"][cols])[:, 1])

    full = auc_with(feats)
    rows = [{"group": "(all features)", "n_features": len(feats), "AUC_only_this_group": full,
             "AUC_without_it": np.nan, "change_if_removed": np.nan}]
    for g in sorted(set(b["column_groups"].values())):
        only = [f for f in feats if b["column_groups"][f] == g]
        keep = [f for f in feats if b["column_groups"][f] != g]
        without = auc_with(keep)
        rows.append({"group": g, "n_features": len(only), "AUC_only_this_group": auc_with(only),
                     "AUC_without_it": without, "change_if_removed": without - full})
    return pd.DataFrame(rows)


def l1_selection(b):
    m = logistic_regression_l1().fit(b["train"][b["features"]], b["train"].churn)
    w = pd.Series(m[-1].coef_[0], index=b["features"])
    out = pd.DataFrame({"feature": w.index, "group": [b["column_groups"][f] for f in w.index],
                        "weight": w.values})
    out["kept"] = out.weight.abs() > 1e-6
    return out.reindex(out.weight.abs().sort_values(ascending=False).index)


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    b = load_datasets()

    print("A. EACH FEATURE ON ITS OWN (test period)")
    a = single_feature_report(b)
    print(a.round(3).to_string(index=False), "\n")

    print(f"B. BEHAVIOUR GROUPS: ONLY THIS GROUP vs WITHOUT IT  (model: {config.CHOSEN_MODEL})")
    g = group_ablation(b)
    print(g.round(3).to_string(index=False), "\n")

    print("C. L1 LOGISTIC REGRESSION: WHICH FEATURES SURVIVE  (weights on standardised features;")
    print("   positive = raises churn risk, 0 = dropped)")
    c = l1_selection(b)
    print(c.round(3).to_string(index=False))
    print(f"\nkept {c.kept.sum()} of {len(c)} features")

    a.to_csv(os.path.join(config.OUTPUT_DIR, "feature_single.csv"), index=False)
    g.to_csv(os.path.join(config.OUTPUT_DIR, "feature_groups.csv"), index=False)
    c.to_csv(os.path.join(config.OUTPUT_DIR, "feature_l1.csv"), index=False)
