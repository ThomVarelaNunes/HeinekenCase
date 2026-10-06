"""
Run the whole pipeline in order. Each step can also be run on its own (python stepN_....py).

    step 2  churn definition      -> how many accounts are active / churned / too small
    step 4  build datasets        -> training + test examples (rebuilds from scratch)
    step 6  compare models        -> outputs/model_comparison.csv
    step 7  feature analysis      -> outputs/feature_*.csv
    step 8  score accounts today  -> outputs/scored_accounts.csv
    step 9  compare rankings      -> outputs/ranking_comparison.csv
"""
import runpy

for step in ["step2_churn_definition", "step4_build_datasets", "step6_compare_models",
             "step7_feature_analysis", "step8_score_accounts", "step9_compare_rankings"]:
    print("\n" + "=" * 100 + f"\n{step}\n" + "=" * 100)
    runpy.run_module(step, run_name="__main__")
