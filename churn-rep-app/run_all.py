"""
Run the whole pipeline in order. Each step can also be run on its own (python stepN_....py).

    step 2  churn definition      -> how many accounts are active / churned / too small
    step 4  build datasets        -> training + test examples (rebuilds from scratch)
    step 6  compare models        -> outputs/model_comparison.csv
    step 7  feature analysis      -> outputs/feature_*.csv
    step 8  score accounts today  -> outputs/scored_accounts.csv
    step 9  compare rankings      -> outputs/ranking_comparison.csv
    step 10 export app data       -> outputs/app_data.json
    app     build the rep app     -> app/tapline.html
    step 11 mock today's route    -> outputs/mobile_data.json
    mobile  build the mobile app  -> app/fieldline.html
"""
import runpy

for step in ["step2_churn_definition", "step4_build_datasets", "step6_compare_models",
             "step7_feature_analysis", "step8_score_accounts", "step9_compare_rankings",
             "step10_export_app_data", "app.build_app",
             "step11_build_route", "app.build_mobile"]:
    print("\n" + "=" * 100 + f"\n{step}\n" + "=" * 100)
    runpy.run_module(step, run_name="__main__")
