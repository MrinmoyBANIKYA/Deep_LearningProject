import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import warnings
import random

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set aesthetic style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300

def main():
    # 1. Load Model and Data
    print("Loading model and data...")
    if not os.path.exists('xgb_best.pkl'):
        print("Error: xgb_best.pkl not found. Please run xgboost_baseline.py first.")
        return
        
    model = joblib.load('xgb_best.pkl')
    df = pd.read_csv('msme_data.csv')
    X = df.drop('default', axis=1)
    y = df['default']

    # 2. Compute SHAP values using Booster workaround
    print("Computing SHAP values using XGBoost internal booster (bypass SHAP/XGBoost compatibility issue)...")
    booster = model.get_booster()
    # predict(..., pred_contribs=True) returns SHAP values
    # Output is (samples, features + 1), where the last column is the expected value
    shap_values_raw = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    
    shap_v = shap_values_raw[:, :-1]
    expected_val = shap_values_raw[:, -1][0]

    # Create Explanation object for modern SHAP plotting
    shap_explanation = shap.Explanation(
        values=shap_v,
        base_values=expected_val,
        data=X.values,
        feature_names=X.columns.tolist()
    )

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # 3. Generate and Save Plots
    print("Generating SHAP plots...")

    # a) Global feature importance bar chart
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_explanation.values, X, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP Bar Plot)")
    plt.tight_layout()
    plt.savefig('data/shap_global_bar.png', dpi=300)
    print("Saved data/shap_global_bar.png")
    plt.close()

    # b) Beeswarm plot showing direction of feature effects
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_explanation.values, X, show=False)
    plt.title("SHAP Beeswarm Plot (Feature Effects)")
    plt.tight_layout()
    plt.savefig('data/shap_beeswarm.png', dpi=300)
    print("Saved data/shap_beeswarm.png")
    plt.close()

    # c) Dependency plot for upi_monthly_vol
    plt.figure(figsize=(10, 6))
    # Correcting column name based on dataset exploration
    col_name = "upi_monthly_vol" if "upi_monthly_vol" in X.columns else "upi_monthly_txn_volume"
    shap.dependence_plot(col_name, shap_explanation.values, X, show=False)
    plt.title(f"SHAP Dependence Plot: {col_name}")
    plt.tight_layout()
    plt.savefig('data/shap_dependency_upi.png', dpi=300)
    print("Saved data/shap_dependency_upi.png")
    plt.close()

    # d) Individual waterfall explanations for 3 randomly selected thin-file borrowers
    print("Generating waterfall plots for 3 random thin-file samples...")
    # Defining thin-file as bureau_credit_age < 24
    thin_file_indices = df[df['bureau_credit_age'] < 24].index.tolist()
    
    if len(thin_file_indices) >= 3:
        selected_indices = random.sample(thin_file_indices, 3)
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(12, 6))
            # waterfall_plot expects an Explanation object for a single sample
            # We construct it from the main explanation object
            single_explanation = shap_explanation[idx]
            shap.plots.waterfall(single_explanation, show=False)
            plt.title(f"SHAP Waterfall: Thin-File Borrower (Sample {idx})")
            plt.tight_layout()
            plt.savefig(f'data/shap_waterfall_thinfile_{i+1}.png', dpi=300)
            print(f"Saved data/shap_waterfall_thinfile_{i+1}.png")
            plt.close()
    else:
        print(f"Warning: Only {len(thin_file_indices)} thin-file samples found. Skipping waterfall plots.")

    print("\nSHAP analysis enhancement complete. All plots saved to research/data/")

if __name__ == "__main__":
    main()
