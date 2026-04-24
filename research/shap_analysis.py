import pandas as pd
import numpy as np
import shap
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import warnings

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

    # 2. Compute SHAP values
    print("Computing SHAP values using XGBoost internal booster (bypass SHAP/XGBoost 3.x compatibility issue)...")
    booster = model.get_booster()
    # predict(..., pred_contribs=True) returns SHAP values
    # The output is (samples, features + 1), where the last column is the expected value (bias)
    shap_values_raw = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    
    shap_v = shap_values_raw[:, :-1]
    expected_val = shap_values_raw[:, -1][0] # Base value is the same for all samples

    # Create Explanation object for compatibility with SHAP plotting functions
    shap_explanation = shap.Explanation(
        values=shap_v,
        base_values=expected_val,
        data=X.values,
        feature_names=X.columns.tolist()
    )

    # 3. Generate and Save Plots
    print("Generating SHAP plots...")

    # a) Global bar plot (mean |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_explanation, show=False)
    plt.title("Global Feature Importance (Mean |SHAP|)")
    plt.tight_layout()
    plt.savefig('shap_global_bar.png', dpi=300)
    print("Saved shap_global_bar.png")
    plt.close()

    # b) Beeswarm summary plot (top 15 features)
    plt.figure(figsize=(12, 8))
    # Note: beeswarm plot in recent SHAP versions uses the Explanation object
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm Summary Plot (Top 15 Features)")
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=300)
    print("Saved shap_beeswarm.png")
    plt.close()

    # c) SHAP dependence plot for gst_filing_rate
    plt.figure(figsize=(10, 6))
    # dependence_plot still works with arrays
    shap.dependence_plot("gst_filing_rate", shap_v, X, show=False)
    plt.title("SHAP Dependence Plot: GST Filing Rate")
    plt.tight_layout()
    plt.savefig('shap_dep_gst.png', dpi=300)
    print("Saved shap_dep_gst.png")
    plt.close()

    # d) SHAP dependence plot for upi_consistency with interaction=cash_flow_seasonality
    plt.figure(figsize=(10, 6))
    shap.dependence_plot("upi_consistency", shap_v, X, 
                         interaction_index="cash_flow_seasonality", show=False)
    plt.title("SHAP Dependence: UPI Consistency (Interaction with Cash Flow Seasonality)")
    plt.tight_layout()
    plt.savefig('shap_dep_upi.png', dpi=300)
    print("Saved shap_dep_upi.png")
    plt.close()

    # 4. Print Top 10 Features Table
    mean_abs_shap = np.abs(shap_v).mean(0)
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Mean |SHAP|': mean_abs_shap
    }).sort_values(by='Mean |SHAP|', ascending=False)
    
    print("\n" + "="*40)
    print("TOP 10 FEATURES BY MEAN |SHAP|")
    print("="*40)
    print(importance_df.head(10).to_string(index=False))
    print("="*40 + "\n")

    # 5. Waterfall Plots for Thin-File Samples
    print("Generating waterfall plots for thin-file samples...")
    thin_file_mask = df['bureau_credit_age'] < 24
    
    try:
        default_indices = df[thin_file_mask & (df['default'] == 1)].index
        nondefault_indices = df[thin_file_mask & (df['default'] == 0)].index
        
        if len(default_indices) > 0 and len(nondefault_indices) > 0:
            default_idx = default_indices[0]
            nondefault_idx = nondefault_indices[0]

            # Save Default Waterfall
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_explanation[default_idx], show=False)
            plt.title(f"SHAP Waterfall: Thin-File Default (Sample {default_idx})")
            plt.tight_layout()
            plt.savefig('shap_waterfall_default.png', dpi=300)
            print("Saved shap_waterfall_default.png")
            plt.close()

            # Save Non-Default Waterfall
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_explanation[nondefault_idx], show=False)
            plt.title(f"SHAP Waterfall: Thin-File Non-Default (Sample {nondefault_idx})")
            plt.tight_layout()
            plt.savefig('shap_waterfall_nondefault.png', dpi=300)
            print("Saved shap_waterfall_nondefault.png")
            plt.close()
        else:
            print("Warning: Could not find thin-file samples for both classes.")
            
    except Exception as e:
        print(f"Error generating waterfall plots: {e}")

if __name__ == "__main__":
    main()
