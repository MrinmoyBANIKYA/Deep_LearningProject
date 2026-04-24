import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os

# IEEE Publication Style Settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "legend.fontsize": 8,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

def main():
    print("Generating figures and tables for IEEE publication...")
    
    # 1. Load Data
    df = pd.read_csv('msme_data.csv')
    X = df.drop('default', axis=1)
    y = df['default']
    feature_names = X.columns.tolist()
    
    # Load previously generated results
    res_df = pd.read_csv('results.csv')
    
    # Stratified 5-Fold CV setup (Same as baseline)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metrics containers
    model_metrics = {
        'Logistic Regression': {'auc': [], 'ap': [], 'f1': [], 'brier': []},
        'Random Forest': {'auc': [], 'ap': [], 'f1': [], 'brier': []},
        'XGBoost': {'auc': [], 'ap': [], 'f1': [], 'brier': []},
        'TabNet': {'auc': [], 'ap': [], 'f1': [], 'brier': []},
        'Hybrid Ensemble': {'auc': [], 'ap': [], 'f1': [], 'brier': []}
    }
    
    # Probability containers for ROC curves
    all_probs = {
        'Logistic Regression': np.zeros(len(y)),
        'Random Forest': np.zeros(len(y)),
        'XGBoost': res_df['xgb_prob'].values,
        'TabNet': res_df['tabnet_prob'].values,
        'Hybrid Ensemble': res_df['hybrid_stacking_prob'].values
    }

    # Generate LR and RF predictions for comparison
    print("Running baseline models (LR, RF) for comparison...")
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # LR
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_probs = lr.predict_proba(X_test)[:, 1]
        all_probs['Logistic Regression'][test_idx] = lr_probs
        
        # RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_probs = rf.predict_proba(X_test)[:, 1]
        all_probs['Random Forest'][test_idx] = rf_probs

    # Calculate metrics for all models
    for name, probs in all_probs.items():
        for train_idx, test_idx in skf.split(X, y):
            y_true_fold = y.iloc[test_idx]
            y_prob_fold = probs[test_idx]
            
            # Use Youden's J for F1
            fpr, tpr, thresholds = roc_curve(y_true_fold, y_prob_fold)
            best_thresh = thresholds[np.argmax(tpr - fpr)]
            y_pred_fold = (y_prob_fold >= best_thresh).astype(int)
            
            model_metrics[name]['auc'].append(roc_auc_score(y_true_fold, y_prob_fold))
            model_metrics[name]['ap'].append(average_precision_score(y_true_fold, y_prob_fold))
            model_metrics[name]['f1'].append(f1_score(y_true_fold, y_pred_fold))
            model_metrics[name]['brier'].append(brier_score_loss(y_true_fold, y_prob_fold))

    # --- 1. ROC Comparison Plot ---
    print("Generating fig_roc_comparison.png...")
    plt.figure(figsize=(5, 4.5))
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    colors = ['#000000', '#333333', '#666666', '#999999', '#CCCCCC']
    
    for i, (name, probs) in enumerate(all_probs.items()):
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})', 
                 linestyle=line_styles[i], color='black', alpha=0.8, linewidth=1.2)
        
    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fig_roc_comparison.png', dpi=600)
    plt.close()

    # --- 2. Feature Importance Comparison ---
    print("Generating fig_feature_importance.png...")
    # Get XGBoost importance (Gain)
    xgb_model = joblib.load('xgb_best.pkl')
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    
    # Get SHAP importance (from previous run)
    # Since I don't want to re-run SHAP in this script (too slow), I'll use the values I just saw
    # or re-calculate for just the top 10 if needed. 
    # Actually, let's just use the values from the previous turn's output for accuracy.
    shap_imp_data = {
        'repayment_flag': 1.2241, 'cash_flow_seasonality': 0.4283, 
        'upi_gst_interaction': 0.3549, 'upi_monthly_vol': 0.2467, 
        'gst_filing_rate': 0.1987, 'log_upi_vol': 0.1138, 
        'credit_util_ratio': 0.1110, 'upi_consistency': 0.1046, 
        'bureau_credit_age': 0.0949, 'alt_signal_composite': 0.0930
    }
    shap_imp = pd.Series(shap_imp_data).sort_values(ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left: XGBoost Gain
    xgb_imp.plot(kind='barh', ax=ax1, color='#444444')
    ax1.set_title('XGBoost Importance (Gain)')
    ax1.set_xlabel('Score')
    ax1.invert_yaxis()
    
    # Right: SHAP
    shap_imp.plot(kind='barh', ax=ax2, color='#888888')
    ax2.set_title('SHAP Importance (Mean |SHAP|)')
    ax2.set_xlabel('Mean |SHAP|')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('fig_feature_importance.png', dpi=600)
    plt.close()

    # --- 3. Ablation Bar Chart ---
    print("Generating fig_ablation.png...")
    # Ablation results from stacking_ensemble.py output
    ablation_data = {
        'Traditional only': 0.8371,
        'Alternative only': 0.6373,
        'Integrated (Full set)': 0.8526
    }
    ablation_df = pd.Series(ablation_data)
    
    plt.figure(figsize=(6, 3))
    bars = plt.barh(ablation_df.index, ablation_df.values, color=['#CCCCCC', '#888888', '#444444'], edgecolor='black')
    plt.xlim(0.5, 0.9)
    plt.xlabel('AUC-ROC Score')
    plt.title('Feature Ablation Study Results')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center')
        
    plt.tight_layout()
    plt.savefig('fig_ablation.png', dpi=600)
    plt.close()

    # --- 4. LaTeX Tables ---
    print("\n" + "="*50)
    print("TABLE II: PERFORMANCE COMPARISON")
    print("="*50)
    
    latex_table_ii = """
\\begin{table}[h]
\\caption{Comparison of Model Performance (Mean $\\pm$ SD)}
\\centering
\\begin{tabular}{lcccc}
\\hline
Model & AUC-ROC & Avg. Precision & F1-Score & Brier Score \\\\
\\hline
"""
    for name in model_metrics.keys():
        m = model_metrics[name]
        latex_table_ii += f"{name} & {np.mean(m['auc']):.4f}\\pm{np.std(m['auc']):.4f} & {np.mean(m['ap']):.4f}\\pm{np.std(m['ap']):.4f} & {np.mean(m['f1']):.4f}\\pm{np.std(m['f1']):.4f} & {np.mean(m['brier']):.4f}\\pm{np.std(m['brier']):.4f} \\\\\n"
    
    latex_table_ii += """\\hline
\\end{tabular}
\\end{table}
"""
    print(latex_table_ii)

    print("\n" + "="*50)
    print("TABLE III: FEATURE ABLATION STUDY")
    print("="*50)
    
    full_auc = ablation_data['Integrated (Full set)']
    latex_table_iii = f"""
\\begin{{table}}[h]
\\caption{{Feature Ablation Study (AUC-ROC Metrics)}}
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
Feature Subset & AUC-ROC & $\\Delta$ Full \\\\
\\hline
Traditional (Bureau + Derived) & {ablation_data['Traditional only']:.4f} & {ablation_data['Traditional only'] - full_auc:+.4f} \\\\
Alternative (Digital + Derived) & {ablation_data['Alternative only']:.4f} & {ablation_data['Alternative only'] - full_auc:+.4f} \\\\
Integrated (19 Features) & {full_auc:.4f} & -- \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
    print(latex_table_iii)

if __name__ == "__main__":
    main()
