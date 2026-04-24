import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             brier_score_loss, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

warnings.filterwarnings('ignore')

def find_optimal_threshold(y_true, y_probs):
    """Find the optimal probability threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def get_stacking_data(X_df, y_df, n_splits=5):
    """
    Generate OOF predictions for XGBoost and TabNet to use as meta-features.
    Also returns the base model metrics and fold indices.
    """
    X = X_df.values
    y = y_df.values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    
    fold_metrics = {'xgb_auc': [], 'tab_auc': []}
    
    # Load best XGB params if available, otherwise use defaults from provided config
    xgb_params = {
        'n_estimators': 330, 'max_depth': 10, 'learning_rate': 0.0176,
        'subsample': 0.686, 'colsample_bytree': 0.693,
        'reg_alpha': 0.233, 'reg_lambda': 0.0058,
        'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': 42
    }
    
    if os.path.exists('xgb_best.pkl'):
        try:
            best_model = joblib.load('xgb_best.pkl')
            # Extract params but keep scale_pos_weight dynamic
            loaded_params = best_model.get_params()
            for k in xgb_params.keys():
                if k in loaded_params:
                    xgb_params[k] = loaded_params[k]
        except:
            pass

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # XGBoost training
        current_xgb_params = xgb_params.copy()
        current_xgb_params['scale_pos_weight'] = len(y_train[y_train==0]) / len(y_train[y_train==1])
        xgb_model = xgb.XGBClassifier(**current_xgb_params)
        xgb_model.fit(X_train, y_train)
        
        xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = xgb_val_probs
        fold_metrics['xgb_auc'].append(roc_auc_score(y_val, xgb_val_probs))
        
        # TabNet training
        tab_model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.3, n_independent=2, n_shared=2,
            lambda_sparse=0.001, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02),
            scheduler_params=dict(gamma=0.95, step_size=1000),
            scheduler_fn=StepLR, verbose=0
        )
        
        tab_model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)], eval_metric=['auc'],
            max_epochs=200, patience=20, batch_size=1024, virtual_batch_size=128,
            weights=1
        )
        
        tab_val_probs = tab_model.predict_proba(X_val)[:, 1]
        oof_tab[val_idx] = tab_val_probs
        fold_metrics['tab_auc'].append(roc_auc_score(y_val, tab_val_probs))
        
        print(f"Fold {fold+1} Completed: XGB AUC={fold_metrics['xgb_auc'][-1]:.4f}, TabNet AUC={fold_metrics['tab_auc'][-1]:.4f}")
        
    return oof_xgb, oof_tab, fold_metrics

def run_ablation(df, trad_feats, alt_feats, all_feats, y):
    """Run the three ablation scenarios."""
    scenarios = [
        ("(a) Traditional features only", trad_feats),
        ("(b) Alternative features only", alt_feats),
        ("(c) All 19 features", all_feats)
    ]
    
    ablation_results = {}
    
    for label, feats in scenarios:
        print(f"\n--- Running Ablation: {label} ---")
        xgb_oof, tab_oof, _ = get_stacking_data(df[feats], y)
        
        # Train meta-learner on full OOF
        X_meta = np.column_stack((xgb_oof, tab_oof))
        lr_meta = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
        
        # Evaluate using CV on meta-features to get AUC
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(lr_meta, X_meta, y, cv=skf, scoring='roc_auc')
        mean_auc = np.mean(cv_res['test_score'])
        
        ablation_results[label] = mean_auc
        print(f"Stacking AUC for {label}: {mean_auc:.4f}")
        
    return ablation_results

def main():
    # 1. Load Data
    print("Loading msme_data.csv...")
    df = pd.read_csv('msme_data.csv')
    y = df['default']
    
    # Feature sets
    bureau_4 = ['bureau_credit_age', 'existing_loan_count', 'repayment_flag', 'credit_util_ratio']
    alt_12 = ['upi_monthly_vol', 'upi_consistency', 'upi_avg_txn_val', 'gst_filing_rate', 
              'gst_revenue_trend', 'digital_pmt_ratio', 'utility_payment_hist', 'mobile_recharge_freq', 
              'cash_flow_seasonality', 'inventory_turnover_px', 'merchant_accept_score', 'bank_balance_stability']
    derived_3 = ['log_upi_vol', 'upi_gst_interaction', 'alt_signal_composite']
    
    trad_features = bureau_4 + derived_3
    alt_features = alt_12 + derived_3
    all_features = alt_12 + bureau_4 + derived_3
    
    # 2. Run Ablation
    ablation_results = run_ablation(df, trad_features, alt_features, all_features, y)
    
    # 3. Full Stacking Ensemble (All 19 Features)
    print("\n--- Training Final Stacking Ensemble (All 19 Features) ---")
    xgb_oof, tab_oof, base_fold_metrics = get_stacking_data(df[all_features], y)
    
    # Meta-learner: Logistic Regression
    X_meta = np.column_stack((xgb_oof, tab_oof))
    lr_meta = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    
    # Stratified 5-fold CV on meta-features for evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ensemble_metrics = {'auc': [], 'ap': [], 'f1': [], 'brier': []}
    oof_hybrid = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta, y)):
        X_train_meta, X_val_meta = X_meta[train_idx], X_meta[val_idx]
        y_train_meta, y_val_meta = y[train_idx], y[val_idx]
        
        lr_meta.fit(X_train_meta, y_train_meta)
        fold_probs = lr_meta.predict_proba(X_val_meta)[:, 1]
        oof_hybrid[val_idx] = fold_probs
        
        threshold = find_optimal_threshold(y_val_meta, fold_probs)
        fold_preds = (fold_probs >= threshold).astype(int)
        
        ensemble_metrics['auc'].append(roc_auc_score(y_val_meta, fold_probs))
        ensemble_metrics['ap'].append(average_precision_score(y_val_meta, fold_probs))
        ensemble_metrics['f1'].append(f1_score(y_val_meta, fold_preds))
        ensemble_metrics['brier'].append(brier_score_loss(y_val_meta, fold_probs))
    
    # 4. Final Reporting
    print("\n--- Stacking Ensemble Evaluation (Mean ± Std) ---")
    for k, v in ensemble_metrics.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    # 5. Save Results to results.csv
    print("\nSaving results to results.csv...")
    df_results = pd.DataFrame({
        'true_label': y,
        'xgb_prob': xgb_oof,
        'tabnet_prob': tab_oof,
        'hybrid_stacking_prob': oof_hybrid
    })
    
    # Add fold metrics as a summary at the end or separate file? 
    # Prompt says "Save ... fold metrics to results.csv"
    # We'll create a summary dataframe and append it or just add columns.
    # Let's add them as columns but only filled for the first 5 rows to be cleaner.
    df_results['fold_auc'] = np.nan
    df_results['fold_ap'] = np.nan
    df_results['fold_f1'] = np.nan
    df_results['fold_brier'] = np.nan
    
    for i in range(5):
        df_results.loc[i, 'fold_auc'] = ensemble_metrics['auc'][i]
        df_results.loc[i, 'fold_ap'] = ensemble_metrics['ap'][i]
        df_results.loc[i, 'fold_f1'] = ensemble_metrics['f1'][i]
        df_results.loc[i, 'fold_brier'] = ensemble_metrics['brier'][i]
    
    df_results.to_csv('results.csv', index=False)
    
    # 6. Comparison Bar Chart
    print("\nGenerating comparison.png...")
    # Get baseline performance
    # LR and RF baselines
    cv_lr = cross_validate(LogisticRegression(max_iter=1000), df[all_features], y, cv=skf, scoring='roc_auc')
    cv_rf = cross_validate(RandomForestClassifier(n_estimators=100, random_state=42), df[all_features], y, cv=skf, scoring='roc_auc')
    
    results_to_plot = {
        'Logistic Regression': np.mean(cv_lr['test_score']),
        'Random Forest': np.mean(cv_rf['test_score']),
        'XGBoost': np.mean(base_fold_metrics['xgb_auc']),
        'TabNet': np.mean(base_fold_metrics['tab_auc']),
        'Hybrid (Stacking)': np.mean(ensemble_metrics['auc'])
    }
    
    plt.figure(figsize=(10, 6))
    models = list(results_to_plot.keys())
    aucs = list(results_to_plot.values())
    
    colors = sns.color_palette("viridis", len(models))
    bars = plt.barh(models, aucs, color=colors)
    plt.xlim(0.7, 1.0)
    plt.xlabel('AUC-ROC Score')
    plt.title('Comparison of Models for MSME Default Prediction')
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center')
        
    plt.tight_layout()
    plt.savefig('comparison.png')
    print("Saved comparison.png")

if __name__ == "__main__":
    main()
