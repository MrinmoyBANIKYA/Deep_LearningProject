import pandas as pd
import numpy as np
import xgboost as xgb
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from evaluation_utils import compute_ks_statistic
from config import RANDOM_SEED
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def run_hybrid_cred_cv(X_df, y, n_splits=5):
    """
    Train a Hybrid ensemble (XGBoost + TabNet) with 5-fold CV.
    Returns mean AUC and mean KS statistic.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    aucs = []
    ks_stats = []
    
    # Preprocess categorical features for XGBoost
    X_xgb = X_df.copy()
    cat_cols = X_xgb.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        X_xgb[col] = le.fit_transform(X_xgb[col].astype(str))
    
    X_xgb_values = X_xgb.values
    y_values = y.values
    
    # Preprocess for TabNet (Scaling)
    scaler = StandardScaler()
    X_tab_values = scaler.fit_transform(X_xgb_values)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_xgb_values, y_values)):
        X_tr_xgb, X_val_xgb = X_xgb_values[train_idx], X_xgb_values[val_idx]
        X_tr_tab, X_val_tab = X_tab_values[train_idx], X_tab_values[val_idx]
        y_tr, y_val = y_values[train_idx], y_values[val_idx]
        
        # 1. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=RANDOM_SEED
        )
        xgb_model.fit(X_tr_xgb, y_tr)
        xgb_probs = xgb_model.predict_proba(X_val_xgb)[:, 1]
        
        # 2. TabNet
        tab_model = TabNetClassifier(
            verbose=0,
            seed=RANDOM_SEED,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
        )
        tab_model.fit(
            X_train=X_tr_tab, y_train=y_tr,
            eval_set=[(X_val_tab, y_val)],
            eval_metric=['auc'],
            max_epochs=20,
            patience=5,
            batch_size=1024,
            virtual_batch_size=128
        )
        tab_probs = tab_model.predict_proba(X_val_tab)[:, 1]
        
        # 3. Hybrid (Simple Averaging)
        hybrid_probs = (xgb_probs + tab_probs) / 2
        
        fold_auc = roc_auc_score(y_val, hybrid_probs)
        fold_ks = compute_ks_statistic(y_val, hybrid_probs)
        
        aucs.append(fold_auc)
        ks_stats.append(fold_ks)
        
    return np.mean(aucs), np.mean(ks_stats)

def main():
    # 1. Load Data
    data_path = '../msme_credit_dataset_75k.csv'
    if not os.path.exists(data_path):
        data_path = 'msme_credit_dataset_75k.csv'
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Use a subset for faster ablation study if dataset is large
    if len(df) > 20000:
        print(f"Subsampling 20,000 rows for faster execution...")
        df = df.sample(20000, random_state=RANDOM_SEED).reset_index(drop=True)
    
    target = 'default_12m'
    y = df[target]
    X = df.drop(target, axis=1)
    all_features = X.columns.tolist()
    
    # 2. Define Configurations
    configs = {
        "FULL": all_features,
        "NO_UPI": [f for f in all_features if f not in [
            'upi_monthly_txn_volume', 'upi_avg_txn_value_inr', 'upi_txn_growth_3m', 'digital_payment_ratio'
        ]],
        "NO_GST": [f for f in all_features if f != 'gst_filing_consistency_score'],
        "BUREAU_ONLY": [
            'cibil_score', 'credit_age_months', 'num_active_loans', 'num_dpd_30', 
            'num_dpd_90', 'credit_utilization_ratio', 'secured_loan_ratio', 'bureau_enquiries_6m'
        ],
        "NO_GEO": [f for f in all_features if f not in [
            'state', 'tier', 'gender_of_proprietor', 'rbi_priority_sector'
        ]]
    }
    
    results = []
    full_auc = 0
    
    # 3. Run Study
    print("\nStarting Ablation Study (5-fold CV for each config)...")
    
    # Run FULL first to get baseline for Delta
    print(f"Config: FULL ({len(configs['FULL'])} features)")
    mean_auc, mean_ks = run_hybrid_cred_cv(X[configs['FULL']], y)
    full_auc = mean_auc
    results.append({
        "Configuration": "FULL",
        "AUC-ROC": mean_auc,
        "KS-Statistic": mean_ks,
        "Delta-AUC": 0.0
    })
    print(f"Result -> AUC: {mean_auc:.4f}, KS: {mean_ks:.4f}")
    
    for name in ["NO_UPI", "NO_GST", "BUREAU_ONLY", "NO_GEO"]:
        feat_list = configs[name]
        print(f"\nConfig: {name} ({len(feat_list)} features)")
        mean_auc, mean_ks = run_hybrid_cred_cv(X[feat_list], y)
        delta = mean_auc - full_auc
        results.append({
            "Configuration": name,
            "AUC-ROC": mean_auc,
            "KS-Statistic": mean_ks,
            "Delta-AUC": delta
        })
        print(f"Result -> AUC: {mean_auc:.4f}, KS: {mean_ks:.4f}, Delta: {delta:.4f}")
        
    # 4. Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv('ablation_results.csv', index=False)
    print("\nResults saved to ablation_results.csv")
    
    # 5. Plot Results
    print("Generating ablation_chart.png...")
    plt.figure(figsize=(10, 6))
    
    # Sort for better visualization (except FULL which should stay at top or bottom)
    plot_df = results_df.copy()
    
    colors = sns.color_palette("magma", len(plot_df))
    bars = plt.barh(plot_df['Configuration'], plot_df['AUC-ROC'], color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlim(0.6, 1.0)
    plt.xlabel('AUC-ROC Score')
    plt.title('Feature Ablation Study: HybridCred Performance')
    
    # Add text labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ablation_chart.png')
    print("Chart saved as ablation_chart.png")

if __name__ == "__main__":
    main()
