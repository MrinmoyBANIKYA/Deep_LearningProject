import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from evaluation_utils import compute_ks_statistic, compute_gini

warnings.filterwarnings('ignore')

def main():
    # 1. Load Data
    print("Loading msme_credit_dataset_75k.csv for OOF Stacking...")
    df = pd.read_csv('../msme_credit_dataset_75k.csv')
    
    # Preprocessing
    from sklearn.preprocessing import LabelEncoder
    cat_cols = ['sector', 'state', 'tier', 'gender_of_proprietor']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop('default_12m', axis=1).values
    y = df['default_12m'].values
    feature_names = df.drop('default_12m', axis=1).columns.tolist()

    # Split into Train (85%) and Test (15%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # 2. OOF Stacking Configuration
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(y_train))
    oof_tabnet = np.zeros(len(y_train))
    
    test_probs_xgb = np.zeros((len(y_test), n_splits))
    test_probs_tabnet = np.zeros((len(y_test), n_splits))

    print(f"\nStarting {n_splits}-Fold OOF Stacking...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold + 1} ---")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # A. Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, 
            scale_pos_weight=len(y_tr[y_tr==0])/len(y_tr[y_tr==1]),
            use_label_encoder=False, eval_metric='logloss',
            random_state=42
        )
        xgb_model.fit(X_tr, y_tr)
        
        # B. Train TabNet
        tabnet = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-4, momentum=0.02, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            epsilon=1e-15, verbose=0
        )
        # Use a subset of training data for TabNet validation if needed, 
        # but here we use the fold's validation set for early stopping.
        tabnet.fit(
            X_train=X_tr, y_train=y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=100, patience=20,
            batch_size=1024, virtual_batch_size=128
        )
        
        # C. Generate OOF Predictions
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        oof_tabnet[val_idx] = tabnet.predict_proba(X_val)[:, 1]
        
        # D. Generate Test Predictions for this fold
        test_probs_xgb[:, fold] = xgb_model.predict_proba(X_test)[:, 1]
        test_probs_tabnet[:, fold] = tabnet.predict_proba(X_test)[:, 1]
        
        print(f"Fold {fold+1} XGB AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")
        print(f"Fold {fold+1} Tab AUC: {roc_auc_score(y_val, oof_tabnet[val_idx]):.4f}")

    # 3. Create Meta-Features (Level 2 Training Matrix)
    # Passthrough features: Index 0 (CIBIL), 7 (UPI/Enquiries), 9 (GST/Age)
    print("\nPreparing meta-features for Level 2 learner...")
    X_meta = np.column_stack([
        oof_xgb, 
        oof_tabnet, 
        X_train[:, [0, 7, 9]]
    ])
    
    # 4. Train Meta-Learner (Logistic Regression)
    print("Training Logistic Regression meta-learner...")
    meta_model = LogisticRegression(solver='lbfgs', random_state=42)
    meta_model.fit(X_meta, y_train)
    
    # 5. Generate Test Predictions (Stacking)
    # Average base model predictions from all folds
    mean_test_xgb = test_probs_xgb.mean(axis=1)
    mean_test_tabnet = test_probs_tabnet.mean(axis=1)
    
    X_meta_test = np.column_stack([
        mean_test_xgb, 
        mean_test_tabnet, 
        X_test[:, [0, 7, 9]]
    ])
    
    final_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    final_preds = (final_probs > 0.5).astype(int)
    
    # 6. Final Evaluation
    stacking_auc = roc_auc_score(y_test, final_probs)
    xgb_mean_auc = roc_auc_score(y_test, mean_test_xgb)
    tab_mean_auc = roc_auc_score(y_test, mean_test_tabnet)
    stacking_gini = compute_gini(y_test, final_probs)
    stacking_ks = compute_ks_statistic(y_test, final_probs)
    
    print(f"\n--- Final Stacking Evaluation ---")
    print(f"Mean XGBoost AUC: {xgb_mean_auc:.4f}")
    print(f"Mean TabNet AUC: {tab_mean_auc:.4f}")
    print(f"Stacking Ensemble AUC: {stacking_auc:.4f}")
    print(f"Stacking Gini Coefficient: {stacking_gini:.4f}")
    print(f"Stacking KS Statistic: {stacking_ks:.4f}")
    
    # 7. Visualization: Model Comparison
    plt.figure(figsize=(10, 6))
    models = ['Avg XGBoost', 'Avg TabNet', 'OOF Stacking (LR)']
    aucs = [xgb_mean_auc, tab_mean_auc, stacking_auc]
    sns.barplot(x=models, y=aucs, palette='viridis')
    plt.ylim(0.8, 1.0)
    plt.ylabel('AUC-ROC Score')
    plt.title('Performance Comparison: Base Models vs Stacking')
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.005, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig('stacking_comparison.png')
    print("\nComparison plot saved as stacking_comparison.png")

    # 8. Save Models
    joblib.dump(meta_model, 'stacking_meta_learner.pkl')
    print("Meta-learner saved as stacking_meta_learner.pkl")

if __name__ == "__main__":
    main()
