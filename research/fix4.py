import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json

def main():
    print("Loading msme_credit_dataset_75k.csv for Ensemble Comparison...")
    df = pd.read_csv('../msme_credit_dataset_75k.csv')
    
    # Preprocessing
    cat_cols = ['sector', 'state', 'tier', 'gender_of_proprietor']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('default_12m', axis=1).values
    y = df['default_12m'].values
    
    # 1. Generate OOF Predictions for Stacking
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y))
    oof_tab = np.zeros(len(y))
    
    print("Generating OOF predictions (5-fold)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
            use_label_encoder=False, eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        
        # TabNet
        tab_model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5, gamma=1.5, momentum=0.02,
            verbose=0
        )
        tab_model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)], eval_metric=['auc'],
            max_epochs=20, batch_size=1024
        )
        oof_tab[val_idx] = tab_model.predict_proba(X_val)[:, 1]
        print(f"  Fold {fold+1} completed.")

    # 2. Weighted Average AUC (0.4/0.6)
    weighted_probs = (0.4 * oof_tab) + (0.6 * oof_xgb)
    weighted_auc = roc_auc_score(y, weighted_probs)
    
    # 3. Stacking AUC (Logistic Regression)
    X_meta = np.column_stack((oof_xgb, oof_tab))
    lr_meta = LogisticRegression()
    
    # Evaluate stacking via nested CV
    stacking_aucs = []
    for train_idx, val_idx in skf.split(X_meta, y):
        lr_meta.fit(X_meta[train_idx], y[train_idx])
        stacking_aucs.append(roc_auc_score(y[val_idx], lr_meta.predict_proba(X_meta[val_idx])[:, 1]))
    
    stacking_auc = np.mean(stacking_aucs)
    
    # 4. Result Comparison
    winner = "Stacking" if stacking_auc > weighted_auc else "Weighted-avg"
    diff = abs(stacking_auc - weighted_auc)
    
    print("\n--- Ensemble Justification ---")
    print(f"Stacking AUC: {stacking_auc:.4f} | Weighted-avg AUC: {weighted_auc:.4f} | Winner: {winner} | Difference: {diff:.4f}")
    
    results = {
        "fix4_stacking_auc": float(stacking_auc),
        "fix4_weighted_auc": float(weighted_auc),
        "fix4_winner": winner,
        "fix4_diff": float(diff)
    }
    with open('fix4_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
