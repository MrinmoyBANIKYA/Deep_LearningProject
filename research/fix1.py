import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import json

def main():
    # 1. Load Data
    print("Loading msme_credit_dataset_75k.csv for Fix 1...")
    df = pd.read_csv('../msme_credit_dataset_75k.csv')
    
    # Preprocessing
    cat_cols = ['sector', 'state', 'tier', 'gender_of_proprietor']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('default_12m', axis=1).values
    y = df['default_12m'].values
    
    # Split into Train (70%), Val (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Train TabNet (Deep Learning) - CORRECTED PARAMS
    print("\nTraining TabNet Model (Fix 1: n_d=64, n_a=64, n_steps=5)...")
    tabnet = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, momentum=0.02,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=1
    )

    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['auc'],
        max_epochs=50, patience=10,
        batch_size=1024, virtual_batch_size=128
    )

    # 3. Train XGBoost (Gradient Boosting)
    print("\nTraining XGBoost Model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, 
        scale_pos_weight=len(y[y==0])/len(y[y==1]),
        use_label_encoder=False, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # 4. Evaluation
    tabnet_probs = tabnet.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    hybrid_probs = (0.4 * tabnet_probs) + (0.6 * xgb_probs)

    tabnet_auc = roc_auc_score(y_test, tabnet_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    hybrid_auc = roc_auc_score(y_test, hybrid_probs)

    print(f"\nFix 1 Results:")
    print(f"TabNet AUC: {tabnet_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    print(f"Hybrid AUC: {hybrid_auc:.4f}")

    # Store results for results_summary.json
    results = {
        "fix1_tabnet_auc": float(tabnet_auc),
        "fix1_xgb_auc": float(xgb_auc),
        "fix1_hybrid_auc": float(hybrid_auc)
    }
    with open('fix1_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
