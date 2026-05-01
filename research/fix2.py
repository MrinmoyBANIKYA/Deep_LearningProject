import pandas as pd
import numpy as np
import torch
import optuna
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json

def objective(trial):
    # 1. Load Data
    df = pd.read_csv('../msme_credit_dataset_75k.csv')
    
    # Preprocessing
    cat_cols = ['sector', 'state', 'tier', 'gender_of_proprietor']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('default_12m', axis=1).values
    y = df['default_12m'].values
    
    # Hyperparameters to search
    n_da = trial.suggest_categorical('n_da', [16, 32, 64])
    n_steps = trial.suggest_categorical('n_steps', [3, 5, 7])
    lr = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
    lambda_sparse = trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    
    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    
    # For speed in 30 trials, we can use a smaller subset or fewer folds, 
    # but the requirement says 5-fold CV. 
    # To keep it runnable within reasonable time, we'll use a subset of 20k rows for tuning
    # if the dataset is too large, but let's try full first or 1/3.
    # Actually, 75k rows with TabNet in 30 trials * 5 folds = 150 fits. 
    # That will take hours. I will use a 20k subset for the Optuna study to ensure it finishes.
    df_sample = df.sample(20000, random_state=42)
    X_s = df_sample.drop('default_12m', axis=1).values
    y_s = df_sample['default_12m'].values
    
    for train_idx, val_idx in skf.split(X_s, y_s):
        X_train, X_val = X_s[train_idx], X_s[val_idx]
        y_train, y_val = y_s[train_idx], y_s[val_idx]
        
        model = TabNetClassifier(
            n_d=n_da, n_a=n_da, n_steps=n_steps,
            lambda_sparse=lambda_sparse,
            optimizer_params=dict(lr=lr),
            verbose=0
        )
        
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=20, patience=5,
            batch_size=batch_size, virtual_batch_size=128
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, preds))
        
    return np.mean(auc_scores)

def main():
    print("Starting TabNet Optuna Study (30 trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("\nBest Trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save best params
    with open('fix2_best_params.json', 'w') as f:
        json.dump(study.best_params, f)
    
    results = {
        "fix2_best_auc": float(study.best_value),
        "fix2_best_params": study.best_params
    }
    with open('fix2_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
