import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, roc_curve)
from evaluation_utils import compute_ks_statistic, compute_gini
import os

def find_optimal_threshold(y_true, y_probs):
    """Find the optimal probability threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def main():
    # 1. Load Data
    # Using the 75k dataset which has 26 columns (25 features + 1 target)
    # The prompt mentions "26-feature set", we'll use all available features in this dataset.
    data_path = '../msme_credit_dataset_75k.csv'
    if not os.path.exists(data_path):
        # Fallback to local if not found (unlikely given the environment)
        data_path = 'msme_credit_dataset_75k.csv'
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Identify target and features
    target_col = 'default_12m'
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Initial features: {X.shape[1]}")
    
    # 2. Preprocessing
    # Handle Categorical Columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {cat_cols}")
    
    # Use LabelEncoder as seen in fairness_audit.py to keep feature count stable
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Apply RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Hyperparameter Tuning (5-fold CV)
    print("Starting 5-fold CV for Logistic Regression (L2 regularization)...")
    param_grid = {'C': [0.1, 0.5, 1.0, 5.0]}
    lr = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_scaled, y)
    
    best_lr = grid_search.best_estimator_
    print(f"Best C value: {grid_search.best_params_['C']}")
    
    # 4. Evaluation (using OOF or simple Train/Test? Prompt says "trained on full... Evaluate using...")
    # Usually, metrics for a baseline are reported via CV. 
    # We'll use the best model to get predictions and calculate metrics.
    # To be thorough, we'll calculate metrics on the full dataset with the best model (or OOF).
    # The prompt says "Evaluate using AUC-ROC, Gini..., KS statistic... and F1-score at the optimal threshold."
    
    y_probs = best_lr.predict_proba(X_scaled)[:, 1]
    
    # AUC-ROC
    auc = roc_auc_score(y, y_probs)
    
    # Gini Coefficient
    gini = compute_gini(y, y_probs)
    
    # KS Statistic
    ks_stat = compute_ks_statistic(y, y_probs)
    
    # F1-score at optimal threshold
    optimal_threshold = find_optimal_threshold(y, y_probs)
    y_preds = (y_probs >= optimal_threshold).astype(int)
    f1 = f1_score(y, y_preds)
    
    metrics = {
        "best_C": grid_search.best_params_['C'],
        "auc_roc": round(auc, 4),
        "gini_coefficient": round(gini, 4),
        "ks_statistic": round(ks_stat, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "f1_score": round(f1, 4)
    }
    
    print("\n--- Baseline Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # 5. Save to JSON
    with open('baseline_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nMetrics saved to baseline_metrics.json")

if __name__ == "__main__":
    main()
