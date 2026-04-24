import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight

def find_optimal_threshold(y_true, y_probs):
    """Find the optimal probability threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def objective(trial, X, y):
    """Optuna objective function for XGBoost tuning."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'scale_pos_weight': len(y[y == 0]) / len(y[y == 1]), # Auto from class ratio
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # 3-fold inner CV for tuning speed
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    losses = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        losses.append(brier_score_loss(y_val, preds))
        
    return np.mean(losses)

def main():
    # 1. Load Data
    print("Loading msme_data.csv...")
    df = pd.read_csv('msme_data.csv')
    X = df.drop('default', axis=1)
    y = df['default']
    
    # 2. Hyperparameter Tuning with Optuna
    print("\nStarting Optuna optimization (50 trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)
    
    print("\nBest Hyperparameters:")
    print(study.best_params)
    
    # 3. Stratified 5-Fold Cross-Validation
    print("\nStarting 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {
        'auc': [], 'ap': [], 'f1': [], 'brier': []
    }
    
    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    
    # Base params with class weight
    best_params = study.best_params
    best_params['scale_pos_weight'] = len(y[y == 0]) / len(y[y == 1])
    best_params['eval_metric'] = 'logloss'

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_probs = model.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshold(y_test, y_probs)
        y_preds = (y_probs >= threshold).astype(int)
        
        # Calculate Metrics
        metrics['auc'].append(roc_auc_score(y_test, y_probs))
        metrics['ap'].append(average_precision_score(y_test, y_probs))
        metrics['f1'].append(f1_score(y_test, y_preds))
        metrics['brier'].append(brier_score_loss(y_test, y_probs))
        
        print(f"Fold {i+1}: AUC={metrics['auc'][-1]:.4f}, F1={metrics['f1'][-1]:.4f}")
        
        # Plot ROC for this fold
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {i+1} (AUC = {metrics["auc"][-1]:.2f})')
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    # 4. Final Reporting
    print("\n--- Final Performance (Mean ± Std) ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # 5. Save Plots & Model
    # a) ROC Curve
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(metrics['auc'])
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2, alpha=.8)
    plt.title('Receiver Operating Characteristic - XGBoost Baseline')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig('xgb_roc.png')
    print("\nROC curve saved as xgb_roc.png")

    # b) Feature Importance
    plt.figure(figsize=(12, 8))
    # Retrain on full data for final importance
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('xgb_importance.png')
    print("Feature importance saved as xgb_importance.png")

    # c) Save Model
    joblib.dump(final_model, 'xgb_best.pkl')
    print("Best model saved as xgb_best.pkl")

if __name__ == "__main__":
    main()
