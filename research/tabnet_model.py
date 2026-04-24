import pandas as pd
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, 
                             brier_score_loss, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def find_optimal_threshold(y_true, y_probs):
    """Find the optimal probability threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def main():
    # 1. Load Data
    print("Loading msme_data.csv for TabNet modeling...")
    df = pd.read_csv('msme_data.csv')
    X = df.drop('default', axis=1).values
    y = df['default'].values
    feature_names = df.drop('default', axis=1).columns.tolist()

    # 2. Stratified 5-Fold CV setup
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = {'auc': [], 'ap': [], 'f1': [], 'brier': []}
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    best_auc = 0
    
    plt.figure(figsize=(10, 8))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Training Fold {fold + 1} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Calculate class weights for imbalance
        class_0_count = np.sum(y_train == 0)
        class_1_count = np.sum(y_train == 1)
        weights = {0: 1, 1: class_0_count / class_1_count}

        # Initialize TabNet
        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.3, n_independent=2, n_shared=2,
            lambda_sparse=0.001,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=0.02),
            scheduler_params=dict(gamma=0.95, step_size=1000),
            scheduler_fn=StepLR,
            verbose=0 # Keep console clean
        )

        # Fit Model
        model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_test, y_test)],
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=200, patience=20,
            batch_size=1024, virtual_batch_size=128,
            weights=1 # Using built-in auto-balancing
        )

        # Predictions
        y_probs = model.predict_proba(X_test)[:, 1]
        threshold = find_optimal_threshold(y_test, y_probs)
        y_preds = (y_probs >= threshold).astype(int)

        # Metrics
        fold_auc = roc_auc_score(y_test, y_probs)
        metrics['auc'].append(fold_auc)
        metrics['ap'].append(average_precision_score(y_test, y_probs))
        metrics['f1'].append(f1_score(y_test, y_preds))
        metrics['brier'].append(brier_score_loss(y_test, y_probs))

        print(f"Fold {fold+1} Results: AUC={fold_auc:.4f}, AP={metrics['ap'][-1]:.4f}")

        # Save ROC for fold
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold+1} (AUC = {fold_auc:.2f})')

        # Save Best Model
        if fold_auc > best_auc:
            best_auc = fold_auc
            model.save_model('tabnet_best')
            # Extract attention for global visualization
            best_model_for_attn = model

    # 3. Final Reporting
    print("\n--- Final Performance (Mean ± Std) ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")

    # 4. Save Plots
    # a) ROC Curve
    plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Chance')
    mean_tpr = np.mean(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {np.mean(metrics["auc"]):.2f})', lw=2)
    plt.title('ROC Curve - TabNet Individual Study')
    plt.legend()
    plt.savefig('tabnet_roc.png')
    print("\nSaved tabnet_roc.png")

    # b) Feature Importance
    plt.figure(figsize=(12, 8))
    feat_importances = pd.Series(best_model_for_attn.feature_importances_, index=feature_names)
    feat_importances.nlargest(19).plot(kind='barh', color='teal')
    plt.title('TabNet Feature Importance')
    plt.tight_layout()
    plt.savefig('tabnet_importance.png')
    print("Saved tabnet_importance.png")

    # c) Attention Heatmap
    plt.figure(figsize=(14, 10))
    explain_matrix, masks = best_model_for_attn.explain(X)
    # Average across all samples and all steps
    mean_attn = np.mean(explain_matrix, axis=0).reshape(1, -1)
    
    # Actually, let's plot the mean mask for each step to show the "Sequential" nature
    steps_attn = []
    for step in range(len(masks)):
        steps_attn.append(np.mean(masks[step], axis=0))
    
    attn_df = pd.DataFrame(steps_attn, columns=feature_names)
    sns.heatmap(attn_df, annot=False, cmap='YlGnBu')
    plt.title('TabNet Sequential Attention Heatmap (Mean per Step)')
    plt.xlabel('Features')
    plt.ylabel('Attention Step')
    plt.tight_layout()
    plt.savefig('tabnet_attention.png')
    print("Saved tabnet_attention.png")

if __name__ == "__main__":
    main()
