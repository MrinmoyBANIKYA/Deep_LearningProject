import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def main():
    # 1. Load Data
    print("Loading msme_data.csv for Hybrid Framework...")
    df = pd.read_csv('msme_data.csv')
    X = df.drop('default', axis=1).values
    y = df['default'].values
    feature_names = df.drop('default', axis=1).columns.tolist()

    # Split into Train (70%), Val (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Train TabNet (Deep Learning)
    print("\nTraining TabNet Model...")
    tabnet = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        lambda_sparse=1e-4, momentum=0.3, clip_value=2.,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        epsilon=1e-15
    )

    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=100, patience=20,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0, drop_last=False
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

    # 4. Hybrid Ensemble Logic (Weighted Average)
    print("\nEvaluating Hybrid Ensemble...")
    tabnet_probs = tabnet.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    # Weighted ensemble (0.4 TabNet + 0.6 XGBoost)
    hybrid_probs = (0.4 * tabnet_probs) + (0.6 * xgb_probs)
    hybrid_preds = (hybrid_probs > 0.5).astype(int)

    # 5. Metrics Comparison
    tabnet_auc = roc_auc_score(y_test, tabnet_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    hybrid_auc = roc_auc_score(y_test, hybrid_probs)

    print(f"\nResults on Test Set:")
    print(f"TabNet AUC: {tabnet_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    print(f"Hybrid AUC: {hybrid_auc:.4f}")

    # 6. Visualization 1: Feature Importance (TabNet Attention)
    print("\nGenerating TabNet Attention Plot...")
    explain_matrix, masks = tabnet.explain(X_test)
    
    plt.figure(figsize=(12, 6))
    importance = np.mean(explain_matrix, axis=0)
    indices = np.argsort(importance)[::-1]
    
    sns.barplot(x=importance[indices], y=[feature_names[i] for i in indices], palette='magma')
    plt.title('TabNet Sequential Attention (Global Feature Importance)')
    plt.tight_layout()
    plt.savefig('tabnet_attention.png')
    print("Saved tabnet_attention.png")

    # 7. Visualization 2: Model Comparison
    plt.figure(figsize=(10, 6))
    models = ['TabNet', 'XGBoost', 'Hybrid (Ensemble)']
    aucs = [tabnet_auc, xgb_auc, hybrid_auc]
    sns.barplot(x=models, y=aucs, palette='Blues_d')
    plt.ylim(0.7, 1.0)
    plt.ylabel('AUC-ROC Score')
    plt.title('Model Performance Comparison')
    plt.savefig('model_comparison.png')
    print("Saved model_comparison.png")

    # 8. Save Models
    joblib.dump(xgb_model, 'hybrid_xgb.pkl')
    tabnet.save_model('hybrid_tabnet')
    print("\nModels saved as hybrid_xgb.pkl and hybrid_tabnet.zip")

if __name__ == "__main__":
    main()
