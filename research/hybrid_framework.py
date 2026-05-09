import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculate Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine points in the current bin
        in_bin = (y_probs > bin_lower) & (y_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def main():
    # 1. Load Data
    print("Loading msme_credit_dataset_75k.csv for Hybrid Framework...")
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

    # Split into Train (70%), Val (15%), Test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # 2. Train TabNet (Deep Learning)
    print("\nTraining TabNet Model...")
    tabnet = TabNetClassifier(
        n_d=64, n_a=64, n_steps=5,
        gamma=1.5, n_independent=2, n_shared=2,
        lambda_sparse=1e-4, momentum=0.02, clip_value=2.,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":50, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        epsilon=1e-15
    )

    tabnet.fit(
        X_train=X_train, y_train=y_train,
        eval_set=[(X_val, y_val)],
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
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        use_label_encoder=False, eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)

    # 4. Platt Scaling Calibration
    print("\nApplying Platt Scaling (CalibratedClassifierCV)...")
    
    # Calibrate TabNet
    tabnet_calibrated = CalibratedClassifierCV(tabnet, method='sigmoid', cv='prefit')
    tabnet_calibrated.fit(X_val, y_val)
    
    # Calibrate XGBoost
    xgb_calibrated = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
    xgb_calibrated.fit(X_val, y_val)

    # 5. Calibration Evaluation & Plotting
    print("\nEvaluating Calibration and Generating Reliability Diagrams...")
    
    # Get probabilities on Test Set
    tab_uncal = tabnet.predict_proba(X_test)[:, 1]
    tab_cal = tabnet_calibrated.predict_proba(X_test)[:, 1]
    xgb_uncal = xgb_model.predict_proba(X_test)[:, 1]
    xgb_cal = xgb_calibrated.predict_proba(X_test)[:, 1]
    
    # Calculate ECE
    ece_tab_uncal = calculate_ece(y_test, tab_uncal)
    ece_tab_cal = calculate_ece(y_test, tab_cal)
    ece_xgb_uncal = calculate_ece(y_test, xgb_uncal)
    ece_xgb_cal = calculate_ece(y_test, xgb_cal)
    
    print(f"TabNet ECE: Uncalibrated={ece_tab_uncal:.4f}, Calibrated={ece_tab_cal:.4f}")
    print(f"XGBoost ECE: Uncalibrated={ece_xgb_uncal:.4f}, Calibrated={ece_xgb_cal:.4f}")
    
    # Plot Reliability Diagrams
    plt.figure(figsize=(12, 10))
    
    # TabNet Calibration Curve
    plt.subplot(2, 1, 1)
    fop_uncal, mpv_uncal = calibration_curve(y_test, tab_uncal, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_test, tab_cal, n_bins=10)
    plt.plot(mpv_uncal, fop_uncal, marker='o', linewidth=1, label=f'Uncalibrated (ECE={ece_tab_uncal:.3f})')
    plt.plot(mpv_cal, fop_cal, marker='s', linewidth=1, label=f'Platt Scaled (ECE={ece_tab_cal:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram: TabNet')
    plt.legend()
    
    # XGBoost Calibration Curve
    plt.subplot(2, 1, 2)
    fop_uncal, mpv_uncal = calibration_curve(y_test, xgb_uncal, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_test, xgb_cal, n_bins=10)
    plt.plot(mpv_uncal, fop_uncal, marker='o', linewidth=1, label=f'Uncalibrated (ECE={ece_xgb_uncal:.3f})')
    plt.plot(mpv_cal, fop_cal, marker='s', linewidth=1, label=f'Platt Scaled (ECE={ece_xgb_cal:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram: XGBoost')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('calibration_curves.png')
    print("Saved calibration_curves.png")

    # 6. Hybrid Ensemble Logic (Weighted Average using Calibrated Probs)
    print("\nEvaluating Hybrid Ensemble (Calibrated)...")
    # Weighted ensemble (0.4 TabNet + 0.6 XGBoost) using calibrated probabilities
    hybrid_probs = (0.4 * tab_cal) + (0.6 * xgb_cal)
    hybrid_preds = (hybrid_probs > 0.5).astype(int)

    # 7. Metrics Comparison
    tabnet_auc = roc_auc_score(y_test, tab_cal)
    xgb_auc = roc_auc_score(y_test, xgb_cal)
    hybrid_auc = roc_auc_score(y_test, hybrid_probs)

    print(f"\nResults on Test Set (Calibrated):")
    print(f"TabNet AUC: {tabnet_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    print(f"Hybrid AUC: {hybrid_auc:.4f}")

    # 8. Visualization 1: Feature Importance (TabNet Attention)
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

    # 9. Visualization 2: Model Comparison
    plt.figure(figsize=(10, 6))
    models = ['TabNet (Cal)', 'XGBoost (Cal)', 'Hybrid (Ensemble)']
    aucs = [tabnet_auc, xgb_auc, hybrid_auc]
    sns.barplot(x=models, y=aucs, palette='Blues_d')
    plt.ylim(0.7, 1.0)
    plt.ylabel('AUC-ROC Score')
    plt.title('Model Performance Comparison (Calibrated)')
    plt.savefig('model_comparison.png')
    print("Saved model_comparison.png")

    # 10. Save Models
    joblib.dump(xgb_calibrated, 'hybrid_xgb_calibrated.pkl')
    # TabNet calibrated needs a different approach to save since it's a wrapper
    # We'll save the base model and the calibration wrapper separately or just use joblib
    joblib.dump(tabnet_calibrated, 'hybrid_tabnet_calibrated.pkl')
    print("\nCalibrated models saved as hybrid_xgb_calibrated.pkl and hybrid_tabnet_calibrated.pkl")

if __name__ == "__main__":
    main()
