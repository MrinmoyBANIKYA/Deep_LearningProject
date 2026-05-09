import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_score
from sklearn.preprocessing import LabelEncoder
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
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def precision_at_top_k(y_true, y_probs, k=0.2):
    """Calculate precision at the top k% of predicted risk scores."""
    n = len(y_true)
    k_count = int(n * k)
    if k_count == 0: return 0
    # Sort indices by probability descending
    top_indices = np.argsort(y_probs)[::-1][:k_count]
    return np.mean(np.array(y_true)[top_indices])

def evaluate_on_group(y_true, y_probs, name="Full"):
    auc = roc_auc_score(y_true, y_probs)
    ks = compute_ks_statistic(y_true, y_probs)
    p20 = precision_at_top_k(y_true, y_probs, k=0.2)
    return {"Group": name, "AUC": auc, "KS": ks, "Prec@20%": p20}

def main():
    # 1. Load Data
    data_path = '../msme_credit_dataset_75k.csv'
    if not os.path.exists(data_path):
        data_path = 'msme_credit_dataset_75k.csv'
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Preprocessing
    target = 'default_12m'
    X = df.drop(target, axis=1)
    y = df[target]
    
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    
    # 4. Define Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=RANDOM_SEED, use_label_encoder=False, eval_metric='logloss')
    }
    
    # 5. Main Evaluation
    print("\nTraining and evaluating models...")
    all_results = []
    model_probs = {}
    
    for name, model in models.items():
        print(f"Running {name}...")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        model_probs[name] = probs
        
        # 6. Thin-File Section
        thin_file_mask = (X_test['cibil_score'] < 650) | (X_test['credit_age_months'] < 24)
        thick_file_mask = ~thin_file_mask
        
        # Calculate for each subgroup
        full_res = evaluate_on_group(y_test, probs, "Full")
        thin_res = evaluate_on_group(y_test[thin_file_mask], probs[thin_file_mask], "Thin-file")
        thick_res = evaluate_on_group(y_test[thick_file_mask], probs[thick_file_mask], "Thick-file")
        
        for res in [full_res, thin_res, thick_res]:
            res['Model'] = name
            all_results.append(res)

    # 7. Comparison Table
    results_df = pd.DataFrame(all_results)
    pivot_df = results_df.pivot(index='Model', columns='Group', values=['AUC', 'KS', 'Prec@20%'])
    
    print("\n--- Subgroup Performance Comparison ---")
    print(results_df.to_string(index=False))
    
    # 8. Save Thin-file ROC curves
    print("\nGenerating thinfile_roc_comparison.png...")
    plt.figure(figsize=(8, 6))
    
    thin_file_mask = (X_test['cibil_score'] < 650) | (X_test['credit_age_months'] < 24)
    y_test_thin = y_test[thin_file_mask]
    
    for name, probs in model_probs.items():
        probs_thin = probs[thin_file_mask]
        fpr, tpr, _ = roc_curve(y_test_thin, probs_thin)
        auc_thin = roc_auc_score(y_test_thin, probs_thin)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_thin:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Thin-file Borrowers (CIBIL < 650 or Age < 24m)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('thinfile_roc_comparison.png')
    print("Saved thinfile_roc_comparison.png")

if __name__ == "__main__":
    main()
