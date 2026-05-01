import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import json

def run_experiment(X_train, X_test, y_train, y_test, thin_file_mask_test):
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, 
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        use_label_encoder=False, eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    
    # Full Test Set
    res_full = {
        "AUC": roc_auc_score(y_test, probs),
        "F1": f1_score(y_test, preds),
        "Prec": precision_score(y_test, preds),
        "Rec": recall_score(y_test, preds)
    }
    
    # Thin-file Subgroup
    probs_thin = probs[thin_file_mask_test]
    y_thin = y_test[thin_file_mask_test]
    preds_thin = preds[thin_file_mask_test]
    
    res_thin = {
        "AUC": roc_auc_score(y_thin, probs_thin),
        "F1": f1_score(y_thin, preds_thin),
        "Prec": precision_score(y_thin, preds_thin),
        "Rec": recall_score(y_thin, preds_thin)
    }
    
    return res_full, res_thin

def main():
    print("Loading msme_credit_dataset_75k.csv for Ablation Study...")
    df = pd.read_csv('../msme_credit_dataset_75k.csv')
    
    # Feature sets
    bureau_cols = ['cibil_score', 'credit_age_months', 'num_active_loans', 'num_dpd_30', 'num_dpd_90', 'credit_utilization_ratio', 'secured_loan_ratio', 'bureau_enquiries_6m']
    alt_cols = ['gst_filing_consistency_score', 'upi_monthly_txn_volume', 'upi_avg_txn_value_inr', 'upi_txn_growth_3m', 'nach_mandate_active', 'bank_statement_avg_balance_lakhs', 'digital_payment_ratio']
    
    # Preprocessing for combined
    cat_cols = ['sector', 'state', 'tier', 'gender_of_proprietor']
    le = LabelEncoder()
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])
    
    all_cols = [c for c in df_enc.columns if c != 'default_12m']
    
    # Split
    df_train, df_test = train_test_split(df_enc, test_size=0.3, random_state=42, stratify=df_enc['default_12m'])
    
    # Thin-file mask for test set
    thin_file_mask_test = (df_test['cibil_score'] < 650) | (df_test['credit_age_months'] < 24)
    
    results = []
    
    # Variant A
    print("Running Variant A: Bureau Only...")
    full_a, thin_a = run_experiment(df_train[bureau_cols], df_test[bureau_cols], df_train['default_12m'], df_test['default_12m'], thin_file_mask_test)
    results.append({"Variant": "Bureau Features Only", "Subgroup": "Full", **full_a})
    results.append({"Variant": "Bureau Features Only", "Subgroup": "Thin-file", **thin_a})
    
    # Variant B
    print("Running Variant B: Alt Only...")
    full_b, thin_b = run_experiment(df_train[alt_cols], df_test[alt_cols], df_train['default_12m'], df_test['default_12m'], thin_file_mask_test)
    results.append({"Variant": "Alt Features Only", "Subgroup": "Full", **full_b})
    results.append({"Variant": "Alt Features Only", "Subgroup": "Thin-file", **thin_b})
    
    # Variant C
    print("Running Variant C: Combined...")
    full_c, thin_c = run_experiment(df_train[all_cols], df_test[all_cols], df_train['default_12m'], df_test['default_12m'], thin_file_mask_test)
    results.append({"Variant": "Combined (All)", "Subgroup": "Full", **full_c})
    results.append({"Variant": "Combined (All)", "Subgroup": "Thin-file", **thin_c})
    
    res_df = pd.DataFrame(results)
    print("\n--- Ablation Results (Table 3) ---")
    print(res_df.to_string())
    
    # LaTeX format (Manual to avoid jinja2 error)
    print("\n--- LaTeX Table ---")
    header = " & ".join(res_df.columns) + " \\\\"
    print(header)
    print("\\hline")
    for _, row in res_df.iterrows():
        row_str = f"{row['Variant']} & {row['Subgroup']} & {row['AUC']:.4f} & {row['F1']:.4f} & {row['Prec']:.4f} & {row['Rec']:.4f} \\\\"
        print(row_str)
    
    # Save results
    with open('fix3_results.json', 'w') as f:
        json.dump(res_df.to_dict(orient='records'), f)

if __name__ == "__main__":
    main()
