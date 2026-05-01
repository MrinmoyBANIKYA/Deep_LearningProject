import numpy as np
import pandas as pd
from scipy.stats import norm
import os

def generate_msme_dataset(n_samples=75000, output_path="msme_credit_dataset_75k.csv"):
    print(f"--- Generating Synthetic MSME Dataset ({n_samples} rows) ---")
    np.random.seed(42)

    # 1. Define Core Latent Variables and Correlation Matrix
    # Indices: 0: cibil_latent, 1: gst_latent, 2: upi_latent, 3: risk_latent, 4: age_latent
    corr_matrix = np.array([
        [1.00, 0.20, 0.15, -0.55, 0.30], # cibil_latent
        [0.20, 1.00, 0.45, -0.35, 0.10], # gst_latent
        [0.15, 0.45, 1.00, -0.25, 0.05], # upi_latent
        [-0.55, -0.35, -0.25, 1.00, -0.20], # risk_latent (high risk -> high default)
        [0.30, 0.10, 0.05, -0.20, 1.00]  # age_latent
    ])

    # Cholesky Decomposition
    L = np.linalg.cholesky(corr_matrix)
    latent_data = np.random.normal(0, 1, size=(n_samples, 5))
    correlated_latent = latent_data @ L.T

    # Convert latent variables to Uniform [0, 1] for mapping to marginals
    u = norm.cdf(correlated_latent)

    df = pd.DataFrame()

    # --- BUREAU FEATURES (8) ---
    # Adjust mapping to skew scores higher to meet thin-file target (35-40%)
    # P(u**0.35 * 600 + 300 < 650) -> P(u**0.35 < 0.583) -> P(u < 0.21) -> ~21%
    df['cibil_score'] = (u[:, 0]**0.35 * 600 + 300).astype(int)
    # P(u**0.5 * 240 < 24) -> P(u**0.5 < 0.1) -> P(u < 0.01) -> ~1%
    # Let's adjust Age to be more uniform but slightly skewed
    df['credit_age_months'] = (u[:, 4]**0.7 * 240).astype(int)
    
    # Logic: Lower CIBIL -> More active loans/DPD
    df['num_active_loans'] = (np.random.poisson(2, n_samples) + (1 - u[:, 0]) * 6).astype(int)
    df['num_dpd_30'] = np.random.binomial(12, (1 - u[:, 0])**2 * 0.4)
    df['num_dpd_90'] = np.random.binomial(6, (1 - u[:, 0])**2 * 0.2)
    df['credit_utilization_ratio'] = np.clip((1 - u[:, 0]) * 0.8 + np.random.normal(0.1, 0.05, n_samples), 0, 1)
    df['secured_loan_ratio'] = np.clip(u[:, 0] * 0.6 + np.random.normal(0.2, 0.1, n_samples), 0, 1)
    df['bureau_enquiries_6m'] = np.random.poisson(1, n_samples) + (df['num_active_loans'] / 4).astype(int)

    # --- BUSINESS FUNDAMENTALS (6) ---
    sectors = ['Manufacturing', 'Trade', 'Services', 'Food', 'Textile']
    df['sector'] = np.random.choice(sectors, n_samples)
    
    df['business_age_years'] = (u[:, 4] * 30).astype(int)
    df['annual_turnover_lakhs'] = np.exp(np.random.normal(4.5, 0.8, n_samples)).clip(1, 500)
    df['udyam_registration'] = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])
    df['employee_count'] = (df['annual_turnover_lakhs'] / 1.5 + np.random.normal(5, 5, n_samples)).clip(1, 500).astype(int)
    
    # Manufacturing bias: higher loan amounts
    base_loan = np.random.normal(40, 15, n_samples).clip(1, 100)
    df['loan_amount_requested_lakhs'] = np.where(df['sector'] == 'Manufacturing', base_loan * 1.3, base_loan).clip(1, 100)

    # --- ALTERNATIVE / DIGITAL SIGNALS (7) ---
    df['gst_filing_consistency_score'] = (u[:, 1] * 100).astype(int)
    df['upi_monthly_txn_volume'] = (u[:, 2] * 5000).astype(int)
    
    # Services bias: higher UPI volume
    df['upi_monthly_txn_volume'] = np.where(df['sector'] == 'Services', df['upi_monthly_txn_volume'] * 1.25, df['upi_monthly_txn_volume']).clip(0, 5000)
    
    df['upi_avg_txn_value_inr'] = np.random.uniform(1000, 50000, n_samples)
    df['upi_txn_growth_3m'] = np.random.uniform(-1, 2, n_samples)
    df['nach_mandate_active'] = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    df['bank_statement_avg_balance_lakhs'] = (u[:, 1] * 18 + np.random.normal(1, 0.5, n_samples)).clip(0, 20)
    df['digital_payment_ratio'] = np.clip(u[:, 2] * 0.7 + np.random.uniform(0, 0.3, n_samples), 0, 1)

    # --- RBI COMPLIANCE & GEOGRAPHY (4) ---
    states = [
        'Uttar Pradesh', 'Maharashtra', 'Tamil Nadu', 'Gujarat', 'Karnataka', 
        'West Bengal', 'Rajasthan', 'Madhya Pradesh', 'Bihar', 'Andhra Pradesh', 
        'Telangana', 'Kerala', 'Odisha', 'Punjab', 'Haryana', 'Chhattisgarh', 
        'Jharkhand', 'Assam', 'Uttarakhand', 'Himachal Pradesh', 'Tripura', 
        'Meghalaya', 'Manipur', 'Nagaland', 'Goa', 'Arunachal Pradesh', 'Mizoram', 'Sikkim'
    ]
    # Weighted by MSME density (UP/MH/TN/GJ overweighted)
    weights = [0.14, 0.13, 0.12, 0.10] + [0.51 / 24] * 24
    df['state'] = np.random.choice(states, n_samples, p=weights)
    
    df['tier'] = np.random.choice(['Tier1', 'Tier2', 'Tier3', 'Rural'], n_samples, p=[0.25, 0.3, 0.25, 0.2])
    df['gender_of_proprietor'] = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.65, 0.33, 0.02])
    df['rbi_priority_sector'] = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])

    # Geographic Bias: Tier 3/Rural have lower CIBIL and lower digital ratio
    mask_rural = df['tier'].isin(['Tier3', 'Rural'])
    df.loc[mask_rural, 'cibil_score'] = (df.loc[mask_rural, 'cibil_score'] * 0.85).astype(int).clip(300, 900)
    df.loc[mask_rural, 'digital_payment_ratio'] = df.loc[mask_rural, 'digital_payment_ratio'] * 0.65

    # Textile bias: higher credit age
    df.loc[df['sector'] == 'Textile', 'credit_age_months'] = (df.loc[df['sector'] == 'Textile', 'credit_age_months'] * 1.3).astype(int).clip(0, 240)

    # --- THIN-FILE BORROWERS (35-40%) ---
    # cibil_score < 650 OR credit_age_months < 24
    thin_file_mask = (df['cibil_score'] < 650) | (df['credit_age_months'] < 24)
    thin_file_rate = thin_file_mask.mean()
    print(f"Thin-file borrower rate: {thin_file_rate:.2%}")

    # --- TARGET (1): default_12m ---
    # Strengthen influence of CIBIL to hit -0.55
    risk_latent = correlated_latent[:, 3]
    cibil_norm = (df['cibil_score'] - 300) / 600
    
    # Increase cibil weight to push correlation closer to -0.55
    risk_score_combined = 0.5 * risk_latent + 0.5 * (1 - cibil_norm)
    
    # Inject thin-file specific correlation for GST
    # Reduce weight from -0.35 to -0.15 to hit ~-0.38 correlation
    gst_normalized = (df['gst_filing_consistency_score'] - 50) / 25
    risk_score_combined[thin_file_mask] += -0.18 * gst_normalized[thin_file_mask]
    
    # Calibrate default rate to 18-22% (using 20% target)
    threshold = np.percentile(risk_score_combined, 80) # Top 20% will default
    df['default_12m'] = (risk_score_combined >= threshold).astype(int)

    # --- ADD NOISE AND OUTLIERS (TOP 2%) ---
    continuous_cols = [
        'annual_turnover_lakhs', 'upi_monthly_txn_volume', 'bank_statement_avg_balance_lakhs',
        'loan_amount_requested_lakhs', 'upi_avg_txn_value_inr'
    ]
    for col in continuous_cols:
        # Add random noise (±5%)
        df[col] = df[col] * np.random.uniform(0.95, 1.05, n_samples)
        # Outliers: Top 2%
        outlier_threshold = df[col].quantile(0.98)
        outlier_mask = df[col] > outlier_threshold
        df.loc[outlier_mask, col] = df.loc[outlier_mask, col] * np.random.uniform(1.5, 3.0)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

    # --- VALIDATION SUMMARY ---
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Default Rate: {df['default_12m'].mean():.2%}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    
    print("\n--- Key Correlations ---")
    cibil_def_corr = df['cibil_score'].corr(df['default_12m'])
    gst_upi_corr = df['gst_filing_consistency_score'].corr(df['upi_monthly_txn_volume'])
    
    thin_data = df[thin_file_mask]
    gst_def_thin_corr = thin_data['gst_filing_consistency_score'].corr(thin_data['default_12m'])
    
    print(f"CIBIL Score vs Default: {cibil_def_corr:.4f} (Target: -0.55)")
    print(f"GST Score vs UPI Volume: {gst_upi_corr:.4f} (Target: 0.45)")
    print(f"GST Score vs Default (Thin-file): {gst_def_thin_corr:.4f} (Target: -0.38)")
    
    print("\n--- Geographic & Sectoral Check ---")
    print(f"UP Sample %: {(df['state'] == 'Uttar Pradesh').mean():.2%}")
    print(f"MH Sample %: {(df['state'] == 'Maharashtra').mean():.2%}")
    print(f"Mfg Avg Loan: {df[df['sector'] == 'Manufacturing']['loan_amount_requested_lakhs'].mean():.2f}")
    print(f"Textile Avg Credit Age: {df[df['sector'] == 'Textile']['credit_age_months'].mean():.2f}")
    print(f"Tier 3/Rural CIBIL Avg: {df[df['tier'].isin(['Tier3', 'Rural'])]['cibil_score'].mean():.2f}")
    print("="*50)

if __name__ == "__main__":
    generate_msme_dataset()
