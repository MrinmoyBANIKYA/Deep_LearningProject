import numpy as np
import pandas as pd
from scipy.stats import lognorm, beta, bernoulli, uniform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Set fixed random seed for reproducibility
np.random.seed(42)
N = 10000

print(f"Generating synthetic dataset for {N} MSME applications...")

# --- ALTERNATIVE DIGITAL FEATURES (12) ---

# upi_monthly_vol: LogNormal(mean=4.2, sigma=0.8), range clip [1, 500]
# Note: LogNormal parameters in scipy are s (sigma) and scale (exp(mean))
upi_monthly_vol = lognorm.rvs(s=0.8, scale=np.exp(4.2), size=N)
upi_monthly_vol = np.clip(upi_monthly_vol, 1, 500)

# upi_consistency: Beta(a=5, b=2), range [0,1]
upi_consistency = beta.rvs(5, 2, size=N)

# upi_avg_txn_val: Uniform [100, 5000] (Typical Indian MSME ticket size)
upi_avg_txn_val = uniform.rvs(100, 4900, size=N)

# gst_filing_rate: Beta(a=6, b=2), range [0,1]
gst_filing_rate = beta.rvs(6, 2, size=N)

# gst_revenue_trend: Normal distributed around 0 (growth/decline percentage)
gst_revenue_trend = np.random.normal(0.05, 0.15, size=N)

# digital_pmt_ratio: Beta(a=4, b=2) (Percentage of digital vs cash)
digital_pmt_ratio = beta.rvs(4, 2, size=N)

# utility_payment_hist: Beta(a=7, b=1.5) (High compliance usually)
utility_payment_hist = beta.rvs(7, 1.5, size=N)

# mobile_recharge_freq: Poisson centered around 4 recharges/month
mobile_recharge_freq = np.random.poisson(4, size=N)

# cash_flow_seasonality: Uniform [0, 1] (0 = stable, 1 = highly seasonal)
cash_flow_seasonality = uniform.rvs(0, 1, size=N)

# inventory_turnover_px: Normal(mean=12, std=4) (Turns per year)
inventory_turnover_px = np.random.normal(12, 4, size=N)
inventory_turnover_px = np.clip(inventory_turnover_px, 1, 50)

# merchant_accept_score: Beta(a=5, b=2) (Internal credit proxy from aggregator)
merchant_accept_score = beta.rvs(5, 2, size=N)

# bank_balance_stability: Uniform [0, 1]
bank_balance_stability = uniform.rvs(0, 1, size=N)

# --- BUREAU STUBS (4) ---

# bureau_credit_age: Uniform [0, 120] months
bureau_credit_age = uniform.rvs(0, 120, size=N)

# existing_loan_count: Poisson(1.5)
existing_loan_count = np.random.poisson(1.5, size=N)

# repayment_flag: Bernoulli(p=0.18) [1 = any 90+ DPD in history]
repayment_flag = bernoulli.rvs(0.18, size=N)

# credit_util_ratio: Beta(a=2, b=5) (Lower is generally better)
credit_util_ratio = beta.rvs(2, 5, size=N)

# --- CREATE INITIAL DATAFRAME ---

df = pd.DataFrame({
    'upi_monthly_vol': upi_monthly_vol,
    'upi_consistency': upi_consistency,
    'upi_avg_txn_val': upi_avg_txn_val,
    'gst_filing_rate': gst_filing_rate,
    'gst_revenue_trend': gst_revenue_trend,
    'digital_pmt_ratio': digital_pmt_ratio,
    'utility_payment_hist': utility_payment_hist,
    'mobile_recharge_freq': mobile_recharge_freq,
    'cash_flow_seasonality': cash_flow_seasonality,
    'inventory_turnover_px': inventory_turnover_px,
    'merchant_accept_score': merchant_accept_score,
    'bank_balance_stability': bank_balance_stability,
    'bureau_credit_age': bureau_credit_age,
    'existing_loan_count': existing_loan_count,
    'repayment_flag': repayment_flag,
    'credit_util_ratio': credit_util_ratio
})

# --- DERIVED FEATURES (3) ---

# log_upi_vol
df['log_upi_vol'] = np.log1p(df['upi_monthly_vol'])

# upi_gst_interaction
df['upi_gst_interaction'] = df['upi_consistency'] * df['gst_filing_rate']

# alt_signal_composite: 1st Principal Component of the 12 alt features
alt_features = [
    'upi_monthly_vol', 'upi_consistency', 'upi_avg_txn_val', 'gst_filing_rate',
    'gst_revenue_trend', 'digital_pmt_ratio', 'utility_payment_hist', 
    'mobile_recharge_freq', 'cash_flow_seasonality', 'inventory_turnover_px',
    'merchant_accept_score', 'bank_balance_stability'
]

scaler = StandardScaler()
alt_scaled = scaler.fit_transform(df[alt_features])
pca = PCA(n_components=1)
df['alt_signal_composite'] = pca.fit_transform(alt_scaled)

# --- GENERATE TARGET: DEFAULT LABEL (Logistic Response) ---

# Define weights for the logistic function
# Strongest predictors: repayment_flag (+ve), low gst (-ve), low upi_consistency (-ve), high seasonality (+ve)
# Intercept is tuned to get ~22% default rate
intercept = 2.5 
z = (intercept + 
     3.5 * df['repayment_flag'] + 
     -2.8 * df['gst_filing_rate'] + 
     -2.2 * df['upi_consistency'] + 
     1.8 * df['cash_flow_seasonality'] +
     0.8 * df['credit_util_ratio'] -
     0.5 * df['log_upi_vol'])

# Convert log-odds to probability
prob_default = 1 / (1 + np.exp(-z))

# Generate binary label
df['default'] = (np.random.rand(N) < prob_default).astype(int)

# --- SAVE AND REPORT ---

df.to_csv('msme_data.csv', index=False)

print("\n--- Dataset Summary ---")
print(f"Shape: {df.shape}")
print(f"Default Rate: {df['default'].mean():.2%}")
print("\nFirst 5 rows:")
print(df.head())

print("\nFeature List (19 total + target):")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col}")
