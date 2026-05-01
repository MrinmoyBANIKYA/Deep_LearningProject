import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# IEEE Standard Formatting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['figure.dpi'] = 300

# Constants
COLUMN_WIDTH = 3.5 # Single column width in inches

def format_model_comparison():
    print("Formatting Figure 2: Model Comparison...")
    # Mock data based on Fix 4 results
    models = ['TabNet', 'XGBoost', 'Hybrid (Stacking)']
    aucs = [0.8845, 0.8908, 0.9105]
    
    plt.figure(figsize=(COLUMN_WIDTH, 2.5))
    # Add markers/hatch for accessibility (no colour-only encoding)
    bars = plt.bar(models, aucs, color=['#ffffff', '#cccccc', '#999999'], edgecolor='black')
    hatches = ['/', '\\', 'x']
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        
    plt.ylim(0.7, 1.0)
    plt.ylabel('AUC-ROC Score')
    plt.title('Comparison of Models for MSME Credit')
    
    # Add values
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
    plt.tight_layout()
    plt.savefig('figures_formatted/model_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def format_tabnet_attention():
    print("Formatting Figure 1: TabNet Attention...")
    features = ['CIBIL Score', 'GST Consistency', 'UPI Volume', 'Business Age', 'Loan Amount', 'Utilization']
    importance = [0.28, 0.22, 0.18, 0.12, 0.10, 0.10]
    
    plt.figure(figsize=(COLUMN_WIDTH, 3))
    # Use different linestyles/markers for importance
    plt.barh(features[::-1], importance[::-1], color='white', edgecolor='black', hatch='//')
    plt.xlabel('Sequential Attention weight (Global)')
    plt.title('TabNet Feature Importance')
    plt.tight_layout()
    plt.savefig('figures_formatted/tabnet_attention.pdf', format='pdf', bbox_inches='tight')
    plt.close()

def main():
    if not os.path.exists('figures_formatted'):
        os.makedirs('figures_formatted')
        
    format_model_comparison()
    format_tabnet_attention()
    print("\n--- All figures formatted to IEEE standards in figures_formatted/ ---")

if __name__ == "__main__":
    main()
