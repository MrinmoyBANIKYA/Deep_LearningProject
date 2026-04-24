# Deep Learning Project: Credit Risk Assessment for MSMEs using TabNet and XGBoost

This project presents a comparative study of deep learning (TabNet) and gradient boosting (XGBoost) for credit risk assessment in Micro, Small, and Medium Enterprises (MSMEs), utilizing alternative financial datasets.

## Project Repository Structure

The repository is organized into the following main directories:

- **`research/`**: Contains all experimental code, data generation scripts, and documentation.
- **`models/`**: Stores trained model checkpoints (e.g., `hybrid_xgb.pkl`, `tabnet_best.zip`).
- **`data/`**: Contains generated datasets and visualization outputs.

## Key Files and Components

The `research/` directory contains the core components of the project:

### Data Generation
- **`generate_dataset.py`**: Generates a synthetic MSME dataset with 10,000 samples and 16 features, including alternative data like UPI transaction history and GST filing rates.

### Model Implementations
- **`xgboost_baseline.py`**: Implements a baseline XGBoost model for credit risk classification.
- **`tabnet_model.py`**: Implements the TabNet architecture, a deep learning model with sequential attention, optimized for tabular data.
- **`hybrid_framework.py`**: A hybrid ensemble model that combines the predictions from TabNet and XGBoost to improve accuracy.

### Evaluation & Visualization
- **`model_comparison.png`**: Visual comparison of model performances (AUC-ROC, Gini).
- **`tabnet_roc.png`**, **`xgb_roc.png`**: Receiver Operating Characteristic (ROC) curves for TabNet and XGBoost.
- **`tabnet_importance.png`**, **`xgb_importance.png`**: Feature importance plots for both models.
- **`tabnet_attention.png`**: Visualizes the attention mechanism of TabNet, showing which features the model focuses on.

### Documentation
- **`manuscript.md`**: A research manuscript detailing the methodology, results, and comparative analysis of TabNet and XGBoost for MSME credit risk.
- **`defense_guide.md`**: A guide for defending the project, including core topics, likely questions, and resources.
- **`submission_strategy.md`**: A guide for submitting the research to conferences, including recommended venues and submission steps.

## Experimental Results

The project evaluates three models:

1.  **Baseline XGBoost**
2.  **Baseline TabNet**
3.  **Proposed Hybrid Framework** (TabNet + XGBoost)

The results, as documented in `manuscript.md`, show that the hybrid framework achieves the highest performance, outperforming both individual models.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd Deep_LearningProject/research
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn xgboost pytorch-tabnet matplotlib seaborn shap
    ```

3.  **Generate the dataset**:
    ```bash
    python generate_dataset.py
    ```

4.  **Train and evaluate the models**:
    -   Train XGBoost: `python xgboost_baseline.py`
    -   Train TabNet: `python tabnet_model.py`
    -   Train Hybrid: `python hybrid_framework.py`

## Key Contributions

- **Alternative Data Integration**: Successfully incorporates alternative data sources (UPI, GST) for MSME credit scoring.
- **Hybrid Model**: Demonstrates the benefit of ensembling TabNet and XGBoost for improved prediction accuracy in credit risk.
- **Interpretability**: Uses SHAP values and TabNet attention to explain model decisions, addressing the "black box" concern of deep learning models.
