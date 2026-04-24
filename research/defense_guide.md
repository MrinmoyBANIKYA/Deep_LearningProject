# Defense Guide: MSME Credit Risk Hybrid Model

## 🧠 1. How the Model Works (The Logic)
Your model is an **Ensemble** of two distinct paradigms:
1.  **XGBoost (Gradient Boosting)**: Excellent at finding patterns in tabular data by building trees that correct the errors of previous trees.
2.  **TabNet (Deep Learning)**: A special neural network designed for tabular data. It uses **Sequential Attention** to focus on specific features at each step, much like a person reading a spreadsheet.

**The Hybrid Benefit**: TabNet captures complex, deep interactions between alternative signals, while XGBoost provides a solid baseline for traditional features. Combining them (via weighted average) reduces variance and improves robustness.

## 📚 2. Core Topics to Study
- **TabNet Sequential Attention**: Understand how the "Masks" work. It doesn't look at all features at once; it "attends" to the most relevant ones iteratively.
- **SHAP Values**: Based on game theory (Shapley values). It explains how much each feature contributes to the final prediction.
- **Credit Risk Metrics**: AUC-ROC (Area Under Curve), Precision-Recall, F1-Score, and **Gini Coefficient** (2 * AUC - 1).
- **MSME Context**: Why GST and UPI are better proxies for cash flow than a static credit score for small businesses.

## ❓ 3. Likely Viva Questions
1.  **"Why TabNet instead of a simple MLP (Multi-Layer Perceptron)?"**
    *   *Answer*: Standard MLPs struggle with tabular data because they don't have built-in feature selection. TabNet's attention mechanism allows it to mimic decision trees while remaining differentiable (trainable like a neural net).
2.  **"Why use synthetic data?"**
    *   *Answer*: Real-world MSME financial data is highly sensitive and protected by privacy laws (DPDP Act in India). Synthetic data allows for architectural validation without ethical or legal risks.
3.  **"How do you handle the class imbalance (22% defaults)?"**
    *   *Answer*: We use appropriate metrics (AUC-ROC/F1) and can implement SMOTE or weighted loss functions if the imbalance is severe.

## 🔗 4. Resources
- **Paper**: "TabNet: Attentive Interpretable Tabular Learning" (Google Cloud AI)
- **Tool**: `shap` library documentation for explainability plots.
- **Video**: Search for "StatQuest XGBoost" and "TabNet explained" on YouTube.
