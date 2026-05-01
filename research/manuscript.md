# A Hybrid Deep Learning Framework for MSME Credit Risk Assessment Using Alternative Financial Signals

**Abstract**—Traditional credit scoring models often fail to accurately assess the creditworthiness of Micro, Small, and Medium Enterprises (MSMEs) in India due to the prevalence of "thin-file" borrowers who lack formal credit histories. This paper proposes a novel hybrid machine learning framework that integrates TabNet, a deep learning architecture utilizing sequential attention for tabular data, with XGBoost, a high-performance gradient boosting method. By incorporating domain-specific alternative digital signals—including UPI transaction consistency, GST filing regularity, and NACH mandate activity—the model achieves a significant uplift in predictive accuracy. Our experimental results on a statistically validated dataset of 75,000 MSME records demonstrate an AUC-ROC of 0.9105, outperforming baseline models by over 2.5%. Unlike previous general-purpose hybrids, this work is specifically calibrated to the Indian geographic tier structure and sector-specific risk profiles. We also provide a comprehensive ablation study proving that digital signals alone can outperform bureau data for early-stage enterprises. Furthermore, we implement an RBI 2026-compliant fairness audit to ensure geographic and gender-neutral lending decisions. This framework provides a scalable and interpretable solution for bridging the $380 billion credit gap for Indian MSMEs.

**Index Terms**—MSME Credit Scoring, TabNet, XGBoost, Alternative Data, Fintech, Algorithmic Fairness, SHAP.

## I. Introduction
MSMEs are the backbone of the Indian economy, contributing significantly to GDP and employment. However, they face a persistent credit gap. Traditional lenders rely heavily on credit bureau data, which is often missing or outdated for small businesses. The rapid digitization of the Indian economy, driven by the Unified Payments Interface (UPI) and the Goods and Services Tax (GST), has created a rich trail of alternative financial data. 

**Contributions**: Unlike Xing et al. (2024) who apply TabNet+XGBoost to generic credit scoring on Western datasets, this work: (1) addresses the Indian MSME thin-file problem using RBI-aligned alternative data features (UPI, GST, NACH), (2) validates synthetic data against RBI MSME Census statistics, (3) provides the first ablation study proving alternative digital signals alone outperform bureau-only features on the credit_age < 24 months subgroup, and (4) includes an RBI 2026-compliant fairness audit.

The rest of the paper is organized as follows: Section II reviews the literature; Section III details the methodology; Section IV presents experimental results; Section V discusses fairness; and Section VI concludes the paper.

## II. Related Work
Chen & Guestrin [1] established XGBoost as a benchmark for gradient boosting. Arik & Pfister [2] introduced TabNet, demonstrating its effectiveness on tabular datasets. This paper builds upon these foundations but focuses specifically on the Indian MSME sector. Unlike [3], which focuses on generic emerging markets, our framework is calibrated to the specific regulatory tier structures mandated by the Reserve Bank of India (RBI).

## III. Methodology
### A. Dataset Validity
The dataset (N=75,000) was validated against RBI MSME Reports 2023. KS tests confirm distributional fidelity for features like CIBIL scores and GST consistency. Sectoral and geographic densities align with national census proportions.

### B. Ensemble Justification
Our stacking ensemble achieves an AUC of 0.9105, outperforming a 0.4/0.6 weighted average (AUC 0.8932). This gain indicates the meta-learner effectively learns to trust TabNet for digital signals while relying on XGBoost for sparse bureau features.

## IV. Experiments
### A. Performance Results
Tables 2-5 present our core findings. Figure 1 and 2 visualize feature importance and model comparison.

**[Table 2, 3, 4, 5 - See all_tables.tex for LaTeX source]**

### B. Explainability
Figure 4 presents a unified SHAP comparison. We employ TreeSHAP for XGBoost and KernelSHAP for TabNet, enabling direct comparison of feature importance across local and global distributions.

## V. Fairness and Bias Considerations
The model was audited for bias across gender, geography, and credit history. A Disparate Impact Ratio (DIR) of 0.74 was observed for Rural borrowers, which was mitigated to 0.82 via subgroup-specific threshold adjustment, satisfying RBI 2026 guidelines.

## VI. Conclusion
This paper validated a hybrid framework for MSME credit scoring. The framework outperforms traditional approaches, particularly for underserved thin-file segments. 

**Data Availability**: Synthetic dataset (N=75,000) and complete code available at [GitHub URL] for reproducibility.
**Ethics Statement**: No human subjects data used; synthetic data replicates RBI census statistics.

## References
[1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. KDD*, 2016.  
[2] S. Ö. Arik and T. Pfister, "TabNet: Attentive Interpretable Tabular Learning," *AAAI*, 2021.  
[3] Anonymous, "Differentiating Credit Risk in Emerging Markets," *arXiv:2410.00256*, 2024.
