# MSME India 75k Final Dataset Card

## Dataset Summary
A high-fidelity synthetic dataset of 75,000 Indian MSME credit applications, calibrated to RBI MSME Annual Report 2023 and SIDBI MSME Pulse statistics.

## Generation Method
- **Framework**: Multivariate Gaussian Copula.
- **Structural Priors**: Correlation structure derived from Kaggle Home Credit Default Risk and Give Me Some Credit.
- **Calibration**: Marginals fitted to RBI 2023 distributional benchmarks.

## Column Metadata
- **Bureau Features**: CIBIL (300-900), Credit Age, DPD history, Utilization.
- **Alternative Signals**: GST Consistency Score (0-100), UPI Transaction Volumes, NACH status.
- **Demographics**: Sector (RBI Census weighted), City Tier, State.

## Statistical Integrity
- **Default Rate**: 19.0% (NPA aligned).
- **Thin-file Density**: 34.9%.
- **Validation**: KS and Chi-squared tests passed against RBI reference marginals.
