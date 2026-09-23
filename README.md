# Customer Transaction Prediction (PRCP-1003)

Predicting whether a bank customer will make a specific transaction in the future, using 200 anonymized numeric features. Built as part of a CDS capstone project.

## Problem
- **Domain:** Banking
- **Data:** 200,000 customers, 200 anonymized features (`var_0`...`var_199`), binary target (1 = will transact, 0 = won't)
- **Challenge:** ~90/10 class imbalance and no feature descriptions to guide EDA

## Approach
1. **Data analysis** — checked data quality (no missing values/duplicates), quantified the class imbalance, and examined feature-level statistics and correlations since the anonymized features ruled out domain-driven EDA.
2. **Modeling** — trained and compared three models with class-imbalance handling (`class_weight='balanced'` / `scale_pos_weight`):
   - Logistic Regression (baseline)
   - LightGBM
   - XGBoost
3. **Evaluation** — used ROC-AUC and PR-AUC instead of accuracy, given the imbalance.

## Results

| Model | Validation ROC-AUC |
|---|---|
| Logistic Regression | 0.860 |
| LightGBM | 0.892 |
| XGBoost | **0.893** |

XGBoost was selected as the best-performing model, closely followed by LightGBM — both substantially outperform the linear baseline, consistent with the weak, non-linear signal spread across the 200 features.

## Key challenges
- Heavy class imbalance handled via metric choice + class weighting rather than resampling
- No domain-driven EDA possible (anonymized features) — focused on statistical properties instead
- Dataset scale (200k rows × 200 features) required early stopping and a single train/val split rather than full grid search / k-fold CV, to keep runtime practical

See the notebook for the full data analysis report, model comparison, and detailed challenges/techniques write-up.

## Tech stack
Python · pandas · scikit-learn · LightGBM · XGBoost · matplotlib · seaborn

## Files
- `PRCP-1003-CustTransPred.ipynb` — full notebook (EDA, modeling, comparison, challenges report)
