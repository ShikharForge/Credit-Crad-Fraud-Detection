# Credit Card Fraud Detection & Risk Analysis

## Overview

Credit card fraud detection is a classic machine learning problem characterized by extreme class imbalance. In standard payment datasets, fraudulent transactions typically account for less than 0.2% of all activity. Under such severe imbalance, conventional evaluation metrics like accuracy (which often exceeds 99.8% by simply predicting all transactions as legitimate) are misleading and uninformative.

The objective of this project is to develop a fraud classification system that maximizes fraud detection (recall) while keeping false alarms (false positives) low to avoid alert fatigue in fraud operations. The project enforces strict data leakage prevention by keeping a 20% held-out test split completely sealed during all exploration, feature engineering, resampling, cross-validation, stacking ensemble design, and threshold tuning.

We evaluate three base classification algorithms (Logistic Regression, Random Forest, and XGBoost) across multiple class-imbalance strategies (unweighted baselines, cost-sensitive class weighting, and SMOTE resampling). The final selected model is a stacking ensemble combining all three base architectures with a logistic regression meta-learner, operating at a tuned decision threshold of 0.25.

## Dataset

This project uses the standard ULB Credit Card Fraud Detection benchmark dataset:
- **Total Transactions**: 284,807 transactions recorded over two days in September 2013 by European cardholders.
- **Fraudulent Transactions**: 492 cases (~0.172% fraud rate).
- **Features**:
  - `Time`: Elapsed seconds from the first transaction in the dataset.
  - `V1` to `V28`: 28 principal components obtained via PCA due to confidentiality constraints.
  - `Amount`: Transaction amount in Euros.
  - `Class`: Binary ground truth target (0 = Legitimate, 1 = Fraud).

*Note: The raw CSV (`data/creditcard.csv`, 144 MB) is excluded from this repository. You can download the dataset directly from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).*

## Approach

1. **Data Audit & EDA (`01_data_audit.ipynb`)**: Verified column data types, missing values (0 missing), duplicate rows (1,081 verified duplicates identified), amount skewness, and diurnal transaction distributions.
2. **Preprocessing & Feature Engineering (`02_preprocessing_feature_engineering.ipynb`, `src/preprocessing.py`)**:
   - Removed 1,081 exact duplicate records prior to splitting.
   - Engineered two domain features: `Log_Amount` (log1p transform to compress heavy right-tail amount skew) and `Hour` (diurnal hour-of-day mapped from `Time`).
   - Created a stratified 80/20 train/test split (226,980 training samples, 56,746 test samples; 378 train frauds, 95 test frauds).
   - Fitted `RobustScaler` on `['Amount', 'Time', 'Log_Amount', 'Hour']` strictly on training data (leaving PCA features unscaled).
3. **Model Benchmarking (`03_model_benchmarking.ipynb`)**:
   - 5-fold Stratified Cross-Validation on the training set using leakage-safe pipelines.
   - Compared baseline, class-weighted, and SMOTE configurations across Logistic Regression, Random Forest, and XGBoost.
   - Evaluated training-derived winsorisation (outlier capping) and concluded original amounts should be preserved.
4. **Stacking Ensemble & Threshold Optimization (`04_stacking_threshold_optimization.ipynb`)**:
   - Generated 5-fold Out-Of-Fold (OOF) prediction probabilities from the three base learners.
   - Trained a Logistic Regression meta-model on OOF probabilities.
   - Evaluated decision thresholds from 0.05 to 0.60 on OOF predictions, selecting $\tau = 0.25$ for operational deployment.
5. **Final Evaluation & Explainability (`05_final_evaluation_explainability.ipynb`, `src/final_pipeline.py`)**:
   - Evaluated the final stacking ensemble once on the sealed held-out test set.
   - Global and local feature attributions using SHAP on XGBoost.
   - Analyzed individual transaction-level cases (True Positive, False Negative, True Negative, False Positive).

## Models

We benchmarked three classification models:
- **Logistic Regression**: Linear decision boundary serving as a calibrated probability baseline.
- **Random Forest**: 100-tree bagging ensemble with max depth 12, offering high precision.
- **XGBoost**: 100-tree gradient boosted trees with max depth 4 and learning rate 0.1.
- **Stacking Ensemble**: Meta-logistic regression model combining out-of-fold probability outputs from Logistic Regression, Random Forest, and XGBoost.

*Imbalance Technique Finding*: SMOTE synthetic oversampling was evaluated across all three model families inside CV folds. SMOTE consistently degraded precision (dropping Logistic Regression to 5.7%, Random Forest to 70.8%, and XGBoost to 23.2%) and multiplied training time without improving PR-AUC. Consequently, SMOTE was rejected for the final architecture.

## Results

### Final Held-Out Test Performance

Evaluated on the completely untouched held-out test set ($N = 56,746$ transactions, 95 actual fraud cases):

| Metric | Random Forest Baseline (@ 0.50) | Final Stacking Ensemble (@ 0.25) | Absolute Difference |
| :--- | :--- | :--- | :--- |
| **Precision** | 97.14% | **95.83%** | -1.31% |
| **Recall** | 71.58% (68 / 95) | **72.63% (69 / 95)** | **+1.05% (+1 fraud intercepted)** |
| **F1-Score** | 0.8242 | **0.8263** | **+0.0021** |
| **PR-AUC** | 0.7909 | **0.8042** | **+0.0133 (+1.33% ranking gain)** |
| **ROC-AUC** | 0.9686 | **0.9705** | **+0.0019** |
| **False Positives (FP)** | 2 | **3** | +1 false alarm |
| **False Negatives (FN)** | 27 | **26** | -1 missed fraud |
| **True Positives (TP)** | 68 | **69** | +1 detected fraud |
| **True Negatives (TN)** | 56,649 | **56,648** | -1 |
| **Accuracy** | 99.9489% | **99.9489%** | 0.0000% |

### Operational Interpretation
At the selected operating threshold ($\tau = 0.25$), the stacking ensemble intercepted 69 of 95 frauds (72.63% recall) with only 3 false alarms across 56,651 legitimate transactions (95.83% alert precision).

## Explainability

- **SHAP Global Feature Importance**: `V14`, `V10`, `V4`, `V12`, `V11`, and `V8` emerged as the most influential components in predicting fraud.
- **Engineered Feature Attribution**: The engineered `Hour` feature ranked in the top 10 SHAP features, confirming that non-uniform overnight fraud patterns provide useful signal.
- **PCA Anonymization Note**: Features `V1` through `V28` are anonymized principal components from the original dataset provider, so their statistical importance is reported without assigning speculative domain meanings.
- **Transaction-Level Explanations**: Post-hoc SHAP analysis on specific test cases illustrated how positive and negative feature contributions pushed predictions above or below the decision threshold.

## Repository Structure

```
credit-card-fraud-detection/
├── data/
│   └── .gitkeep                                   # Raw CSV excluded (download from Kaggle)
├── notebooks/
│   ├── 01_data_audit.ipynb                        # Phase 1: Data Audit & EDA
│   ├── 02_preprocessing_feature_engineering.ipynb # Phase 2: Preprocessing & Leakage-Safe Splits
│   ├── 03_model_benchmarking.ipynb                # Phase 3: Model Benchmarking & Imbalance Experiments
│   ├── 04_stacking_threshold_optimization.ipynb   # Phase 4: Stacking Ensemble & Threshold Tuning
│   └── 05_final_evaluation_explainability.ipynb   # Phase 5: SHAP Explainability & Risk Analysis
├── src/
│   ├── preprocessing.py                           # Reusable preprocessing pipeline
│   └── final_pipeline.py                          # Reusable Stacking Ensemble scoring pipeline
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShikharForge/Credit-Crad-Fraud-Detection.git
   cd Credit-Crad-Fraud-Detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder:
   ```
   data/creditcard.csv
   ```

4. **Run the notebooks in sequence**:
   - `01_data_audit.ipynb`
   - `02_preprocessing_feature_engineering.ipynb`
   - `03_model_benchmarking.ipynb`
   - `04_stacking_threshold_optimization.ipynb`
   - `05_final_evaluation_explainability.ipynb`

   Or import the reusable inference pipeline in Python:
   ```python
   from src.final_pipeline import load_and_train_final_pipeline

   pipeline = load_and_train_final_pipeline('data/creditcard.csv', threshold=0.25)
   scored_df = pipeline.predict_transactions(new_transactions_df)
   ```

## Limitations

- **Feature Anonymization**: PCA features `V1`–`V28` obscure specific merchant categories, terminal locations, and user history.
- **Single Benchmark Snapshot**: The dataset covers a 2-day European transaction window from 2013 and does not reflect modern fraud mechanisms or evolving cardholder behaviors.
- **Concept Drift**: In real-world payment environments, fraud patterns evolve rapidly, requiring continuous monitoring and periodic model retraining.
- **Scope**: This repository is an offline research and benchmarking analysis, not an end-to-end production payment processing system.
