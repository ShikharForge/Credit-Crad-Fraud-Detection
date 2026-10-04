"""
src/final_pipeline.py
=====================
Production-ready, reusable inference pipeline for the frozen Credit Card Fraud
Detection Stacking Ensemble model.

Frozen Configuration
--------------------
* Base Learners:
    - LogisticRegression (max_iter=1000, random_state=42)
    - RandomForestClassifier (n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    - XGBClassifier (n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=42, n_jobs=-1)
* Meta-Learner:
    - LogisticRegression (random_state=42) trained on 5-fold Out-Of-Fold (OOF) base probabilities.
* Preprocessing:
    - RobustScaler applied to ['Amount', 'Time', 'Log_Amount', 'Hour'].
* Operating Threshold:
    - tau = 0.25 (selected operational threshold balancing fraud capture and low alert fatigue).

Usage Example
-------------
    from src.final_pipeline import FraudDetectionPipeline

    # Initialize and fit on training data
    pipeline = FraudDetectionPipeline(threshold=0.25)
    pipeline.fit(X_train_raw, y_train)

    # Score new transactions
    results_df = pipeline.predict_transactions(new_raw_df)
"""

from __future__ import annotations

import os
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

from src.preprocessing import (
    drop_exact_duplicates,
    engineer_features,
    COLS_TO_SCALE,
    RANDOM_SEED,
    TARGET_COL
)

# ---------------------------------------------------------------------------
# Global Defaults
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.25
RISK_LABEL_LOW:    str   = "LOW_RISK"
RISK_LABEL_HIGH:   str   = "HIGH_RISK"


class FraudDetectionPipeline:
    """
    End-to-end stacking ensemble pipeline for credit card fraud detection.
    
    Provides leakage-free training, OOF meta-model calibration, and high-throughput
    batch/single-record inference with standardized risk labels.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        random_state: int = RANDOM_SEED,
        n_cv_splits: int = 5
    ) -> None:
        """
        Initialize the fraud detection pipeline.

        Parameters
        ----------
        threshold : float, default=0.25
            Classification decision threshold for fraud flagging.
        random_state : int, default=42
            Seed for deterministic reproducibility.
        n_cv_splits : int, default=5
            Number of stratified folds for OOF meta-model training.
        """
        self.threshold = threshold
        self.random_state = random_state
        self.n_cv_splits = n_cv_splits

        # Preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', RobustScaler(), COLS_TO_SCALE)
            ],
            remainder='passthrough'
        )

        # Base Learners
        self.base_lr = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )
        self.base_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.base_xgb = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric='logloss',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Meta-Model
        self.meta_model = LogisticRegression(random_state=self.random_state)
        
        self.is_fitted: bool = False
        self.feature_names_: list[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> FraudDetectionPipeline:
        """
        Fit the preprocessor, base learners, and stacking meta-model on training data.

        Meta-model is trained strictly on 5-fold cross-validated out-of-fold predictions
        to eliminate meta-layer data leakage.

        Parameters
        ----------
        X_train : pd.DataFrame
            Unscaled predictor matrix containing raw numerical columns and engineered features.
        y_train : pd.Series
            Binary target series (0 = Legitimate, 1 = Fraud).

        Returns
        -------
        FraudDetectionPipeline
            Fitted instance.
        """
        # 1. Fit preprocessor on training data
        X_tr_scaled = self.preprocessor.fit_transform(X_train)
        self.feature_names_ = [
            c.split('__')[1] if '__' in c else c
            for c in self.preprocessor.get_feature_names_out()
        ]
        X_tr_df = pd.DataFrame(X_tr_scaled, columns=self.feature_names_)

        # 2. Generate 5-fold OOF base predictions for meta-model training
        cv = StratifiedKFold(
            n_splits=self.n_cv_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        oof_lr = np.zeros(len(y_train))
        oof_rf = np.zeros(len(y_train))
        oof_xgb = np.zeros(len(y_train))

        for train_idx, val_idx in cv.split(X_tr_df, y_train):
            X_fold_tr, y_fold_tr = X_tr_df.iloc[train_idx], y_train.iloc[train_idx]
            X_fold_va = X_tr_df.iloc[val_idx]

            fold_lr = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X_fold_tr, y_fold_tr)
            fold_rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=self.random_state, n_jobs=-1).fit(X_fold_tr, y_fold_tr)
            fold_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=self.random_state, n_jobs=-1).fit(X_fold_tr, y_fold_tr)

            oof_lr[val_idx] = fold_lr.predict_proba(X_fold_va)[:, 1]
            oof_rf[val_idx] = fold_rf.predict_proba(X_fold_va)[:, 1]
            oof_xgb[val_idx] = fold_xgb.predict_proba(X_fold_va)[:, 1]

        # 3. Train Meta-Model on OOF Predictions
        X_meta_train = np.column_stack([oof_lr, oof_rf, oof_xgb])
        self.meta_model.fit(X_meta_train, y_train)

        # 4. Fit final base models on the FULL training set
        self.base_lr.fit(X_tr_df, y_train)
        self.base_rf.fit(X_tr_df, y_train)
        self.base_xgb.fit(X_tr_df, y_train)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate continuous fraud probabilities from the stacking ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor matrix matching feature schema.

        Returns
        -------
        np.ndarray
            1D array of fraud probabilities in range [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

        # Ensure engineered features exist
        X_proc = X.copy()
        if "Log_Amount" not in X_proc.columns or "Hour" not in X_proc.columns:
            X_proc = engineer_features(X_proc)

        # Transform using pre-fitted preprocessor
        X_scaled = self.preprocessor.transform(X_proc)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names_)

        # Base model predictions
        prob_lr  = self.base_lr.predict_proba(X_df)[:, 1]
        prob_rf  = self.base_rf.predict_proba(X_df)[:, 1]
        prob_xgb = self.base_xgb.predict_proba(X_df)[:, 1]

        # Meta-model prediction
        X_meta = np.column_stack([prob_lr, prob_rf, prob_xgb])
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Generate binary fraud classifications applying the decision threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor matrix.
        threshold : float, optional
            Custom decision threshold. Defaults to self.threshold (0.25).

        Returns
        -------
        np.ndarray
            Binary predictions (0 = Legitimate, 1 = Fraud).
        """
        th = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= th).astype(int)

    def predict_transactions(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Score a batch of transactions and return a formatted risk assessment DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Raw transaction DataFrame (with or without 'Class' target).
        threshold : float, optional
            Decision threshold (default = 0.25).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - transaction_index: Original row identifier
            - fraud_probability: Float probability [0.0, 1.0]
            - prediction: Binary flag (0 or 1)
            - risk_label: 'LOW_RISK' or 'HIGH_RISK'
        """
        th = self.threshold if threshold is None else threshold
        
        # Isolate features if target exists
        X_eval = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()
        
        probs = self.predict_proba(X_eval)
        preds = (probs >= th).astype(int)
        labels = [RISK_LABEL_HIGH if p == 1 else RISK_LABEL_LOW for p in preds]

        return pd.DataFrame({
            "transaction_index": df.index,
            "fraud_probability": np.round(probs, 6),
            "prediction": preds,
            "risk_label": labels
        })


def load_and_train_final_pipeline(
    csv_path: str,
    threshold: float = DEFAULT_THRESHOLD
) -> FraudDetectionPipeline:
    """
    Convenience helper to reconstruct and train the final pipeline directly from raw CSV.

    Parameters
    ----------
    csv_path : str
        Path to data/creditcard.csv.
    threshold : float, default=0.25
        Frozen operational threshold.

    Returns
    -------
    FraudDetectionPipeline
        Trained, production-ready pipeline instance.
    """
    from src.preprocessing import build_full_pipeline

    data_dict = build_full_pipeline(csv_path, verbose=False)
    X_train = data_dict["X_train_raw"]
    y_train = data_dict["y_train"]

    pipeline = FraudDetectionPipeline(threshold=threshold, random_state=RANDOM_SEED)
    pipeline.fit(X_train, y_train)
    return pipeline
