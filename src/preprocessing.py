"""
src/preprocessing.py
====================
Reusable, leakage-free preprocessing utilities for the Credit Card Fraud
Detection & Risk Analysis project.

Design principles
-----------------
* RANDOM_SEED = 42 for all stochastic operations.
* Scaler parameters are ALWAYS fitted on training data only.
* No model training, no SMOTE — those belong in the modelling phase.
* All functions accept DataFrames and return DataFrames/arrays so they
  compose naturally in any downstream pipeline.

Usage example
-------------
    from src.preprocessing import build_full_pipeline

    results = build_full_pipeline('data/creditcard.csv')
    X_train = results['X_train_scaled']
    X_test  = results['X_test_scaled']
    y_train = results['y_train']
    y_test  = results['y_test']
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
RANDOM_SEED:   int   = 42
TARGET_COL:    str   = "Class"
TEST_SIZE:     float = 0.20

# Columns that require scaling.
# V1-V28 are PCA-transformed and already approximately zero-centred;
# re-scaling them is unnecessary and could distort their variance structure.
# Amount is highly right-skewed; Time spans a large numerical range.
# Log_Amount and Hour are the two engineered features that also need scaling.
COLS_TO_SCALE: list = ["Amount", "Time", "Log_Amount", "Hour"]


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw creditcard.csv into a DataFrame.

    Parameters
    ----------
    csv_path : str
        Absolute or relative path to creditcard.csv.

    Returns
    -------
    pd.DataFrame
        Raw dataset, completely unchanged.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. "
            "Ensure creditcard.csv is in the data/ directory."
        )
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# 2. Duplicate Handling
# ---------------------------------------------------------------------------

def audit_duplicates(df: pd.DataFrame) -> dict:
    """
    Audit exact duplicate rows and return a structured report.

    Checks:
    - Total non-first-occurrence duplicate rows.
    - Class distribution within duplicated rows.
    - Whether any rows are feature-identical but have conflicting Class labels
      (same features, different target) — a data integrity concern.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame including the 'Class' column.

    Returns
    -------
    dict with keys:
        n_total_rows, n_duplicate_rows, pct_duplicate,
        dup_class_breakdown, conflicting_pairs
    """
    n_dup = int(df.duplicated().sum())
    pct   = n_dup / len(df) * 100

    dup_mask        = df.duplicated(keep=False)
    dup_rows        = df[dup_mask]
    class_breakdown = dup_rows[TARGET_COL].value_counts().sort_index()

    # Conflict detection: same feature hash -> more than one unique Class value
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    df_temp = df.copy()
    df_temp["_hash"] = df_temp[feature_cols].apply(
        lambda row: hash(tuple(row)), axis=1
    )
    conflict_hashes = (
        df_temp.groupby("_hash")[TARGET_COL]
        .nunique()
        .pipe(lambda s: s[s > 1].index)
    )
    conflicting_pairs = df_temp[df_temp["_hash"].isin(conflict_hashes)].drop(
        columns=["_hash"]
    )

    return {
        "n_total_rows":        len(df),
        "n_duplicate_rows":    n_dup,
        "pct_duplicate":       float(pct),
        "dup_class_breakdown": class_breakdown,
        "conflicting_pairs":   conflicting_pairs,
    }


def drop_exact_duplicates(
    df: pd.DataFrame,
    keep: str = "first",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove exact duplicate rows (all columns identical) from the DataFrame.

    Decision rationale (Phase 1 conclusion):
    - Duplicates provide no additional information.
    - Identical rows in both train and test sets cause data leakage.
    - The original CSV is never modified; this operates on an in-memory copy.

    Parameters
    ----------
    df     : Input DataFrame.
    keep   : Which occurrence to keep: 'first' (default) or 'last'.
    verbose: Print audit summary if True.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame with a reset integer index.
    """
    report = audit_duplicates(df)

    if verbose:
        print("  Duplicate audit:")
        print(f"    Total rows               : {report['n_total_rows']:,}")
        print(f"    Duplicate rows (non-1st) : {report['n_duplicate_rows']:,}  "
              f"({report['pct_duplicate']:.4f}%)")
        print(f"    Class breakdown in dups:")
        for cls, cnt in report["dup_class_breakdown"].items():
            label = "Genuine" if cls == 0 else "Fraud"
            print(f"      Class {cls} ({label}): {cnt:,}")

    if not report["conflicting_pairs"].empty:
        warnings.warn(
            f"\n[WARNING] {len(report['conflicting_pairs'])} rows are "
            "feature-identical but have DIFFERENT Class labels. "
            "NOT auto-removed. Review manually.\n"
            + report["conflicting_pairs"].head(10).to_string(),
            UserWarning,
            stacklevel=2,
        )

    df_clean = df.drop_duplicates(keep=keep).reset_index(drop=True)

    if verbose:
        removed = report["n_total_rows"] - len(df_clean)
        print(f"    Rows removed             : {removed:,}")
        print(f"    Rows remaining           : {len(df_clean):,}")

    return df_clean


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a small set of defensible transaction-level features.

    Each feature:
    - Contains NO target information (no look-ahead / label leakage).
    - Is available at real-time prediction time.
    - Has a clear business or statistical motivation from Phase 1 EDA.

    Features created
    ----------------
    Log_Amount : float
        log1p(Amount).
        Motivation: Amount is highly right-skewed (Phase 1 Task 4).
        The log transformation compresses the long tail and helps linear
        models separate fraud from non-fraud on this dimension.
        log1p handles Amount=0 safely.

    Hour : float
        Approximate hour-of-day (0.0 to 23.999) derived from Time.
        Motivation: Phase 1 Task 5 showed genuine transactions follow a
        diurnal cycle while fraud is more uniformly spread across hours.
        Formula: (Time % 86400) / 3600  — maps elapsed seconds to a
        position within a 24-hour window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Amount' and 'Time' columns.

    Returns
    -------
    pd.DataFrame
        Original columns preserved; two new columns appended.
    """
    df = df.copy()
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Hour"]       = (df["Time"] % 86_400) / 3_600
    return df


# ---------------------------------------------------------------------------
# 4. Feature / Target Separation
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple:
    """
    Separate predictor matrix X from target vector y.

    Parameters
    ----------
    df         : Full DataFrame including target column.
    target_col : Name of target column (default: 'Class').

    Returns
    -------
    (X, y) : (pd.DataFrame, pd.Series)

    Raises
    ------
    KeyError   : If target_col not in df.
    ValueError : If target contains values other than {0, 1}.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    invalid = set(df[target_col].unique()) - {0, 1}
    if invalid:
        raise ValueError(
            f"Target contains unexpected values: {invalid}. Expected 0 and 1."
        )

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return X, y


# ---------------------------------------------------------------------------
# 5. Train / Test Split
# ---------------------------------------------------------------------------

def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Create a stratified 80/20 train/test split.

    Stratification ensures the ~0.17% fraud fraction is faithfully preserved
    in both subsets, preventing any accidental under/over-representation of
    the minority class.

    Parameters
    ----------
    X            : Predictor matrix.
    y            : Binary target Series.
    test_size    : Fraction reserved for testing (default 0.20).
    random_state : Reproducibility seed (default 42).

    Returns
    -------
    (X_train, X_test, y_train, y_test) with reset integer indices.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 6. Scaling
# ---------------------------------------------------------------------------

def fit_scaler(
    X_train: pd.DataFrame,
    cols_to_scale: Optional[list] = None,
) -> tuple:
    """
    Fit a RobustScaler on specified columns of the TRAINING SET ONLY.

    Scaler choice — RobustScaler:
    - Amount has extreme right-skew with high-value outliers (Phase 1 Task 8).
    - RobustScaler uses median + IQR, making it robust to outliers.
    - Preserves potential fraud signal in extreme Amount values.

    V1-V28 are NOT scaled here (already PCA-normalised).

    Parameters
    ----------
    X_train       : Training predictor matrix. Scaler fitted HERE ONLY.
    cols_to_scale : Columns to scale. Defaults to COLS_TO_SCALE.

    Returns
    -------
    (scaler, actual_cols) : (RobustScaler, list[str])
        scaler      — fitted scaler.
        actual_cols — columns actually scaled (exist in X_train).
    """
    if cols_to_scale is None:
        cols_to_scale = COLS_TO_SCALE

    actual_cols = [c for c in cols_to_scale if c in X_train.columns]

    if not actual_cols:
        warnings.warn(
            "No scale columns found in X_train. Returning unfitted scaler.",
            UserWarning, stacklevel=2,
        )
        return RobustScaler(), []

    scaler = RobustScaler()
    scaler.fit(X_train[actual_cols])   # <<< training data ONLY
    return scaler, actual_cols


def apply_scaler(
    X: pd.DataFrame,
    scaler: RobustScaler,
    cols_to_scale: list,
) -> pd.DataFrame:
    """
    Apply a pre-fitted RobustScaler to specified columns of X.

    NEVER refits the scaler. Only transforms.
    Call fit_scaler() on training data first, then apply_scaler() on
    train/validation/test/new-data separately.

    Parameters
    ----------
    X             : Predictor matrix (any split).
    scaler        : Pre-fitted RobustScaler from fit_scaler().
    cols_to_scale : Must match the columns the scaler was fitted on.

    Returns
    -------
    pd.DataFrame — copy of X with scaled columns replaced.
    """
    X_out = X.copy()
    actual = [c for c in cols_to_scale if c in X_out.columns]
    if actual:
        X_out[actual] = scaler.transform(X_out[actual])
    return X_out


# ---------------------------------------------------------------------------
# 7. Outlier Strategy
# ---------------------------------------------------------------------------

def get_outlier_thresholds(
    X_train: pd.DataFrame,
    cols: Optional[list] = None,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict:
    """
    Compute percentile-based winsorisation thresholds from TRAINING DATA ONLY.

    Phase 1 Outlier Strategy:
    - Extreme Amount outliers exist (max $25,691; P99 ~ $2,125).
    - Outliers are NOT auto-removed — fraud may manifest as extreme values.
    - This function provides thresholds for OPTIONAL experimentation in Phase 3.
    - Default pipeline behaviour: preserve original values.

    Parameters
    ----------
    X_train   : Training predictor matrix. Thresholds derived here only.
    cols      : Columns to analyse. Defaults to ['Amount', 'Log_Amount'].
    lower_pct : Lower percentile for floor.
    upper_pct : Upper percentile for ceiling.

    Returns
    -------
    dict: {column: {'lower': float, 'upper': float}}
    """
    if cols is None:
        cols = ["Amount", "Log_Amount"]
    thresholds = {}
    for col in cols:
        if col in X_train.columns:
            thresholds[col] = {
                "lower": float(np.percentile(X_train[col], lower_pct)),
                "upper": float(np.percentile(X_train[col], upper_pct)),
            }
    return thresholds


def apply_winsorisation(
    X: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    """
    Apply winsorisation using pre-computed training-set thresholds.

    IMPORTANT: This is an OPTIONAL experimental transformation.
    - Default pipeline does NOT call this function.
    - Must be tested against the default in Phase 3 model comparisons.
    - The comparison result must come from held-out test performance, NOT
      from test-set snooping.

    Parameters
    ----------
    X          : Predictor matrix.
    thresholds : Output of get_outlier_thresholds() (training-derived).

    Returns
    -------
    pd.DataFrame — copy of X with winsorised columns.
    """
    X_out = X.copy()
    for col, bounds in thresholds.items():
        if col in X_out.columns:
            X_out[col] = X_out[col].clip(lower=bounds["lower"],
                                          upper=bounds["upper"])
    return X_out


# ---------------------------------------------------------------------------
# 8. Full Pipeline
# ---------------------------------------------------------------------------

def build_full_pipeline(
    csv_path: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> dict:
    """
    Execute the complete leakage-free preprocessing pipeline in one call.

    Steps
    -----
    1  Load raw CSV.
    2  Audit and drop exact duplicates (before any split).
    3  Engineer features (Log_Amount, Hour).
    4  Separate X / y.
    5  Stratified train/test split.
    6  Fit RobustScaler on X_train ONLY.
    7  Transform X_train and X_test with the fitted scaler.
    8  Compute outlier thresholds from X_train (for Phase 3 experiments).

    Parameters
    ----------
    csv_path     : Path to data/creditcard.csv.
    test_size    : Test fraction (default 0.20).
    random_state : Reproducibility seed (default 42).
    verbose      : Print progress if True.

    Returns
    -------
    dict with keys:
        X_train_scaled, X_test_scaled   — scaled predictor matrices
        X_train_raw,    X_test_raw      — unscaled predictor matrices
        y_train,        y_test          — target vectors
        scaler                          — fitted RobustScaler
        cols_scaled                     — list of scaled column names
        thresholds                      — outlier bounds from training data
        n_original                      — row count before dedup
        n_after_dedup                   — row count after dedup
        n_removed_dup                   — rows removed
    """
    if verbose:
        print("[1/8] Loading raw data ...")
    df_raw     = load_raw_data(csv_path)
    n_original = len(df_raw)

    if verbose:
        print("[2/8] Auditing and dropping exact duplicates ...")
    df_clean      = drop_exact_duplicates(df_raw, verbose=verbose)
    n_after_dedup = len(df_clean)
    n_removed     = n_original - n_after_dedup

    if verbose:
        print("[3/8] Engineering features ...")
    df_feat = engineer_features(df_clean)

    if verbose:
        print("[4/8] Separating features and target ...")
    X, y = split_features_target(df_feat)

    if verbose:
        print("[5/8] Stratified train/test split ...")
    X_train_raw, X_test_raw, y_train, y_test = make_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print("[6/8] Fitting RobustScaler on X_train only ...")
    scaler, cols_scaled = fit_scaler(X_train_raw)

    if verbose:
        print("[7/8] Transforming train and test sets ...")
    X_train_scaled = apply_scaler(X_train_raw, scaler, cols_scaled)
    X_test_scaled  = apply_scaler(X_test_raw,  scaler, cols_scaled)

    if verbose:
        print("[8/8] Computing outlier thresholds from training data ...")
    thresholds = get_outlier_thresholds(X_train_raw)

    if verbose:
        print("\nPipeline complete.")
        print(f"  Original rows  : {n_original:,}")
        print(f"  After dedup    : {n_after_dedup:,}  ({n_removed:,} removed)")
        print(f"  X_train shape  : {X_train_scaled.shape}")
        print(f"  X_test  shape  : {X_test_scaled.shape}")
        print(f"  Fraud in train : {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
        print(f"  Fraud in test  : {y_test.sum():,} ({y_test.mean()*100:.4f}%)")
        print(f"  Cols scaled    : {cols_scaled}")

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled":  X_test_scaled,
        "X_train_raw":    X_train_raw,
        "X_test_raw":     X_test_raw,
        "y_train":        y_train,
        "y_test":         y_test,
        "scaler":         scaler,
        "cols_scaled":    cols_scaled,
        "thresholds":     thresholds,
        "n_original":     n_original,
        "n_after_dedup":  n_after_dedup,
        "n_removed_dup":  n_removed,
    }
