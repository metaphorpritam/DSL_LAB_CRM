"""
data.py — loads and preprocesses the Give Me Some Credit dataset.

Returns:
    load_data()      → preprocessed DataFrame
    make_splits()    → (X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, class_ratio)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, RENAME, FEATURE_COLS,
    SEED, TEST_SIZE, VAL_SIZE,
)

# ── Sentinel / delinquency columns ────────────────────────────────────────────
DELINQ_COLS = ["delinq_30_59", "delinq_60_89", "delinq_90"]


def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Read CSV, rename columns, apply the shared cleaning pipeline.

    Returns a DataFrame with 13 feature columns + 'defaulted' target.
    """
    path = Path(data_path or DATA_PATH)
    raw = pd.read_csv(path, index_col=0).rename(columns=RENAME)

    df = raw.copy()
    df = df[df["age"] > 0]

    # Sentinel treatment: flag 96/98 codes, cap all delinq cols at 10
    df["delinq_sentinel"] = (df["delinq_90"] >= 96).astype(int)
    for col in DELINQ_COLS:
        df[col] = df[col].clip(upper=10)

    # Missing indicators (derived from raw — safe against imputation order)
    df["monthly_income_missing"] = raw.loc[df.index, "monthly_income"].isna().astype(int)
    df["dependents_missing"]     = raw.loc[df.index, "dependents"].isna().astype(int)

    # Imputation
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
    df["dependents"]     = df["dependents"].fillna(0)

    print(f"Dataset shape  : {df.shape}")
    print(f"Default rate   : {df['defaulted'].mean():.3%}")
    print(f"Sentinel rows  : {df['delinq_sentinel'].sum():,}")

    return df


def make_splits(df: pd.DataFrame):
    """Stratified 70/15/15 train/val/test split with StandardScaler.

    Returns:
        X_train_s, X_val_s, X_test_s  — scaled feature arrays (float32)
        y_train, y_val, y_test         — label arrays (float32)
        scaler                         — fitted StandardScaler
        class_ratio                    — n_neg / n_pos (for class_weights)
    """
    np.random.seed(SEED)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["defaulted"].values.astype(np.float32)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED, stratify=y_temp
    )

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    n_pos       = int(y_train.sum())
    n_neg       = int(len(y_train) - n_pos)
    class_ratio = n_neg / n_pos

    print(f"\nTrain : {X_train_s.shape}  default={y_train.mean():.3%}")
    print(f"Val   : {X_val_s.shape}   default={y_val.mean():.3%}")
    print(f"Test  : {X_test_s.shape}  default={y_test.mean():.3%}")
    print(f"class_ratio (n_neg/n_pos): {class_ratio:.2f}")

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, class_ratio
