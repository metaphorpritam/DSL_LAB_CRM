"""
data_preprocessing.py — quick-look loader for the cleaned training data.

Applies the shared cleaning pipeline (sentinel flagging, delinquency capping,
missing-value indicators, median/zero imputation) and prints a short summary.
The data path is resolved relative to this file, so the script runs from any
working directory.

Usage:
    python scripts/data_preprocessing.py
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "Data" / "cs-training.csv"

RENAME = {
    "SeriousDlqin2yrs":                     "defaulted",
    "RevolvingUtilizationOfUnsecuredLines":  "unsecured_credit",
    "age":                                   "age",
    "NumberOfTime30-59DaysPastDueNotWorse":  "delinq_30_59",
    "DebtRatio":                             "debt_ratio",
    "MonthlyIncome":                         "monthly_income",
    "NumberOfOpenCreditLinesAndLoans":       "open_credit",
    "NumberOfTimes90DaysLate":               "delinq_90",
    "NumberRealEstateLoansOrLines":          "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse":  "delinq_60_89",
    "NumberOfDependents":                    "dependents",
}

DELINQ_COLS = ["delinq_30_59", "delinq_60_89", "delinq_90"]


def load_clean(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and clean the training data, returning the prepared DataFrame."""
    raw = pd.read_csv(data_path, index_col=0).rename(columns=RENAME)

    df = raw.copy()
    df = df[df["age"] > 0]

    # Sentinel treatment: flag codes 96/98, cap delinquency counts at 10
    df["delinq_sentinel"] = (df["delinq_90"] >= 96).astype(int)
    for col in DELINQ_COLS:
        df[col] = df[col].clip(upper=10)

    # Missing indicators (derived from raw, before imputation)
    df["monthly_income_missing"] = raw.loc[df.index, "monthly_income"].isna().astype(int)
    df["dependents_missing"]     = raw.loc[df.index, "dependents"].isna().astype(int)

    # Imputation
    df["monthly_income"] = df["monthly_income"].fillna(df["monthly_income"].median())
    df["dependents"]     = df["dependents"].fillna(0)

    return df


if __name__ == "__main__":
    df = load_clean()
    print(f"Shape          : {df.shape}")
    print(f"Default rate   : {df['defaulted'].mean():.3%}")
    print(f"Sentinel rows  : {df['delinq_sentinel'].sum():,}")
