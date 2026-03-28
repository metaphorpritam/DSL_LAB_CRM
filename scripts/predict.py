"""
CatBoost Tuned — Credit Default Prediction + CIBIL Score
Reads borrower data from a Google Sheet, runs the saved CatBoost model,
computes a CIBIL-aligned credit score (300-900), and outputs JSON.

Designed for n8n integration (Execute Command node).

Usage:
    python predict.py
    python predict.py --sheet-id <GOOGLE_SHEET_ID>

The Google Sheet must be publicly readable (Anyone with the link → Viewer).
Input is read from **row 2** (row 1 = headers).
"""

import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from catboost import CatBoostClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"

DEFAULT_SHEET_ID = "1uTb3CsFbyO6TKJBkXLxNNmjhGep8NPgozKgf2GHwbBg"

# Column mapping: Google Sheet (original Kaggle names) → internal names
COL_MAP = {
    "RevolvingUtilizationOfUnsecuredLines": "unsecured_credit",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "delinq_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit",
    "NumberOfTimes90DaysLate": "delinq_90",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "delinq_60_89",
    "NumberOfDependents": "dependents",
}

# --- Load model and metadata ---
_model = CatBoostClassifier()
_model.load_model(str(MODEL_DIR / "catboost_tuned.cbm"))
_meta = joblib.load(MODEL_DIR / "catboost_tuned_meta.joblib")
_scaler = _meta["scaler"]
_feature_cols = _meta["feature_cols"]
_thresh_f1 = _meta["thresh_f1"]
_thresh_prec = _meta["thresh_prec"]

MEDIAN_INCOME = 5400.0


# ---------------------------------------------------------------------------
# Google Sheet reader
# ---------------------------------------------------------------------------
def read_google_sheet(sheet_id: str) -> dict:
    """Read row 2 from the public Google Sheet and return a dict with internal names."""
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError("Google Sheet has no data rows")

    # The input row is the last row (CreditScore column is empty / NaN)
    row = df.iloc[-1]

    raw = {}
    for sheet_col, internal_name in COL_MAP.items():
        val = row.get(sheet_col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            raw[internal_name] = None
        else:
            raw[internal_name] = float(val)
    return raw


# ---------------------------------------------------------------------------
# Preprocessing (same pipeline as training)
# ---------------------------------------------------------------------------
def preprocess(raw: dict) -> np.ndarray:
    monthly_income = raw.get("monthly_income")
    dependents = raw.get("dependents")

    monthly_income_missing = int(monthly_income is None)
    dependents_missing = int(dependents is None)

    if monthly_income is None:
        monthly_income = MEDIAN_INCOME
    if dependents is None:
        dependents = 0.0

    delinq_90_raw = raw.get("delinq_90") or 0
    delinq_sentinel = int(delinq_90_raw >= 96)

    delinq_30_59 = min(raw.get("delinq_30_59") or 0, 10)
    delinq_60_89 = min(raw.get("delinq_60_89") or 0, 10)
    delinq_90 = min(delinq_90_raw, 10)

    features = {
        "unsecured_credit": raw.get("unsecured_credit") or 0,
        "age": raw.get("age") or 0,
        "delinq_30_59": delinq_30_59,
        "debt_ratio": raw.get("debt_ratio") or 0,
        "monthly_income": monthly_income,
        "open_credit": raw.get("open_credit") or 0,
        "delinq_90": delinq_90,
        "real_estate_loans": raw.get("real_estate_loans") or 0,
        "delinq_60_89": delinq_60_89,
        "dependents": dependents,
        "monthly_income_missing": monthly_income_missing,
        "dependents_missing": dependents_missing,
        "delinq_sentinel": delinq_sentinel,
    }

    row = np.array([[features[c] for c in _feature_cols]], dtype=np.float64)
    return _scaler.transform(row)


# ---------------------------------------------------------------------------
# CIBIL credit score (300-900) per CIBIL_Scoring_Methodology.md
# ---------------------------------------------------------------------------
def cibil_score(raw: dict) -> dict:
    """Compute CIBIL-aligned credit score from raw borrower inputs."""
    delinq_90 = raw.get("delinq_90") or 0
    delinq_60_89 = raw.get("delinq_60_89") or 0
    delinq_30_59 = raw.get("delinq_30_59") or 0
    utilization = raw.get("unsecured_credit") or 0
    age = raw.get("age") or 0
    open_credit = raw.get("open_credit") or 0
    real_estate = raw.get("real_estate_loans") or 0
    debt_ratio = raw.get("debt_ratio") or 0
    monthly_income = raw.get("monthly_income") or 0

    # Component 1: Payment History (35%)
    has_default = int(delinq_90 >= 96)  # sentinel codes = default flag
    payment_score = min(100, max(0,
        100
        - delinq_90 * 25
        - delinq_60_89 * 15
        - delinq_30_59 * 7
        - has_default * 100
    ))

    # Component 2: Credit Utilization (30%)
    utilization_score = max(0, min(100, 100 - utilization * 150))

    # Component 3: Credit Mix & Duration (25%)
    age_factor = 60.0 if age >= 25 else (age / 25.0) * 60.0
    mix_score = min(100, open_credit * 2 + real_estate * 10 + age_factor)

    # Component 4: Other Factors (10%)
    debt_penalty = -30 if debt_ratio > 0.80 else 0
    income_bonus = min(50, monthly_income / 10000.0)
    other_score = min(100, max(0, 50 + debt_penalty + income_bonus))

    # Final weighted score → 300-900 scale
    weighted = (
        payment_score * 0.35
        + utilization_score * 0.30
        + mix_score * 0.25
        + other_score * 0.10
    )
    score = max(300, min(900, round(300 + weighted * 6)))

    # Grade assignment
    if score >= 750:
        grade = "Excellent"
    elif score >= 700:
        grade = "Very Good"
    elif score >= 650:
        grade = "Good"
    elif score >= 600:
        grade = "Fair"
    else:
        grade = "Poor"

    return {
        "cibil_score": score,
        "cibil_grade": grade,
        "components": {
            "payment_history": round(payment_score, 2),
            "credit_utilization": round(utilization_score, 2),
            "credit_mix_duration": round(mix_score, 2),
            "other_factors": round(other_score, 2),
        },
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(raw: dict) -> dict:
    X = preprocess(raw)
    prob = float(_model.predict_proba(X)[0, 1])

    cibil = cibil_score(raw)

    return {
        "default_probability": round(prob, 6),
        "prediction_f1": int(prob >= _thresh_f1),
        "prediction_conservative": int(prob >= _thresh_prec),
        "threshold_f1": round(_thresh_f1, 4),
        "threshold_conservative": round(_thresh_prec, 4),
        "cibil_score": cibil["cibil_score"],
        "cibil_grade": cibil["cibil_grade"],
        "cibil_components": cibil["components"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit default prediction + CIBIL score")
    parser.add_argument("--sheet-id", default=DEFAULT_SHEET_ID,
                        help="Google Sheet ID (default: project sheet)")
    parser.add_argument("json_input", nargs="?", default=None,
                        help="Optional JSON string input (skips Google Sheet)")
    args = parser.parse_args()

    if args.json_input:
        raw_input = json.loads(args.json_input)
    else:
        raw_input = read_google_sheet(args.sheet_id)

    result = predict(raw_input)
    print(json.dumps(result, indent=2))
