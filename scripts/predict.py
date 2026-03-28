"""
CatBoost Tuned — Credit Default Prediction Script
Loads the saved CatBoost tuned model and generates predictions from raw borrower inputs.
Designed for integration with n8n (Execute Command / Python Function nodes).

Usage:
    python predict.py '<json_input>'

Example:
    python predict.py '{
        "unsecured_credit": 0.8,
        "age": 45,
        "delinq_30_59": 0,
        "debt_ratio": 0.3,
        "monthly_income": 5000,
        "open_credit": 8,
        "delinq_90": 0,
        "real_estate_loans": 1,
        "delinq_60_89": 0,
        "dependents": 2
    }'

Input fields (all numeric):
    unsecured_credit   - Revolving utilisation of unsecured lines (ratio, can exceed 1.0)
    age                - Borrower age in years
    delinq_30_59       - Number of times 30-59 days past due
    debt_ratio         - Debt ratio (monthly debt payments / monthly income)
    monthly_income     - Monthly income (nullable → pass null or omit for missing)
    open_credit        - Number of open credit lines and loans
    delinq_90          - Number of times 90+ days late
    real_estate_loans  - Number of real estate loans or lines
    delinq_60_89       - Number of times 60-89 days past due
    dependents         - Number of dependents (nullable → pass null or omit for missing)

Output (JSON):
    {
        "default_probability": 0.073,
        "prediction_f1": 0,
        "prediction_conservative": 0,
        "threshold_f1": 0.752,
        "threshold_conservative": 0.905
    }
"""

import sys
import json
import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier

MODEL_DIR = Path(__file__).resolve().parent / "models"

# --- Load model and metadata once at import time ---
_model = CatBoostClassifier()
_model.load_model(str(MODEL_DIR / "catboost_tuned.cbm"))
_meta = joblib.load(MODEL_DIR / "catboost_tuned_meta.joblib")
_scaler = _meta["scaler"]
_feature_cols = _meta["feature_cols"]
_thresh_f1 = _meta["thresh_f1"]
_thresh_prec = _meta["thresh_prec"]

DELINQ_COLS = ["delinq_30_59", "delinq_60_89", "delinq_90"]
MEDIAN_INCOME = 5400.0  # training-set median (used for imputation)


def preprocess(raw: dict) -> np.ndarray:
    """Apply the same pipeline as training: sentinel flag, clipping,
    missing indicators, imputation, then StandardScaler."""

    monthly_income = raw.get("monthly_income")
    dependents = raw.get("dependents")

    # Missing indicators (before imputation)
    monthly_income_missing = int(monthly_income is None)
    dependents_missing = int(dependents is None)

    # Imputation
    if monthly_income is None:
        monthly_income = MEDIAN_INCOME
    if dependents is None:
        dependents = 0.0

    delinq_90_raw = raw.get("delinq_90", 0)

    # Sentinel flag: codes 96/98 in the original data
    delinq_sentinel = int(delinq_90_raw >= 96)

    # Clip delinquency columns at 10
    delinq_30_59 = min(raw.get("delinq_30_59", 0), 10)
    delinq_60_89 = min(raw.get("delinq_60_89", 0), 10)
    delinq_90 = min(delinq_90_raw, 10)

    # Build feature vector in the exact order the model expects
    features = {
        "unsecured_credit": raw.get("unsecured_credit", 0),
        "age": raw.get("age", 0),
        "delinq_30_59": delinq_30_59,
        "debt_ratio": raw.get("debt_ratio", 0),
        "monthly_income": monthly_income,
        "open_credit": raw.get("open_credit", 0),
        "delinq_90": delinq_90,
        "real_estate_loans": raw.get("real_estate_loans", 0),
        "delinq_60_89": delinq_60_89,
        "dependents": dependents,
        "monthly_income_missing": monthly_income_missing,
        "dependents_missing": dependents_missing,
        "delinq_sentinel": delinq_sentinel,
    }

    row = np.array([[features[c] for c in _feature_cols]], dtype=np.float64)
    row_scaled = _scaler.transform(row)
    return row_scaled


def predict(raw: dict) -> dict:
    """Return default probability and binary predictions under both thresholds."""
    X = preprocess(raw)
    prob = float(_model.predict_proba(X)[0, 1])
    return {
        "default_probability": round(prob, 6),
        "prediction_f1": int(prob >= _thresh_f1),
        "prediction_conservative": int(prob >= _thresh_prec),
        "threshold_f1": round(_thresh_f1, 4),
        "threshold_conservative": round(_thresh_prec, 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py '<json_input>'", file=sys.stderr)
        sys.exit(1)

    raw_input = json.loads(sys.argv[1])
    result = predict(raw_input)
    print(json.dumps(result))
