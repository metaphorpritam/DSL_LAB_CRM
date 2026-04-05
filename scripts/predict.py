"""
CatBoost Tuned — Credit Default Prediction + CIBIL Score + SHAP Explanation
Reads borrower data from a Google Sheet, runs the saved CatBoost model,
computes a CIBIL-aligned credit score (300-900), and provides per-feature
SHAP explanations with actionable improvement recommendations.

Designed for n8n integration (Execute Command node).

Usage:
    python predict.py
    python predict.py --sheet-id <GOOGLE_SHEET_ID>

The Google Sheet must be publicly readable (Anyone with the link → Viewer).
Input is read from the last row (the row awaiting prediction).
"""

import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
import joblib
import shap
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

# --- SHAP explainer (TreeExplainer is exact & fast for CatBoost) ---
_explainer = shap.TreeExplainer(_model)

# Human-readable feature labels
_FEATURE_LABELS = {
    "unsecured_credit": "Revolving Credit Utilization",
    "age": "Age",
    "delinq_30_59": "30-59 Days Past Due",
    "debt_ratio": "Debt Ratio",
    "monthly_income": "Monthly Income",
    "open_credit": "Open Credit Lines",
    "delinq_90": "90+ Days Late",
    "real_estate_loans": "Real Estate Loans",
    "delinq_60_89": "60-89 Days Past Due",
    "dependents": "Dependents",
    "monthly_income_missing": "Income Data Missing",
    "dependents_missing": "Dependents Data Missing",
    "delinq_sentinel": "Severe Default Flag",
}

# Improvement tips keyed by feature name; only shown when feature pushes risk UP
_IMPROVEMENT_TIPS = {
    "unsecured_credit": "Reduce revolving credit utilization below 30% — pay down balances or request a credit limit increase.",
    "delinq_30_59": "Avoid any new late payments. Set up auto-pay or payment reminders for all accounts.",
    "delinq_60_89": "Prioritize clearing any 60-89 day overdue accounts immediately.",
    "delinq_90": "Resolve all 90+ day delinquencies — negotiate settlements or payment plans with creditors.",
    "debt_ratio": "Lower your debt-to-income ratio below 40% — pay down high-interest debt first.",
    "monthly_income": "Increase documented income (salary revision, secondary income) to improve repayment capacity.",
    "monthly_income_missing": "Provide income documentation — missing income data is treated as a risk signal.",
    "dependents": "Higher dependents reduce disposable income; ensure debt obligations account for household size.",
    "dependents_missing": "Provide dependents information to avoid conservative risk assumptions.",
    "delinq_sentinel": "This flag indicates a serious default event. Focus on rebuilding clean payment history over 12-24 months.",
    "open_credit": "Maintain a healthy mix of credit lines. Avoid opening too many new accounts at once.",
    "real_estate_loans": "Secured loans (real estate) can positively diversify your credit mix.",
    "age": "Credit history length improves naturally over time — maintain existing accounts.",
}


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
def cibil_score(raw: dict, default_probability: float = 0.0) -> dict:
    """Compute CIBIL-aligned credit score from raw borrower inputs.

    The composite formula weights:
      - Default Probability  50%  (ML model P(default) mapped to 0-100)
      - Payment History      25%  (delinquency penalties)
      - Credit Utilisation   15%
      - Credit Mix/Duration  10%
    """
    delinq_90 = raw.get("delinq_90") or 0
    delinq_60_89 = raw.get("delinq_60_89") or 0
    delinq_30_59 = raw.get("delinq_30_59") or 0
    utilization = raw.get("unsecured_credit") or 0
    age = raw.get("age") or 0
    open_credit = raw.get("open_credit") or 0
    real_estate = raw.get("real_estate_loans") or 0

    # Component 1: Payment History (25%)
    has_default = int(delinq_90 >= 96)  # sentinel codes = default flag
    payment_score = min(100, max(0,
        100
        - delinq_90 * 25
        - delinq_60_89 * 15
        - delinq_30_59 * 7
        - has_default * 100
    ))

    # Component 2: Credit Utilization (15%)
    utilization_score = max(0, min(100, 100 - utilization * 150))

    # Component 3: Credit Mix & Duration (10%)
    age_factor = 60.0 if age >= 25 else (age / 25.0) * 60.0
    mix_score = min(100, open_credit * 2 + real_estate * 10 + age_factor)

    # Component 4: Default Probability (50%) — ML model output mapped to 0-100
    # Lower probability = higher score (inverse: good borrower → high component)
    default_component = (1.0 - default_probability) * 100

    # Final weighted score → 300-900 scale
    weighted = (
        payment_score * 0.25
        + utilization_score * 0.15
        + mix_score * 0.10
        + default_component * 0.50
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
            "default_probability_component": round(default_component, 2),
            "payment_history": round(payment_score, 2),
            "credit_utilization": round(utilization_score, 2),
            "credit_mix_duration": round(mix_score, 2),
        },
    }


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------
def explain(X_scaled: np.ndarray) -> dict:
    """Compute per-feature SHAP values and return structured explanation."""
    sv = _explainer.shap_values(X_scaled)
    # TreeExplainer may return a list for binary classification
    if isinstance(sv, list):
        sv = sv[1]  # class-1 (default) SHAP values
    shap_vals = sv[0]  # single row
    base_value = _explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    # Build per-feature breakdown sorted by |SHAP|
    breakdown = []
    for i, col in enumerate(_feature_cols):
        val = float(shap_vals[i])
        breakdown.append({
            "feature": col,
            "label": _FEATURE_LABELS.get(col, col),
            "shap_value": round(val, 6),
            "direction": "increases risk" if val > 0 else "decreases risk",
        })
    breakdown.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # Top risk drivers (positive SHAP = pushes toward default) with tips
    risk_drivers = []
    for item in breakdown:
        if item["shap_value"] > 0.001:
            tip = _IMPROVEMENT_TIPS.get(item["feature"])
            risk_drivers.append({
                "feature": item["label"],
                "impact": round(item["shap_value"], 4),
                "recommendation": tip,
            })

    # Top protective factors (negative SHAP = reduces default risk)
    protective = []
    for item in breakdown:
        if item["shap_value"] < -0.001:
            protective.append({
                "feature": item["label"],
                "impact": round(item["shap_value"], 4),
            })

    return {
        "base_value": round(float(base_value), 6),
        "feature_contributions": breakdown,
        "risk_drivers": risk_drivers,
        "protective_factors": protective,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(raw: dict) -> dict:
    X = preprocess(raw)
    prob = float(_model.predict_proba(X)[0, 1])

    cibil = cibil_score(raw, default_probability=prob)
    shap_explanation = explain(X)

    return {
        "default_probability": round(prob, 6),
        "prediction_f1": int(prob >= _thresh_f1),
        "prediction_conservative": int(prob >= _thresh_prec),
        "threshold_f1": round(_thresh_f1, 4),
        "threshold_conservative": round(_thresh_prec, 4),
        "cibil_score": cibil["cibil_score"],
        "cibil_grade": cibil["cibil_grade"],
        "cibil_components": cibil["components"],
        "shap_explanation": shap_explanation,
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
