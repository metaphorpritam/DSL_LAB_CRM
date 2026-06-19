"""
Streamlit Credit Risk Dashboard
Probability of default, CIBIL score, SHAP rationale, and improvement guidance.

Run:  streamlit run scripts/app.py
"""

import sys
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup paths & imports from predict.py helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from predict import (
    _model,
    _scaler,
    _feature_cols,
    _thresh_f1,
    _thresh_prec,
    _explainer,
    _FEATURE_LABELS,
    _IMPROVEMENT_TIPS,
    preprocess,
    cibil_score,
    explain,
    predict,
    MEDIAN_INCOME,
)

# ---------------------------------------------------------------------------
# Random customer generator (realistic distributions)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng()

FEATURE_RANGES = {
    "unsecured_credit": (0.0, 1.5),
    "age": (21, 80),
    "delinq_30_59": (0, 5),
    "debt_ratio": (0.0, 2.0),
    "monthly_income": (1000.0, 25000.0),
    "open_credit": (1, 25),
    "delinq_90": (0, 5),
    "real_estate_loans": (0, 5),
    "delinq_60_89": (0, 5),
    "dependents": (0, 5),
}


def random_customer() -> dict:
    """Generate a plausible random borrower profile."""
    return {
        "unsecured_credit": round(float(_RNG.exponential(0.3)), 4),
        "age": int(_RNG.integers(21, 80)),
        "delinq_30_59": int(_RNG.choice([0, 0, 0, 0, 1, 1, 2, 3])),
        "debt_ratio": round(float(np.clip(_RNG.exponential(0.4), 0, 5)), 4),
        "monthly_income": round(float(np.clip(_RNG.lognormal(8.5, 0.8), 1000, 50000)), 2),
        "open_credit": int(np.clip(_RNG.poisson(8), 1, 30)),
        "delinq_90": int(_RNG.choice([0, 0, 0, 0, 0, 0, 1, 2])),
        "real_estate_loans": int(_RNG.choice([0, 0, 1, 1, 2, 3])),
        "delinq_60_89": int(_RNG.choice([0, 0, 0, 0, 0, 1, 1, 2])),
        "dependents": int(_RNG.choice([0, 0, 0, 1, 1, 2, 3])),
    }


# ---------------------------------------------------------------------------
# Score color helpers
# ---------------------------------------------------------------------------
def score_color(score: int) -> str:
    if score >= 750:
        return "#2ecc71"  # green
    if score >= 700:
        return "#27ae60"
    if score >= 650:
        return "#f39c12"  # orange
    if score >= 600:
        return "#e67e22"
    return "#e74c3c"  # red


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", page_icon="$", layout="wide")

st.title("Credit Risk Dashboard")
st.caption("CatBoost model + CIBIL scoring + SHAP explanations")

# --- Sidebar: customer inputs ---
st.sidebar.header("Borrower Profile")

if st.sidebar.button("Randomize Customer", type="primary", use_container_width=True):
    st.session_state["customer"] = random_customer()

# Initialize defaults
if "customer" not in st.session_state:
    st.session_state["customer"] = random_customer()

c = st.session_state["customer"]

unsecured_credit = st.sidebar.slider(
    "Revolving Credit Utilization", 0.0, 3.0, float(c["unsecured_credit"]), 0.01,
    help="Total balance on revolving credit / credit limit"
)
age = st.sidebar.slider("Age", 18, 100, int(c["age"]))
delinq_30_59 = st.sidebar.slider("30-59 Days Past Due (count)", 0, 10, int(c["delinq_30_59"]))
delinq_60_89 = st.sidebar.slider("60-89 Days Past Due (count)", 0, 10, int(c["delinq_60_89"]))
delinq_90 = st.sidebar.slider("90+ Days Late (count)", 0, 10, int(c["delinq_90"]))
debt_ratio = st.sidebar.slider("Debt Ratio", 0.0, 5.0, float(c["debt_ratio"]), 0.01,
                                help="Monthly debt payments / monthly income")
monthly_income = st.sidebar.number_input("Monthly Income ($)", 0.0, 100000.0,
                                          float(c["monthly_income"]), 100.0)
open_credit = st.sidebar.slider("Open Credit Lines", 0, 40, int(c["open_credit"]))
real_estate_loans = st.sidebar.slider("Real Estate Loans", 0, 10, int(c["real_estate_loans"]))
dependents = st.sidebar.slider("Dependents", 0, 10, int(c["dependents"]))

# Build raw input dict
raw = {
    "unsecured_credit": unsecured_credit,
    "age": age,
    "delinq_30_59": delinq_30_59,
    "debt_ratio": debt_ratio,
    "monthly_income": monthly_income,
    "open_credit": open_credit,
    "delinq_90": delinq_90,
    "real_estate_loans": real_estate_loans,
    "delinq_60_89": delinq_60_89,
    "dependents": dependents,
}

# --- Run prediction ---
result = predict(raw)
prob = result["default_probability"]
cibil = result["cibil_score"]
grade = result["cibil_grade"]
components = result["cibil_components"]
shap_data = result["shap_explanation"]

# ---------------------------------------------------------------------------
# Top metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Default Probability", f"{prob:.1%}")
    verdict = "DEFAULT" if result["prediction_f1"] else "NO DEFAULT"
    color = "#e74c3c" if result["prediction_f1"] else "#2ecc71"
    st.markdown(f"**Prediction:** <span style='color:{color};font-size:1.2em'>{verdict}</span>",
                unsafe_allow_html=True)

with col2:
    sc = score_color(cibil)
    st.markdown(f"### CIBIL Score")
    st.markdown(
        f"<span style='font-size:3em;font-weight:bold;color:{sc}'>{cibil}</span>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown("### Grade")
    st.markdown(
        f"<span style='font-size:2.5em;font-weight:bold;color:{sc}'>{grade}</span>",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown("### Thresholds")
    st.markdown(f"- **F1-optimal:** {result['threshold_f1']:.4f}")
    st.markdown(f"- **Conservative:** {result['threshold_conservative']:.4f}")

st.divider()

# ---------------------------------------------------------------------------
# CIBIL Score Components
# ---------------------------------------------------------------------------
st.subheader("CIBIL Score Components")

comp_cols = st.columns(4)
# Keys + weights must match predict.cibil_score(): default-prob 50%, payment 25%,
# utilisation 15%, mix/duration 10%.
comp_names = [
    ("Default Probability (ML)", "default_probability_component", "50%"),
    ("Payment History", "payment_history", "25%"),
    ("Credit Utilization", "credit_utilization", "15%"),
    ("Credit Mix & Duration", "credit_mix_duration", "10%"),
]
for col, (label, key, weight) in zip(comp_cols, comp_names):
    val = components[key]
    col.metric(f"{label} ({weight})", f"{val:.0f}/100")

st.divider()

# ---------------------------------------------------------------------------
# SHAP Explanation — Waterfall-style bar chart
# ---------------------------------------------------------------------------
st.subheader("Feature Contributions (SHAP)")

contributions = shap_data["feature_contributions"]
df_shap = pd.DataFrame(contributions)
df_shap["abs_shap"] = df_shap["shap_value"].abs()
df_shap = df_shap.sort_values("abs_shap", ascending=True)

# Horizontal bar chart with color coding
import altair as alt

bars = (
    alt.Chart(df_shap)
    .mark_bar()
    .encode(
        x=alt.X("shap_value:Q", title="SHAP Value (impact on default risk)"),
        y=alt.Y("label:N", sort=None, title=""),
        color=alt.condition(
            alt.datum.shap_value > 0,
            alt.value("#e74c3c"),  # risk = red
            alt.value("#2ecc71"),  # protective = green
        ),
        tooltip=["label:N", "shap_value:Q", "direction:N"],
    )
    .properties(height=400)
)

st.altair_chart(bars, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Risk Drivers + Improvement Guidance
# ---------------------------------------------------------------------------
col_risk, col_protect = st.columns(2)

with col_risk:
    st.subheader("Risk Drivers")
    risk_drivers = shap_data["risk_drivers"]
    if risk_drivers:
        for item in risk_drivers:
            with st.expander(f"**{item['feature']}** — impact: +{item['impact']:.4f}"):
                if item.get("recommendation"):
                    st.info(item["recommendation"])
                else:
                    st.write("No specific recommendation available.")
    else:
        st.success("No significant risk drivers identified.")

with col_protect:
    st.subheader("Protective Factors")
    protective = shap_data["protective_factors"]
    if protective:
        for item in protective:
            st.markdown(f"- **{item['feature']}** — impact: {item['impact']:.4f}")
    else:
        st.warning("No significant protective factors identified.")

st.divider()

# ---------------------------------------------------------------------------
# Improvement Guidance Summary
# ---------------------------------------------------------------------------
st.subheader("Improvement Guidance")

if risk_drivers:
    st.markdown("Based on the model's analysis, here are the top actions to improve this borrower's credit profile:")
    for i, item in enumerate(risk_drivers[:5], 1):
        rec = item.get("recommendation", "Maintain current standing.")
        st.markdown(f"**{i}. {item['feature']}:** {rec}")
else:
    st.success("This profile has strong credit standing. Maintain current financial habits.")

# ---------------------------------------------------------------------------
# Raw input summary
# ---------------------------------------------------------------------------
with st.expander("Raw Borrower Inputs"):
    st.json(raw)
