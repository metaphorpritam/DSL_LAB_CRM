"""
config.py — constants and hyperparameters for the CatBoost pipeline.
"""

from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Data ─────────────────────────────────────────────────────────────────────
# Resolved relative to this file (…/scripts/catboost_pipeline/) so the pipeline
# loads the data regardless of the current working directory.
DATA_PATH = str(Path(__file__).resolve().parents[2] / "Data" / "cs-training.csv")

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

FEATURE_COLS = [
    "unsecured_credit", "age", "delinq_30_59", "debt_ratio",
    "monthly_income", "open_credit", "delinq_90", "real_estate_loans",
    "delinq_60_89", "dependents", "monthly_income_missing", "dependents_missing",
    "delinq_sentinel",
]

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SIZE  = 0.15   # 70 / 15 / 15 stratified split
VAL_SIZE   = 0.15 / 0.85   # fraction of temp set

# ── Threshold sweep ───────────────────────────────────────────────────────────
PREC_TARGET = 0.50   # minimum Default-class precision for Strategy B

# ── CatBoost baseline ─────────────────────────────────────────────────────────
CATBOOST_BASELINE = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    eval_metric="AUC",
    early_stopping_rounds=30,
    random_seed=SEED,
    verbose=100,
)

# ── Optuna ────────────────────────────────────────────────────────────────────
N_TRIALS = 50

OPTUNA_PARAM_SPACE = {
    "learning_rate":      ("float_log",  0.01, 0.30),
    "depth":              ("int",        4,    10),
    "iterations":         ("int",        500,  3000),
    "l2_leaf_reg":        ("float",      1.0,  10.0),
    "bagging_temperature":("float",      0.0,  1.0),
    "random_strength":    ("float",      0.0,  10.0),
    "border_count":       ("int",        32,   255),
}

OPTUNA_FIXED = dict(
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=SEED,
    verbose=0,
)
