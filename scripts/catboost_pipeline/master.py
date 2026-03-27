"""
master.py — self-contained CatBoost pipeline (baseline + Optuna tuning).

Combines: config, data, evaluate, train, tune, run into a single script.

Usage:
    cd scripts/catboost_pipeline
    python master.py                        # 50 Optuna trials (default)
    python master.py --n-trials 20          # quick smoke test
    python master.py --data /path/to/data.csv --n-trials 50
"""

from __future__ import annotations

import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score, classification_report,
)
from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42

# Path relative to this file; override with --data CLI flag
_HERE     = Path(__file__).parent
DATA_PATH = _HERE / "../../Data/cs-training.csv"

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

DELINQ_COLS = ["delinq_30_59", "delinq_60_89", "delinq_90"]

TEST_SIZE   = 0.15
VAL_SIZE    = 0.15 / 0.85   # fraction of temp → gives 70/15/15 overall
PREC_TARGET = 0.50           # minimum Default-class precision for Strategy B

# CatBoost baseline hyperparameters
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

# Optuna search space: name → (kind, low, high)
N_TRIALS = 50
OPTUNA_PARAM_SPACE = {
    "learning_rate":       ("float_log", 0.01, 0.30),
    "depth":               ("int",       4,    10),
    "iterations":          ("int",       500,  3000),
    "l2_leaf_reg":         ("float",     1.0,  10.0),
    "bagging_temperature": ("float",     0.0,  1.0),
    "random_strength":     ("float",     0.0,  10.0),
    "border_count":        ("int",       32,   255),
}
OPTUNA_FIXED = dict(
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=SEED,
    verbose=0,
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Read CSV, rename columns, apply the shared cleaning pipeline."""
    path = Path(data_path) if data_path else DATA_PATH
    raw  = pd.read_csv(path, index_col=0).rename(columns=RENAME)

    df = raw.copy()
    df = df[df["age"] > 0]

    # Sentinel treatment: flag 96/98 codes; cap all delinq cols at 10
    df["delinq_sentinel"] = (df["delinq_90"] >= 96).astype(int)
    for col in DELINQ_COLS:
        df[col] = df[col].clip(upper=10)

    # Missing indicators (derived from raw before imputation)
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
    """Stratified 70/15/15 split with StandardScaler.

    Returns:
        X_train_s, X_val_s, X_test_s  — scaled float32 arrays
        y_train, y_val, y_test         — float32 label arrays
        scaler                         — fitted StandardScaler
        class_ratio                    — n_neg / n_pos
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


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def get_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    prec_target: float = PREC_TARGET,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep 499 thresholds; return two operating points.

    Strategy A — Max-F1  : maximises Default-class F1.
    Strategy B — Prec≥T  : highest recall s.t. precision >= prec_target.
    """
    thr = np.linspace(0.01, 0.99, 499)
    pv  = np.array([precision_score(y_val, val_probs >= t, zero_division=0) for t in thr])
    rv  = np.array([recall_score(y_val,    val_probs >= t, zero_division=0) for t in thr])
    fv  = np.array([f1_score(y_val,        val_probs >= t, zero_division=0) for t in thr])

    idx_f1    = int(np.argmax(fv))
    thresh_f1 = thr[idx_f1]

    feasible = pv >= prec_target
    if feasible.any():
        idx_prec    = feasible.nonzero()[0][int(np.argmax(rv[feasible]))]
        thresh_prec = thr[idx_prec]
    else:
        idx_prec    = int(np.argmax(pv))
        thresh_prec = thr[idx_prec]
        print(
            f"  Warning: precision target {prec_target:.0%} never achieved on val; "
            f"using max-precision threshold ({thresh_prec:.3f})."
        )

    print(
        f"  Strategy A — Max-F1    : t={thresh_f1:.3f}  "
        f"prec={pv[idx_f1]:.3f}  rec={rv[idx_f1]:.3f}  F1={fv[idx_f1]:.3f}"
    )
    print(
        f"  Strategy B — Prec≥{prec_target:.0%} : t={thresh_prec:.3f}  "
        f"prec={pv[idx_prec]:.3f}  rec={rv[idx_prec]:.3f}"
    )
    return thresh_f1, thresh_prec, thr, pv, rv


def print_evaluation(
    name: str,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    thresh_f1: float,
    thresh_prec: float,
    prec_target: float = PREC_TARGET,
) -> tuple[float, float]:
    auc = roc_auc_score(y_test, test_probs)
    ap  = average_precision_score(y_test, test_probs)

    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Avg Precision : {ap:.4f}")
    print(f"\n  ── Strategy A: Max-F1 (t={thresh_f1:.3f}) ──")
    print(classification_report(
        y_test, (test_probs >= thresh_f1).astype(int),
        target_names=["No Default", "Default"],
    ))
    print(f"  ── Strategy B: Prec≥{prec_target:.0%} (t={thresh_prec:.3f}) ──")
    print(classification_report(
        y_test, (test_probs >= thresh_prec).astype(int),
        target_names=["No Default", "Default"],
    ))
    return auc, ap


def plot_evaluation(
    name: str,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    thresh_f1: float,
    thresh_prec: float,
    prec_target: float = PREC_TARGET,
    color: str = "steelblue",
) -> None:
    """4-panel plot: two confusion matrices, ROC curve, PR curve."""
    pf  = (test_probs >= thresh_f1).astype(int)
    pp  = (test_probs >= thresh_prec).astype(int)
    auc = roc_auc_score(y_test, test_probs)
    ap  = average_precision_score(y_test, test_probs)

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))

    for ax, preds, label in [
        (axes[0], pf, f"Max-F1\n(t={thresh_f1:.3f})"),
        (axes[1], pp, f"Prec≥{prec_target:.0%}\n(t={thresh_prec:.3f})"),
    ]:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax, cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            linewidths=0.5,
        )
        ax.set_title(f"{name}\nConfusion Matrix — {label}")
        ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    axes[2].plot(fpr, tpr, color=color, lw=2, label=f"AUC = {auc:.4f}")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1)
    axes[2].set_title(f"{name} — ROC Curve")
    axes[2].set_xlabel("FPR"); axes[2].set_ylabel("TPR"); axes[2].legend()

    pc, rc, _ = precision_recall_curve(y_test, test_probs)
    axes[3].plot(rc, pc, color=color, lw=2, label=f"AP = {ap:.4f}")
    axes[3].axhline(y_test.mean(), color="gray", linestyle="--", label="Baseline (prior)")
    axes[3].scatter(
        [recall_score(y_test, pf)], [precision_score(y_test, pf)],
        color="tomato", s=90, zorder=5, label=f"Max-F1 t={thresh_f1:.3f}",
    )
    axes[3].scatter(
        [recall_score(y_test, pp)], [precision_score(y_test, pp)],
        color="green", s=90, marker="D", zorder=5,
        label=f"Prec≥{prec_target:.0%} t={thresh_prec:.3f}",
    )
    axes[3].set_title(f"{name} — PR Curve")
    axes[3].set_xlabel("Recall"); axes[3].set_ylabel("Precision")
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model: CatBoostClassifier,
    pool: Pool,
    feature_cols: list[str],
    title: str = "CatBoost Feature Importance",
) -> None:
    """Side-by-side PredictionValuesChange and LossFunctionChange bars."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, imp_type in zip(axes, ["PredictionValuesChange", "LossFunctionChange"]):
        imp   = model.get_feature_importance(pool, type=imp_type)
        order = np.argsort(imp)
        ax.barh(
            np.array(feature_cols)[order], imp[order],
            color="#1565C0", edgecolor="black", linewidth=0.5,
        )
        ax.set_title(imp_type); ax.set_xlabel("Importance")
    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


def plot_optuna_history(study: optuna.Study, baseline_auc: float) -> None:
    """Trial history + fANOVA parameter importances."""
    trial_nums  = [t.number for t in study.trials]
    trial_aucs  = [t.value  for t in study.trials]
    best_so_far = [max(trial_aucs[: i + 1]) for i in range(len(trial_aucs))]
    importances = optuna.importance.get_param_importances(study)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(trial_nums, trial_aucs, s=20, alpha=0.5, color="steelblue", label="Trial AUC")
    axes[0].plot(trial_nums, best_so_far, color="crimson", lw=2, label="Best so far")
    axes[0].axhline(
        baseline_auc, color="orange", lw=1.5, ls="--",
        label=f"Baseline AUC {baseline_auc:.4f}",
    )
    axes[0].set_xlabel("Trial"); axes[0].set_ylabel("Val ROC-AUC")
    axes[0].set_title("Optuna Trial History"); axes[0].legend()

    params_list = list(importances.keys())
    imp_vals    = list(importances.values())
    axes[1].barh(params_list[::-1], imp_vals[::-1], color="teal")
    axes[1].set_xlabel("Importance")
    axes[1].set_title("Hyperparameter Importances (fANOVA)")

    plt.tight_layout(); plt.show()


def plot_model_comparison(
    models_info: list[tuple],
    y_test: np.ndarray,
    prec_target: float = PREC_TARGET,
) -> None:
    """Overlaid ROC + PR curves and grouped AUC/AP bar chart."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, probs, tf1, tprec, color in models_info:
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} (AUC={roc_auc_score(y_test, probs):.4f})")

        pc, rc, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rc, pc, color=color, lw=2,
                     label=f"{name} (AP={average_precision_score(y_test, probs):.4f})")
        axes[1].scatter(
            [recall_score(y_test, (probs >= tf1).astype(int))],
            [precision_score(y_test, (probs >= tf1).astype(int))],
            color=color, s=80, zorder=5,
        )
        axes[1].scatter(
            [recall_score(y_test, (probs >= tprec).astype(int))],
            [precision_score(y_test, (probs >= tprec).astype(int))],
            color=color, s=80, marker="D", zorder=5,
        )

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_title("ROC Curves"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].axhline(y_test.mean(), color="gray", linestyle="--", label="Baseline")
    axes[1].legend(
        handles=axes[1].lines + [
            Line2D([0], [0], marker="o", color="gray", label="Max-F1 t",
                   markersize=7, linestyle=""),
            Line2D([0], [0], marker="D", color="gray",
                   label=f"Prec≥{prec_target:.0%} t", markersize=7, linestyle=""),
        ],
        fontsize=8,
    )
    axes[1].set_title("PR Curves"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")

    metric_labels = ["ROC-AUC", "Avg Precision"]
    x     = np.arange(len(metric_labels))
    width = 0.8 / max(len(models_info), 1)
    for i, (name, probs, _, _, color) in enumerate(models_info):
        vals = [roc_auc_score(y_test, probs), average_precision_score(y_test, probs)]
        bars = axes[2].bar(x + i * width, vals, width,
                           label=name, color=color, edgecolor="black", linewidth=0.6)
        for bar, v in zip(bars, vals):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8,
            )
    axes[2].set_xticks(x + width * (len(models_info) - 1) / 2)
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_title("Threshold-Independent Metrics")
    axes[2].set_ylim(0, 1.05); axes[2].legend()

    plt.suptitle("CatBoost Baseline vs Tuned — Test Set Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN  (baseline)
# ══════════════════════════════════════════════════════════════════════════════

def train_baseline(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    class_ratio: float,
) -> tuple[CatBoostClassifier, Pool, Pool]:
    """Train the baseline CatBoost model."""
    train_pool = Pool(X_train_s, y_train, feature_names=FEATURE_COLS)
    val_pool   = Pool(X_val_s,   y_val,   feature_names=FEATURE_COLS)

    params = CATBOOST_BASELINE.copy()
    params["class_weights"] = [1, class_ratio]

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    print(f"\nBest iteration : {model.get_best_iteration()}")
    print(f"Best val AUC   : {model.get_best_score()['validation']['AUC']:.4f}")

    return model, train_pool, val_pool


# ══════════════════════════════════════════════════════════════════════════════
# TUNE  (Optuna)
# ══════════════════════════════════════════════════════════════════════════════

def _build_trial_params(trial: optuna.Trial, class_ratio: float) -> dict:
    params: dict = {}
    for name, spec in OPTUNA_PARAM_SPACE.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
    params["class_weights"] = [1, class_ratio]
    params.update(OPTUNA_FIXED)
    return params


def run_tuning(
    train_pool: Pool,
    val_pool: Pool,
    class_ratio: float,
    n_trials: int = N_TRIALS,
) -> optuna.Study:
    """Run the Optuna TPE search and return the completed study."""
    def objective(trial: optuna.Trial) -> float:
        model = CatBoostClassifier(**_build_trial_params(trial, class_ratio))
        model.fit(train_pool, eval_set=val_pool)
        return model.best_score_["validation"]["AUC"]

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest val AUC : {study.best_value:.5f}")
    print("Best params  :")
    for k, v in study.best_params.items():
        print(f"  {k:<25} {v}")

    return study


def retrain_tuned(
    study: optuna.Study,
    train_pool: Pool,
    val_pool: Pool,
    class_ratio: float,
) -> CatBoostClassifier:
    """Retrain CatBoost with the best Optuna params (early stopping active)."""
    params = study.best_params.copy()
    params["class_weights"] = [1, class_ratio]
    params.update({
        "eval_metric":           "AUC",
        "early_stopping_rounds": 50,
        "random_seed":           SEED,
        "verbose":               100,
    })

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)

    print(f"\nTuned best iteration : {model.get_best_iteration()}")
    print(f"Tuned best val AUC   : {model.get_best_score()['validation']['AUC']:.4f}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE  (orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(data_path: str | None = None, n_trials: int = N_TRIALS) -> None:
    t0 = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 1 — Data loading & preprocessing")
    print("=" * 68)
    df = load_data(data_path)
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, _, class_ratio = make_splits(df)

    train_pool = Pool(X_train_s, y_train, feature_names=FEATURE_COLS)
    val_pool   = Pool(X_val_s,   y_val,   feature_names=FEATURE_COLS)

    # ── 2. Baseline ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 2 — CatBoost baseline")
    print("=" * 68)
    baseline, _, _ = train_baseline(X_train_s, y_train, X_val_s, y_val, class_ratio)

    baseline_val_probs  = baseline.predict_proba(X_val_s)[:, 1]
    baseline_test_probs = baseline.predict_proba(X_test_s)[:, 1]
    baseline_auc        = float(baseline.get_best_score()["validation"]["AUC"])

    print("\n── Threshold sweep ──")
    cat_tf1, cat_tprec, *_ = get_thresholds(y_val, baseline_val_probs)
    print_evaluation("CatBoost Baseline", y_test, baseline_test_probs, cat_tf1, cat_tprec)
    plot_evaluation("CatBoost Baseline", y_test, baseline_test_probs, cat_tf1, cat_tprec,
                    color="#1565C0")
    plot_feature_importance(baseline, train_pool, FEATURE_COLS,
                            title="CatBoost Baseline — Feature Importance")

    # ── 3. Optuna search ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  STEP 3 — Optuna hyperparameter search ({n_trials} trials)")
    print("=" * 68)
    study = run_tuning(train_pool, val_pool, class_ratio, n_trials=n_trials)
    plot_optuna_history(study, baseline_auc)

    # ── 4. Tuned model ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 4 — Retrain with best params")
    print("=" * 68)
    cat_tuned = retrain_tuned(study, train_pool, val_pool, class_ratio)

    tuned_val_probs  = cat_tuned.predict_proba(X_val_s)[:, 1]
    tuned_test_probs = cat_tuned.predict_proba(X_test_s)[:, 1]

    print("\n── Threshold sweep ──")
    tuned_tf1, tuned_tprec, *_ = get_thresholds(y_val, tuned_val_probs)
    print_evaluation("CatBoost Tuned", y_test, tuned_test_probs, tuned_tf1, tuned_tprec)
    plot_evaluation("CatBoost Tuned", y_test, tuned_test_probs, tuned_tf1, tuned_tprec,
                    color="darkcyan")
    plot_feature_importance(cat_tuned, train_pool, FEATURE_COLS,
                            title="CatBoost Tuned — Feature Importance")

    # ── 5. Comparison ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 5 — Baseline vs Tuned comparison")
    print("=" * 68)
    plot_model_comparison(
        [
            ("CatBoost Baseline", baseline_test_probs, cat_tf1,    cat_tprec,    "#1565C0"),
            ("CatBoost Tuned",    tuned_test_probs,    tuned_tf1,  tuned_tprec,  "darkcyan"),
        ],
        y_test,
    )

    elapsed = time.time() - t0
    print(f"\nPipeline complete — total time: {elapsed / 60:.1f} min")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-contained CatBoost pipeline.")
    parser.add_argument("--data",     default=None,    help="Path to cs-training.csv")
    parser.add_argument("--n-trials", default=N_TRIALS, type=int,
                        help=f"Optuna trial count (default: {N_TRIALS})")
    args = parser.parse_args()
    run_pipeline(data_path=args.data, n_trials=args.n_trials)
