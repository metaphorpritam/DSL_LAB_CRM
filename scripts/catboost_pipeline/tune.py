"""
tune.py — Optuna hyperparameter search for CatBoost.

Usage (standalone):
    cd scripts/catboost_pipeline
    python tune.py [--data ../../Data/cs-training.csv] [--n-trials 50]

Returns the study object and best CatBoostClassifier when imported.
"""

from __future__ import annotations

import argparse

import numpy as np
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier, Pool

from config import (
    SEED, FEATURE_COLS, N_TRIALS,
    OPTUNA_PARAM_SPACE, OPTUNA_FIXED,
)
from data import load_data, make_splits
from evaluate import get_thresholds, print_evaluation, plot_evaluation
from evaluate import plot_feature_importance, plot_optuna_history, plot_model_comparison
from train import train_baseline


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Objective ─────────────────────────────────────────────────────────────────

def _build_trial_params(trial: optuna.Trial, class_ratio: float) -> dict:
    """Translate OPTUNA_PARAM_SPACE entries into a CatBoost params dict."""
    params: dict = {}
    for name, spec in OPTUNA_PARAM_SPACE.items():
        kind = spec[0]
        if kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            raise ValueError(f"Unknown param kind: {kind}")
    params["class_weights"] = [1, class_ratio]
    params.update(OPTUNA_FIXED)
    return params


def make_objective(
    train_pool: Pool,
    val_pool: Pool,
    class_ratio: float,
):
    """Return an Optuna objective that is closed over the data pools."""
    def objective(trial: optuna.Trial) -> float:
        params = _build_trial_params(trial, class_ratio)
        model  = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)
        return model.best_score_["validation"]["AUC"]

    return objective


# ── Main tuning function ───────────────────────────────────────────────────────

def run_tuning(
    train_pool: Pool,
    val_pool: Pool,
    class_ratio: float,
    n_trials: int = N_TRIALS,
) -> optuna.Study:
    """Run the Optuna TPE search and return the completed study."""
    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        make_objective(train_pool, val_pool, class_ratio),
        n_trials=n_trials,
        show_progress_bar=True,
    )

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
    """Retrain CatBoost with the best params found by Optuna."""
    best_params = study.best_params.copy()
    best_params["class_weights"] = [1, class_ratio]
    best_params.update({
        "eval_metric":           "AUC",
        "early_stopping_rounds": 50,
        "random_seed":           SEED,
        "verbose":               100,
    })

    cat_tuned = CatBoostClassifier(**best_params)
    cat_tuned.fit(train_pool, eval_set=val_pool)

    print(f"\nTuned best iteration : {cat_tuned.get_best_iteration()}")
    print(f"Tuned best val AUC   : {cat_tuned.get_best_score()['validation']['AUC']:.4f}")

    return cat_tuned


# ── CLI entry point ────────────────────────────────────────────────────────────

def main(data_path: str | None = None, n_trials: int = N_TRIALS) -> None:
    df = load_data(data_path)
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, _, class_ratio = make_splits(df)

    train_pool = Pool(X_train_s, y_train, feature_names=FEATURE_COLS)
    val_pool   = Pool(X_val_s,   y_val,   feature_names=FEATURE_COLS)
    test_pool  = Pool(X_test_s,  feature_names=FEATURE_COLS)

    # ── Baseline (needed for comparison) ─────────────────────────────────────
    print("\n── CatBoost Baseline ──────────────────────────────────────────")
    baseline, _, _ = train_baseline(X_train_s, y_train, X_val_s, y_val, class_ratio)
    baseline_val_probs  = baseline.predict_proba(X_val_s)[:, 1]
    baseline_test_probs = baseline.predict_proba(X_test_s)[:, 1]
    baseline_auc = float(baseline.get_best_score()["validation"]["AUC"])

    print("\n── Threshold sweep (baseline) ──")
    cat_thresh_f1, cat_thresh_prec, *_ = get_thresholds(y_val, baseline_val_probs)
    print_evaluation("CatBoost Baseline", y_test, baseline_test_probs, cat_thresh_f1, cat_thresh_prec)

    # ── Optuna search ─────────────────────────────────────────────────────────
    print(f"\n── Optuna search ({n_trials} trials) ──────────────────────────────")
    study = run_tuning(train_pool, val_pool, class_ratio, n_trials=n_trials)

    print("\n── Visualising search history ──")
    plot_optuna_history(study, baseline_auc)

    # ── Retrain with best params ──────────────────────────────────────────────
    print("\n── Retraining with best params ────────────────────────────────")
    cat_tuned = retrain_tuned(study, train_pool, val_pool, class_ratio)
    tuned_val_probs  = cat_tuned.predict_proba(X_val_s)[:, 1]
    tuned_test_probs = cat_tuned.predict_proba(X_test_s)[:, 1]

    print("\n── Threshold sweep (tuned) ──")
    thresh_f1_tuned, thresh_prec_tuned, *_ = get_thresholds(y_val, tuned_val_probs)

    print("\n── Test-set evaluation (tuned) ──")
    print_evaluation("CatBoost Tuned", y_test, tuned_test_probs, thresh_f1_tuned, thresh_prec_tuned)
    plot_evaluation("CatBoost Tuned", y_test, tuned_test_probs, thresh_f1_tuned, thresh_prec_tuned, color="darkcyan")

    print("\n── Feature importance (tuned) ──")
    plot_feature_importance(cat_tuned, train_pool, FEATURE_COLS, title="CatBoost Tuned — Feature Importance")

    # ── Baseline vs Tuned comparison ─────────────────────────────────────────
    print("\n── Baseline vs Tuned comparison ────────────────────────────────")
    models_info = [
        ("CatBoost Baseline", baseline_test_probs, cat_thresh_f1,    cat_thresh_prec,    "#1565C0"),
        ("CatBoost Tuned",    tuned_test_probs,    thresh_f1_tuned,   thresh_prec_tuned,  "darkcyan"),
    ]
    plot_model_comparison(models_info, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuning for CatBoost.")
    parser.add_argument("--data",     default=None, help="Path to cs-training.csv")
    parser.add_argument("--n-trials", default=N_TRIALS, type=int, help="Number of Optuna trials")
    args = parser.parse_args()
    main(data_path=args.data, n_trials=args.n_trials)
