"""
run.py — end-to-end CatBoost pipeline orchestrator.

Runs the full pipeline in sequence:
    1. Data loading & preprocessing
    2. CatBoost baseline training & evaluation
    3. Optuna hyperparameter search
    4. Retraining tuned model & evaluation
    5. Baseline vs Tuned comparison plots

Usage:
    cd scripts/catboost_pipeline
    python run.py                           # default settings
    python run.py --n-trials 20            # quick smoke test
    python run.py --data /path/to/data.csv --n-trials 50
"""

from __future__ import annotations

import argparse
import time

from catboost import Pool

from config import FEATURE_COLS, N_TRIALS
from data import load_data, make_splits
from evaluate import (
    get_thresholds,
    print_evaluation,
    plot_evaluation,
    plot_feature_importance,
    plot_optuna_history,
    plot_model_comparison,
)
from train import train_baseline
from tune import run_tuning, retrain_tuned


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

    # ── 2. Baseline CatBoost ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 2 — CatBoost baseline")
    print("=" * 68)
    baseline, _, _ = train_baseline(X_train_s, y_train, X_val_s, y_val, class_ratio)

    baseline_val_probs  = baseline.predict_proba(X_val_s)[:, 1]
    baseline_test_probs = baseline.predict_proba(X_test_s)[:, 1]
    baseline_auc        = float(baseline.get_best_score()["validation"]["AUC"])

    print("\n── Threshold sweep (validation set) ──")
    cat_thresh_f1, cat_thresh_prec, *_ = get_thresholds(y_val, baseline_val_probs)

    print("\n── Test-set evaluation ──")
    print_evaluation("CatBoost Baseline", y_test, baseline_test_probs, cat_thresh_f1, cat_thresh_prec)
    plot_evaluation("CatBoost Baseline", y_test, baseline_test_probs, cat_thresh_f1, cat_thresh_prec, color="#1565C0")

    print("\n── Feature importance ──")
    plot_feature_importance(baseline, train_pool, FEATURE_COLS, title="CatBoost Baseline — Feature Importance")

    # ── 3. Optuna search ──────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  STEP 3 — Optuna hyperparameter search ({n_trials} trials)")
    print("=" * 68)
    study = run_tuning(train_pool, val_pool, class_ratio, n_trials=n_trials)

    print("\n── Visualising search history ──")
    plot_optuna_history(study, baseline_auc)

    # ── 4. Tuned model ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 4 — Retraining with best params")
    print("=" * 68)
    cat_tuned = retrain_tuned(study, train_pool, val_pool, class_ratio)

    tuned_val_probs  = cat_tuned.predict_proba(X_val_s)[:, 1]
    tuned_test_probs = cat_tuned.predict_proba(X_test_s)[:, 1]

    print("\n── Threshold sweep (validation set) ──")
    thresh_f1_tuned, thresh_prec_tuned, *_ = get_thresholds(y_val, tuned_val_probs)

    print("\n── Test-set evaluation ──")
    print_evaluation("CatBoost Tuned", y_test, tuned_test_probs, thresh_f1_tuned, thresh_prec_tuned)
    plot_evaluation("CatBoost Tuned", y_test, tuned_test_probs, thresh_f1_tuned, thresh_prec_tuned, color="darkcyan")

    print("\n── Feature importance ──")
    plot_feature_importance(cat_tuned, train_pool, FEATURE_COLS, title="CatBoost Tuned — Feature Importance")

    # ── 5. Comparison ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  STEP 5 — Baseline vs Tuned comparison")
    print("=" * 68)
    models_info = [
        ("CatBoost Baseline", baseline_test_probs, cat_thresh_f1,  cat_thresh_prec,  "#1565C0"),
        ("CatBoost Tuned",    tuned_test_probs,    thresh_f1_tuned, thresh_prec_tuned, "darkcyan"),
    ]
    plot_model_comparison(models_info, y_test)

    elapsed = time.time() - t0
    print(f"\nPipeline complete — total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end CatBoost pipeline.")
    parser.add_argument("--data",     default=None, help="Path to cs-training.csv")
    parser.add_argument("--n-trials", default=N_TRIALS, type=int, help="Optuna trial count (default: 50)")
    args = parser.parse_args()
    run_pipeline(data_path=args.data, n_trials=args.n_trials)
