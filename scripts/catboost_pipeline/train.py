"""
train.py — CatBoost baseline training.

Usage (standalone):
    cd scripts/catboost_pipeline
    python train.py [--data ../../Data/cs-training.csv]
"""

from __future__ import annotations

import argparse

import numpy as np
from catboost import CatBoostClassifier, Pool

from config import CATBOOST_BASELINE, FEATURE_COLS
from data import load_data, make_splits
from evaluate import get_thresholds, print_evaluation, plot_evaluation, plot_feature_importance


def build_baseline(class_ratio: float) -> CatBoostClassifier:
    """Return a freshly-instantiated baseline CatBoost model."""
    params = CATBOOST_BASELINE.copy()
    params["class_weights"] = [1, class_ratio]
    return CatBoostClassifier(**params)


def train_baseline(
    X_train_s: np.ndarray,
    y_train: np.ndarray,
    X_val_s: np.ndarray,
    y_val: np.ndarray,
    class_ratio: float,
) -> tuple[CatBoostClassifier, Pool, Pool]:
    """Train the baseline CatBoost model; return (model, train_pool, val_pool)."""
    train_pool = Pool(X_train_s, y_train, feature_names=FEATURE_COLS)
    val_pool   = Pool(X_val_s,   y_val,   feature_names=FEATURE_COLS)

    model = build_baseline(class_ratio)
    model.fit(train_pool, eval_set=val_pool)

    print(f"\nBest iteration : {model.get_best_iteration()}")
    print(f"Best val AUC   : {model.get_best_score()['validation']['AUC']:.4f}")

    return model, train_pool, val_pool


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(data_path: str | None = None) -> None:
    df = load_data(data_path)
    X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, _, class_ratio = make_splits(df)

    print("\n── CatBoost Baseline ──────────────────────────────────────────")
    model, train_pool, _ = train_baseline(X_train_s, y_train, X_val_s, y_val, class_ratio)

    val_probs  = model.predict_proba(X_val_s)[:, 1]
    test_probs = model.predict_proba(X_test_s)[:, 1]

    print("\n── Threshold sweep (validation set) ──")
    thresh_f1, thresh_prec, *_ = get_thresholds(y_val, val_probs)

    print("\n── Test-set evaluation ──")
    print_evaluation("CatBoost Baseline", y_test, test_probs, thresh_f1, thresh_prec)
    plot_evaluation("CatBoost Baseline", y_test, test_probs, thresh_f1, thresh_prec, color="#1565C0")

    print("\n── Feature importance ──")
    plot_feature_importance(model, train_pool, FEATURE_COLS, title="CatBoost Baseline — Feature Importance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CatBoost baseline model.")
    parser.add_argument("--data", default=None, help="Path to cs-training.csv")
    args = parser.parse_args()
    main(data_path=args.data)
