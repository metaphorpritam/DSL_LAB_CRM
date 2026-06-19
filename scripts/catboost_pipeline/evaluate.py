"""
evaluate.py — threshold sweep, metrics, and plotting helpers.

Public API:
    get_thresholds(y_val, val_probs)         → thresh_f1, thresh_prec, thr, pv, rv
    print_evaluation(name, ...)
    plot_evaluation(name, ...)
    plot_feature_importance(model, pool, feature_cols, title)
    plot_optuna_history(study, baseline_auc)
    plot_model_comparison(models_info, y_test)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, precision_score, recall_score, classification_report,
)

from config import PREC_TARGET

plt.rcParams["figure.dpi"] = 120
sns.set_style("whitegrid")


# ── Threshold helpers ─────────────────────────────────────────────────────────

def get_thresholds(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    prec_target: float = PREC_TARGET,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep 499 thresholds on the validation set and return two operating points.

    Strategy A — Max-F1  : threshold that maximises F1 for the Default class.
    Strategy B — Prec≥T  : highest recall subject to Default-class precision >= prec_target.

    Returns:
        thresh_f1, thresh_prec, thr_array, precision_array, recall_array
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


# ── Evaluation helpers ────────────────────────────────────────────────────────

def print_evaluation(
    name: str,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    thresh_f1: float,
    thresh_prec: float,
    prec_target: float = PREC_TARGET,
) -> tuple[float, float]:
    """Print classification reports for both threshold strategies."""
    auc = roc_auc_score(y_test, test_probs)
    ap  = average_precision_score(y_test, test_probs)

    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  Avg Precision : {ap:.4f}")

    print(f"\n  ── Strategy A: Max-F1 (t={thresh_f1:.3f}) ──")
    print(
        classification_report(
            y_test,
            (test_probs >= thresh_f1).astype(int),
            target_names=["No Default", "Default"],
        )
    )
    print(f"  ── Strategy B: Prec≥{prec_target:.0%} (t={thresh_prec:.3f}) ──")
    print(
        classification_report(
            y_test,
            (test_probs >= thresh_prec).astype(int),
            target_names=["No Default", "Default"],
        )
    )
    return auc, ap


def plot_evaluation(
    name: str,
    y_test: np.ndarray,
    test_probs: np.ndarray,
    thresh_f1: float,
    thresh_prec: float,
    prec_target: float = PREC_TARGET,
    color: str = "steelblue",
    save_path: str | None = None,
) -> None:
    """4-panel plot: two confusion matrices, ROC curve, PR curve."""
    pf = (test_probs >= thresh_f1).astype(int)
    pp = (test_probs >= thresh_prec).astype(int)
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
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

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
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    pool,
    feature_cols: list[str],
    title: str = "CatBoost Feature Importance",
    save_path: str | None = None,
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
        ax.set_title(f"{imp_type}")
        ax.set_xlabel("Importance")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── Optuna visualisation ──────────────────────────────────────────────────────

def plot_optuna_history(
    study,
    baseline_auc: float,
    save_path: str | None = None,
) -> None:
    """Trial history + fANOVA parameter importances."""
    import optuna

    trial_nums  = [t.number for t in study.trials]
    trial_aucs  = [t.value  for t in study.trials]
    best_so_far = [max(trial_aucs[: i + 1]) for i in range(len(trial_aucs))]

    # fANOVA importances need at least two completed trials; skip gracefully
    # for single-trial smoke runs (e.g. --n-trials 1).
    importances = (
        optuna.importance.get_param_importances(study)
        if len(study.trials) > 1 else {}
    )

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

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()


# ── Cross-model comparison ────────────────────────────────────────────────────

def plot_model_comparison(
    models_info: list[tuple],   # (name, test_probs, thresh_f1, thresh_prec, color)
    y_test: np.ndarray,
    prec_target: float = PREC_TARGET,
    save_path: str | None = None,
) -> None:
    """Overlaid ROC, PR curves + grouped bar chart for all models."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, probs, tf1, tprec, color in models_info:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.4f})")

        pc, rc, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        axes[1].plot(rc, pc, color=color, lw=2, label=f"{name} (AP={ap:.4f})")
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
        handles=axes[1].lines
        + [
            Line2D([0], [0], marker="o",  color="gray", label="Max-F1 t",             markersize=7, linestyle=""),
            Line2D([0], [0], marker="D",  color="gray", label=f"Prec≥{prec_target:.0%} t", markersize=7, linestyle=""),
        ],
        fontsize=8,
    )
    axes[1].set_title("PR Curves"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")

    metric_labels = ["ROC-AUC", "Avg Precision"]
    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(models_info), 1)
    for i, (name, probs, _, _, color) in enumerate(models_info):
        vals = [roc_auc_score(y_test, probs), average_precision_score(y_test, probs)]
        bars = axes[2].bar(
            x + i * width, vals, width,
            label=name, color=color, edgecolor="black", linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            axes[2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8,
            )
    axes[2].set_xticks(x + width * (len(models_info) - 1) / 2)
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_title("Threshold-Independent Metrics")
    axes[2].set_ylim(0, 1.05)
    axes[2].legend()

    plt.suptitle("CatBoost Baseline vs Tuned — Test Set Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
