import base64
from io import BytesIO
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FRAUD_COLOR = "#e74c3c"
LEGIT_COLOR = "#2ecc71"
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold")
    return _fig_to_b64(fig)


def plot_roc_curves(models_probs: Dict, y_true) -> str:
    """models_probs: {model_name: y_prob_array}"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, y_prob), color in zip(models_probs.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2, color=color)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def plot_pr_curves(models_probs: Dict, y_true) -> str:
    """models_probs: {model_name: y_prob_array}"""
    baseline = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=baseline, color="k", linestyle="--", linewidth=1,
               label=f"Baseline (fraud rate={baseline:.4f})")
    for (name, y_prob), color in zip(models_probs.items(), COLORS):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2, color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _fig_to_b64(fig)


def plot_metrics_comparison(results: Dict) -> str:
    """Bar chart comparing all models across Recall, F1, PR-AUC, ROC-AUC on val set."""
    models = list(results.keys())
    metric_keys = ["recall", "f1", "pr_auc", "roc_auc"]
    metric_labels = ["Recall", "F1 Score", "PR-AUC", "ROC-AUC"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [results[m]["val_metrics"][metric] for m in models]
        offset = (i - len(metric_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=COLORS[i])
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Validation Set", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models])
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X, y, model_name: str = "Model", threshold: float = 0.5) -> Dict:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "metrics": {
            "recall":            round(float(recall_score(y, y_pred)), 4),
            "precision":         round(float(precision_score(y, y_pred)), 4),
            "f1":                round(float(f1_score(y, y_pred)), 4),
            "pr_auc":            round(float(average_precision_score(y, y_prob)), 4),
            "roc_auc":           round(float(roc_auc_score(y, y_prob)), 4),
            "specificity":       round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
            "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "plots": {
            "confusion_matrix": plot_confusion_matrix(y, y_pred, model_name),
        },
    }
