import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from .config import EXPECTED_COLUMNS, TARGET_COL

sns.set_theme(style="darkgrid", palette="muted")
FRAUD_COLOR = "#e74c3c"
LEGIT_COLOR = "#2ecc71"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ---------------------------------------------------------------------------
# Validation & cleaning
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    if df[TARGET_COL].nunique() != 2:
        return False, f"'{TARGET_COL}' must be binary (0/1)."
    return True, ""


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True), n_removed


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_class_distribution(df: pd.DataFrame) -> str:
    n_legit = int((df[TARGET_COL] == 0).sum())
    n_fraud = int((df[TARGET_COL] == 1).sum())
    total = len(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold")

    bars = ax1.bar(
        ["Legitimate", "Fraud"],
        [n_legit, n_fraud],
        color=[LEGIT_COLOR, FRAUD_COLOR],
        edgecolor="white",
        linewidth=2,
    )
    ax1.set_ylabel("Count")
    ax1.set_title("Transaction Counts")
    for bar, count in zip(bars, [n_legit, n_fraud]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count / total * 100:.3f}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax2.pie(
        [n_legit, n_fraud],
        labels=["Legitimate", "Fraud"],
        colors=[LEGIT_COLOR, FRAUD_COLOR],
        autopct="%1.3f%%",
        startangle=90,
        explode=(0, 0.12),
        shadow=True,
    )
    ax2.set_title("Class Proportion")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_amount_distribution(df: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Amount by Class", fontsize=15, fontweight="bold")

    for cls, label, color in [(0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
        subset = df[df[TARGET_COL] == cls]["Amount"]
        ax1.hist(subset, bins=60, alpha=0.7, label=label, color=color, density=True)
    ax1.set_xlabel("Amount ($)")
    ax1.set_ylabel("Density (log scale)")
    ax1.set_title("Amount Distribution")
    ax1.set_yscale("log")
    ax1.legend()

    df_box = df[[TARGET_COL, "Amount"]].copy()
    df_box[TARGET_COL] = df_box[TARGET_COL].map({0: "Legitimate", 1: "Fraud"})
    sns.boxplot(
        data=df_box,
        x=TARGET_COL,
        y="Amount",
        palette={"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR},
        ax=ax2,
        showfliers=False,
    )
    ax2.set_title("Amount Box Plot (no outliers)")
    ax2.set_xlabel("Class")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_time_distribution(df: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Transaction Time by Class", fontsize=15, fontweight="bold")

    for cls, label, color in [(0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
        ax1.hist(
            df[df[TARGET_COL] == cls]["Time"],
            bins=48,
            alpha=0.7,
            label=label,
            color=color,
            density=True,
        )
    ax1.set_xlabel("Time (seconds from first transaction)")
    ax1.set_ylabel("Density")
    ax1.set_title("Time Distribution")
    ax1.legend()

    df_copy = df.copy()
    df_copy["time_bin"] = pd.cut(df_copy["Time"], bins=24, labels=False)
    fraud_rate = df_copy.groupby("time_bin")[TARGET_COL].mean()
    ax2.plot(
        fraud_rate.index,
        fraud_rate.values,
        color=FRAUD_COLOR,
        marker="o",
        linewidth=2,
        markersize=4,
    )
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.3%}"))
    ax2.set_xlabel("Time Bin (24 equal intervals)")
    ax2.set_ylabel("Fraud Rate")
    ax2.set_title("Fraud Rate Over Time")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_correlation_heatmap(df: pd.DataFrame) -> str:
    v_cols = [f"V{i}" for i in range(1, 29)]
    corr = (
        df[v_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 12))
    colors = [FRAUD_COLOR if x > 0 else LEGIT_COLOR for x in corr.values]
    ax.barh(corr.index, corr.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="white", linewidth=1, linestyle="--")
    ax.set_xlabel("Pearson Correlation with Class (Fraud=1)")
    ax.set_title("Feature Correlation with Fraud", fontsize=14, fontweight="bold")

    plt.tight_layout()
    return _fig_to_b64(fig)


def _plot_top_features(df: pd.DataFrame, n: int = 6) -> str:
    v_cols = [f"V{i}" for i in range(1, 29)]
    corr = (
        df[v_cols + [TARGET_COL]]
        .corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
    )
    top_features = corr.head(n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Top {n} Features Most Correlated with Fraud",
        fontsize=14,
        fontweight="bold",
    )

    for feat, ax in zip(top_features, axes.flatten()):
        for cls, label, color in [(0, "Legitimate", LEGIT_COLOR), (1, "Fraud", FRAUD_COLOR)]:
            ax.hist(
                df[df[TARGET_COL] == cls][feat],
                bins=50,
                alpha=0.6,
                label=label,
                color=color,
                density=True,
            )
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylabel("Density")

    plt.tight_layout()
    return _fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def generate_eda_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_fraud = int((df[TARGET_COL] == 1).sum())
    n_legit = int((df[TARGET_COL] == 0).sum())

    def _stats(series: pd.Series) -> Dict:
        return {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "median": round(float(series.median()), 4),
            "max": round(float(series.max()), 4),
        }

    return {
        "summary": {
            "total_transactions": len(df),
            "n_legitimate": n_legit,
            "n_fraud": n_fraud,
            "fraud_rate": round(float(df[TARGET_COL].mean()), 6),
            "n_features": len(df.columns) - 1,
            "missing_values": int(df.isnull().sum().sum()),
            "amount_all": _stats(df["Amount"]),
            "amount_fraud": _stats(df[df[TARGET_COL] == 1]["Amount"]),
            "amount_legit": _stats(df[df[TARGET_COL] == 0]["Amount"]),
        },
        "plots": {
            "class_distribution": _plot_class_distribution(df),
            "amount_distribution": _plot_amount_distribution(df),
            "time_distribution": _plot_time_distribution(df),
            "correlation_heatmap": _plot_correlation_heatmap(df),
            "top_features": _plot_top_features(df),
        },
    }
