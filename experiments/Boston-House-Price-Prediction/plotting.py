"""Visualization utilities for the Boston house price analysis workflow."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats.api as sms
from matplotlib.figure import Figure
from scipy.stats import pearsonr
from statsmodels.compat import lzip


def _slugify(text: str) -> str:
    """Convert arbitrary text to a filesystem-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def save_figure(fig: Figure, filename: str, plot_dir: Path) -> None:
    """Persist figure to disk under the result directory."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")


def run_univariate_analysis(df: pd.DataFrame, plot_dir: Path) -> None:
    """Plot variable distributions, including the log-transformed target."""
    sns.set_theme(style="whitegrid")
    for column in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        fig.tight_layout()
        save_figure(fig, f"distribution_{_slugify(column)}.png", plot_dir)
        plt.close(fig)

    df["MEDV_log"] = np.log(df["MEDV"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="MEDV_log", kde=True, ax=ax)
    ax.set_title("Log-transformed MEDV distribution")
    fig.tight_layout()
    save_figure(fig, "distribution_medv_log.png", plot_dir)
    plt.close(fig)


def run_bivariate_analysis(df: pd.DataFrame, plot_dir: Path) -> None:
    """Visualize pairwise relationships used in the notebook."""
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap=cmap, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    save_figure(fig, "correlation_heatmap.png", plot_dir)
    plt.close(fig)

    scatter_pairs = [
        ("AGE", "DIS"),
        ("RAD", "TAX"),
        ("INDUS", "TAX"),
        ("RM", "MEDV"),
        ("LSTAT", "MEDV"),
        ("INDUS", "NOX"),
        ("AGE", "NOX"),
        ("DIS", "NOX"),
    ]
    for x_col, y_col in scatter_pairs:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.scatterplot(x=x_col, y=y_col, data=df, ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout()
        filename = f"scatter_{_slugify(y_col)}_vs_{_slugify(x_col)}.png"
        save_figure(fig, filename, plot_dir)
        plt.close(fig)

    df_sub = df[df["TAX"] < 600]
    corr_value = pearsonr(df_sub["TAX"], df_sub["RAD"])[0]
    print(f"Correlation between TAX and RAD after filtering high tax values: {corr_value:.4f}")


def residual_analysis(model, X_train, y_train, plot_dir: Path):
    """Run the residual diagnostics replicated from the notebook."""
    residuals = model.resid
    print(f"\nMean of residuals: {residuals.mean():.6f}")

    test_names = ["F statistic", "p-value"]
    test_results = sms.het_goldfeldquandt(y_train, X_train)
    print("\nGoldfeld-Quandt test for heteroscedasticity:")
    print(dict(lzip(test_names, test_results)))

    fitted_vals = model.fittedvalues
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.residplot(x=fitted_vals, y=residuals, color="lightblue", lowess=True, ax=ax)
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual vs Fitted Plot")
    fig.tight_layout()
    save_figure(fig, "residual_vs_fitted.png", plot_dir)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Residual distribution")
    fig.tight_layout()
    save_figure(fig, "residual_distribution.png", plot_dir)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    stats.probplot(residuals, dist="norm", plot=plt)
    fig.suptitle("Residual Q-Q Plot")
    fig.tight_layout()
    save_figure(fig, "residual_qq_plot.png", plot_dir)
    plt.close(fig)
    return residuals


__all__ = [
    "run_univariate_analysis",
    "run_bivariate_analysis",
    "residual_analysis",
]
