"""
P-Value Distribution Visualizations

Plots statistical significance distributions across causal methods,
helping identify robust causal relationships and assess overall discovery power.
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def plot_pvalue_distribution(
    results_df: pd.DataFrame,
    method: str = "Granger",
    alpha: float = 0.05,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> Optional[Path]:
    """
    Plot histogram of p-values for a single method.

    Parameters:
        results_df (pd.DataFrame): Results with 'p_value' column
        method (str): Method name (for title)
        alpha (float): Significance threshold
        output_path (Optional[Path]): Save path (.pdf, .png, .svg)
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure or None
    """
    if results_df is None or len(results_df) == 0:
        logger.warning(f"No data for p-value plot ({method})")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    p_values = results_df["p_value"].dropna()

    # Histogram
    ax.hist(p_values, bins=30, alpha=0.7, color="steelblue", edgecolor="black")

    # Significance threshold
    ax.axvline(alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}")

    # Formatting
    ax.set_xlabel("P-value", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{method}: P-value Distribution (n={len(p_values)})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Statistics text
    n_sig = (p_values < alpha).sum()
    pct_sig = 100 * n_sig / len(p_values)
    stats_text = f"Significant: {n_sig}/{len(p_values)} ({pct_sig:.1f}%)\nMean p: {p_values.mean():.4f}"
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="wheat",
            alpha=0.85,
            edgecolor="orange",
            linewidth=1.5,
        ),
    )

    plt.tight_layout(pad=1.0)

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None


def plot_pvalue_comparison(
    results_dict: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
) -> Optional[Path]:
    """
    Compare p-value distributions across multiple methods.

    Parameters:
        results_dict (Dict[str, pd.DataFrame]): Dict with method names as keys and results as values
        alpha (float): Significance threshold
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    if n_methods == 0:
        logger.warning("No methods provided for comparison")
        return None

    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        df = results_dict[method]
        if df is None or len(df) == 0:
            continue

        # Check if p_value column exists
        if "p_value" not in df.columns:
            ax.text(
                0.5,
                0.5,
                "No 'p_value'\ncolumn available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"{method}\n(no p-value data)", fontsize=12, fontweight="bold")
            continue

        p_values = df["p_value"].dropna()

        ax.hist(p_values, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(
            alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}"
        )

        n_sig = (p_values < alpha).sum()
        pct_sig = 100 * n_sig / len(p_values)

        ax.set_xlabel("P-value", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{method}\n{n_sig}/{len(p_values)} significant ({pct_sig:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    fig.suptitle(
        "P-value Distributions: Method Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(pad=1.2)

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None
