"""
Lag Distribution Visualizations

Plots temporal relationships (lags in days/weeks) discovered by causal methods,
helping understand climate system memory and response timescales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def plot_lag_histogram(
    results_df: pd.DataFrame,
    method: str = "Granger",
    lag_column: str = "delay",
    time_unit: str = "days",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> Optional[Path]:
    """
    Plot histogram of time lags for discovered causal relationships.

    Parameters:
        results_df (pd.DataFrame): Results with lag column
        method (str): Method name (for title)
        lag_column (str): Column name for lag values
        time_unit (str): "days", "weeks", or "months"
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if results_df is None or len(results_df) == 0:
        logger.warning(f"No data for lag plot ({method})")
        return None

    # Filter only significant relationships
    if "is_significant" in results_df.columns:
        significant_df = results_df[results_df["is_significant"].astype(bool)]
    else:
        significant_df = results_df.copy()

    if len(significant_df) == 0:
        logger.warning(f"No significant relationships for lag plot ({method})")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    lags = significant_df[lag_column].dropna().values

    # Determine bins
    if len(np.unique(lags)) < 20:
        bins = len(np.unique(lags))
    else:
        bins = 30

    # Histogram
    counts, edges, patches = ax.hist(
        lags, bins=bins, alpha=0.7, color="darkgreen", edgecolor="black"
    )

    # Color code bars by intensity
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
    for count, patch in zip(counts, patches):
        patch.set_facecolor(plt.cm.Greens(norm(count)))

    # Formatting
    ax.set_xlabel(f"Lag ({time_unit})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{method}: Lag Distribution (n={len(lags)} significant relationships)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Statistics
    stats_text = f"Mean lag: {lags.mean():.1f} {time_unit}\nMedian lag: {np.median(lags):.1f} {time_unit}\nRange: {lags.min()}-{lags.max()}"
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
            facecolor="lightgreen",
            alpha=0.85,
            edgecolor="darkgreen",
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


def plot_lag_distribution(
    results_dict: Dict[str, pd.DataFrame],
    lag_column: str = "delay",
    time_unit: str = "days",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
) -> Optional[Path]:
    """
    Compare lag distributions across multiple methods.

    Parameters:
        results_dict (Dict[str, pd.DataFrame]): Dict with method names and results
        lag_column (str): Column name for lag values
        time_unit (str): "days", "weeks", or "months"
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)

    if n_methods == 0:
        logger.warning("No methods provided for lag comparison")
        return None

    fig, axes = plt.subplots(1, n_methods, figsize=figsize, sharey=True)
    if n_methods == 1:
        axes = [axes]

    all_lags = []

    for ax, method in zip(axes, methods):
        df = results_dict[method]
        if df is None or len(df) == 0:
            continue

        # Filter significant - check if column exists
        has_is_significant = "is_significant" in df.columns
        if has_is_significant:
            sig_df = df[df["is_significant"]]
        else:
            sig_df = df  # Use all rows if is_significant column doesn't exist

        if len(sig_df) == 0:
            ax.text(
                0.5,
                0.5,
                "No significant\nrelationships",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"{method}\n(0 significant)", fontsize=12, fontweight="bold")
            continue

        # Check if lag column exists
        if lag_column not in sig_df.columns:
            ax.text(
                0.5,
                0.5,
                f"No '{lag_column}'\ncolumn available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"{method}\n(no lag data)", fontsize=12, fontweight="bold")
            continue

        lags = sig_df[lag_column].dropna().values
        all_lags.extend(lags)

        bins = max(10, len(np.unique(lags)))
        ax.hist(lags, bins=bins, alpha=0.7, color="darkgreen", edgecolor="black")

        ax.set_xlabel(f"Lag ({time_unit})", fontsize=11, fontweight="bold")
        ax.set_title(
            f"{method}\n(n={len(lags)} significant)", fontsize=12, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    fig.suptitle(
        f"Lag Distributions: Method Comparison ({time_unit})",
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
