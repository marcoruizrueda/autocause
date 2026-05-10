"""
Method Comparison Visualizations

Plots multi-panel comparisons between causal discovery methods,
including agreement matrices and side-by-side performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def plot_method_comparison(
    results_dict: Dict[str, pd.DataFrame],
    metrics: list = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (16, 10),
) -> Optional[Path]:
    """
    Create multi-panel comparison of causal discovery methods.

    Parameters:
        results_dict (Dict[str, pd.DataFrame]): Dict with method names and results
        metrics (list): Metrics to compare (e.g., ["n_edges", "mean_pvalue", "n_significant"])
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if metrics is None:
        metrics = ["n_edges", "n_significant", "mean_pvalue"]

    methods = list(results_dict.keys())
    n_methods = len(methods)

    if n_methods == 0:
        logger.warning("No methods provided for comparison")
        return None

    # Compute metrics
    method_metrics = {}

    for method, df in results_dict.items():
        if df is None or len(df) == 0:
            method_metrics[method] = {m: 0 for m in metrics}
            continue

        # Check for is_significant column
        has_is_significant = "is_significant" in df.columns
        is_significant = (
            df["is_significant"] if has_is_significant else pd.Series([False] * len(df))
        )

        # Check for delay/lag column
        if "delay" in df.columns:
            lag_col = df["delay"]
        elif "lag" in df.columns:
            lag_col = df["lag"]
        else:
            lag_col = pd.Series([1] * len(df))

        m_dict = {
            "n_edges": len(df),
            "n_significant": is_significant.sum(),
            "mean_pvalue": df["p_value"].mean() if "p_value" in df.columns else 0,
            "median_pvalue": df["p_value"].median() if "p_value" in df.columns else 0,
            "n_positive": ((lag_col > 0) & is_significant).sum(),
        }
        method_metrics[method] = {m: m_dict.get(m, 0) for m in metrics}

    # Create comparison plots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [method_metrics[m].get(metric, 0) for m in methods]
        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))

        bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}" if isinstance(val, float) else f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Method", fontsize=11, fontweight="bold")
        ax.set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.suptitle("Method Comparison", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(pad=1.0)

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None


def plot_agreement_matrix(
    consensus_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> Optional[Path]:
    """
    Plot heatmap of method agreement for discovered edges.

    Parameters:
        consensus_df (pd.DataFrame): Consensus results with voting information
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if consensus_df is None or len(consensus_df) == 0:
        logger.warning("No consensus data for agreement matrix")
        return None

    # Extract method votes
    voting_data = []
    methods = ["Granger", "TransferEntropy", "PCMCI+"]

    for _, row in consensus_df.iterrows():
        edge_label = f"{row['source']} → {row['target']}"
        votes = {m: 0 for m in methods}

        # Parse method votes from string
        methods_str = row.get("methods_voting", "")
        for m in methods:
            if m in methods_str:
                votes[m] = 1

        voting_data.append({"edge": edge_label, **votes})

    if not voting_data:
        logger.warning("No voting data to plot")
        return None

    voting_df = pd.DataFrame(voting_data).set_index("edge")

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(voting_df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(np.arange(len(voting_df.columns)))
    ax.set_yticks(np.arange(len(voting_df)))
    ax.set_xticklabels(voting_df.columns, fontsize=11, fontweight="bold")
    ax.set_yticklabels(voting_df.index, fontsize=9)

    # Values in cells
    for i in range(len(voting_df)):
        for j in range(len(voting_df.columns)):
            ax.text(
                j,
                i,
                int(voting_df.values[i, j]),
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                fontsize=10,
            )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label="Method Agreement")
    cbar.set_label("Agreement", fontsize=11, fontweight="bold")

    ax.set_title(
        "Method Agreement Matrix (Consensus Edges)", fontsize=13, fontweight="bold"
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
