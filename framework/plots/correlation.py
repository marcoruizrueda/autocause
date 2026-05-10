"""
Correlation Analysis Plotting Module

Provides comprehensive visualizations for correlation analysis results:
- Correlation matrices (heatmaps)
- Scatter plots with regression lines
- Comparison plots across correlation methods
- Confidence interval plots
- Correlation networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def plot_correlation_matrix(
    df_corr: pd.DataFrame,
    method: str = "pearson",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    annotate: bool = True,
    mask_insignificant: bool = False,
    alpha: float = 0.05,
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.

    Parameters:
        df_corr: DataFrame with correlation results from batch_correlation_analysis
        method: Correlation method to plot ("pearson", "spearman", "kendall")
        output_path: Optional path to save figure (SVG format)
        figsize: Figure size (width, height)
        cmap: Colormap name
        vmin: Minimum correlation value for color scale
        vmax: Maximum correlation value for color scale
        annotate: Show correlation values on heatmap
        mask_insignificant: Gray out non-significant correlations
        alpha: Significance level for masking

    Returns:
        Matplotlib figure object
    """
    logger.info(f"Creating {method} correlation matrix...")

    # Extract correlation values and pivot to matrix form
    corr_col = (
        f"{method}_r"
        if method == "pearson"
        else f"{method}_rho"
        if method == "spearman"
        else f"{method}_tau"
    )
    p_col = f"{method}_p"

    if corr_col not in df_corr.columns:
        raise ValueError(f"Column {corr_col} not found in dataframe")

    # Create matrix
    variables = sorted(set(df_corr["var1"].tolist() + df_corr["var2"].tolist()))
    matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)

    for _, row in df_corr.iterrows():
        var1, var2 = row["var1"], row["var2"]
        corr = row[corr_col]
        matrix.loc[var1, var2] = corr
        matrix.loc[var2, var1] = corr  # Symmetric

    # Create significance mask
    if mask_insignificant and p_col in df_corr.columns:
        sig_matrix = pd.DataFrame(
            np.ones((len(variables), len(variables))),
            index=variables,
            columns=variables,
        )
        for _, row in df_corr.iterrows():
            var1, var2 = row["var1"], row["var2"]
            is_sig = row[p_col] < alpha
            sig_matrix.loc[var1, var2] = is_sig
            sig_matrix.loc[var2, var1] = is_sig
        mask = sig_matrix == 0
    else:
        mask = None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Gray out insignificant cells
    if mask is not None:
        for i in range(len(variables)):
            for j in range(len(variables)):
                if mask.iloc[i, j]:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            fill=True,
                            facecolor="lightgray",
                            edgecolor="none",
                            alpha=0.5,
                        )
                    )

    # Annotate cells
    if annotate:
        for i in range(len(variables)):
            for j in range(len(variables)):
                value = matrix.iloc[i, j]
                if not np.isnan(value):
                    text_color = "white" if abs(value) > 0.5 else "black"
                    if mask is not None and mask.iloc[i, j]:
                        text_color = "darkgray"
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{method.capitalize()} Correlation", fontsize=12)

    # Axis labels
    ax.set_xticks(range(len(variables)))
    ax.set_yticks(range(len(variables)))
    ax.set_xticklabels(variables, rotation=45, ha="right")
    ax.set_yticklabels(variables)

    # Title
    title = f"{method.capitalize()} Correlation Matrix"
    if mask_insignificant:
        title += f" (α={alpha}, non-significant grayed out)"
    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout(pad=1.0)

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as SVG (high resolution)
        fig.savefig(
            output_path.with_suffix(".svg"), format="svg", dpi=300, bbox_inches="tight"
        )
        logger.info(f"✅ Saved: {output_path.with_suffix('.svg')}")

    return fig


def plot_correlation_comparison(
    df_corr: pd.DataFrame,
    var1: str,
    var2: str,
    df_data: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> plt.Figure:
    """
    Plot comparison of all correlation methods for a single variable pair.

    Shows scatter plot with regression lines for different methods and
    a bar chart comparing correlation coefficients.

    Parameters:
        df_corr: DataFrame with correlation results
        var1: First variable name
        var2: Second variable name
        df_data: Optional DataFrame with raw data for scatter plot
        output_path: Optional path to save figure (SVG format)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    logger.info(f"Creating correlation comparison for {var1} ↔ {var2}...")

    # Find the pair in results
    pair_results = df_corr[
        ((df_corr["var1"] == var1) & (df_corr["var2"] == var2))
        | ((df_corr["var1"] == var2) & (df_corr["var2"] == var1))
    ]

    if len(pair_results) == 0:
        raise ValueError(f"Pair {var1} ↔ {var2} not found in correlation results")

    pair_results = pair_results.iloc[0]

    # Create figure
    if df_data is not None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Scatter plot
        ax_scatter = axes[0]
        data = df_data[[var1, var2]].dropna()
        x = data[var1].values
        y = data[var2].values

        ax_scatter.scatter(x, y, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)

        # Add linear regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax_scatter.plot(
            x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit"
        )

        ax_scatter.set_xlabel(var1, fontsize=11)
        ax_scatter.set_ylabel(var2, fontsize=11)
        ax_scatter.set_title("Scatter Plot", fontsize=12)
        ax_scatter.legend()
        ax_scatter.grid(True, alpha=0.3)

        # Correlation coefficients bar chart
        ax_bar = axes[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_bar = axes[0]

    # Extract correlation values
    methods = []
    correlations = []
    p_values = []

    if "pearson_r" in pair_results:
        methods.append("Pearson")
        correlations.append(pair_results["pearson_r"])
        p_values.append(pair_results.get("pearson_p", np.nan))

    if "spearman_rho" in pair_results:
        methods.append("Spearman")
        correlations.append(pair_results["spearman_rho"])
        p_values.append(pair_results.get("spearman_p", np.nan))

    if "kendall_tau" in pair_results:
        methods.append("Kendall")
        correlations.append(pair_results["kendall_tau"])
        p_values.append(pair_results.get("kendall_p", np.nan))

    if "dcor" in pair_results:
        methods.append("dCor")
        correlations.append(pair_results["dcor"])
        p_values.append(np.nan)  # dCor doesn't have p-value in simplified version

    # Bar colors based on significance
    colors = []
    for p in p_values:
        if np.isnan(p):
            colors.append("gray")
        elif p < 0.001:
            colors.append("darkgreen")
        elif p < 0.01:
            colors.append("green")
        elif p < 0.05:
            colors.append("lightgreen")
        else:
            colors.append("lightcoral")

    bars = ax_bar.bar(
        methods, correlations, color=colors, edgecolor="black", linewidth=1.5
    )
    ax_bar.axhline(y=0, color="k", linestyle="-", linewidth=0.8)
    ax_bar.set_ylabel("Correlation Coefficient", fontsize=11)
    ax_bar.set_title("Correlation Methods Comparison", fontsize=12)
    ax_bar.set_ylim([-1.1, 1.1])
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Add p-value annotations
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        if not np.isnan(p):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.05 * np.sign(height),
                f"p={p:.3f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=8,
            )

    # P-values bar chart
    ax_idx = 2 if df_data is not None else 1
    ax_p = axes[ax_idx]

    valid_methods = [m for m, p in zip(methods, p_values) if not np.isnan(p)]
    valid_p_values = [p for p in p_values if not np.isnan(p)]

    if valid_p_values:
        # Plot -log10(p) for better visualization
        log_p = [-np.log10(p) if p > 0 else 10 for p in valid_p_values]
        ax_p.bar(
            valid_methods, log_p, color="steelblue", edgecolor="black", linewidth=1.5
        )

        # Significance thresholds
        ax_p.axhline(
            y=-np.log10(0.05),
            color="orange",
            linestyle="--",
            linewidth=2,
            label="α=0.05",
        )
        ax_p.axhline(
            y=-np.log10(0.01), color="red", linestyle="--", linewidth=2, label="α=0.01"
        )

        ax_p.set_ylabel("-log10(p-value)", fontsize=11)
        ax_p.set_title("Statistical Significance", fontsize=12)
        ax_p.legend(fontsize=9)
        ax_p.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Correlation Analysis: {var1} ↔ {var2}", fontsize=14, y=0.995)
    plt.tight_layout(pad=1.0)

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path.with_suffix(".svg"), format="svg", dpi=300, bbox_inches="tight"
        )
        logger.info(f"✅ Saved: {output_path.with_suffix('.svg')}")

    return fig


def plot_correlation_network(
    df_corr: pd.DataFrame,
    method: str = "pearson",
    threshold: float = 0.3,
    alpha: float = 0.05,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot correlation network graph showing significant relationships.

    Nodes are variables, edges are correlations above threshold.
    Edge width/color represents correlation strength.

    Parameters:
        df_corr: DataFrame with correlation results
        method: Correlation method ("pearson", "spearman", "kendall")
        threshold: Minimum absolute correlation to show edge
        alpha: Significance level for filtering edges
        output_path: Optional path to save figure (SVG format)
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error(
            "networkx is required for network plots. Install with: pip install networkx"
        )
        return None

    logger.info(f"Creating correlation network ({method}, threshold={threshold})...")

    # Filter significant correlations above threshold
    corr_col = (
        f"{method}_r"
        if method == "pearson"
        else f"{method}_rho"
        if method == "spearman"
        else f"{method}_tau"
    )
    p_col = f"{method}_p"

    df_filtered = df_corr.copy()
    if p_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[p_col] < alpha]
    df_filtered = df_filtered[df_filtered[corr_col].abs() >= threshold]

    if len(df_filtered) == 0:
        logger.warning(f"No significant correlations above threshold {threshold}")
        return None

    # Build network
    G = nx.Graph()

    for _, row in df_filtered.iterrows():
        var1, var2 = row["var1"], row["var2"]
        corr = row[corr_col]
        G.add_edge(var1, var2, weight=abs(corr), correlation=corr)

    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=1500,
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )

    # Draw edges with width/color based on correlation
    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    correlations = [G[u][v]["correlation"] for u, v in edges]

    # Color edges: positive = blue, negative = red
    edge_colors = ["blue" if c > 0 else "red" for c in correlations]
    edge_widths = [w * 5 for w in weights]  # Scale for visibility

    nx.draw_networkx_edges(
        G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.6, ax=ax
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Add edge labels (correlation values)
    edge_labels = {(u, v): f"{G[u][v]['correlation']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    # Legend
    pos_patch = mpatches.Patch(color="blue", label="Positive correlation")
    neg_patch = mpatches.Patch(color="red", label="Negative correlation")
    ax.legend(handles=[pos_patch, neg_patch], loc="upper left", fontsize=10)

    # Title
    ax.set_title(
        f"{method.capitalize()} Correlation Network\n(|r| ≥ {threshold}, α={alpha})",
        fontsize=14,
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout(pad=1.0)

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path.with_suffix(".svg"), format="svg", dpi=300, bbox_inches="tight"
        )
        logger.info(f"✅ Saved: {output_path.with_suffix('.svg')}")

    return fig


def plot_all_correlation_visualizations(
    df_corr: pd.DataFrame,
    df_data: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
    methods: List[str] = ["pearson", "spearman", "kendall"],
    threshold: float = 0.3,
    alpha: float = 0.05,
) -> Dict[str, plt.Figure]:
    """
    Create all correlation visualizations and save to output directory.

    Parameters:
        df_corr: DataFrame with correlation results
        df_data: Optional DataFrame with raw data
        output_dir: Directory to save figures
        methods: List of methods to visualize
        threshold: Threshold for network plot
        alpha: Significance level

    Returns:
        Dict of figure names to figure objects
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Correlation matrices for each method
    for method in methods:
        try:
            fig = plot_correlation_matrix(
                df_corr,
                method=method,
                output_path=output_dir / f"correlation_matrix_{method}.svg"
                if output_dir
                else None,
                mask_insignificant=True,
                alpha=alpha,
            )
            figures[f"matrix_{method}"] = fig
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to create {method} matrix: {e}")

    # Correlation network
    try:
        fig = plot_correlation_network(
            df_corr,
            method="pearson",
            threshold=threshold,
            alpha=alpha,
            output_path=output_dir / "correlation_network.svg" if output_dir else None,
        )
        if fig:
            figures["network"] = fig
            plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to create correlation network: {e}")

    logger.info(f"\n✅ Created {len(figures)} correlation visualizations")
    if output_dir:
        logger.info(f"   Saved to: {output_dir}")

    return figures
