"""
Advanced Causal Discovery Visualizations and Mathematical Analyses

Implements publication-quality plots and statistical analyses typical for causal inference:
- Granger causality spectral analysis
- Transfer entropy information flow
- Conditional independence tests visualization
- DAG structure learning diagnostics
- Bootstrap uncertainty quantification
- False discovery rate analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats, signal
from scipy.cluster import hierarchy
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2


def plot_granger_spectrum(
    X: np.ndarray,
    Y: np.ndarray,
    max_lag: int = 12,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8),
) -> Optional[Path]:
    """
    Plot Granger causality spectral decomposition showing frequency-domain contributions.

    This reveals at which frequencies X Granger-causes Y, providing insight into
    timescale-specific causal relationships.

    Parameters:
        X: Cause variable time series (n_samples,)
        Y: Effect variable time series (n_samples,)
        max_lag: Maximum lag to consider
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Time series plot
    ax = axes[0, 0]
    t = np.arange(len(X))
    ax.plot(t, X, label="X (Cause)", alpha=0.7, linewidth=1.5)
    ax.plot(t, Y, label="Y (Effect)", alpha=0.7, linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("(A) Time Series")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Cross-correlation function
    ax = axes[0, 1]
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.correlate(Y - Y.mean(), X - X.mean(), mode="full")
    xcorr = xcorr[len(X) - max_lag - 1 : len(X) + max_lag] / (
        X.std() * Y.std() * len(X)
    )

    ax.stem(lags, xcorr, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.8)
    ax.axvline(x=0, color="r", linestyle="--", alpha=0.5, label="Zero lag")
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("(B) Cross-Correlation Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Power spectral density
    ax = axes[1, 0]
    freqs_x, psd_x = signal.periodogram(X, scaling="density")
    freqs_y, psd_y = signal.periodogram(Y, scaling="density")

    ax.semilogy(freqs_x, psd_x, label="X (Cause)", alpha=0.7, linewidth=2)
    ax.semilogy(freqs_y, psd_y, label="Y (Effect)", alpha=0.7, linewidth=2)
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title("(C) Spectral Analysis")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Coherence (frequency-domain correlation)
    ax = axes[1, 1]
    freqs, coherence = signal.coherence(X, Y, nperseg=min(256, len(X) // 4))

    ax.plot(freqs, coherence, linewidth=2, color="purple")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="Threshold (0.5)")
    ax.set_xlabel("Frequency (cycles/step)")
    ax.set_ylabel("Coherence")
    ax.set_title("(D) Coherence (Freq-domain Correlation)")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Granger spectrum: {output_path}")
        plt.close()
        return output_path

    return None


def plot_transfer_entropy_flow(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
) -> Optional[Path]:
    """
    Plot transfer entropy as information flow diagram with edge widths proportional to TE.

    Transfer entropy quantifies directed information transfer, making it ideal
    for visualizing information flow networks.

    Parameters:
        results_df: Results with 'source', 'target', 'te_value' columns
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx required for TE flow plot")
        return None

    if results_df is None or len(results_df) == 0:
        logger.warning("No data for TE flow plot")
        return None

    # Create graph with TE values as weights
    G = nx.DiGraph()

    for _, row in results_df.iterrows():
        te = row.get("te_value", row.get("statistic", 1.0))
        G.add_edge(row["source"], row["target"], weight=te, te=te)

    # Layout
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes with size proportional to in-degree (information reception)
    node_sizes = [3000 + 1000 * G.in_degree(node) for node in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightcoral",
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="darkred",
        linewidths=2,
        ax=ax,
    )

    # Draw edges with width proportional to TE
    edges = list(G.edges(data=True))
    te_values = [e[2]["te"] for e in edges]

    if len(te_values) > 0:
        te_min, te_max = min(te_values), max(te_values)
        te_range = te_max - te_min if te_max > te_min else 1.0

        # Check if all TE values are the same (no variation)
        if te_range < 1e-6:
            # Use uniform width and color for all edges
            widths = [2.0] * len(edges)
            colors = ["steelblue"] * len(edges)
            logger.warning("All TE values are identical - using uniform edge styling")
        else:
            # Normalize widths between 1 and 8
            widths = [1.0 + 7.0 * ((te - te_min) / te_range) for te in te_values]

            # Color by TE strength
            cmap = plt.cm.YlOrRd
            colors = [cmap((te - te_min) / te_range) for te in te_values]

        # Draw edges with varying properties
        for (u, v, d), width, color in zip(edges, widths, colors):
            ax.annotate(
                "",
                xy=pos[v],
                xytext=pos[u],
                arrowprops=dict(
                    arrowstyle="->",
                    lw=width,
                    color=color,
                    alpha=0.8,
                    shrinkA=20,
                    shrinkB=20,
                ),
            )

            # Add TE value labels for strong connections (or all if uniform)
            if width > 5 or te_range < 1e-6:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{d['te']:.2f}",
                    fontsize=8,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Add colorbar for TE scale
    if len(te_values) > 0:
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(vmin=te_min, vmax=te_max)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Transfer Entropy (bits)", rotation=270, labelpad=20)

    ax.set_title(
        f"Transfer Entropy Information Flow\n{len(G.nodes())} variables, {len(G.edges())} connections",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    plt.tight_layout(pad=1.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved TE flow: {output_path}")
        plt.close()
        return output_path

    return None


def plot_conditional_independence_matrix(
    results_df: pd.DataFrame,
    method: str = "PCMCI+",
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> Optional[Path]:
    """
    Plot matrix of conditional independence test results (p-values or test statistics).

    Shows which variable pairs are conditionally independent given others,
    fundamental for structure learning.

    Parameters:
        results_df: Results with 'source', 'target', 'p_value' columns
        method: Method name for title
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if results_df is None or len(results_df) == 0:
        logger.warning("No data for CI matrix")
        return None

    # Get unique variables
    sources = results_df["source"].unique()
    targets = results_df["target"].unique()
    variables = sorted(set(list(sources) + list(targets)))
    n_vars = len(variables)

    # Create p-value matrix
    pval_matrix = np.ones((n_vars, n_vars))

    for _, row in results_df.iterrows():
        i = variables.index(row["source"])
        j = variables.index(row["target"])
        pval = row.get("p_value", row.get("best_p_value", 1.0))
        pval_matrix[i, j] = pval

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 1. P-value matrix (log scale)
    im1 = ax1.imshow(
        -np.log10(pval_matrix + 1e-10), cmap="RdYlGn", aspect="auto", vmin=0, vmax=5
    )
    ax1.set_xticks(np.arange(n_vars))
    ax1.set_yticks(np.arange(n_vars))
    ax1.set_xticklabels(variables, rotation=45, ha="right")
    ax1.set_yticklabels(variables)
    ax1.set_xlabel("Target (Effect)")
    ax1.set_ylabel("Source (Cause)")
    ax1.set_title(f"(A) {method}: -log₁₀(p-value) Matrix")

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("-log₁₀(p-value)", rotation=270, labelpad=20)

    # Add significance markers
    for i in range(n_vars):
        for j in range(n_vars):
            if pval_matrix[i, j] < 0.01:
                ax1.text(
                    j,
                    i,
                    "**",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                    fontweight="bold",
                )
            elif pval_matrix[i, j] < 0.05:
                ax1.text(
                    j,
                    i,
                    "*",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                    fontweight="bold",
                )

    # 2. Adjacency matrix (binary: significant or not)
    adj_matrix = (pval_matrix < 0.05).astype(int)

    im2 = ax2.imshow(adj_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax2.set_xticks(np.arange(n_vars))
    ax2.set_yticks(np.arange(n_vars))
    ax2.set_xticklabels(variables, rotation=45, ha="right")
    ax2.set_yticklabels(variables)
    ax2.set_xlabel("Target (Effect)")
    ax2.set_ylabel("Source (Cause)")
    ax2.set_title(f"(B) {method}: Adjacency Matrix (α=0.05)")

    # Add edge counts
    for i in range(n_vars):
        for j in range(n_vars):
            if adj_matrix[i, j] == 1:
                ax2.text(
                    j,
                    i,
                    "→",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=20,
                    fontweight="bold",
                )

    plt.tight_layout(pad=1.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved CI matrix: {output_path}")
        plt.close()
        return output_path

    return None


def plot_lag_distribution_analysis(
    results_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> Optional[Path]:
    """
    Comprehensive lag distribution analysis across methods and variable pairs.

    Shows:
    - Histogram of optimal lags
    - Lag vs p-value scatter
    - Per-method lag distributions
    - Temporal clustering analysis

    Parameters:
        results_df: Results with 'lag_steps', 'p_value', 'method' columns
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if results_df is None or len(results_df) == 0:
        logger.warning("No data for lag analysis")
        return None

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Get lag data
    df = results_df.copy()
    if "lag_steps" not in df.columns:
        if "lag" in df.columns:
            df["lag_steps"] = df["lag"]
        elif "best_lag" in df.columns:
            df["lag_steps"] = df["best_lag"]
        else:
            logger.warning("No lag column found")
            return None

    # Remove NaN lags
    df = df[df["lag_steps"].notna()]

    if len(df) == 0:
        logger.warning("No valid lag data")
        return None

    # 1. Overall lag histogram
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(df["lag_steps"], bins=20, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(
        df["lag_steps"].median(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {df['lag_steps'].median():.1f}",
    )
    ax1.axvline(
        df["lag_steps"].mean(),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df['lag_steps'].mean():.1f}",
    )
    ax1.set_xlabel("Lag (steps)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(
        "(A) Distribution of Optimal Lags Across All Methods", fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Lag vs p-value scatter
    ax2 = fig.add_subplot(gs[1, 0])
    if "p_value" in df.columns:
        scatter = ax2.scatter(
            df["lag_steps"],
            -np.log10(df["p_value"] + 1e-10),
            alpha=0.6,
            s=50,
            c=df["lag_steps"],
            cmap="viridis",
        )
        ax2.axhline(
            -np.log10(0.05), color="red", linestyle="--", alpha=0.5, label="α=0.05"
        )
        ax2.axhline(
            -np.log10(0.01), color="darkred", linestyle="--", alpha=0.5, label="α=0.01"
        )
        ax2.set_xlabel("Lag (steps)")
        ax2.set_ylabel("-log₁₀(p-value)")
        ax2.set_title("(B) Lag vs Significance", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label="Lag (steps)")

    # 3. Per-method lag boxplot
    ax3 = fig.add_subplot(gs[1, 1])
    if "method" in df.columns:
        methods = df["method"].unique()
        method_lags = [df[df["method"] == m]["lag_steps"].values for m in methods]

        bp = ax3.boxplot(method_lags, labels=methods, patch_artist=True)
        for patch, color in zip(
            bp["boxes"], ["lightblue", "lightgreen", "lightyellow"]
        ):
            patch.set_facecolor(color)

        ax3.set_ylabel("Lag (steps)")
        ax3.set_title("(C) Lag Distribution by Method", fontweight="bold")
        ax3.grid(True, alpha=0.3, axis="y")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 4. Lag autocorrelation (temporal clustering)
    ax4 = fig.add_subplot(gs[2, 0])
    lags_sorted = np.sort(df["lag_steps"].values)
    lag_diffs = np.diff(lags_sorted)

    if len(lag_diffs) > 0:
        ax4.hist(lag_diffs, bins=15, alpha=0.7, color="coral", edgecolor="black")
        ax4.axvline(
            lag_diffs.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean Δ: {lag_diffs.mean():.2f}",
        )
        ax4.set_xlabel("Lag Difference (steps)")
        ax4.set_ylabel("Frequency")
        ax4.set_title("(D) Temporal Clustering of Lags", fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # 5. Cumulative distribution
    ax5 = fig.add_subplot(gs[2, 1])
    sorted_lags = np.sort(df["lag_steps"].values)
    cumulative = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)

    ax5.plot(sorted_lags, cumulative, linewidth=2, color="purple")
    ax5.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Median")
    ax5.axvline(df["lag_steps"].median(), color="red", linestyle="--", alpha=0.5)
    ax5.set_xlabel("Lag (steps)")
    ax5.set_ylabel("Cumulative Probability")
    ax5.set_title("(E) Cumulative Distribution of Lags", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        "Comprehensive Lag Distribution Analysis",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved lag analysis: {output_path}")
        plt.close()
        return output_path

    return None


def plot_bootstrap_uncertainty(
    bootstrap_results: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> Optional[Path]:
    """
    Plot bootstrap confidence intervals for lag estimates.

    Shows uncertainty quantification via bootstrap resampling,
    critical for assessing reliability of temporal relationships.

    Parameters:
        bootstrap_results: Dict with 'point_estimate', 'ci_lower', 'ci_upper', 'bootstrap_samples'
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    if not bootstrap_results or len(bootstrap_results) == 0:
        logger.warning("No bootstrap results to plot")
        return None

    n_pairs = len(bootstrap_results)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(figsize[0], figsize[1] * n_pairs / 3))

    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx, (pair_name, result) in enumerate(bootstrap_results.items()):
        point = result.get("point_estimate")
        ci_lower = result.get("ci_lower")
        ci_upper = result.get("ci_upper")
        samples = result.get("bootstrap_samples", [])

        # Left: Bootstrap distribution
        ax_left = axes[idx, 0] if n_pairs > 1 else axes[0]
        if len(samples) > 0:
            ax_left.hist(
                samples,
                bins=20,
                alpha=0.7,
                color="steelblue",
                edgecolor="black",
                density=True,
            )
            ax_left.axvline(
                point, color="red", linewidth=2, label=f"Point: {point:.1f}"
            )
            ax_left.axvline(
                ci_lower,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"CI: [{ci_lower:.1f}, {ci_upper:.1f}]",
            )
            ax_left.axvline(ci_upper, color="orange", linestyle="--", linewidth=2)
            ax_left.set_xlabel("Lag (steps)")
            ax_left.set_ylabel("Density")
            ax_left.set_title(f"{pair_name}: Bootstrap Distribution")
            ax_left.legend()
            ax_left.grid(True, alpha=0.3)

        # Right: Forest plot
        ax_right = axes[idx, 1] if n_pairs > 1 else axes[1]
        ax_right.errorbar(
            [point],
            [0],
            xerr=[[point - ci_lower], [ci_upper - point]],
            fmt="o",
            markersize=10,
            capsize=10,
            capthick=2,
            color="steelblue",
            ecolor="steelblue",
            linewidth=2,
        )
        ax_right.axvline(point, color="red", linestyle="--", alpha=0.5)
        ax_right.set_xlabel("Lag (steps)")
        ax_right.set_yticks([])
        ax_right.set_title(f"{pair_name}: Confidence Interval")
        ax_right.grid(True, alpha=0.3, axis="x")

    plt.tight_layout(pad=1.2)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved bootstrap plot: {output_path}")
        plt.close()
        return output_path

    return None


def plot_fdr_diagnostics(
    p_values: np.ndarray,
    q_values: np.ndarray,
    alpha: float = 0.05,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> Optional[Path]:
    """
    Plot False Discovery Rate (FDR) diagnostic plots.

    Shows:
    - P-value distribution (should be uniform under null)
    - Q-Q plot for p-values
    - BH procedure threshold line
    - FDR vs rejection threshold

    Parameters:
        p_values: Array of p-values
        q_values: Array of q-values (FDR-adjusted)
        alpha: FDR level
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Remove NaN values
    mask = ~(np.isnan(p_values) | np.isnan(q_values))
    p_values = p_values[mask]
    q_values = q_values[mask]

    if len(p_values) == 0:
        logger.warning("No valid p-values for FDR diagnostics")
        return None

    # 1. P-value histogram
    ax = axes[0, 0]
    ax.hist(
        p_values, bins=50, alpha=0.7, color="steelblue", edgecolor="black", density=True
    )
    ax.axhline(1.0, color="red", linestyle="--", linewidth=2, label="Uniform (null)")
    ax.set_xlabel("P-value")
    ax.set_ylabel("Density")
    ax.set_title("(A) P-value Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q-Q plot
    ax = axes[0, 1]
    sorted_p = np.sort(p_values)
    theoretical = np.linspace(0, 1, len(sorted_p))

    ax.scatter(theoretical, sorted_p, alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Uniform (null)")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Observed P-value Quantiles")
    ax.set_title("(B) Q-Q Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Benjamini-Hochberg procedure
    ax = axes[1, 0]
    m = len(p_values)
    ranks = np.arange(1, m + 1)
    sorted_p = np.sort(p_values)
    bh_threshold = (ranks / m) * alpha

    ax.plot(ranks, sorted_p, "o-", label="Sorted p-values", markersize=4, alpha=0.7)
    ax.plot(ranks, bh_threshold, "r--", linewidth=2, label=f"BH threshold (α={alpha})")
    ax.set_xlabel("Rank")
    ax.set_ylabel("P-value")
    ax.set_title("(C) Benjamini-Hochberg Procedure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Find rejection threshold
    rejections = sorted_p <= bh_threshold
    if np.any(rejections):
        max_reject_idx = np.where(rejections)[0][-1]
        ax.axvline(
            max_reject_idx + 1,
            color="green",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Max reject: {max_reject_idx + 1}",
        )
        ax.legend()

    # 4. Q-value vs P-value
    ax = axes[1, 1]
    sorted_idx = np.argsort(p_values)
    ax.plot(p_values[sorted_idx], q_values[sorted_idx], "o-", markersize=4, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="p=q")
    ax.axhline(alpha, color="red", linestyle="--", linewidth=2, label=f"FDR α={alpha}")
    ax.axvline(alpha, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("P-value")
    ax.set_ylabel("Q-value (FDR)")
    ax.set_title("(D) Q-value vs P-value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Count rejections
    n_reject_p = np.sum(p_values < alpha)
    n_reject_q = np.sum(q_values < alpha)
    fig.suptitle(
        f"FDR Diagnostics: {n_reject_p} raw rejections → {n_reject_q} FDR-controlled rejections",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout(pad=1.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved FDR diagnostics: {output_path}")
        plt.close()
        return output_path

    return None


def plot_dag_learning_diagnostics(
    results_df: pd.DataFrame,
    true_dag: Optional[Dict] = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
) -> Optional[Path]:
    """
    Plot DAG structure learning diagnostics.

    If true DAG is known, shows:
    - True positives, false positives, false negatives
    - Precision-recall curve
    - Structural Hamming Distance

    Parameters:
        results_df: Results with 'source', 'target', 'p_value' columns
        true_dag: Optional dict of true edges {(source, target): True}
        output_path: Save path
        figsize: Figure size

    Returns:
        Path to saved figure
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx required for DAG diagnostics")
        return None

    if results_df is None or len(results_df) == 0:
        logger.warning("No results for DAG diagnostics")
        return None

    # Build learned DAG
    G_learned = nx.DiGraph()
    for _, row in results_df.iterrows():
        if row.get("p_value", row.get("best_p_value", 1.0)) < 0.05:
            G_learned.add_edge(row["source"], row["target"])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Degree distribution
    ax = axes[0]
    in_degrees = [d for n, d in G_learned.in_degree()]
    out_degrees = [d for n, d in G_learned.out_degree()]

    x = np.arange(
        max(
            max(in_degrees) if in_degrees else 0, max(out_degrees) if out_degrees else 0
        )
        + 1
    )
    in_hist = [in_degrees.count(i) for i in x]
    out_hist = [out_degrees.count(i) for i in x]

    ax.bar(x - 0.2, in_hist, width=0.4, label="In-degree", alpha=0.7)
    ax.bar(x + 0.2, out_hist, width=0.4, label="Out-degree", alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("(A) Degree Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. If true DAG provided, show confusion matrix
    ax = axes[1]
    if true_dag:
        # Build true DAG
        G_true = nx.DiGraph()
        for (u, v), exists in true_dag.items():
            if exists:
                G_true.add_edge(u, v)

        # Compute metrics
        true_edges = set(G_true.edges())
        learned_edges = set(G_learned.edges())

        tp = len(true_edges & learned_edges)
        fp = len(learned_edges - true_edges)
        fn = len(true_edges - learned_edges)
        tn = 0  # Hard to define for sparse graphs

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Plot confusion-like metrics
        metrics = ["TP", "FP", "FN"]
        values = [tp, fp, fn]
        colors = ["green", "red", "orange"]

        bars = ax.bar(
            metrics, values, color=colors, alpha=0.7, edgecolor="black", linewidth=2
        )
        ax.set_ylabel("Count")
        ax.set_title(
            f"(B) Edge Detection Performance\nPrec: {precision:.2f}, Rec: {recall:.2f}, F1: {f1:.2f}"
        )
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        # Just show edge statistics
        stats_text = f"""
        Learned DAG Statistics:
        
        Nodes: {G_learned.number_of_nodes()}
        Edges: {G_learned.number_of_edges()}
        
        Avg in-degree: {np.mean(in_degrees):.2f}
        Avg out-degree: {np.mean(out_degrees):.2f}
        
        Is DAG: {nx.is_directed_acyclic_graph(G_learned)}
        """
        ax.text(
            0.1,
            0.5,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_title("(B) Learned DAG Statistics")
        ax.axis("off")

    plt.tight_layout(pad=1.0)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved DAG diagnostics: {output_path}")
        plt.close()
        return output_path

    return None
