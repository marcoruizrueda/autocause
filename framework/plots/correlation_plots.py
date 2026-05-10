"""
Robust Correlation Visualization

Creates plots comparing multiple correlation methods:
- Distance correlation (nonlinear dependence)
- Spearman correlation (rank-based, outlier-robust)
- Kendall correlation (tau, better for small samples)
- Pearson correlation (standard, for comparison)

All plots are dataset-agnostic and work with arbitrary time series.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_correlation_heatmap(
    data: pd.DataFrame,
    methods: list = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 4),
    title: str = "Correlation Matrix Comparison",
) -> plt.Figure:
    """
    Create side-by-side heatmaps comparing correlation methods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data (rows=time, columns=variables)
    methods : list, optional
        Correlation methods to compare
        Options: 'distance', 'spearman', 'kendall', 'pearson'
        Default: all four methods
    output_path : Path, optional
        Where to save figure
    figsize : tuple, default=(16, 4)
        Figure size (width, height)
    title : str
        Main title

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100),
    ...     'Z': np.random.randn(100)
    ... })
    >>> fig = plot_correlation_heatmap(data)
    >>> plt.close(fig)
    """
    from framework.core.robust_correlation import correlation_matrix

    if methods is None:
        methods = ["distance", "spearman", "kendall", "pearson"]

    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Compute correlation matrix
        corr_mat = correlation_matrix(data, method=method)

        # Plot heatmap
        sns.heatmap(
            corr_mat,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1 if method != "distance" else 0,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )

        method_name = method.capitalize()
        ax.set_title(f"{method_name} Correlation", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(pad=1.0)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correlation heatmap to {output_path}")

    return fig


def plot_correlation_scatter(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    methods: list = None,
    lag: int = 0,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 3),
) -> plt.Figure:
    """
    Create scatter plots with correlation values for multiple methods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var1 : str
        First variable name
    var2 : str
        Second variable name
    methods : list, optional
        Correlation methods to show
    lag : int, default=0
        Lag to apply to var1 (positive = var1 leads var2)
    output_path : Path, optional
        Where to save figure
    figsize : tuple, default=(12, 3)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100)
    ... })
    >>> fig = plot_correlation_scatter(data, 'X', 'Y')
    >>> plt.close(fig)
    """
    from framework.core.robust_correlation import compute_all_correlations

    if methods is None:
        methods = ["distance", "spearman", "kendall", "pearson"]

    # Prepare data with lag
    x = data[var1].values
    y = data[var2].values

    if lag != 0:
        if lag > 0:
            x = x[:-lag]
            y = y[lag:]
        else:
            x = x[-lag:]
            y = y[:lag]

    # Remove NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Compute all correlations
    corr_results = compute_all_correlations(x, y, lag=0)  # Already lagged

    # Create plots
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)

    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        # Scatter plot
        ax.scatter(x, y, alpha=0.5, s=20, edgecolors="k", linewidths=0.5)

        # Add regression line for Pearson/Spearman
        if method in ["pearson", "spearman"]:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        # Title with correlation value
        corr_dict = corr_results.get(method, {})
        corr_value = corr_dict.get("correlation", np.nan)
        method_name = method.capitalize()
        ax.set_title(f"{method_name}: {corr_value:.3f}", fontweight="bold")

        lag_text = f" (lag={lag})" if lag != 0 else ""
        ax.set_xlabel(f"{var1}{lag_text}")
        ax.set_ylabel(var2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correlation scatter to {output_path}")

    return fig


def plot_lagged_correlation(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    max_lag: int = 10,
    methods: list = None,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot correlation as function of lag for multiple methods.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    var1 : str
        Predictor variable
    var2 : str
        Target variable
    max_lag : int, default=10
        Maximum lag to test
    methods : list, optional
        Correlation methods
    output_path : Path, optional
        Where to save figure
    figsize : tuple, default=(10, 6)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100)
    ... })
    >>> fig = plot_lagged_correlation(data, 'X', 'Y', max_lag=5)
    >>> plt.close(fig)
    """
    from framework.core.robust_correlation import compute_all_correlations

    if methods is None:
        methods = ["distance", "spearman", "kendall", "pearson"]

    x = data[var1].values
    y = data[var2].values

    # Compute correlations at each lag
    results = {method: [] for method in methods}
    lags = list(range(0, max_lag + 1))

    for lag in lags:
        x_lagged = x[:-lag] if lag > 0 else x
        y_subset = y[lag:] if lag > 0 else y

        # Remove NaN
        mask = ~(np.isnan(x_lagged) | np.isnan(y_subset))
        x_clean = x_lagged[mask]
        y_clean = y_subset[mask]

        corr_results = compute_all_correlations(x_clean, y_clean, lag=0)

        for method in methods:
            corr_dict = corr_results.get(method, {})
            corr_value = corr_dict.get("correlation", np.nan)
            results[method].append(corr_value)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "distance": "#e74c3c",
        "spearman": "#3498db",
        "kendall": "#2ecc71",
        "pearson": "#9b59b6",
    }

    for method in methods:
        values = results[method]
        color = colors.get(method, "black")
        label = method.capitalize()
        ax.plot(lags, values, marker="o", linewidth=2, label=label, color=color)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag (time steps)", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title(f"Lagged Correlation: {var1} → {var2}", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved lagged correlation plot to {output_path}")

    return fig


def plot_method_comparison(
    data: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Compare all correlation methods in a grid layout.

    Creates a matrix comparing each method with every other method
    to show where they agree/disagree.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    output_path : Path, optional
        Where to save figure
    figsize : tuple, default=(12, 10)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100),
    ...     'Z': np.random.randn(100)
    ... })
    >>> fig = plot_method_comparison(data)
    >>> plt.close(fig)
    """
    from framework.core.robust_correlation import correlation_matrix

    methods = ["distance", "spearman", "kendall", "pearson"]
    n_methods = len(methods)

    # Compute correlation matrices
    matrices = {}
    for method in methods:
        matrices[method] = correlation_matrix(data, method=method)

    # Create comparison plot
    fig, axes = plt.subplots(n_methods, n_methods, figsize=figsize)

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show the correlation matrix
                sns.heatmap(
                    matrices[method1],
                    annot=True,
                    fmt=".2f",
                    cmap="RdBu_r",
                    center=0,
                    vmin=-1 if method1 != "distance" else 0,
                    vmax=1,
                    square=True,
                    ax=ax,
                    cbar=False,
                )
                ax.set_title(method1.capitalize(), fontweight="bold")
            else:
                # Off-diagonal: scatter plot comparing methods
                mat1 = matrices[method1].values
                mat2 = matrices[method2].values

                # Get upper triangle (exclude diagonal)
                mask = np.triu(np.ones_like(mat1), k=1).astype(bool)
                vals1 = mat1[mask]
                vals2 = mat2[mask]

                ax.scatter(vals1, vals2, alpha=0.6, s=30)
                ax.plot([-1, 1], [-1, 1], "r--", alpha=0.5)

                ax.set_xlim(-1 if method1 != "distance" else 0, 1)
                ax.set_ylim(-1 if method2 != "distance" else 0, 1)

                if j == 0:
                    ax.set_ylabel(method1.capitalize())
                if i == n_methods - 1:
                    ax.set_xlabel(method2.capitalize())

                ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Correlation Method Comparison", fontsize=16, fontweight="bold", y=0.995
    )
    plt.tight_layout(pad=1.0)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved method comparison to {output_path}")

    return fig


def plot_partial_correlation_network(
    data: pd.DataFrame,
    threshold: float = 0.3,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Plot network graph of partial correlations (conditional independence).

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    threshold : float, default=0.3
        Minimum absolute correlation to show edge
    output_path : Path, optional
        Where to save figure
    figsize : tuple, default=(10, 10)
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100),
    ...     'Z': np.random.randn(100)
    ... })
    >>> fig = plot_partial_correlation_network(data)
    >>> plt.close(fig)
    """
    from framework.core.robust_correlation import partial_correlation

    variables = data.columns.tolist()
    n_vars = len(variables)

    # Compute partial correlations
    partial_corrs = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                partial_corrs[i, j] = 1.0
            elif i < j:
                # Control for all other variables
                control_vars = [v for v in variables if v not in [var1, var2]]

                if len(control_vars) > 0:
                    pcorr, _ = partial_correlation(
                        data[var1].values,
                        data[var2].values,
                        data[control_vars].values,
                    )
                else:
                    # No control variables, just use regular correlation
                    from framework.core.robust_correlation import pearson_correlation

                    pcorr, _ = pearson_correlation(
                        data[var1].values,
                        data[var2].values,
                    )

                partial_corrs[i, j] = pcorr
                partial_corrs[j, i] = pcorr

    # Create network layout
    fig, ax = plt.subplots(figsize=figsize)

    # Position nodes in circle
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
    positions = {i: (np.cos(angle), np.sin(angle)) for i, angle in enumerate(angles)}

    # Draw edges
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            corr = partial_corrs[i, j]

            if abs(corr) >= threshold:
                x1, y1 = positions[i]
                x2, y2 = positions[j]

                # Edge width proportional to correlation
                width = abs(corr) * 3

                # Edge color: red for positive, blue for negative
                color = "#e74c3c" if corr > 0 else "#3498db"
                alpha = min(abs(corr), 1.0)

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=width,
                    alpha=alpha,
                    zorder=1,
                )

    # Draw nodes
    for i, var in enumerate(variables):
        x, y = positions[i]
        ax.scatter(x, y, s=1000, c="white", edgecolors="black", linewidths=2, zorder=2)
        ax.text(
            x,
            y,
            var,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=3,
        )

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="#e74c3c", linewidth=3, label="Positive"),
        Line2D([0], [0], color="#3498db", linewidth=3, label="Negative"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True, shadow=True)

    ax.set_title(
        f"Partial Correlation Network (threshold={threshold})",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout(pad=1.0)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved partial correlation network to {output_path}")

    return fig


def create_all_correlation_plots(
    data: pd.DataFrame,
    output_dir: Path,
    variable_pairs: list = None,
    max_lag: int = 10,
) -> Dict[str, Path]:
    """
    Generate all correlation plots and save to directory.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    output_dir : Path
        Output directory
    variable_pairs : list, optional
        List of (var1, var2) tuples to plot
        If None, plots all pairs
    max_lag : int, default=10
        Maximum lag for lagged correlation plots

    Returns
    -------
    Dict[str, Path]
        Mapping of plot type to file path

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'X': np.random.randn(100),
    ...     'Y': np.random.randn(100)
    ... })
    >>> output_dir = Path('/tmp/plots')
    >>> paths = create_all_correlation_plots(data, output_dir)
    >>> 'heatmap' in paths
    True
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # 1. Correlation heatmap
    heatmap_path = output_dir / "correlation_heatmap.png"
    plot_correlation_heatmap(data, output_path=heatmap_path)
    plot_paths["heatmap"] = heatmap_path

    # 2. Method comparison
    comparison_path = output_dir / "method_comparison.png"
    plot_method_comparison(data, output_path=comparison_path)
    plot_paths["comparison"] = comparison_path

    # 3. Partial correlation network
    network_path = output_dir / "partial_correlation_network.png"
    plot_partial_correlation_network(data, output_path=network_path)
    plot_paths["network"] = network_path

    # 4. Variable pair plots
    if variable_pairs is None:
        variables = data.columns.tolist()
        variable_pairs = [
            (v1, v2) for i, v1 in enumerate(variables) for v2 in variables[i + 1 :]
        ]

    for var1, var2 in variable_pairs:
        # Scatter plot
        scatter_path = output_dir / f"scatter_{var1}_{var2}.png"
        plot_correlation_scatter(data, var1, var2, output_path=scatter_path)
        plot_paths[f"scatter_{var1}_{var2}"] = scatter_path

        # Lagged correlation
        lagged_path = output_dir / f"lagged_{var1}_{var2}.png"
        plot_lagged_correlation(
            data, var1, var2, max_lag=max_lag, output_path=lagged_path
        )
        plot_paths[f"lagged_{var1}_{var2}"] = lagged_path

    logger.info(f"Created {len(plot_paths)} correlation plots in {output_dir}")

    return plot_paths
