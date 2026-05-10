"""
Causal Network Graph Visualizations

Plots directed causal graphs showing variable relationships and information flow,
using spring-layout or hierarchical positioning for clarity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import networkx for graph visualization
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Install with: pip install networkx")


def plot_causal_graph(
    results_df: pd.DataFrame,
    method: str = "Consensus",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 12),
    node_size: int = 3000,
    alpha_threshold: float = 0.05,
    edge_label: str = "lag",
    arrowsize: int = 22,
    edge_alpha: float = 0.8,
    show_legend: bool = True,
    layout: str = "spring",
) -> Optional[Path]:
    """
    Plot publication-quality directed causal network graph.

    Parameters:
        results_df (pd.DataFrame): Results with 'source', 'target', 'p_value' columns
        method (str): Method name (for title)
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size
        node_size (int): Node size in points
        alpha_threshold (float): Significance threshold for edge inclusion
        edge_label (str): 'lag', 'lag_p', 'lag_q', or None
        arrowsize (int): Arrow size
        edge_alpha (float): Edge transparency
        show_legend (bool): Show method legend
        layout (str): 'spring', 'circular', 'hierarchical', or 'kamada_kawai'

    Returns:
        Optional[Path]: Path to saved figure
    """
    if not NETWORKX_AVAILABLE:
        logger.warning("networkx required for graph visualization. Skipping plot.")
        return None

    if results_df is None or len(results_df) == 0:
        logger.warning(f"No data for causal graph ({method})")
        return None

    # Normalize common columns for robustness across methods
    df = results_df.copy()
    if "p_value" not in df.columns and "best_p_value" in df.columns:
        df["p_value"] = df["best_p_value"]
    if "lag_steps" not in df.columns:
        # Try alternate lag column names
        if "lag" in df.columns:
            df["lag_steps"] = df["lag"]
        elif "delay" in df.columns:
            df["lag_steps"] = df["delay"]
        else:
            df["lag_steps"] = np.nan

    # Filter significant edges
    if "p_value" in df.columns:
        sig_edges = df[df["p_value"] < alpha_threshold]
    else:
        sig_edges = df

    if len(sig_edges) == 0:
        logger.warning(f"No significant edges for graph plot ({method})")
        return None

    # Create directed graph
    G = nx.DiGraph()

    # Add edges with weights (inverse of p-value)
    for _, row in sig_edges.iterrows():
        source = row.get("source", row.get("cause"))
        target = row.get("target", row.get("effect"))
        if source is None or target is None:
            continue
        p_val = row.get("p_value", np.nan)

        # Weight: higher weight for lower p-value (stronger evidence)
        weight = -np.log10(p_val + 1e-10) if not np.isnan(p_val) else 1.0

        G.add_edge(
            source,
            target,
            weight=weight,
            p_value=p_val,
            q_value=row.get("q_value", np.nan),
            lag=row.get("lag_steps", np.nan),
            method=row.get("method", method),
        )

    # Choose layout algorithm
    if layout == "spring":
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "hierarchical":
        # Attempt topological sort for DAG
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except Exception:
            # Fallback to spring if graphviz not available
            pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    # Compute node colors based on in-degree and out-degree
    node_colors = []
    node_edge_colors = []
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)

        # Color based on role: source (high out), sink (high in), or intermediate
        if out_deg > in_deg and out_deg > 0:
            node_colors.append("#ff7f0e")  # Orange for sources
            node_edge_colors.append("#d62728")
        elif in_deg > out_deg and in_deg > 0:
            node_colors.append("#1f77b4")  # Blue for sinks
            node_edge_colors.append("#0c5a8a")
        else:
            node_colors.append("#2ca02c")  # Green for intermediates
            node_edge_colors.append("#1a7018")

    # Draw nodes with enhanced styling
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        ax=ax,
        edgecolors=node_edge_colors,
        linewidths=3,
        alpha=0.9,
    )

    # Draw edges with varying width and method-based colors
    edges = list(G.edges(data=True))
    weights = [e[2]["weight"] for e in edges]
    max_weight = max(weights) if weights else 1
    widths = [max(2.0, 5.0 * (w / max_weight)) for w in weights]

    # Method color map with distinct colors
    method_colors = {
        "Granger": "#2b8cbe",  # steelblue
        "TransferEntropy": "#ffb000",  # goldenrod
        "PCMCI+": "#e6550d",  # dark orange
        "Consensus": "#2ca25f",  # green
    }

    # Collect methods present
    methods_present = []

    # Draw edges with enhanced curved arrows
    for (u, v, d), width in zip(edges, widths):
        m = d.get("method", method)
        if m not in methods_present:
            methods_present.append(m)
        color = method_colors.get(m, "#1b7837")

        # Add curvature for better visibility of multiple edges
        rad = 0.2 if G.has_edge(v, u) else 0.0  # Curve if bidirectional

        ax.annotate(
            "",
            xy=pos[v],
            xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->",
                lw=width,
                color=color,
                alpha=edge_alpha,
                shrinkA=25,
                shrinkB=25,
                connectionstyle=f"arc3,rad={rad}",
            ),
        )

    # Draw node labels with background boxes for readability
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node,
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.9,
            ),
        )

    # Edge labels (lags and/or p/q values)
    if edge_label in {"lag", "lag_p", "lag_q"} and len(edges) > 0:
        edge_labels = {}
        for u, v, d in edges:
            lag = d.get("lag", np.nan)
            p = d.get("p_value", np.nan)
            q = d.get("q_value", np.nan)
            label = None
            if edge_label == "lag" and not np.isnan(lag):
                label = f"τ={int(lag)}"
            elif edge_label == "lag_p":
                parts = []
                if not np.isnan(lag):
                    parts.append(f"τ={int(lag)}")
                if not np.isnan(p):
                    parts.append(f"p={p:.2g}")
                label = ", ".join(parts) if parts else None
            elif edge_label == "lag_q":
                parts = []
                if not np.isnan(lag):
                    parts.append(f"τ={int(lag)}")
                if not np.isnan(q):
                    parts.append(f"q={q:.2g}")
                elif not np.isnan(p):
                    parts.append(f"p={p:.2g}")
                label = ", ".join(parts) if parts else None

            if label:
                edge_labels[(u, v)] = label

        if edge_labels:
            # Draw edge labels with better positioning and background
            for (u, v), label in edge_labels.items():
                # Calculate midpoint
                x = (pos[u][0] + pos[v][0]) / 2
                y = (pos[u][1] + pos[v][1]) / 2

                ax.text(
                    x,
                    y,
                    label,
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="lightyellow",
                        edgecolor="gray",
                        alpha=0.8,
                        linewidth=1,
                    ),
                )

    # Title and formatting with enhanced styling
    ax.set_title(
        f"{method}: Causal Network Graph (α < {alpha_threshold})\n{len(G.nodes())} variables, {len(G.edges())} edges",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    # Add a subtle background
    ax.set_facecolor("#f8f8f8")

    # Legend for methods if applicable
    legend_method = None
    if show_legend and len(methods_present) > 1:
        from matplotlib.lines import Line2D

        handles = [
            Line2D(
                [0],
                [0],
                color=method_colors.get(m, "#1b7837"),
                lw=4,
                label=m,
                alpha=edge_alpha,
            )
            for m in methods_present
        ]
        legend_method = ax.legend(
            handles=handles,
            title="Method",
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            title_fontsize=12,
            ncol=1,
            borderpad=1,
        )
        legend_method.get_frame().set_alpha(0.95)
        legend_method.get_frame().set_edgecolor("black")
        legend_method.get_frame().set_linewidth(1.5)

    # Add node role legend
    if show_legend:
        from matplotlib.lines import Line2D

        role_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#ff7f0e",
                markersize=12,
                label="Source (high out-degree)",
                markeredgecolor="#d62728",
                markeredgewidth=2,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#1f77b4",
                markersize=12,
                label="Sink (high in-degree)",
                markeredgecolor="#0c5a8a",
                markeredgewidth=2,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2ca02c",
                markersize=12,
                label="Intermediate",
                markeredgecolor="#1a7018",
                markeredgewidth=2,
            ),
        ]
        role_legend = ax.legend(
            handles=role_handles,
            title="Node Role",
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=11,
            title_fontsize=12,
            ncol=1,
            borderpad=1,
        )
        role_legend.get_frame().set_alpha(0.95)
        role_legend.get_frame().set_edgecolor("black")
        role_legend.get_frame().set_linewidth(1.5)

        # Add method legend back if needed
        if show_legend and legend_method is not None and len(methods_present) > 1:
            ax.add_artist(legend_method)

    plt.tight_layout(pad=1.5)

    if output_path:
        output_path = Path(output_path)
        # Save both PNG and SVG for quality and editability
        save_path_png = output_path.parent / f"{output_path.stem}.png"
        save_path_svg = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(
            save_path_png, format="png", dpi=300, bbox_inches="tight", facecolor="white"
        )
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path_png} and {save_path_svg}")
        plt.close()
        return save_path_svg

    return None


def plot_network_structure(
    results_df: pd.DataFrame,
    layout: str = "hierarchical",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
) -> Optional[Path]:
    """
    Plot network structure with hierarchical or circular layout.

    Parameters:
        results_df (pd.DataFrame): Results with 'source', 'target' columns
        layout (str): "hierarchical", "circular", or "spring"
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if not NETWORKX_AVAILABLE:
        logger.warning("networkx required for network structure plot. Skipping.")
        return None

    if results_df is None or len(results_df) == 0:
        logger.warning("No data for network structure plot")
        return None

    # Create graph
    G = nx.DiGraph()

    for _, row in results_df.iterrows():
        G.add_edge(row["source"], row["target"])

    # Choose layout
    if layout == "hierarchical":
        # Topological sort based layout (if DAG)
        try:
            pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:  # spring
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightyellow",
        node_size=2000,
        ax=ax,
        edgecolors="black",
        linewidths=2,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        ax=ax,
        alpha=0.6,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)

    # Title
    ax.set_title(
        f"Network Structure ({layout.capitalize()} Layout)\n{len(G.nodes())} nodes, {len(G.edges())} edges",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None
