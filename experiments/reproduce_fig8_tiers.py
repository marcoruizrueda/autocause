#!/usr/bin/env python3
"""
Reproduce Fig 8: Tier Example on DGP-Atlas F8 (dgp_002)
=========================================================
Generates the method-count support visualization for one DGP-Atlas instance
from the non-Gaussian F8 family (dgp_002).

Paper description (Fig 8):
  - 6 variables, 9 reference connections
  - Panel (a): Reference graph
  - Panel (b): Tier-1 links (>=3 methods): 8/9 reference, precision=0.89
  - Panel (c): Tier-2 links (=2 methods): 1/2 matches reference
  - Panel (d): Tier-3 links (=1 method): 0 match reference
  - Orange edges: absent from reference
  - Grey dashed: missed reference connections

Usage:
  python experiments/reproduce_fig8_tiers.py --data-dir data/dgp_atlas --output-dir experiments/figures
  python experiments/reproduce_fig8_tiers.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.core.run_workflow import run_causal_discovery_workflow
from framework.core.graph_metrics import binary_metrics_undirected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration for the specific example
FAMILY = "F8"
DATASET_IDX = 2  # dgp_002
TAU_MAX = 5
ALPHA = 0.05

METHOD_CONFIG = {
    "granger": {"enabled": True},
    "transfer_entropy": {"enabled": True},
    "pcmci": {"enabled": True, "allow_contemporaneous": False},
    "varlingam": {"enabled": True},
    "lpcmci": {"enabled": False},
    "correlation": {"enabled": True},
    "predictive_baseline": {"enabled": True},
}


def load_dataset(data_dir: Path) -> tuple:
    """Load the F8/dgp_002 dataset."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from run_dgp_atlas import load_dgp_dataset
    return load_dgp_dataset(data_dir, FAMILY, DATASET_IDX)


def extract_tier_edges(result_dir: Path) -> dict:
    """Extract edges grouped by tier from consensus output."""
    consensus_path = result_dir / "consensus" / "5-tiers" / "consensus_with_tiers.csv"
    if not consensus_path.exists():
        # Try alternative paths
        for alt in [
            result_dir / "consensus_with_tiers.csv",
            result_dir / "ensemble_edges.csv",
        ]:
            if alt.exists():
                consensus_path = alt
                break

    if not consensus_path.exists():
        return {}

    df = pd.read_csv(consensus_path)
    tier_edges = {}

    for _, row in df.iterrows():
        src = row.get("source", "")
        tgt = row.get("target", "")
        n_methods = row.get("n_methods_found", row.get("tier", 1))

        # Assign tier from method count
        if n_methods >= 3:
            tier = 1
        elif n_methods == 2:
            tier = 2
        else:
            tier = 3

        tier_edges.setdefault(tier, []).append((src, tgt))

    return tier_edges


def plot_tier_graphs(
    true_edges: set,
    tier_edges: dict,
    var_names: list,
    output_dir: Path,
):
    """Create the 4-panel tier visualization (Fig 8)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Layout for the graph (consistent across panels)
    G_ref = nx.DiGraph()
    G_ref.add_nodes_from(var_names)
    G_ref.add_edges_from(true_edges)
    pos = nx.spring_layout(G_ref, seed=42, k=2)

    true_undirected = {frozenset(e) for e in true_edges}

    # --- Panel (a): Reference graph ---
    ax = axes[0]
    ax.set_title("(a) Reference graph", fontweight="bold")
    nx.draw_networkx_nodes(G_ref, pos, ax=ax, node_color="#3498db",
                           node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G_ref, pos, ax=ax, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G_ref, pos, ax=ax, edge_color="#2c3e50",
                           arrows=True, arrowsize=15, width=1.5)
    ax.axis("off")

    # --- Panels (b)-(d): Tier graphs ---
    tier_configs = [
        (1, "(b) Tier-1 (>=3 methods)", "#2ecc71"),
        (2, "(c) Tier-2 (=2 methods)", "#f39c12"),
        (3, "(d) Tier-3 (=1 method)", "#e74c3c"),
    ]

    for panel_idx, (tier, title, color) in enumerate(tier_configs):
        ax = axes[panel_idx + 1]
        ax.set_title(title, fontweight="bold")

        edges = tier_edges.get(tier, [])
        discovered_undirected = {frozenset(e) for e in edges}

        # Draw nodes
        nx.draw_networkx_nodes(G_ref, pos, ax=ax, node_color="#3498db",
                               node_size=500, alpha=0.9)
        nx.draw_networkx_labels(G_ref, pos, ax=ax, font_size=8, font_weight="bold")

        # Classify edges
        tp_edges = []  # True positives (in reference)
        fp_edges = []  # False positives (not in reference)

        for edge in edges:
            edge_fs = frozenset(edge)
            if edge_fs in true_undirected:
                tp_edges.append(edge)
            else:
                fp_edges.append(edge)

        # Draw TP edges (green/tier color)
        if tp_edges:
            G_tp = nx.DiGraph()
            G_tp.add_edges_from(tp_edges)
            nx.draw_networkx_edges(G_tp, pos, ax=ax, edge_color=color,
                                   arrows=True, arrowsize=12, width=2.0)

        # Draw FP edges (orange)
        if fp_edges:
            G_fp = nx.DiGraph()
            G_fp.add_edges_from(fp_edges)
            nx.draw_networkx_edges(G_fp, pos, ax=ax, edge_color="#e67e22",
                                   arrows=True, arrowsize=12, width=1.5,
                                   style="solid")

        # Draw missed edges (grey dashed)
        missed = true_undirected - discovered_undirected
        if missed:
            missed_directed = []
            for edge_fs in missed:
                nodes = list(edge_fs)
                missed_directed.append(tuple(nodes))
            G_missed = nx.DiGraph()
            G_missed.add_edges_from(missed_directed)
            nx.draw_networkx_edges(G_missed, pos, ax=ax, edge_color="#bdc3c7",
                                   arrows=True, arrowsize=10, width=1.0,
                                   style="dashed")

        # Compute metrics for annotation
        tp = len(tp_edges)
        fp = len(fp_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ax.text(0.05, 0.05, f"TP={tp}, FP={fp}\nPrec={precision:.2f}",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="True positive (in reference)"),
        mpatches.Patch(color="#e67e22", label="False positive (not in reference)"),
        mpatches.Patch(facecolor="white", edgecolor="#bdc3c7",
                       linestyle="--", label="Missed (reference edge not found)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Fig 8: Method-count support for DGP-Atlas {FAMILY}/dgp_{DATASET_IDX:03d} "
        f"({len(var_names)} vars, {len(true_edges)} reference edges)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "fig8_causal_graph_example.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig8_causal_graph_example.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"  Fig 8 saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 8: Tier example on DGP-Atlas F8/dgp_002"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/dgp_atlas"),
        help="Root directory containing DGP-Atlas datasets",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/figures"),
        help="Output directory for the figure",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Fig 8 reproduction")
        logger.info(f"  Dataset: {FAMILY}/dgp_{DATASET_IDX:03d}")
        logger.info(f"  Expected: 6 vars, 9 reference edges")
        logger.info(f"  Expected Tier-1 precision: 0.89 (8/9)")
        try:
            import matplotlib
            import networkx
            from framework.core.run_workflow import run_causal_discovery_workflow
            logger.info("  Imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading {FAMILY}/dgp_{DATASET_IDX:03d}...")
    df, true_edges = load_dataset(args.data_dir)
    var_names = df.columns.tolist()
    logger.info(f"  {len(var_names)} variables, {len(true_edges)} true edges")

    # Run workflow
    result_dir = args.output_dir / "fig8_workdir"
    logger.info("Running causal discovery workflow...")
    result = run_causal_discovery_workflow(
        data_df=df,
        output_dir=result_dir,
        tau_max=TAU_MAX,
        alpha=ALPHA,
        sampling_days=1,
        date_col=None,
        method_config=METHOD_CONFIG.copy(),
        enable_consensus=True,
        enable_causal_audit=True,
        apply_audit_recommendation=True,
        true_edges=true_edges,
        undirected_eval=True,
        enable_preprocessing=True,
        enable_distribution_tests=True,
        enable_strength_analysis=False,
        enable_temporal_validation=False,
        enable_tracking=True,
    )

    # Extract tier edges
    tier_edges = extract_tier_edges(result_dir)
    if not tier_edges:
        logger.error("No tier edges found in workflow output")
        return 1

    for tier, edges in sorted(tier_edges.items()):
        logger.info(f"  Tier-{tier}: {len(edges)} edges")

    # Generate figure
    plot_tier_graphs(true_edges, tier_edges, var_names, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
