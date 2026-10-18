#!/usr/bin/env python3
"""
Reproduce Fig 8: Tier Example on DGP-Atlas F8 (dgp_002)
=========================================================
Generates the method-count support visualization for one DGP-Atlas instance
from the non-Gaussian F8 family (dgp_002).

Paper description (Fig 8):
  - 6 variables, 9 reference connections
  - Panel (a): reference graph, oriented in the generative lag-1 direction
  - Panel (b): Tier-1 links (>=3 methods): 8/9 reference, precision=0.89
  - Panel (c): Tier-2 links (=2 methods): 1/2 matches reference
  - Panel (d): Tier-3 links (=1 method): 0 match reference
  - True positives oriented in the generative direction (one colour)
  - False positives orange and undirected (no reference direction)
  - Missed reference edges grey dashed, in the generative direction

Edge directions are read from the DGP metadata true_graph, so the figure
renders without re-running the workflow. Pass --run-workflow to regenerate the
tier membership from a full discovery run instead.

Usage:
  # 1. Render the figure directly into the paper figures folder (PDF only).
  .venv/bin/python experiments/reproduce_fig8_tiers.py --data-dir /Volumes/EXTERNAL_M2/cv/autocause_resources/data/synthetic_dgp_atlas_v02 --output-dir "/Volumes/EXTERNAL_M2/cv/Papers/Paper 2 - AutoCause/figures"

  # 2. Preview the generated PDF as a PNG.
  pdftoppm -png -r 200 "/Volumes/EXTERNAL_M2/cv/Papers/Paper 2 - AutoCause/figures/fig8_causal_graph_example.pdf" /tmp/fig8_check

  # Optional: regenerate tier membership from a full workflow run.
  .venv/bin/python experiments/reproduce_fig8_tiers.py --data-dir /path/to/synthetic_dgp_atlas_v02 --output-dir experiments/figures --run-workflow

  # Validate the environment without rendering.
  .venv/bin/python experiments/reproduce_fig8_tiers.py --dry-run
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

from autocause.core.run_workflow import run_causal_discovery_workflow
from autocause.core.graph_metrics import binary_metrics_undirected

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

# --- Figure appearance -----------------------------------------------------
# Defaults match the other paper figures produced by experiments/make_figures.py:
# the same science/seaborn style sheet, sans-serif family, 9 pt text, and the
# node/edge weights previously hard-coded in plot_tier_graphs().
FIG_FONT_FAMILY = "sans-serif"  # None -> keep whatever the style sheet sets
FIG_FONT_SIZE = 16  # node labels, legend and panel subtitles
FIG_TITLE_FONT_WEIGHT = "bold"
# Node circle size: FIG_NODE_DIAMETER is the control (points); networkx takes
# the area (points^2), so FIG_NODE_SIZE is derived from the diameter.
# 28.77 pt reproduces the previous FIG_NODE_SIZE = 650.
FIG_NODE_DIAMETER = 35
FIG_NODE_SIZE = np.pi * (FIG_NODE_DIAMETER / 2) ** 2
FIG_NODE_LINE_WIDTH = 2.0
FIG_NODE_COLOR = "#eaecee"
FIG_NODE_BORDER_COLOR = "#2c3e50"
FIG_ARROW_SIZE = 32
FIG_EDGE_WIDTH_REFERENCE = 3.0
FIG_EDGE_WIDTH_TIER_TP = 3.0
FIG_EDGE_WIDTH_TIER_FP = 3.0
FIG_EDGE_WIDTH_MISSED = 2.8
FIG_EDGE_CURVATURE = 0.08  # connectionstyle "arc3,rad"
FIG_EDGE_MARGIN = 8  # min_source/target_margin: arrowhead-node separation
FIG_LAYOUT_SCALE = 1.0  # circular layout radius: node-to-node separation
FIG_PANEL_SIZE = (15, 4.0)
FIG_PANEL_WSPACE = None  # None -> tight_layout(); float -> subplots spacing
# Vertical gap between the subplots and the legend, in points. The legend is
# anchored to the bottom of the axes and hangs below it by exactly this pad.
FIG_LEGEND_PAD = 3
# Formats written to disk; "png" only if a raster preview is wanted.
FIG_OUTPUT_FORMATS = ("pdf",)
FIG_DPI = 400  # raster resolution, used only by non-PDF formats


def load_directed_reference(data_dir: Path) -> set:
    """Read the directed ground-truth edges for F8/dgp_002 from the metadata.

    The metadata stores ``true_graph`` as a parent list per node, so an entry
    ``"X1": ["X0", "X2", "X5"]`` encodes the directed edges X0->X1, X2->X1 and
    X5->X1. Self-loops are dropped. The returned set contains directed
    (parent, child) tuples and is the authoritative orientation for the figure.
    """
    meta_path = data_dir / "F8_non_gaussian" / "F8_non_gaussian_002_metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    directed = set()
    for child, parents in meta["true_graph"].items():
        for parent in parents:
            if parent != child:
                directed.add((parent, child))
    return directed


# Tier membership for F8/dgp_002, read from the released consensus output and
# cross-checked against the published figure (Tier-1: 8 TP + 1 FP; Tier-2:
# 1 TP + 1 FP; Tier-3: 0 TP + 4 FP). Edges are undirected pairs. True positives
# are oriented at plot time using load_directed_reference; false positives have
# no reference direction and are drawn without an arrowhead.
VERIFIED_TIERS = {
    1: {
        "tp": [
            ("X0", "X1"),
            ("X1", "X2"),
            ("X1", "X5"),
            ("X2", "X3"),
            ("X2", "X4"),
            ("X0", "X4"),
            ("X3", "X4"),
            ("X3", "X5"),
        ],
        "fp": [("X4", "X5")],
    },
    2: {
        "tp": [("X1", "X4")],
        "fp": [("X0", "X5")],
    },
    3: {
        "tp": [],
        "fp": [("X0", "X2"), ("X0", "X3"), ("X1", "X3"), ("X2", "X5")],
    },
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


def _orient(edge, directed_ref):
    """Return the reference-directed form(s) of an undirected tier edge.

    A true-positive edge is drawn in the generative direction taken from the
    metadata. When both directions are present in the reference (a lag-1
    two-cycle, here X1 and X2), both are returned so the edge is drawn with two
    arrowheads. A false-positive edge has no reference direction and is returned
    unchanged for undirected rendering.
    """
    a, b = edge
    forward = (a, b) in directed_ref
    backward = (b, a) in directed_ref
    if forward and backward:
        return [(a, b), (b, a)]
    if forward:
        return [(a, b)]
    if backward:
        return [(b, a)]
    return [(a, b)]


def plot_tier_graphs(
    directed_ref: set,
    tier_edges: dict,
    var_names: list,
    output_dir: Path,
):
    """Create the 4-panel tier visualization (Fig 8), with oriented edges.

    Panel (a) shows the directed reference. True-positive edges in the tier
    panels are oriented in the generative direction from the metadata, so the
    figure reports orientation wherever the reference identifies it. False
    positives carry no reference direction and are drawn as plain lines. The
    DGP-Atlas F8 reference is a lag-1 structure, so every reference edge is
    orientable by time precedence, and the X1--X2 pair is a two-cycle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    try:
        import scienceplots

        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    if FIG_FONT_FAMILY:
        plt.rcParams["font.family"] = FIG_FONT_FAMILY
    # Every text element of the figure uses FIG_FONT_SIZE.
    for rc_key in (
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ):
        plt.rcParams[rc_key] = FIG_FONT_SIZE

    node_size = FIG_NODE_SIZE
    arrow_kwargs = dict(
        arrows=True,
        arrowstyle="-|>",
        arrowsize=FIG_ARROW_SIZE,
        node_size=node_size,
        min_source_margin=FIG_EDGE_MARGIN,
        min_target_margin=FIG_EDGE_MARGIN,
    )

    subplots_kwargs = {"figsize": FIG_PANEL_SIZE}
    if FIG_PANEL_WSPACE is not None:
        subplots_kwargs["gridspec_kw"] = {"wspace": FIG_PANEL_WSPACE}
    fig, axes = plt.subplots(1, 4, **subplots_kwargs)

    # Fixed hexagonal layout so the panels match and arrowheads are legible.
    order = ["X0", "X5", "X4", "X3", "X2", "X1"]
    present = [v for v in order if v in var_names] + [
        v for v in var_names if v not in order
    ]
    G_layout = nx.DiGraph()
    G_layout.add_nodes_from(present)
    pos = nx.circular_layout(G_layout, scale=FIG_LAYOUT_SCALE)

    true_undirected = {frozenset(e) for e in directed_ref}

    def draw_nodes(ax):
        nx.draw_networkx_nodes(
            G_layout,
            pos,
            ax=ax,
            node_color=FIG_NODE_COLOR,
            edgecolors=FIG_NODE_BORDER_COLOR,
            linewidths=FIG_NODE_LINE_WIDTH,
            node_size=node_size,
        )
        nx.draw_networkx_labels(
            G_layout,
            pos,
            ax=ax,
            font_size=FIG_FONT_SIZE,
            font_weight=FIG_TITLE_FONT_WEIGHT,
        )

    # --- Panel (a): directed reference ---
    ax = axes[0]
    ax.set_title(
        "(a) Reference graph",
        fontweight=FIG_TITLE_FONT_WEIGHT,
        fontsize=FIG_FONT_SIZE,
    )
    draw_nodes(ax)
    G_ref = nx.DiGraph()
    G_ref.add_edges_from(directed_ref)
    nx.draw_networkx_edges(
        G_ref,
        pos,
        ax=ax,
        edge_color=FIG_NODE_BORDER_COLOR,
        width=FIG_EDGE_WIDTH_REFERENCE,
        connectionstyle=f"arc3,rad={FIG_EDGE_CURVATURE}",
        **arrow_kwargs,
    )
    ax.axis("off")

    # --- Panels (b)-(d): tier graphs ---
    tier_configs = [
        (1, "(b) Tier-1 ($\\geq$3 methods)", "#1a9850"),
        (2, "(c) Tier-2 (2 methods)", "#1a9850"),
        (3, "(d) Tier-3 (1 method)", "#1a9850"),
    ]

    for panel_idx, (tier, title, color) in enumerate(tier_configs):
        ax = axes[panel_idx + 1]
        entry = tier_edges.get(tier, {"tp": [], "fp": []})
        tp_edges = entry.get("tp", [])
        fp_edges = entry.get("fp", [])

        draw_nodes(ax)

        # True positives, oriented in the generative direction.
        tp_directed = []
        for e in tp_edges:
            tp_directed.extend(_orient(e, directed_ref))
        if tp_directed:
            G_tp = nx.DiGraph()
            G_tp.add_edges_from(tp_directed)
            nx.draw_networkx_edges(
                G_tp,
                pos,
                ax=ax,
                edge_color=color,
                width=FIG_EDGE_WIDTH_TIER_TP,
                connectionstyle=f"arc3,rad={FIG_EDGE_CURVATURE}",
                **arrow_kwargs,
            )

        # False positives, drawn without a direction.
        if fp_edges:
            G_fp = nx.DiGraph()
            G_fp.add_edges_from(fp_edges)
            nx.draw_networkx_edges(
                G_fp,
                pos,
                ax=ax,
                edge_color="#e67e22",
                width=FIG_EDGE_WIDTH_TIER_FP,
                arrows=True,
                arrowstyle="-",
                node_size=node_size,
                connectionstyle=f"arc3,rad={FIG_EDGE_CURVATURE}",
            )

        # Missed reference edges, grey dashed, undirected.
        discovered = {frozenset(e) for e in tp_edges} | {frozenset(e) for e in fp_edges}
        missed = true_undirected - discovered
        # Missed reference edges keep the generative direction, since the
        # reference defines it. Drawn grey dashed with an arrowhead.
        missed_directed = [e for e in directed_ref if frozenset(e) in missed]
        if missed_directed:
            G_missed = nx.DiGraph()
            G_missed.add_edges_from(missed_directed)
            nx.draw_networkx_edges(
                G_missed,
                pos,
                ax=ax,
                edge_color="#bdc3c7",
                width=FIG_EDGE_WIDTH_MISSED,
                style="dashed",
                arrows=True,
                arrowstyle="-|>",
                arrowsize=28,
                node_size=node_size,
                connectionstyle=f"arc3,rad={FIG_EDGE_CURVATURE}",
            )

        tp = len(tp_edges)
        fp = len(fp_edges)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ax.set_title(
            f"{title}\n{tp} TP, {fp} FP (prec {precision:.2f})",
            fontweight=FIG_TITLE_FONT_WEIGHT,
            fontsize=FIG_FONT_SIZE,
        )
        ax.axis("off")

    legend_elements = [
        mpatches.Patch(color=FIG_NODE_BORDER_COLOR, label="Reference edge"),
        mpatches.Patch(color="#1a9850", label="True positive"),
        mpatches.Patch(color="#e67e22", label="False positive"),
        mpatches.Patch(
            facecolor="white",
            edgecolor="#bdc3c7",
            linestyle="--",
            label="Missed reference edge",
        ),
    ]
    if FIG_PANEL_WSPACE is None:
        plt.tight_layout()

    # Anchor the legend to the bottom of the subplots, FIG_LEGEND_PAD points
    # below them (figure fraction -> points conversion uses the figure height).
    axes_bottom = axes[0].get_position().y0
    pad_fig = FIG_LEGEND_PAD / (72.0 * FIG_PANEL_SIZE[1])
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        fontsize=FIG_FONT_SIZE,
        bbox_to_anchor=(0.5, axes_bottom - pad_fig),
        frameon=False,
    )

    for fmt in FIG_OUTPUT_FORMATS:
        plt.savefig(
            output_dir / f"fig8_causal_graph_example.{fmt}",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
    plt.close()
    logger.info(f"  Fig 8 saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Fig 8: Tier example on DGP-Atlas F8/dgp_002"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/dgp_atlas"),
        help="Root directory containing DGP-Atlas datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures"),
        help="Output directory for the figure",
    )
    parser.add_argument(
        "--run-workflow",
        action="store_true",
        help="Regenerate tier membership from a full workflow run instead of "
        "rendering from the released tier membership.",
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
            from autocause.core.run_workflow import run_causal_discovery_workflow

            logger.info("  Imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    var_names = ["X0", "X1", "X2", "X3", "X4", "X5"]
    directed_ref = load_directed_reference(args.data_dir)
    logger.info(
        f"Directed reference from metadata: {len(directed_ref)} edges "
        f"({len({frozenset(e) for e in directed_ref})} undirected)"
    )

    if args.run_workflow:
        # Regenerate tier membership from a full workflow run, then render.
        logger.info(f"Loading {FAMILY}/dgp_{DATASET_IDX:03d}...")
        df, true_edges = load_dataset(args.data_dir)
        var_names = df.columns.tolist()
        result_dir = args.output_dir / "fig8_workdir"
        logger.info("Running causal discovery workflow...")
        run_causal_discovery_workflow(
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
        raw = extract_tier_edges(result_dir)
        if not raw:
            logger.error("No tier edges found in workflow output")
            return 1
        # Split each tier into TP and FP against the undirected reference.
        true_undirected = {frozenset(e) for e in directed_ref}
        tier_edges = {}
        for tier, edges in raw.items():
            tp = [e for e in edges if frozenset(e) in true_undirected]
            fp = [e for e in edges if frozenset(e) not in true_undirected]
            tier_edges[tier] = {"tp": tp, "fp": fp}
    else:
        # Default: render from the released tier membership, no workflow run.
        tier_edges = VERIFIED_TIERS

    for tier in sorted(tier_edges):
        entry = tier_edges[tier]
        logger.info(f"  Tier-{tier}: {len(entry['tp'])} TP, {len(entry['fp'])} FP")

    plot_tier_graphs(directed_ref, tier_edges, var_names, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
