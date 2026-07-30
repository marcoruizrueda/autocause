#!/usr/bin/env python3
"""
Paper Figure Generation
========================
Generates all figures for the AutoCause paper from experiment results.

Figures:
  Fig 2: Mean F1 per method per benchmark (bar chart with bootstrap CI)
  Fig 3: Consensus-support precision per benchmark
  Fig 4: TimeGraph F1 heatmap (18 categories x methods)
  Fig 5: CausalRivers TPR/FDR by topology class
  Fig 7: DGP-Atlas F1 heatmap (10 families x methods)
  Fig 8: Tier example on DGP-Atlas F8 dgp_002

Usage:
  python experiments/make_figures.py --results-dir experiments/ --output-dir experiments/figures
  python experiments/make_figures.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paper-style plotting configuration
METHODS_CAUSAL = ["granger", "varlingam", "transfer_entropy", "pcmci"]
METHODS_BASELINE = ["correlation", "predictive_baseline"]
ALL_METHODS = METHODS_CAUSAL + METHODS_BASELINE

METHOD_LABELS = {
    "granger": "VAR-Granger",
    "varlingam": "VARLiNGAM",
    "transfer_entropy": "Transfer entropy",
    "pcmci": "PCMCI+ (adaptive CI)",
    "correlation": "Lagged correlation",
    "predictive_baseline": "Random Forest",
}

METHOD_COLORS = {
    "granger": "#1f77b4",
    "varlingam": "#ff7f0e",
    "transfer_entropy": "#2ca02c",
    "pcmci": "#d62728",
    "correlation": "#9467bd",
    "predictive_baseline": "#8c564b",
}


def bootstrap_ci(values, n_bootstrap=2000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


def make_fig2_results_summary(results_dir: Path, output_dir: Path):
    """Fig 2: Mean F1 per method per benchmark with bootstrap 95% CI."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    benchmarks = {
        "DGP-Atlas": results_dir / "dgp_atlas" / "results" / "dgp_atlas_all_metrics.csv",
        "TimeGraph": results_dir / "timegraph_validation" / "results" / "timegraph_all_metrics.csv",
        "CausalRivers": results_dir / "causalrivers_validation" / "results" / "causalrivers_all_metrics.csv",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax_idx, (bench_name, csv_path) in enumerate(benchmarks.items()):
        ax = axes[ax_idx]

        if not csv_path.exists():
            ax.set_title(f"{bench_name}\n(no data)")
            continue

        df = pd.read_csv(csv_path)

        means, lowers, uppers = [], [], []
        labels = []
        colors = []

        for method in ALL_METHODS:
            col = f"{method}_f1"
            if col in df.columns:
                values = df[col].dropna().values
                mean, lower, upper = bootstrap_ci(values)
                means.append(mean)
                lowers.append(mean - lower)
                uppers.append(upper - mean)
                labels.append(METHOD_LABELS.get(method, method))
                colors.append(METHOD_COLORS.get(method, "#333333"))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=[lowers, uppers], capsize=3,
                      color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(bench_name, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Mean F1 (skeleton)")
    fig.suptitle("Fig 2: Mean F1 per method per benchmark (bootstrap 95% CI)", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_results_summary.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig2_results_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Fig 2 saved")


def make_fig3_evidence_tiering(results_dir: Path, output_dir: Path):
    """Fig 3: Consensus-support precision per benchmark."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    # Load tier metrics from recompute script outputs
    tier_file = results_dir / "results" / "tier_metrics_all.csv"
    if not tier_file.exists():
        logger.warning("  Fig 3: tier_metrics_all.csv not found. Run recompute_tier_metrics.py first.")
        return

    tier_df = pd.read_csv(tier_file)

    benchmarks = tier_df["benchmark"].unique()
    fig, axes = plt.subplots(1, len(benchmarks), figsize=(12, 4), sharey=True)
    if len(benchmarks) == 1:
        axes = [axes]

    for ax_idx, bench in enumerate(benchmarks):
        ax = axes[ax_idx]
        bdata = tier_df[tier_df["benchmark"] == bench]

        tiers = sorted(bdata["tier"].unique())
        precisions = [bdata[bdata["tier"] == t]["precision"].values[0] for t in tiers]
        n_edges = [bdata[bdata["tier"] == t]["n_edges"].values[0] for t in tiers]

        bars = ax.bar(range(len(tiers)), precisions, color=["#2ecc71", "#f39c12", "#e74c3c"][:len(tiers)])

        for i, (p, n) in enumerate(zip(precisions, n_edges)):
            ax.text(i, p + 0.02, f"n={n}", ha="center", fontsize=8)

        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels([f"Tier-{t}" for t in tiers])
        ax.set_title(bench, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("Precision")
    fig.suptitle("Fig 3: Consensus-support precision per benchmark", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "fig3_evidence_tiering.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig3_evidence_tiering.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Fig 3 saved")


def make_fig4_timegraph_heatmap(results_dir: Path, output_dir: Path):
    """Fig 4: F1 heatmap for all TimeGraph categories."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    csv_path = results_dir / "timegraph_validation" / "results" / "timegraph_all_metrics.csv"
    if not csv_path.exists():
        logger.warning("  Fig 4: TimeGraph results not found.")
        return

    df = pd.read_csv(csv_path)

    # Build heatmap matrix
    categories = df["category"].tolist()
    matrix = []
    for method in ALL_METHODS:
        col = f"{method}_f1"
        if col in df.columns:
            matrix.append(df[col].values)
        else:
            matrix.append(np.full(len(df), np.nan))

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(ALL_METHODS)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in ALL_METHODS], fontsize=9)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)

    # Add value annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="F1 (skeleton)")
    ax.set_title("Fig 4: TimeGraph F1 per method per category", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig4_timegraph_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig4_timegraph_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Fig 4 saved")


def make_fig5_causalrivers_fdr(results_dir: Path, output_dir: Path):
    """Fig 5: CausalRivers TPR (solid) and FDR (hatched) by topology class."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    csv_path = results_dir / "causalrivers_validation" / "results" / "causalrivers_all_metrics.csv"
    if not csv_path.exists():
        logger.warning("  Fig 5: CausalRivers results not found.")
        return

    df = pd.read_csv(csv_path)
    topology_classes = ["random", "root_cause", "confounder"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax_idx, topo_class in enumerate(topology_classes):
        ax = axes[ax_idx]
        class_data = df[df["topology_class"] == topo_class]

        methods_display = METHODS_CAUSAL + METHODS_BASELINE
        x = np.arange(len(methods_display))
        width = 0.35

        tpr_vals = []
        fdr_vals = []
        for method in methods_display:
            tpr_col = f"{method}_tpr"
            fdr_col = f"{method}_fdr"
            tpr_vals.append(class_data[tpr_col].mean() if tpr_col in class_data.columns else 0)
            fdr_vals.append(class_data[fdr_col].mean() if fdr_col in class_data.columns else 0)

        ax.bar(x - width/2, tpr_vals, width, label="TPR", alpha=0.85,
               color=[METHOD_COLORS.get(m, "#333") for m in methods_display])
        ax.bar(x + width/2, fdr_vals, width, label="FDR", alpha=0.5,
               color=[METHOD_COLORS.get(m, "#333") for m in methods_display],
               hatch="//", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m)[:10] for m in methods_display],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(topo_class.replace("_", " ").title(), fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Rate")
    fig.suptitle("Fig 5: CausalRivers skeleton TPR and graph FDR by topology class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_causalrivers_fdr.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig5_causalrivers_fdr.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Fig 5 saved")


def make_fig7_atlas_heatmap(results_dir: Path, output_dir: Path):
    """Fig 7: DGP-Atlas mean F1 heatmap (10 families x methods)."""
    import matplotlib.pyplot as plt

    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("seaborn-v0_8-paper")

    csv_path = results_dir / "dgp_atlas" / "results" / "dgp_atlas_all_metrics.csv"
    if not csv_path.exists():
        logger.warning("  Fig 7: DGP-Atlas results not found.")
        return

    df = pd.read_csv(csv_path)
    families = sorted(df["family"].unique())

    # Compute mean F1 per family per method
    matrix = []
    for method in ALL_METHODS:
        col = f"{method}_f1"
        row_vals = []
        for family in families:
            fam_data = df[df["family"] == family]
            if col in fam_data.columns:
                row_vals.append(fam_data[col].mean())
            else:
                row_vals.append(np.nan)
        matrix.append(row_vals)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_yticks(range(len(ALL_METHODS)))
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in ALL_METHODS], fontsize=9)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, fontsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Mean F1 (skeleton)")
    ax.set_title("Fig 7: DGP-Atlas mean F1 per method per family", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "fig7_atlas_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig7_atlas_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("  Fig 7 saved")


def main():
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments"),
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/figures"),
        help="Output directory for figures",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Figure generation validation")
        logger.info(f"  Results directory: {args.results_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            logger.info(f"  matplotlib version: {matplotlib.__version__}")
        except ImportError as e:
            logger.error(f"  matplotlib not available: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating paper figures...")
    make_fig2_results_summary(args.results_dir, args.output_dir)
    make_fig3_evidence_tiering(args.results_dir, args.output_dir)
    make_fig4_timegraph_heatmap(args.results_dir, args.output_dir)
    make_fig5_causalrivers_fdr(args.results_dir, args.output_dir)
    make_fig7_atlas_heatmap(args.results_dir, args.output_dir)

    logger.info(f"\nAll figures saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
