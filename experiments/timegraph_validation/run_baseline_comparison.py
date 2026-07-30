#!/usr/bin/env python3
"""
TimeGraph Baseline Comparison (Table 5 in paper)
==================================================
Compares AutoCause PCMCI+ results with published values from Ferdous et al. (2025)
on 10 selected TimeGraph rows.

This is a DIAGNOSTIC comparison, not a component ablation. The two pipelines
differ in preprocessing, graph-matching conventions, and multiple-testing procedures.

Comparison rows (from paper Table 5):
  A1 Gaussian, A1 Student-t, B1 Gaussian, B1 Student-t, C1 Gaussian,
  A1C Gaussian, A1C Student-t, B1C Gaussian, B1C Student-t, C1C Gaussian

Configuration:
- PCMCI+ with ParCorr (aligned with published baseline)
- tau_max = 2 (matching TimeGraph true lag)
- alpha = 0.05
- Benjamini-Hochberg FDR correction

Usage:
  python experiments/timegraph_validation/run_baseline_comparison.py --data-dir data/timegraph
  python experiments/timegraph_validation/run_baseline_comparison.py --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.core.run_workflow import run_causal_discovery_workflow
from framework.core.graph_metrics import binary_metrics_undirected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published PCMCI+ values from Ferdous et al. (2025) - Table 5 in paper
# ---------------------------------------------------------------------------

PUBLISHED_RESULTS = {
    ("A1", "Gaussian"): {"tpr": 1.00, "fdr": 0.00},
    ("A1", "Student-t"): {"tpr": 0.67, "fdr": 0.33},
    ("B1", "Gaussian"): {"tpr": 0.00, "fdr": 1.00},
    ("B1", "Student-t"): {"tpr": 0.33, "fdr": 0.93},
    ("C1", "Gaussian"): {"tpr": 0.00, "fdr": 1.00},
    ("A1C", "Gaussian"): {"tpr": 0.67, "fdr": 0.50},
    ("A1C", "Student-t"): {"tpr": 0.67, "fdr": 0.50},
    ("B1C", "Gaussian"): {"tpr": 0.22, "fdr": 0.00},
    ("B1C", "Student-t"): {"tpr": 0.33, "fdr": 0.94},
    ("C1C", "Gaussian"): {"tpr": 0.33, "fdr": 0.79},
}

# Comparison categories
COMPARISON_CATEGORIES = [
    ("A1", "Gaussian"),
    ("A1", "Student-t"),
    ("B1", "Gaussian"),
    ("B1", "Student-t"),
    ("C1", "Gaussian"),
    ("A1C", "Gaussian"),
    ("A1C", "Student-t"),
    ("B1C", "Gaussian"),
    ("B1C", "Student-t"),
    ("C1C", "Gaussian"),
]

# Configuration aligned with published baseline
METHOD_CONFIG_PCMCI_ONLY = {
    "granger": {"enabled": False},
    "transfer_entropy": {"enabled": False},
    "pcmci": {
        "enabled": True,
        "test_method": "parcorr",  # Fixed ParCorr for comparison
        "allow_contemporaneous": True,
    },
    "varlingam": {"enabled": False},
    "lpcmci": {"enabled": False},
    "correlation": {"enabled": False},
    "predictive_baseline": {"enabled": False},
}


def load_comparison_dataset(data_dir: Path, category: str, noise: str) -> tuple:
    """Load a specific TimeGraph dataset for the baseline comparison.

    The noise variant may be encoded in the filename or subdirectory.
    """
    # Try different naming patterns
    if noise == "Student-t":
        noise_suffix = "_t"
        noise_dir = "student_t"
    else:
        noise_suffix = "_gaussian"
        noise_dir = "gaussian"

    candidates = [
        data_dir / category / noise_dir / "data.csv",
        data_dir / category / f"data{noise_suffix}.csv",
        data_dir / f"{category}{noise_suffix}.csv",
        data_dir / category / "data.csv",  # Single noise variant per dir
    ]

    df = None
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c, index_col=0)
            break

    if df is None:
        raise FileNotFoundError(f"Data not found for {category} ({noise})")

    # Load ground truth (same for both noise variants within a category)
    gt_candidates = [
        data_dir / category / "ground_truth.json",
        data_dir / category / noise_dir / "ground_truth.json",
        data_dir / "ground_truths" / f"{category}.json",
    ]

    true_edges = set()
    for c in gt_candidates:
        if c.exists():
            with open(c) as f:
                gt = json.load(f)
            if isinstance(gt, list):
                true_edges = {tuple(e) for e in gt}
            elif isinstance(gt, dict) and "edges" in gt:
                true_edges = {tuple(e) for e in gt["edges"]}
            break

    return df, true_edges


def run_baseline_comparison(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Run the baseline comparison and produce Table 5."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for category, noise in COMPARISON_CATEGORIES:
        logger.info(f"\n--- {category} ({noise}) ---")

        try:
            df, true_edges = load_comparison_dataset(data_dir, category, noise)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping: {e}")
            rows.append({
                "category": category,
                "noise": noise,
                "error": str(e),
            })
            continue

        # Determine if deseasonalization applies (C categories)
        deseasonalize = category.startswith("C")

        result = run_causal_discovery_workflow(
            data_df=df,
            output_dir=output_dir / f"{category}_{noise.replace('-', '')}",
            tau_max=5,  # Paper uses tau_max=5
            alpha=0.05,
            sampling_days=1,
            date_col=None,
            method_config=METHOD_CONFIG_PCMCI_ONLY.copy(),
            enable_consensus=False,
            enable_causal_audit=True,
            apply_audit_recommendation=False,  # Use fixed ParCorr
            true_edges=true_edges,
            undirected_eval=True,
            deseasonalize=deseasonalize,
            enable_preprocessing=True,
            enable_distribution_tests=False,
            enable_strength_analysis=False,
            enable_temporal_validation=False,
            enable_tracking=True,
        )

        # Get AutoCause PCMCI+ metrics
        metrics_path = output_dir / f"{category}_{noise.replace('-', '')}" / "graph_recovery_metrics.csv"
        ac_tpr, ac_fdr = np.nan, np.nan
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
            pcmci_row = mdf[mdf["method"] == "pcmci"]
            if len(pcmci_row) > 0:
                ac_tpr = pcmci_row.iloc[0].get("tpr", pcmci_row.iloc[0].get("recall", np.nan))
                ac_fdr = pcmci_row.iloc[0].get("fdr", np.nan)

        pub = PUBLISHED_RESULTS.get((category, noise), {})

        row = {
            "category": category,
            "noise": noise,
            "published_tpr": pub.get("tpr", np.nan),
            "published_fdr": pub.get("fdr", np.nan),
            "autocause_tpr": ac_tpr,
            "autocause_fdr": ac_fdr,
        }
        rows.append(row)

        logger.info(
            f"  Published: TPR={pub.get('tpr', '?'):.2f}, FDR={pub.get('fdr', '?'):.2f}"
        )
        logger.info(f"  AutoCause: TPR={ac_tpr:.2f}, FDR={ac_fdr:.2f}")

    # Save comparison table
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    # Print formatted table (matches Table 5 in paper)
    logger.info(f"\n{'='*70}")
    logger.info("BASELINE COMPARISON (Table 5)")
    logger.info(f"{'='*70}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")

    # Compute mean values
    valid = comparison_df.dropna(subset=["autocause_tpr", "autocause_fdr"])
    if len(valid) > 0:
        logger.info(f"\nMean Published:  TPR={valid['published_tpr'].mean():.2f}, FDR={valid['published_fdr'].mean():.2f}")
        logger.info(f"Mean AutoCause:  TPR={valid['autocause_tpr'].mean():.2f}, FDR={valid['autocause_fdr'].mean():.2f}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="TimeGraph baseline comparison (Table 5 in paper)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/timegraph"),
        help="Root directory containing TimeGraph datasets",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/timegraph_validation/baseline_comparison"),
        help="Output directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Baseline comparison configuration")
        logger.info(f"  Comparison rows: {len(COMPARISON_CATEGORIES)}")
        logger.info(f"  Published values available: {len(PUBLISHED_RESULTS)}")
        logger.info(f"  Method: PCMCI+ with ParCorr (fixed)")
        try:
            from framework.core.run_workflow import run_causal_discovery_workflow
            logger.info("  Imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    run_baseline_comparison(args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
