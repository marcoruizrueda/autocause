#!/usr/bin/env python3
"""
TimeGraph Benchmark Experiment
================================
Reproduces the TimeGraph evaluation from the AutoCause paper (Section 4).

Configuration:
- 18 categories (A1, A2, A1C, A2C, B1, B2, B1C, B2C, C1, C2, C1C, C2C, D1, D2, D1C, D2C, plus two additional)
- 4 variables per category
- T = 1000 samples per dataset
- True lag = 2 time steps
- tau_max = 5 (fixed)
- alpha = 0.05
- 4 causal methods + 2 non-causal baselines

Structural families:
  A: Linear structure
  B: Polynomial nonlinearity (x^2, x^3)
  C: Trends and seasonality on linear structure
  D: Missing data blocks
  Suffix 'C': Confounded (one driving variable removed from observed set)

Reference:
  Ferdous et al. (2025) TimeGraph benchmark
  https://github.com/hferdous/TimeGraph

Usage:
  python experiments/run_timegraph.py --data-dir data/timegraph --output-dir experiments/timegraph_validation/results
  python experiments/run_timegraph.py --dry-run
"""

import argparse
import json
import logging
import sys
import time
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

# ---------------------------------------------------------------------------
# Configuration matching the paper (Section 4.1)
# ---------------------------------------------------------------------------

TAU_MAX = 5
ALPHA = 0.05
SAMPLING_DAYS = 1

# TimeGraph categories: 4 structural families x variants
TIMEGRAPH_CATEGORIES = [
    "A1", "A2", "A1C", "A2C",
    "B1", "B2", "B1C", "B2C",
    "C1", "C2", "C1C", "C2C",
    "D1", "D2", "D1C", "D2C",
]

# Category properties from the paper
CATEGORY_INFO = {
    "A1": {"family": "linear", "noise": "Gaussian", "confounded": False},
    "A2": {"family": "linear", "noise": "Student-t", "confounded": False},
    "A1C": {"family": "linear", "noise": "Gaussian", "confounded": True},
    "A2C": {"family": "linear", "noise": "Student-t", "confounded": True},
    "B1": {"family": "nonlinear", "noise": "Gaussian", "confounded": False},
    "B2": {"family": "nonlinear", "noise": "Student-t", "confounded": False},
    "B1C": {"family": "nonlinear", "noise": "Gaussian", "confounded": True},
    "B2C": {"family": "nonlinear", "noise": "Student-t", "confounded": True},
    "C1": {"family": "trend_seasonal", "noise": "Gaussian", "confounded": False},
    "C2": {"family": "trend_seasonal", "noise": "Student-t", "confounded": False},
    "C1C": {"family": "trend_seasonal", "noise": "Gaussian", "confounded": True},
    "C2C": {"family": "trend_seasonal", "noise": "Student-t", "confounded": True},
    "D1": {"family": "missing_data", "noise": "Gaussian", "confounded": False},
    "D2": {"family": "missing_data", "noise": "Student-t", "confounded": False},
    "D1C": {"family": "missing_data", "noise": "Gaussian", "confounded": True},
    "D2C": {"family": "missing_data", "noise": "Student-t", "confounded": True},
}

# Full method configuration: 4 causal + 2 baselines
METHOD_CONFIG = {
    "granger": {"enabled": True},
    "transfer_entropy": {"enabled": True},
    "pcmci": {
        "enabled": True,
        "allow_contemporaneous": True,  # TimeGraph has contemporaneous edges
    },
    "varlingam": {"enabled": True},
    "lpcmci": {"enabled": False},  # Excluded due to wall-time constraint
    "correlation": {"enabled": True},
    "predictive_baseline": {"enabled": True},
}


def load_timegraph_dataset(data_dir: Path, category: str) -> tuple:
    """Load a TimeGraph category dataset and its ground truth.

    Expected directory structure:
        data_dir/
            A1/data.csv (or A1.csv)
            A1/ground_truth.json (or adjacency files)

    Returns
    -------
    tuple of (pd.DataFrame, set of (source, target) true edges)
    """
    # Try multiple directory structures
    cat_dir = data_dir / category

    data_candidates = [
        cat_dir / "data.csv",
        cat_dir / "series.csv",
        data_dir / f"{category}.csv",
        data_dir / f"{category}_data.csv",
    ]

    df = None
    for candidate in data_candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, index_col=0)
            break

    if df is None:
        raise FileNotFoundError(
            f"No data file found for category {category}. "
            f"Searched: {[str(c) for c in data_candidates]}"
        )

    # Load ground truth
    gt_candidates = [
        cat_dir / "ground_truth.json",
        cat_dir / "true_graph.json",
        cat_dir / "adjacency.json",
        data_dir / f"{category}_ground_truth.json",
        data_dir / "ground_truths" / f"{category}.json",
    ]

    true_edges = set()
    for candidate in gt_candidates:
        if candidate.exists():
            with open(candidate) as f:
                gt = json.load(f)
            if isinstance(gt, list):
                true_edges = {tuple(e) for e in gt}
            elif isinstance(gt, dict):
                if "edges" in gt:
                    true_edges = {tuple(e) for e in gt["edges"]}
                elif "adjacency_matrix" in gt:
                    mat = np.array(gt["adjacency_matrix"])
                    var_names = gt.get("var_names", df.columns.tolist())
                    for i in range(mat.shape[0]):
                        for j in range(mat.shape[1]):
                            if mat[i, j] != 0:
                                true_edges.add((var_names[i], var_names[j]))
            break

    if not true_edges:
        logger.warning(f"No ground truth found for category {category}")

    return df, true_edges


def run_single_category(
    df: pd.DataFrame,
    true_edges: set,
    output_dir: Path,
    category: str,
) -> dict:
    """Run the full workflow on a single TimeGraph category."""
    info = CATEGORY_INFO.get(category, {})
    logger.info(
        f"  Category {category}: {info.get('family', '?')}, "
        f"noise={info.get('noise', '?')}, confounded={info.get('confounded', '?')}"
    )
    logger.info(f"    {len(df)} obs, {len(df.columns)} vars, {len(true_edges)} true edges")

    start_time = time.time()

    # Determine if deseasonalization is needed (C categories)
    deseasonalize = info.get("family") == "trend_seasonal"

    result = run_causal_discovery_workflow(
        data_df=df,
        output_dir=output_dir / category,
        tau_max=TAU_MAX,
        alpha=ALPHA,
        sampling_days=SAMPLING_DAYS,
        date_col=None,
        method_config=METHOD_CONFIG.copy(),
        enable_consensus=True,
        enable_causal_audit=True,
        apply_audit_recommendation=True,
        true_edges=true_edges,
        undirected_eval=True,
        deseasonalize=deseasonalize,
        enable_preprocessing=True,
        enable_distribution_tests=True,
        enable_strength_analysis=False,
        enable_temporal_validation=False,
        enable_tracking=True,
    )

    elapsed = time.time() - start_time

    metrics = {
        "category": category,
        "family": info.get("family", "unknown"),
        "noise": info.get("noise", "unknown"),
        "confounded": info.get("confounded", False),
        "n_obs": len(df),
        "n_vars": len(df.columns),
        "n_true_edges": len(true_edges),
        "elapsed_seconds": elapsed,
    }

    # Extract per-method metrics
    metrics_path = output_dir / category / "graph_recovery_metrics.csv"
    if metrics_path.exists():
        method_metrics = pd.read_csv(metrics_path)
        for _, row in method_metrics.iterrows():
            method = row.get("method", "unknown")
            metrics[f"{method}_f1"] = row.get("f1", np.nan)
            metrics[f"{method}_precision"] = row.get("precision", np.nan)
            metrics[f"{method}_recall"] = row.get("recall", np.nan)
            metrics[f"{method}_tpr"] = row.get("tpr", np.nan)
            metrics[f"{method}_fdr"] = row.get("fdr", np.nan)
            metrics[f"{method}_shd"] = row.get("shd", np.nan)

    return metrics


def run_timegraph(data_dir: Path, output_dir: Path, categories: list = None) -> pd.DataFrame:
    """Run the full TimeGraph benchmark.

    Parameters
    ----------
    data_dir : Path
        Root directory containing TimeGraph datasets.
    output_dir : Path
        Output directory for results.
    categories : list, optional
        Subset of categories to run. Default: all 16.

    Returns
    -------
    pd.DataFrame with per-category metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = TIMEGRAPH_CATEGORIES

    all_metrics = []
    total_start = time.time()

    for category in categories:
        logger.info(f"\n{'='*70}")
        logger.info(f"TIMEGRAPH CATEGORY: {category}")
        logger.info(f"{'='*70}")

        try:
            df, true_edges = load_timegraph_dataset(data_dir, category)
            metrics = run_single_category(df, true_edges, output_dir, category)
            all_metrics.append(metrics)
        except FileNotFoundError as e:
            logger.warning(f"  Skipping {category}: {e}")
        except Exception as e:
            logger.error(f"  FAILED {category}: {e}")
            all_metrics.append({"category": category, "error": str(e)})

    total_elapsed = time.time() - total_start

    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "timegraph_all_metrics.csv", index=False)

    # Compute summary table (Fig 4 in paper)
    if len(results_df) > 0:
        summary = _compute_category_summary(results_df)
        summary.to_csv(output_dir / "timegraph_category_summary.csv", index=False)
        logger.info(f"\n{'='*70}")
        logger.info("TIMEGRAPH RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\n{summary.to_string()}")

    logger.info(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {output_dir}")

    return results_df


def _compute_category_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute F1 per method per category (Fig 4 heatmap in paper)."""
    methods = ["granger", "varlingam", "transfer_entropy", "pcmci", "correlation", "predictive_baseline"]
    rows = []

    for _, row in results_df.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            continue
        summary_row = {
            "category": row["category"],
            "family": row.get("family", ""),
            "noise": row.get("noise", ""),
            "confounded": row.get("confounded", False),
        }
        for method in methods:
            summary_row[f"{method}_f1"] = row.get(f"{method}_f1", np.nan)
            summary_row[f"{method}_tpr"] = row.get(f"{method}_tpr", np.nan)
            summary_row[f"{method}_fdr"] = row.get(f"{method}_fdr", np.nan)
        rows.append(summary_row)

    # Add aggregate
    agg = {"category": "MEAN", "family": "ALL", "noise": "", "confounded": ""}
    df_valid = results_df[~results_df.get("error", pd.Series(dtype=str)).notna()]
    for method in methods:
        col = f"{method}_f1"
        if col in df_valid.columns:
            agg[f"{method}_f1"] = df_valid[col].mean()
    rows.append(agg)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run TimeGraph benchmark for AutoCause paper reproduction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/timegraph"),
        help="Root directory containing TimeGraph datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/timegraph_validation/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Subset of categories to run (e.g., A1 B1 C1). Default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running experiments",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Validating TimeGraph experiment configuration")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Categories: {args.categories or TIMEGRAPH_CATEGORIES}")
        logger.info(f"  tau_max: {TAU_MAX}")
        logger.info(f"  alpha: {ALPHA}")
        logger.info(f"  Methods: {[k for k, v in METHOD_CONFIG.items() if v.get('enabled')]}")

        try:
            from framework.core.run_workflow import run_causal_discovery_workflow
            from framework.core.graph_metrics import binary_metrics_undirected
            logger.info("  All imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1

        if args.data_dir.exists():
            contents = [d.name for d in args.data_dir.iterdir()]
            logger.info(f"  Data dir contents: {sorted(contents)[:10]}...")
        else:
            logger.warning(f"  Data directory not found: {args.data_dir}")
            logger.info("  Download TimeGraph from: https://github.com/hferdous/TimeGraph")

        logger.info("  DRY RUN PASSED")
        return 0

    results = run_timegraph(args.data_dir, args.output_dir, args.categories)
    return 0


if __name__ == "__main__":
    sys.exit(main())
