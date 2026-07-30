#!/usr/bin/env python3
"""
DGP-Atlas Benchmark Experiment
================================
Reproduces the DGP-Atlas evaluation from the AutoCause paper (Section 4).

Configuration:
- 97 datasets across 10 families (F1-F10)
- tau_max = 5 (fixed across all methods)
- alpha = 0.05
- Benjamini-Hochberg FDR correction
- 4 causal methods: VAR-Granger, VARLiNGAM, Transfer Entropy, PCMCI+
- 2 non-causal baselines: Lagged Correlation, Random Forest
- Adaptive CI-test selection via pre-discovery diagnostics
- Audit-driven preprocessing (interpolation for F3/F9, deseasonalization for F6)

Datasets:
- DGP-Atlas (Ruiz 2026): 10 families of VAR(1) processes
  F1: Clean VAR (7 vars)
  F2: Structural breaks (6 vars)
  F3: Irregular sampling (5 vars)
  F4: High persistence (5 vars)
  F5: Latent confounders (8 vars)
  F6: Seasonality (7 vars)
  F7: Polynomial nonlinearity (8 vars) — 3 explosive excluded → 7 datasets
  F8: Non-Gaussian noise (5 vars)
  F9: Mixed violations (6 vars)
  F10: Extreme cases (6 vars)

Reference:
  https://zenodo.org/records/19409395

Usage:
  python experiments/run_dgp_atlas.py --data-dir data/dgp_atlas --output-dir experiments/dgp_atlas/results
  python experiments/run_dgp_atlas.py --dry-run   # validate setup without running
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
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

# Fixed parameters across all DGP-Atlas experiments
TAU_MAX = 5
ALPHA = 0.05
SAMPLING_DAYS = 1  # daily resolution in synthetic data

# 10 DGP-Atlas families with their properties
DGP_FAMILIES = {
    "F1": {"name": "clean_var", "n_vars": 7, "n_datasets": 10, "description": "Clean VAR(1)"},
    "F2": {"name": "structural_breaks", "n_vars": 6, "n_datasets": 10, "description": "Structural breaks"},
    "F3": {"name": "irregular_sampling", "n_vars": 5, "n_datasets": 10, "description": "Irregular sampling"},
    "F4": {"name": "high_persistence", "n_vars": 5, "n_datasets": 10, "description": "High persistence"},
    "F5": {"name": "latent_confounders", "n_vars": 8, "n_datasets": 10, "description": "Latent confounders"},
    "F6": {"name": "seasonality", "n_vars": 7, "n_datasets": 10, "description": "Seasonality"},
    "F7": {"name": "nonlinear", "n_vars": 8, "n_datasets": 7, "description": "Polynomial nonlinearity (3 explosive excluded)"},
    "F8": {"name": "non_gaussian", "n_vars": 5, "n_datasets": 10, "description": "Non-Gaussian noise"},
    "F9": {"name": "mixed_violations", "n_vars": 6, "n_datasets": 10, "description": "Mixed violations"},
    "F10": {"name": "extreme_cases", "n_vars": 6, "n_datasets": 10, "description": "Extreme cases"},
}

# Method configuration: all 4 causal methods + 2 baselines enabled
METHOD_CONFIG = {
    "granger": {"enabled": True},
    "transfer_entropy": {"enabled": True},
    "pcmci": {
        "enabled": True,
        "allow_contemporaneous": False,  # DGP-Atlas has no contemporaneous edges
    },
    "varlingam": {"enabled": True},
    "lpcmci": {"enabled": False},  # Excluded due to wall-time constraint
    "correlation": {"enabled": True},  # Non-causal baseline
    "predictive_baseline": {"enabled": True},  # Non-causal baseline
}


def load_dgp_dataset(data_dir: Path, family: str, dataset_idx: int) -> tuple:
    """Load a single DGP-Atlas dataset and its ground truth.

    Expected directory structure:
        data_dir/
            F1/dgp_001/data.csv
            F1/dgp_001/ground_truth.json
            ...

    Returns
    -------
    tuple of (pd.DataFrame, set)
        Data and true edges as set of (source, target) tuples.
    """
    family_dir = data_dir / family
    dataset_name = f"dgp_{dataset_idx:03d}"
    dataset_dir = family_dir / dataset_name

    # Try multiple naming conventions
    data_candidates = [
        dataset_dir / "data.csv",
        dataset_dir / "series.csv",
        family_dir / f"{dataset_name}.csv",
        family_dir / f"data_{dataset_idx:03d}.csv",
    ]

    df = None
    for candidate in data_candidates:
        if candidate.exists():
            df = pd.read_csv(candidate, index_col=0, parse_dates=True)
            break

    if df is None:
        raise FileNotFoundError(
            f"No data file found for {family}/{dataset_name}. "
            f"Searched: {[str(c) for c in data_candidates]}"
        )

    # Load ground truth
    gt_candidates = [
        dataset_dir / "ground_truth.json",
        dataset_dir / "true_graph.json",
        family_dir / f"ground_truth_{dataset_idx:03d}.json",
        family_dir / "ground_truth.json",
    ]

    true_edges = set()
    for candidate in gt_candidates:
        if candidate.exists():
            with open(candidate) as f:
                gt = json.load(f)
            # Accept different formats
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
        logger.warning(f"No ground truth found for {family}/{dataset_name}")

    return df, true_edges


def run_single_dataset(
    df: pd.DataFrame,
    true_edges: set,
    output_dir: Path,
    family: str,
    dataset_name: str,
) -> dict:
    """Run the full workflow on a single DGP-Atlas dataset.

    Returns
    -------
    dict with per-method and consensus metrics.
    """
    logger.info(f"  Running {family}/{dataset_name} ({len(df)} obs, {len(df.columns)} vars)")

    start_time = time.time()

    result = run_causal_discovery_workflow(
        data_df=df,
        output_dir=output_dir / family / dataset_name,
        tau_max=TAU_MAX,
        alpha=ALPHA,
        sampling_days=SAMPLING_DAYS,
        date_col=None,  # Synthetic data uses integer index
        method_config=METHOD_CONFIG.copy(),
        enable_consensus=True,
        enable_causal_audit=True,
        apply_audit_recommendation=True,
        true_edges=true_edges,
        undirected_eval=True,  # Paper evaluates skeleton (undirected)
        enable_preprocessing=True,
        enable_distribution_tests=True,
        enable_strength_analysis=False,  # Not needed for benchmark metrics
        enable_temporal_validation=False,  # Not needed for benchmark metrics
        enable_tracking=True,
    )

    elapsed = time.time() - start_time

    # Collect metrics from the result
    metrics = {
        "family": family,
        "dataset": dataset_name,
        "n_obs": len(df),
        "n_vars": len(df.columns),
        "n_true_edges": len(true_edges),
        "elapsed_seconds": elapsed,
    }

    # Extract per-method metrics from saved graph_recovery_metrics.csv
    metrics_path = output_dir / family / dataset_name / "graph_recovery_metrics.csv"
    if metrics_path.exists():
        method_metrics = pd.read_csv(metrics_path)
        for _, row in method_metrics.iterrows():
            method = row.get("method", "unknown")
            metrics[f"{method}_f1"] = row.get("f1", np.nan)
            metrics[f"{method}_precision"] = row.get("precision", np.nan)
            metrics[f"{method}_recall"] = row.get("recall", np.nan)
            metrics[f"{method}_fdr"] = row.get("fdr", np.nan)
            metrics[f"{method}_shd"] = row.get("shd", np.nan)

    return metrics


def run_dgp_atlas(data_dir: Path, output_dir: Path, families: list = None) -> pd.DataFrame:
    """Run the full DGP-Atlas benchmark.

    Parameters
    ----------
    data_dir : Path
        Root directory containing DGP-Atlas datasets.
    output_dir : Path
        Output directory for results.
    families : list, optional
        Subset of families to run (e.g., ["F1", "F8"]). Default: all.

    Returns
    -------
    pd.DataFrame with per-dataset metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if families is None:
        families = list(DGP_FAMILIES.keys())

    all_metrics = []
    total_start = time.time()

    for family in families:
        family_info = DGP_FAMILIES[family]
        n_datasets = family_info["n_datasets"]
        logger.info(f"\n{'='*70}")
        logger.info(f"FAMILY {family}: {family_info['description']} ({n_datasets} datasets)")
        logger.info(f"{'='*70}")

        for idx in range(1, n_datasets + 1):
            dataset_name = f"dgp_{idx:03d}"
            try:
                df, true_edges = load_dgp_dataset(data_dir, family, idx)
                metrics = run_single_dataset(
                    df, true_edges, output_dir, family, dataset_name
                )
                all_metrics.append(metrics)
            except FileNotFoundError as e:
                logger.warning(f"  Skipping {family}/{dataset_name}: {e}")
            except Exception as e:
                logger.error(f"  FAILED {family}/{dataset_name}: {e}")
                all_metrics.append({
                    "family": family,
                    "dataset": dataset_name,
                    "error": str(e),
                })

    total_elapsed = time.time() - total_start

    # Save aggregated results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "dgp_atlas_all_metrics.csv", index=False)

    # Compute per-family summary (Table in paper)
    if len(results_df) > 0 and "error" not in results_df.columns:
        summary = _compute_family_summary(results_df)
        summary.to_csv(output_dir / "dgp_atlas_family_summary.csv", index=False)
        logger.info(f"\n{'='*70}")
        logger.info("DGP-ATLAS RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\n{summary.to_string()}")

    logger.info(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {output_dir}")

    return results_df


def _compute_family_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean F1 per method per family (Table 3 / Fig 2 in paper)."""
    methods = ["granger", "varlingam", "transfer_entropy", "pcmci", "correlation", "predictive_baseline"]
    rows = []

    for family in sorted(results_df["family"].unique()):
        family_data = results_df[results_df["family"] == family]
        row = {"family": family, "n_datasets": len(family_data)}
        for method in methods:
            col = f"{method}_f1"
            if col in family_data.columns:
                row[f"{method}_f1_mean"] = family_data[col].mean()
                row[f"{method}_f1_std"] = family_data[col].std()
        rows.append(row)

    # Add aggregate row
    agg_row = {"family": "ALL", "n_datasets": len(results_df)}
    for method in methods:
        col = f"{method}_f1"
        if col in results_df.columns:
            agg_row[f"{method}_f1_mean"] = results_df[col].mean()
            agg_row[f"{method}_f1_std"] = results_df[col].std()
    rows.append(agg_row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run DGP-Atlas benchmark for AutoCause paper reproduction"
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
        default=Path("experiments/dgp_atlas/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Subset of families to run (e.g., F1 F8). Default: all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running experiments",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Validating DGP-Atlas experiment configuration")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Families: {args.families or list(DGP_FAMILIES.keys())}")
        logger.info(f"  tau_max: {TAU_MAX}")
        logger.info(f"  alpha: {ALPHA}")
        logger.info(f"  Methods: {[k for k, v in METHOD_CONFIG.items() if v.get('enabled')]}")
        logger.info(f"  Total datasets: {sum(f['n_datasets'] for f in DGP_FAMILIES.values())}")

        # Validate imports
        try:
            from framework.core.run_workflow import run_causal_discovery_workflow
            from framework.core.graph_metrics import binary_metrics_undirected
            logger.info("  All imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1

        # Check data directory
        if args.data_dir.exists():
            families_found = [d.name for d in args.data_dir.iterdir() if d.is_dir()]
            logger.info(f"  Families found in data dir: {families_found}")
        else:
            logger.warning(f"  Data directory not found: {args.data_dir}")
            logger.info("  Download DGP-Atlas from: https://zenodo.org/records/19409395")

        logger.info("  DRY RUN PASSED")
        return 0

    results = run_dgp_atlas(args.data_dir, args.output_dir, args.families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
