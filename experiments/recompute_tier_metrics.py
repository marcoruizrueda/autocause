#!/usr/bin/env python3
"""
Recompute Consensus-Support Tier Metrics
==========================================
Computes precision per consensus tier per benchmark (Fig 3 / Section 4.3).

Tier definitions (Table 4 in paper):
  Tier-1: Detected by >= 3 of 4 causal methods (majority support)
  Tier-2: Detected by exactly 2 causal methods
  Tier-3: Detected by exactly 1 causal method

Reports:
  - Per-tier precision, recall, n_edges for each benchmark
  - Vote-threshold comparison (Table 7): any-2, majority-3, unanimous

Sources:
  - experiments/dgp_atlas/results/*/consensus/5-tiers/consensus_with_tiers.csv
  - experiments/timegraph_validation/results/*/consensus/5-tiers/consensus_with_tiers.csv
  - experiments/causalrivers_validation/results/*/consensus/5-tiers/consensus_with_tiers.csv

Usage:
  python experiments/recompute_tier_metrics.py --results-dir experiments/
  python experiments/recompute_tier_metrics.py --dry-run
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

from framework.core.graph_metrics import binary_metrics_undirected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_consensus_edges(result_dir: Path) -> pd.DataFrame:
    """Load consensus edges with tier labels from a single dataset result."""
    # Try multiple paths
    candidates = [
        result_dir / "consensus" / "5-tiers" / "consensus_with_tiers.csv",
        result_dir / "consensus_with_tiers.csv",
        result_dir / "ensemble_edges.csv",
    ]

    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    return None


def load_true_edges(result_dir: Path) -> set:
    """Load ground truth from experiment result directory."""
    candidates = [
        result_dir / "ground_truth.json",
        result_dir / "true_edges.json",
        result_dir / "experiment_log.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                data = json.load(f)
            if isinstance(data, list):
                return {tuple(e) for e in data}
            elif isinstance(data, dict):
                if "true_edges" in data:
                    return {tuple(e) for e in data["true_edges"]}
                if "edges" in data:
                    return {tuple(e) for e in data["edges"]}
    return set()


def compute_tier_precision(
    consensus_df: pd.DataFrame,
    true_edges: set,
    tier_col: str = "tier",
    src_col: str = "source",
    tgt_col: str = "target",
) -> dict:
    """Compute precision per tier against ground truth (undirected skeleton).

    Returns dict with tier -> {precision, n_edges, tp, fp}.
    """
    if consensus_df is None or true_edges is None or len(true_edges) == 0:
        return {}

    true_undirected = {frozenset(e) for e in true_edges}
    results = {}

    for tier in sorted(consensus_df[tier_col].unique()):
        tier_edges = consensus_df[consensus_df[tier_col] == tier]
        discovered = set()
        for _, row in tier_edges.iterrows():
            if src_col in row and tgt_col in row:
                discovered.add(frozenset([row[src_col], row[tgt_col]]))

        tp = len(discovered & true_undirected)
        fp = len(discovered - true_undirected)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results[tier] = {
            "precision": precision,
            "n_edges": len(discovered),
            "tp": tp,
            "fp": fp,
        }

    return results


def collect_benchmark_tiers(results_dir: Path, benchmark: str) -> pd.DataFrame:
    """Collect tier metrics across all datasets for a benchmark."""
    # Map benchmark to directory
    bench_dirs = {
        "DGP-Atlas": results_dir / "dgp_atlas" / "results",
        "TimeGraph": results_dir / "timegraph_validation" / "results",
        "CausalRivers": results_dir / "causalrivers_validation" / "results",
    }

    bench_dir = bench_dirs.get(benchmark)
    if bench_dir is None or not bench_dir.exists():
        return pd.DataFrame()

    # Aggregate edges across all datasets
    all_tier_edges = []

    for dataset_dir in sorted(bench_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_dir.name.endswith("_metrics") or dataset_dir.name.startswith("."):
            continue

        consensus_df = load_consensus_edges(dataset_dir)
        true_edges = load_true_edges(dataset_dir)

        if consensus_df is None or len(true_edges) == 0:
            continue

        # Add ground truth membership
        true_undirected = {frozenset(e) for e in true_edges}
        for _, row in consensus_df.iterrows():
            src = row.get("source", "")
            tgt = row.get("target", "")
            edge_fs = frozenset([src, tgt])
            all_tier_edges.append({
                "dataset": dataset_dir.name,
                "source": src,
                "target": tgt,
                "tier": row.get("tier", row.get("n_methods_found", 0)),
                "n_methods": row.get("n_methods_found", row.get("tier", 0)),
                "is_true": edge_fs in true_undirected,
            })

    return pd.DataFrame(all_tier_edges)


def compute_aggregate_tier_metrics(tier_edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate precision per tier from pooled edges."""
    if len(tier_edges_df) == 0:
        return pd.DataFrame()

    rows = []
    for tier in sorted(tier_edges_df["tier"].unique()):
        tier_data = tier_edges_df[tier_edges_df["tier"] == tier]
        tp = tier_data["is_true"].sum()
        total = len(tier_data)
        fp = total - tp
        precision = tp / total if total > 0 else 0.0

        rows.append({
            "tier": tier,
            "precision": precision,
            "n_edges": total,
            "tp": tp,
            "fp": fp,
        })

    return pd.DataFrame(rows)


def compute_vote_threshold_comparison(tier_edges_df: pd.DataFrame) -> pd.DataFrame:
    """Compare different vote thresholds (Table 7 in paper).

    Rules: any-2 (>=2), majority-3 (>=3), unanimous (all 4).
    """
    if len(tier_edges_df) == 0:
        return pd.DataFrame()

    rows = []
    thresholds = [
        ("Any two (>=2 methods)", 2),
        ("Majority-3 (>=3 methods)", 3),
        ("Unanimous (all 4 causal methods)", 4),
    ]

    for rule_name, min_methods in thresholds:
        admitted = tier_edges_df[tier_edges_df["n_methods"] >= min_methods]
        tp = admitted["is_true"].sum()
        total = len(admitted)
        fp = total - tp
        precision = tp / total if total > 0 else 0.0

        rows.append({
            "rule": rule_name,
            "min_methods": min_methods,
            "precision": precision,
            "n_edges": total,
            "tp": tp,
            "fp": fp,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Recompute consensus tier metrics for paper Fig 3 and Table 7"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("experiments"),
        help="Root experiments directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/results"),
        help="Output directory for tier metrics",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Tier metrics recomputation")
        logger.info(f"  Results directory: {args.results_dir}")
        logger.info(f"  Benchmarks: DGP-Atlas, TimeGraph, CausalRivers")
        logger.info(f"  Tier definitions: Tier-1 (>=3), Tier-2 (=2), Tier-3 (=1)")
        try:
            from framework.core.graph_metrics import binary_metrics_undirected
            logger.info("  Imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_tier_metrics = []

    for benchmark in ["DGP-Atlas", "TimeGraph", "CausalRivers"]:
        logger.info(f"\n{'='*70}")
        logger.info(f"TIER METRICS: {benchmark}")
        logger.info(f"{'='*70}")

        tier_edges = collect_benchmark_tiers(args.results_dir, benchmark)
        if len(tier_edges) == 0:
            logger.warning(f"  No tier data found for {benchmark}")
            continue

        # Aggregate precision per tier
        tier_metrics = compute_aggregate_tier_metrics(tier_edges)
        tier_metrics["benchmark"] = benchmark
        all_tier_metrics.append(tier_metrics)

        logger.info(f"\n  Tier precision ({benchmark}):")
        for _, row in tier_metrics.iterrows():
            logger.info(f"    Tier-{int(row['tier'])}: precision={row['precision']:.3f} (n={int(row['n_edges'])})")

        # Vote threshold comparison (Table 7)
        vote_comparison = compute_vote_threshold_comparison(tier_edges)
        vote_comparison.to_csv(
            args.output_dir / f"vote_threshold_{benchmark.lower().replace('-', '_')}.csv",
            index=False,
        )
        logger.info(f"\n  Vote threshold comparison ({benchmark}):")
        for _, row in vote_comparison.iterrows():
            logger.info(f"    {row['rule']}: precision={row['precision']:.3f} (n={int(row['n_edges'])})")

    # Save combined tier metrics
    if all_tier_metrics:
        combined = pd.concat(all_tier_metrics, ignore_index=True)
        combined.to_csv(args.output_dir / "tier_metrics_all.csv", index=False)
        logger.info(f"\nTier metrics saved to: {args.output_dir / 'tier_metrics_all.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
