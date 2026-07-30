#!/usr/bin/env python3
"""
Verify CausalRivers Tier Precision Inversion
==============================================
Reproduces the key finding from Section 4.3: on CausalRivers, majority-supported
Tier-1 links do NOT have the highest precision relative to the topology reference.

Paper values (Section 4.3):
  Tier-1 precision: 0.425
  Tier-2 precision: 0.667 (n=6)
  Tier-3 precision: 0.571 (n=7)

This script independently verifies the tier-precision inversion by recomputing
metrics from the CausalRivers experiment outputs.

Usage:
  python experiments/verify_causalrivers_tiers.py --results-dir experiments/causalrivers_validation/results
  python experiments/verify_causalrivers_tiers.py --dry-run
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expected paper values for verification
EXPECTED_TIER_PRECISION = {
    1: 0.425,
    2: 0.667,
    3: 0.571,
}


def load_subgraph_tiers(result_dir: Path) -> tuple:
    """Load consensus tiers and reference edges for a CausalRivers subgraph.

    Returns (consensus_df, true_edges_set)
    """
    # Load consensus edges
    consensus_candidates = [
        result_dir / "consensus" / "5-tiers" / "consensus_with_tiers.csv",
        result_dir / "consensus_with_tiers.csv",
        result_dir / "ensemble_edges.csv",
    ]
    consensus_df = None
    for c in consensus_candidates:
        if c.exists():
            consensus_df = pd.read_csv(c)
            break

    # Load ground truth
    gt_candidates = [
        result_dir / "ground_truth.json",
        result_dir / "true_edges.json",
        result_dir / "experiment_log.json",
    ]
    true_edges = set()
    for c in gt_candidates:
        if c.exists():
            with open(c) as f:
                data = json.load(f)
            if isinstance(data, list):
                true_edges = {tuple(e) for e in data}
            elif isinstance(data, dict):
                if "true_edges" in data:
                    true_edges = {tuple(e) for e in data["true_edges"]}
            break

    return consensus_df, true_edges


def verify_tier_inversion(results_dir: Path) -> dict:
    """Verify the tier-precision inversion on CausalRivers.

    Pools all consensus edges across 30 subgraphs and computes per-tier precision.
    """
    all_edges = []

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}

    for subgraph_dir in sorted(results_dir.iterdir()):
        if not subgraph_dir.is_dir():
            continue
        if subgraph_dir.name.startswith(".") or subgraph_dir.name.endswith(".csv"):
            continue

        consensus_df, true_edges = load_subgraph_tiers(subgraph_dir)
        if consensus_df is None or len(true_edges) == 0:
            continue

        true_undirected = {frozenset(e) for e in true_edges}

        for _, row in consensus_df.iterrows():
            src = row.get("source", "")
            tgt = row.get("target", "")
            edge = frozenset([src, tgt])
            tier = row.get("tier", row.get("n_methods_found", 0))

            # Map n_methods to tier: 1->3, 2->2, >=3->1
            if "n_methods_found" in row and "tier" not in consensus_df.columns:
                n = row["n_methods_found"]
                if n >= 3:
                    tier = 1
                elif n == 2:
                    tier = 2
                else:
                    tier = 3

            all_edges.append({
                "subgraph": subgraph_dir.name,
                "source": src,
                "target": tgt,
                "tier": tier,
                "is_true": edge in true_undirected,
            })

    if not all_edges:
        logger.warning("No tier data found")
        return {}

    edges_df = pd.DataFrame(all_edges)

    # Compute per-tier precision
    tier_results = {}
    for tier in sorted(edges_df["tier"].unique()):
        tier_data = edges_df[edges_df["tier"] == tier]
        tp = tier_data["is_true"].sum()
        total = len(tier_data)
        precision = tp / total if total > 0 else 0.0

        tier_results[tier] = {
            "precision": precision,
            "n_edges": total,
            "tp": tp,
            "fp": total - tp,
        }

    return tier_results


def main():
    parser = argparse.ArgumentParser(
        description="Verify CausalRivers tier-precision inversion (Section 4.3)"
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("experiments/causalrivers_validation/results"),
        help="CausalRivers experiment results directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: CausalRivers tier verification")
        logger.info(f"  Results directory: {args.results_dir}")
        logger.info(f"  Expected inversion: Tier-1 < Tier-2 and Tier-1 < Tier-3")
        logger.info(f"  Paper values: Tier-1={EXPECTED_TIER_PRECISION[1]:.3f}, "
                     f"Tier-2={EXPECTED_TIER_PRECISION[2]:.3f}, "
                     f"Tier-3={EXPECTED_TIER_PRECISION[3]:.3f}")
        try:
            from framework.core.graph_metrics import binary_metrics_undirected
            logger.info("  Imports OK")
        except ImportError as e:
            logger.error(f"  Import failed: {e}")
            return 1
        logger.info("  DRY RUN PASSED")
        return 0

    logger.info("Verifying CausalRivers tier-precision inversion...")
    tier_results = verify_tier_inversion(args.results_dir)

    if not tier_results:
        logger.error("Could not compute tier metrics. Run run_causalrivers.py first.")
        return 1

    logger.info(f"\n{'='*70}")
    logger.info("CAUSALRIVERS TIER PRECISION VERIFICATION")
    logger.info(f"{'='*70}")

    for tier in sorted(tier_results.keys()):
        r = tier_results[tier]
        expected = EXPECTED_TIER_PRECISION.get(tier, None)
        match_str = ""
        if expected is not None:
            diff = abs(r["precision"] - expected)
            match_str = f" (expected {expected:.3f}, diff={diff:.3f})"
        logger.info(
            f"  Tier-{tier}: precision={r['precision']:.3f}, n={r['n_edges']}"
            f"{match_str}"
        )

    # Check inversion
    if 1 in tier_results and 2 in tier_results:
        inverted = tier_results[1]["precision"] < tier_results[2]["precision"]
        logger.info(f"\n  Tier-1 < Tier-2: {inverted}")
        if inverted:
            logger.info("  CONFIRMED: Majority support does NOT select highest-precision tier")
        else:
            logger.info("  NOT CONFIRMED: Tier-1 has higher precision")

    # Save results
    output_path = args.results_dir / "tier_verification.json"
    with open(output_path, "w") as f:
        json.dump(tier_results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
