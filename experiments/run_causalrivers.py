#!/usr/bin/env python3
"""
CausalRivers Benchmark Experiment
====================================
Reproduces the CausalRivers evaluation from the AutoCause paper (Section 4).

Configuration:
- RiversBavaria: 494 gauging stations, water level at 15-min intervals (2019-2023)
- Resampled to 6-hour resolution
- Restricted to calendar year 2021 (~1460 observations per station)
- 30 five-station subgraphs (10 per topology class)
- tau_max = 5 (30-hour search window)
- alpha = 0.05
- Reference: directed river topology (upstream-to-downstream)

Topology classes:
  1. Random: randomly connected 5-station subgraphs
  2. Root-cause chains: longest directed path = 5 stations
  3. Confounder: at least one station with 2+ downstream stations

Evaluation:
- Primary: skeleton (undirected adjacency)
- Secondary: directed F1, identifiable CPDAG F1

Reference:
  Stein et al. (2025) CausalRivers
  https://github.com/CausalRivers/causalrivers

Usage:
  python experiments/run_causalrivers.py --data-dir data/causalrivers --output-dir experiments/causalrivers_validation/results
  python experiments/run_causalrivers.py --dry-run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.core.run_workflow import run_causal_discovery_workflow
from framework.core.graph_metrics import binary_metrics_undirected, binary_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration matching the paper (Section 4.1)
# ---------------------------------------------------------------------------

TAU_MAX = 5  # 30-hour search window at 6h resolution
ALPHA = 0.05
SAMPLING_DAYS = 0.25  # 6 hours = 0.25 days
RESAMPLE_FREQ = "6h"  # Resample from 15-min to 6-hour
YEAR = 2021  # Calendar year restriction
N_SUBGRAPHS_PER_CLASS = 10
N_STATIONS_PER_SUBGRAPH = 5

TOPOLOGY_CLASSES = ["random", "root_cause", "confounder"]

METHOD_CONFIG = {
    "granger": {"enabled": True},
    "transfer_entropy": {"enabled": True},
    "pcmci": {
        "enabled": True,
        "allow_contemporaneous": True,
    },
    "varlingam": {"enabled": True},
    "lpcmci": {"enabled": False},
    "correlation": {"enabled": True},
    "predictive_baseline": {"enabled": True},
}


def resample_to_6h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15-minute discharge data to 6-hour resolution using mean."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for resampling")
    return df.resample(RESAMPLE_FREQ).mean().dropna(how="all")


def restrict_to_year(df: pd.DataFrame, year: int = YEAR) -> pd.DataFrame:
    """Restrict data to a single calendar year."""
    mask = df.index.year == year
    return df[mask]


def sample_subgraphs(
    topology: Dict,
    n_per_class: int = N_SUBGRAPHS_PER_CLASS,
    n_stations: int = N_STATIONS_PER_SUBGRAPH,
    seed: int = 42,
) -> Dict[str, List[List[str]]]:
    """Sample subgraphs from the river topology by class.

    Parameters
    ----------
    topology : dict
        River topology with 'edges' (list of [upstream, downstream]) and
        'stations' (list of station IDs).

    Returns
    -------
    dict mapping topology class to list of subgraph station lists.
    """
    import networkx as nx

    rng = np.random.default_rng(seed)

    edges = topology.get("edges", [])
    G = nx.DiGraph()
    G.add_edges_from(edges)

    stations = list(G.nodes())
    subgraphs = {"random": [], "root_cause": [], "confounder": []}

    # --- Random subgraphs: connected 5-station subgraphs ---
    attempts = 0
    while len(subgraphs["random"]) < n_per_class and attempts < 1000:
        sample = rng.choice(stations, size=n_stations, replace=False).tolist()
        sub = G.subgraph(sample)
        if nx.is_weakly_connected(sub):
            subgraphs["random"].append(sample)
        attempts += 1

    # --- Root-cause chains: longest path = 5 stations ---
    all_paths = []
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                try:
                    for path in nx.all_simple_paths(G, source, target, cutoff=n_stations):
                        if len(path) == n_stations:
                            all_paths.append(path)
                except nx.NetworkXError:
                    pass
            if len(all_paths) > 500:
                break
        if len(all_paths) > 500:
            break

    if all_paths:
        rng.shuffle(all_paths)
        # Select non-overlapping paths
        used_stations = set()
        for path in all_paths:
            if len(subgraphs["root_cause"]) >= n_per_class:
                break
            if not set(path) & used_stations:
                subgraphs["root_cause"].append(list(path))
                used_stations.update(path)

    # --- Confounder subgraphs: at least one node with 2+ downstream ---
    confounders = [n for n in G.nodes() if G.out_degree(n) >= 2]
    attempts = 0
    used_stations = set()
    for conf_node in confounders:
        if len(subgraphs["confounder"]) >= n_per_class:
            break
        downstream = list(G.successors(conf_node))
        if len(downstream) < 2:
            continue
        # Build subgraph: confounder + 2 downstream + 2 more connected
        candidates = [conf_node] + downstream[:2]
        # Add neighbors to fill to 5
        for node in candidates[:]:
            for neighbor in list(G.predecessors(node)) + list(G.successors(node)):
                if neighbor not in candidates:
                    candidates.append(neighbor)
                if len(candidates) >= n_stations:
                    break
            if len(candidates) >= n_stations:
                break

        if len(candidates) >= n_stations:
            subgraph_stations = candidates[:n_stations]
            if not set(subgraph_stations) & used_stations:
                subgraphs["confounder"].append(subgraph_stations)
                used_stations.update(subgraph_stations)

    logger.info(
        f"Sampled subgraphs: random={len(subgraphs['random'])}, "
        f"root_cause={len(subgraphs['root_cause'])}, "
        f"confounder={len(subgraphs['confounder'])}"
    )
    return subgraphs


def get_reference_edges(
    topology: Dict, stations: List[str]
) -> Set[Tuple[str, str]]:
    """Extract reference edges for a subgraph from the river topology."""
    edges = topology.get("edges", [])
    station_set = set(stations)
    ref_edges = set()
    for edge in edges:
        src, tgt = edge[0], edge[1]
        if src in station_set and tgt in station_set:
            ref_edges.add((src, tgt))
    return ref_edges


def load_causalrivers_data(data_dir: Path) -> tuple:
    """Load CausalRivers discharge data and topology.

    Expected structure:
        data_dir/
            discharge/    (or data/)
                station_id.csv ...  (or single merged file)
            topology.json (or edges.json)

    Returns
    -------
    tuple of (pd.DataFrame with station columns, dict topology)
    """
    # Load topology
    topo_candidates = [
        data_dir / "topology.json",
        data_dir / "edges.json",
        data_dir / "graph.json",
        data_dir / "RiversBavaria" / "topology.json",
    ]
    topology = None
    for candidate in topo_candidates:
        if candidate.exists():
            with open(candidate) as f:
                topology = json.load(f)
            break

    if topology is None:
        raise FileNotFoundError(
            f"No topology file found. Searched: {[str(c) for c in topo_candidates]}"
        )

    # Load discharge data
    data_candidates = [
        data_dir / "discharge.parquet",
        data_dir / "discharge.csv",
        data_dir / "data.parquet",
        data_dir / "RiversBavaria" / "discharge.parquet",
    ]

    df = None
    for candidate in data_candidates:
        if candidate.exists():
            if candidate.suffix == ".parquet":
                df = pd.read_parquet(candidate)
            else:
                df = pd.read_csv(candidate, index_col=0, parse_dates=True)
            break

    # Try loading from individual station files
    if df is None:
        station_dirs = [
            data_dir / "discharge",
            data_dir / "data",
            data_dir / "stations",
            data_dir / "RiversBavaria" / "discharge",
        ]
        for station_dir in station_dirs:
            if station_dir.is_dir():
                station_files = sorted(station_dir.glob("*.csv"))
                if station_files:
                    frames = {}
                    for sf in station_files:
                        station_id = sf.stem
                        sdf = pd.read_csv(sf, index_col=0, parse_dates=True)
                        if len(sdf.columns) == 1:
                            frames[station_id] = sdf.iloc[:, 0]
                        else:
                            frames[station_id] = sdf.iloc[:, 0]
                    if frames:
                        df = pd.DataFrame(frames)
                    break

    if df is None:
        raise FileNotFoundError(
            f"No discharge data found in {data_dir}. "
            "Download from: https://github.com/CausalRivers/causalrivers"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    return df, topology


def run_single_subgraph(
    df_full: pd.DataFrame,
    topology: Dict,
    stations: List[str],
    output_dir: Path,
    topology_class: str,
    subgraph_idx: int,
) -> dict:
    """Run the workflow on a single 5-station CausalRivers subgraph."""
    subgraph_name = f"{topology_class}_{subgraph_idx:02d}"

    # Extract station data
    available = [s for s in stations if s in df_full.columns]
    if len(available) < N_STATIONS_PER_SUBGRAPH:
        logger.warning(
            f"  {subgraph_name}: only {len(available)}/{N_STATIONS_PER_SUBGRAPH} "
            f"stations available in data"
        )
        if len(available) < 3:
            return {"subgraph": subgraph_name, "error": "Insufficient stations"}

    df_sub = df_full[available].copy()

    # Resample to 6h and restrict to 2021
    df_sub = resample_to_6h(df_sub)
    df_sub = restrict_to_year(df_sub)
    df_sub = df_sub.dropna(how="all")

    if len(df_sub) < 100:
        return {"subgraph": subgraph_name, "error": f"Too few observations: {len(df_sub)}"}

    # Get reference edges for this subgraph
    true_edges = get_reference_edges(topology, available)

    logger.info(
        f"  {subgraph_name}: {len(available)} stations, {len(df_sub)} obs, "
        f"{len(true_edges)} reference edges"
    )

    start_time = time.time()

    result = run_causal_discovery_workflow(
        data_df=df_sub,
        output_dir=output_dir / subgraph_name,
        tau_max=TAU_MAX,
        alpha=ALPHA,
        sampling_days=SAMPLING_DAYS,
        date_col=None,  # Already has DatetimeIndex
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

    elapsed = time.time() - start_time

    metrics = {
        "subgraph": subgraph_name,
        "topology_class": topology_class,
        "n_stations": len(available),
        "n_obs": len(df_sub),
        "n_reference_edges": len(true_edges),
        "elapsed_seconds": elapsed,
    }

    # Extract per-method metrics
    metrics_path = output_dir / subgraph_name / "graph_recovery_metrics.csv"
    if metrics_path.exists():
        method_metrics = pd.read_csv(metrics_path)
        for _, row in method_metrics.iterrows():
            method = row.get("method", "unknown")
            metrics[f"{method}_f1"] = row.get("f1", np.nan)
            metrics[f"{method}_precision"] = row.get("precision", np.nan)
            metrics[f"{method}_recall"] = row.get("recall", np.nan)
            metrics[f"{method}_tpr"] = row.get("tpr", np.nan)
            metrics[f"{method}_fdr"] = row.get("fdr", np.nan)

    return metrics


def run_causalrivers(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Run the full CausalRivers benchmark.

    Parameters
    ----------
    data_dir : Path
        Root directory containing CausalRivers data.
    output_dir : Path
        Output directory for results.

    Returns
    -------
    pd.DataFrame with per-subgraph metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CausalRivers data...")
    df_full, topology = load_causalrivers_data(data_dir)
    logger.info(f"  Loaded: {df_full.shape[1]} stations, {len(df_full)} observations")
    logger.info(f"  Date range: {df_full.index.min()} to {df_full.index.max()}")

    # Sample subgraphs
    logger.info("Sampling subgraphs...")
    subgraphs = sample_subgraphs(topology, N_SUBGRAPHS_PER_CLASS, N_STATIONS_PER_SUBGRAPH)

    all_metrics = []
    total_start = time.time()

    for topo_class in TOPOLOGY_CLASSES:
        class_subgraphs = subgraphs.get(topo_class, [])
        logger.info(f"\n{'='*70}")
        logger.info(f"TOPOLOGY CLASS: {topo_class.upper()} ({len(class_subgraphs)} subgraphs)")
        logger.info(f"{'='*70}")

        for idx, stations in enumerate(class_subgraphs, 1):
            try:
                metrics = run_single_subgraph(
                    df_full, topology, stations, output_dir, topo_class, idx
                )
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"  FAILED {topo_class}_{idx:02d}: {e}")
                all_metrics.append({
                    "subgraph": f"{topo_class}_{idx:02d}",
                    "topology_class": topo_class,
                    "error": str(e),
                })

    total_elapsed = time.time() - total_start

    # Save results
    results_df = pd.DataFrame(all_metrics)
    results_df.to_csv(output_dir / "causalrivers_all_metrics.csv", index=False)

    # Save subgraph definitions for reproducibility
    with open(output_dir / "subgraph_definitions.json", "w") as f:
        json.dump(subgraphs, f, indent=2)

    # Compute summary by topology class (Fig 5 in paper)
    if len(results_df) > 0:
        summary = _compute_topology_summary(results_df)
        summary.to_csv(output_dir / "causalrivers_topology_summary.csv", index=False)
        logger.info(f"\n{'='*70}")
        logger.info("CAUSALRIVERS RESULTS SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\n{summary.to_string()}")

    logger.info(f"\nTotal runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Results saved to: {output_dir}")

    return results_df


def _compute_topology_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics per method per topology class."""
    methods = ["granger", "varlingam", "transfer_entropy", "pcmci", "correlation", "predictive_baseline"]
    rows = []

    for topo_class in TOPOLOGY_CLASSES:
        class_data = results_df[results_df["topology_class"] == topo_class]
        class_data = class_data[~class_data.get("error", pd.Series(dtype=str)).notna()]
        row = {"topology_class": topo_class, "n_subgraphs": len(class_data)}
        for method in methods:
            f1_col = f"{method}_f1"
            tpr_col = f"{method}_tpr"
            fdr_col = f"{method}_fdr"
            if f1_col in class_data.columns:
                row[f"{method}_f1_mean"] = class_data[f1_col].mean()
            if tpr_col in class_data.columns:
                row[f"{method}_tpr_mean"] = class_data[tpr_col].mean()
            if fdr_col in class_data.columns:
                row[f"{method}_fdr_mean"] = class_data[fdr_col].mean()
        rows.append(row)

    # Aggregate
    valid = results_df[~results_df.get("error", pd.Series(dtype=str)).notna()]
    agg = {"topology_class": "ALL", "n_subgraphs": len(valid)}
    for method in methods:
        f1_col = f"{method}_f1"
        if f1_col in valid.columns:
            agg[f"{method}_f1_mean"] = valid[f1_col].mean()
    rows.append(agg)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run CausalRivers benchmark for AutoCause paper reproduction"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/causalrivers"),
        help="Root directory containing CausalRivers data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/causalrivers_validation/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without running experiments",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: Validating CausalRivers experiment configuration")
        logger.info(f"  Data directory: {args.data_dir}")
        logger.info(f"  Output directory: {args.output_dir}")
        logger.info(f"  Topology classes: {TOPOLOGY_CLASSES}")
        logger.info(f"  Subgraphs per class: {N_SUBGRAPHS_PER_CLASS}")
        logger.info(f"  Stations per subgraph: {N_STATIONS_PER_SUBGRAPH}")
        logger.info(f"  tau_max: {TAU_MAX} ({TAU_MAX * SAMPLING_DAYS * 24:.0f}h window)")
        logger.info(f"  Resample: {RESAMPLE_FREQ}")
        logger.info(f"  Year: {YEAR}")
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
            contents = [p.name for p in args.data_dir.iterdir()]
            logger.info(f"  Data dir contents: {sorted(contents)}")
        else:
            logger.warning(f"  Data directory not found: {args.data_dir}")
            logger.info("  Download from: https://github.com/CausalRivers/causalrivers")

        logger.info("  DRY RUN PASSED")
        return 0

    results = run_causalrivers(args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
