"""
Consensus Causal Discovery

Identifies robust causal relationships by detecting agreement across
multiple causal discovery methods (VAR-based Granger, transfer entropy, PCMCI+).

A consensus causal edge is detected when:
1. At least min_votes methods agree on direction (source → target)
2. Time lags are within lag_tolerance_steps
3. Significance determined by q-value (FDR-corrected) when available, else p-value < alpha
4. Median lag used for consensus (robust to outliers)
5. Vote breakdown tracks which methods agreed

Paradigm diversity: Methods are grouped into paradigm families (Assaad et al.,
2022). Agreement within a family (e.g., Granger + VARLiNGAM, both linear) is
less informative than agreement across families. The paradigm_diversity_score
counts the number of distinct paradigm families that agree, providing a more
conservative measure of robustness than raw vote count.

Outputs:
- Consensus CSV: source, target, lag_steps, lag_days, vote_count, agreeing_methods,
  best_q_value, best_p_value, all_significant, paradigm_diversity
- GraphML: Directed graph with edge attributes (weight=-log10(q), lag_steps, lag_days,
  methods, vote_count)

References:
    - Assaad, C. K., Devijver, E., & Gaussier, E. (2022). Survey and evaluation
      of causal discovery methods for time series. JAIR, 73, 767-819.
    - Pfaff, B., Stigler, M., & Svensson, A. (2013). Testing for causality between
      two variables in a multivariate VAR framework. Journal of Applied Econometrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Paradigm family mapping (Assaad et al. 2022 taxonomy).
# Agreement across families is more informative than within a family.
METHOD_PARADIGM = {
    "Granger": "regression",
    "VARLiNGAM": "regression",  # Both are linear regression-based
    "TransferEntropy": "information_theoretic",
    "PCMCI+": "constraint_based",
    "LPCMCI": "constraint_based",  # Extension of PCMCI for latent confounders
    "RF": "predictive",  # Non-causal baseline
}


def compute_paradigm_diversity(methods: list) -> int:
    """
    Count the number of distinct paradigm families represented.

    Agreement from methods in different paradigm families provides stronger
    evidence than agreement from methods in the same family (e.g., Granger
    and VARLiNGAM are both linear regression-based).

    Parameters
    ----------
    methods : list of str
        Method names (e.g., ["Granger", "PCMCI+", "TransferEntropy"])

    Returns
    -------
    int
        Number of distinct paradigm families (1 to 4).
    """
    families = set()
    for method in methods:
        family = METHOD_PARADIGM.get(method, "unknown")
        families.add(family)
    return len(families)


def harmonize_lag_windows(lag1: int, lag2: int, lag_window: int = 1) -> bool:
    """
    Check if two lags are within tolerance window.

    Parameters:
        lag1 (int): First lag estimate
        lag2 (int): Second lag estimate
        lag_window (int): Maximum lag difference (days or weeks)

    Returns:
        bool: True if lags are within window
    """
    return abs(lag1 - lag2) <= lag_window


def detect_agreement(
    granger_results: pd.DataFrame,
    te_results: pd.DataFrame,
    pcmci_results: pd.DataFrame,
    lag_tolerance_steps: int = 1,
    min_votes: int = 2,
    alpha: float = 0.05,
    sampling_days: int = 1,
    use_lag_bands: bool = False,
) -> pd.DataFrame:
    """
    Detect consensus across causal inference methods with q-value support.

    Combines results from Granger, Transfer Entropy, and PCMCI+ to identify
    robust causal relationships where multiple methods agree.

    Parameters:
        granger_results (pd.DataFrame): Granger causality results
            Required: source, target, lag_steps (or best_lag), p_value, q_value (optional)
        te_results (pd.DataFrame): Transfer Entropy results
            Required: source, target, lag_steps (or delay), p_value, q_value (optional)
        pcmci_results (pd.DataFrame): PCMCI+ results
            Required: source, target, lag_steps (or lag), p_value, q_value (optional)
        lag_tolerance_steps (int): Maximum lag difference in timesteps (default: 1)
        min_votes (int): Minimum number of methods that must agree (2-3)
        alpha (float): Significance threshold for p-values (if q-values unavailable)
        sampling_days (int): Days per timestep for lag_days conversion
        use_lag_bands (bool): If True, use lag-band matching (fast/mid/long) instead of
            exact lag tolerance. Reduces false negatives from exact lag sensitivity.

    Returns:
        pd.DataFrame: Consensus edges with columns:
            - source, target: Variable names
            - lag_steps: Median lag in timesteps
            - lag_days: Median lag in days
            - vote_count: Number of agreeing methods
            - agreeing_methods: Comma-separated method names
            - best_q_value: Minimum q-value across methods (if available)
            - best_p_value: Minimum p-value across methods
            - all_significant: Boolean, all methods significant
            - lag_std_steps: Std dev of lags across methods
            - lag_band: Lag band name if use_lag_bands=True (fast/mid/long)
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(
        f"Consensus Detection (min_votes={min_votes}, lag_tolerance={lag_tolerance_steps} steps, lag_bands={use_lag_bands})"
    )
    logger.info(f"{'=' * 70}")

    # Import lag-band utilities if needed
    if use_lag_bands:
        from framework.core.lagbands import get_lag_band, match_within_band

    # Collect all edges with method labels
    all_edges = []

    # Helper to extract edge data
    def extract_edge(row, method_name):
        # Try different column names for lag
        lag_steps = (
            row.get("lag_steps")
            or row.get("best_lag")
            or row.get("lag")
            or row.get("delay")
        )

        # Try to get significance from q_value first, then p_value
        q_value = row.get("q_value") or row.get("best_q_value")
        p_value = row.get("p_value") or row.get("best_p_value")

        # Optional effect size if provided by method CSV (e.g., Granger partial R²)
        effect_size = row.get("effect_size") if "effect_size" in row.index else None
        try:
            effect_size = (
                float(effect_size)
                if effect_size is not None and not pd.isna(effect_size)
                else np.nan
            )
        except Exception:
            effect_size = np.nan

        # Normalize booleans across method schemas
        is_sig = False
        if q_value is not None and not pd.isna(q_value):
            is_sig = bool(q_value < alpha)
        elif p_value is not None and not pd.isna(p_value):
            is_sig = bool(p_value < alpha)
        else:
            # Fall back to method-specific flags
            is_sig = bool(row.get("is_significant", row.get("is_causal", False)))

        # Normalize source/target naming across methods
        source = row.get("source") or row.get("cause")
        target = row.get("target") or row.get("effect")

        return {
            "source": source,
            "target": target,
            "lag_steps": lag_steps,
            "method": method_name,
            "p_value": p_value,
            "q_value": q_value,
            "is_significant": is_sig,
            "effect_size": effect_size,
        }

    # Process Granger results
    if granger_results is not None and len(granger_results) > 0:
        for _, row in granger_results.iterrows():
            edge = extract_edge(row, "Granger")
            if (
                edge["source"]
                and edge["target"]
                and edge["lag_steps"] is not None
                and not pd.isna(edge["lag_steps"])
            ):
                all_edges.append(edge)

    # Process Transfer Entropy results
    if te_results is not None and len(te_results) > 0:
        for _, row in te_results.iterrows():
            edge = extract_edge(row, "TransferEntropy")
            if (
                edge["source"]
                and edge["target"]
                and edge["lag_steps"] is not None
                and not pd.isna(edge["lag_steps"])
            ):
                all_edges.append(edge)

    # Process PCMCI+ results
    if pcmci_results is not None and len(pcmci_results) > 0:
        for _, row in pcmci_results.iterrows():
            edge = extract_edge(row, "PCMCI+")
            # Skip if explicitly marked as non-causal
            if "causal" in row.index and not row["causal"]:
                continue
            if (
                edge["source"]
                and edge["target"]
                and edge["lag_steps"] is not None
                and not pd.isna(edge["lag_steps"])
            ):
                all_edges.append(edge)

    if not all_edges:
        logger.warning("No edges to consensus-check")
        return pd.DataFrame()

    edges_df = pd.DataFrame(all_edges)
    logger.info(f"Input: {len(edges_df)} edges from all methods")
    logger.info(f"  Significant edges: {edges_df['is_significant'].sum()}")

    # Group edges by (source, target) with lag tolerance
    consensus_edges = []

    # Get unique source-target pairs
    unique_pairs = edges_df.groupby(["source", "target"])

    for (source, target), pair_edges in unique_pairs:
        # Group edges by lag tolerance or lag bands
        lag_groups = []
        for _, edge in pair_edges.iterrows():
            lag = edge["lag_steps"]

            # Find or create lag group
            found_group = False
            for group in lag_groups:
                # Use lag-band matching if enabled, otherwise use tolerance
                if use_lag_bands:
                    # Check if lag is in same band as any lag in group
                    lag_band = get_lag_band(lag)
                    if any(get_lag_band(g_lag) == lag_band for g_lag in group["lags"]):
                        group["lags"].append(lag)
                        group["edges"].append(edge)
                        group["band"] = lag_band
                        found_group = True
                        break
                else:
                    # Original tolerance-based matching
                    if any(
                        abs(lag - g_lag) <= lag_tolerance_steps
                        for g_lag in group["lags"]
                    ):
                        group["lags"].append(lag)
                        group["edges"].append(edge)
                        found_group = True
                        break

            if not found_group:
                new_group = {"lags": [lag], "edges": [edge]}
                if use_lag_bands:
                    new_group["band"] = get_lag_band(lag)
                lag_groups.append(new_group)

        # Process each lag group
        for group in lag_groups:
            group_edges = pd.DataFrame(group["edges"])

            # Count unique methods
            methods_voting = group_edges["method"].unique()
            n_methods = len(methods_voting)

            # Compute best q-value and p-value for this group
            q_values = group_edges["q_value"].dropna()
            best_q_value = q_values.min() if len(q_values) > 0 else np.nan

            p_values = group_edges["p_value"].dropna()
            best_p_value = p_values.min() if len(p_values) > 0 else np.nan

            # Method weights: prioritize Granger for directionality, PCMCI moderate, TE lower
            method_weights = {"Granger": 1.0, "PCMCI+": 0.8, "TransferEntropy": 0.6}

            # Significance score using -log10(p); add effect size contribution (scaled)
            def _edge_score(r: pd.Series) -> float:
                mw = method_weights.get(r.get("method"), 0.5)
                p = r.get("p_value")
                sig = 0.0
                try:
                    if p is not None and not pd.isna(p) and float(p) > 0:
                        sig = max(0.0, -np.log10(float(p)))
                except Exception:
                    sig = 0.0
                eff = r.get("effect_size")
                try:
                    eff = float(eff) if eff is not None and not pd.isna(eff) else 0.0
                except Exception:
                    eff = 0.0
                return mw * (sig + 5.0 * eff)

            group_score = (
                float(group_edges.apply(_edge_score, axis=1).sum())
                if len(group_edges) > 0
                else 0.0
            )

            # High-confidence rule for single-method detections: allow if method is Granger and p < 0.02
            allow_single = False
            if n_methods == 1:
                mname = methods_voting[0]
                try:
                    allow_single = (
                        (mname == "Granger")
                        and (not pd.isna(best_p_value))
                        and (float(best_p_value) < 0.02)
                    )
                except Exception:
                    allow_single = False

            # If global min_votes==1, treat base requirement as 2 to suppress generic singletons
            base_min = 2 if int(min_votes) <= 1 else int(min_votes)
            include_group = (n_methods >= base_min) or allow_single

            if include_group:
                # Use median lag (robust to outliers) among valid lags
                valid_lags = group_edges["lag_steps"].dropna()
                if len(valid_lags) == 0:
                    continue
                median_lag_steps = int(np.round(valid_lags.median()))
                median_lag_days = median_lag_steps * sampling_days
                lag_std_steps = valid_lags.std()

                # Count significant results
                n_significant = int(group_edges["is_significant"].sum())
                all_significant = n_significant == n_methods

                edge_data = {
                    "source": source,
                    "target": target,
                    "lag_steps": median_lag_steps,
                    "lag_days": median_lag_days,
                    "vote_count": n_methods,
                    "paradigm_diversity": compute_paradigm_diversity(
                        list(methods_voting)
                    ),
                    "agreeing_methods": ",".join(sorted(methods_voting)),
                    "best_q_value": best_q_value,
                    "best_p_value": best_p_value,
                    "n_significant": n_significant,
                    "all_significant": all_significant,
                    "lag_std_steps": float(lag_std_steps)
                    if not pd.isna(lag_std_steps)
                    else 0.0,
                    "_score": group_score,
                }

                # Add lag band if using band-based consensus
                if use_lag_bands:
                    from framework.core.lagbands import get_band_description

                    edge_data["lag_band"] = group.get(
                        "band", get_lag_band(median_lag_steps)
                    )
                    edge_data["lag_band_description"] = get_band_description(
                        edge_data["lag_band"]
                    )

                consensus_edges.append(edge_data)

    consensus_df = pd.DataFrame(consensus_edges)

    if len(consensus_df) > 0:
        # Filter reverse causation: for bidirectional edges A→B and B→A, keep only stronger one
        # Prefer higher consensus score; fall back to votes+significance if scores unavailable
        edges_to_remove = set()
        for idx, row in consensus_df.iterrows():
            if idx in edges_to_remove:
                continue

            # Look for reverse edge
            reverse = consensus_df[
                (consensus_df["source"] == row["target"])
                & (consensus_df["target"] == row["source"])
            ]

            if len(reverse) > 0:
                reverse_idx = reverse.index[0]
                if reverse_idx in edges_to_remove:
                    continue

                # If only one direction is supported by Granger, prefer that direction
                f_has_granger = isinstance(row.get("agreeing_methods"), str) and (
                    "Granger" in row.get("agreeing_methods")
                )
                r_has_granger = isinstance(
                    reverse.iloc[0].get("agreeing_methods"), str
                ) and ("Granger" in reverse.iloc[0].get("agreeing_methods"))
                if f_has_granger and not r_has_granger:
                    edges_to_remove.add(reverse_idx)
                    logger.info(
                        f"Filtered reverse edge: {reverse.iloc[0]['source']}→{reverse.iloc[0]['target']} (prefer Granger-supported forward)"
                    )
                    continue
                if r_has_granger and not f_has_granger:
                    edges_to_remove.add(idx)
                    logger.info(
                        f"Filtered reverse edge: {row['source']}→{row['target']} (reverse has Granger support)"
                    )
                    continue

                # Compare using consensus scores if available on both
                f_score = float(row.get("_score", np.nan))
                r_score = float(reverse.iloc[0].get("_score", np.nan))
                if not (pd.isna(f_score) or pd.isna(r_score)):
                    if f_score < r_score:
                        edges_to_remove.add(idx)
                        logger.info(
                            f"Filtered reverse edge: {row['source']}→{row['target']} (lower consensus score than reverse)"
                        )
                    else:
                        edges_to_remove.add(reverse_idx)
                        logger.info(
                            f"Filtered reverse edge: {reverse.iloc[0]['source']}→{reverse.iloc[0]['target']} (lower consensus score than forward)"
                        )
                else:
                    # Fall back: use vote_count first, then best significance
                    forward_score = row["vote_count"] * 1000 - (
                        row["best_q_value"]
                        if not pd.isna(row["best_q_value"])
                        else row["best_p_value"]
                    )
                    reverse_score = reverse.iloc[0]["vote_count"] * 1000 - (
                        reverse.iloc[0]["best_q_value"]
                        if not pd.isna(reverse.iloc[0]["best_q_value"])
                        else reverse.iloc[0]["best_p_value"]
                    )
                    if forward_score < reverse_score:
                        edges_to_remove.add(idx)
                        logger.info(
                            f"Filtered reverse edge: {row['source']}→{row['target']} (weaker than reverse)"
                        )
                    else:
                        edges_to_remove.add(reverse_idx)
                        logger.info(
                            f"Filtered reverse edge: {reverse.iloc[0]['source']}→{reverse.iloc[0]['target']} (weaker than reverse)"
                        )

        if edges_to_remove:
            consensus_df = consensus_df.drop(index=list(edges_to_remove)).reset_index(
                drop=True
            )
            logger.info(f"Removed {len(edges_to_remove)} reverse causation edges")

        # Sort by vote_count (descending), then by best significance using a temp column
        consensus_df["_sort_sig"] = consensus_df["best_q_value"].where(
            ~consensus_df["best_q_value"].isna(), consensus_df["best_p_value"]
        )
        # If available, also sort by consensus score (desc)
        if "_score" in consensus_df.columns:
            consensus_df = consensus_df.sort_values(
                by=["vote_count", "_score", "_sort_sig"], ascending=[False, False, True]
            ).drop(
                columns=[
                    c for c in ["_sort_sig", "_score"] if c in consensus_df.columns
                ]
            )
        else:
            consensus_df = consensus_df.sort_values(
                by=["vote_count", "_sort_sig"], ascending=[False, True]
            ).drop(columns=["_sort_sig"])

        logger.info(f"Output: {len(consensus_df)} consensus edges")
        logger.info(
            f"  - Full agreement (all 3 methods): {(consensus_df['vote_count'] == 3).sum()}"
        )
        logger.info(
            f"  - Partial agreement (2 methods): {(consensus_df['vote_count'] == 2).sum()}"
        )
        logger.info(
            f"  - All methods significant: {consensus_df['all_significant'].sum()}"
        )

        return consensus_df
    else:
        logger.warning(f"No consensus edges found with min_votes={min_votes}")
        return pd.DataFrame()


def export_graphml(
    consensus_df: pd.DataFrame,
    output_path: str,
    alpha: float = 0.05,
) -> None:
    """
    Export consensus edges to GraphML format for network visualization.

    Parameters:
        consensus_df: Consensus edges DataFrame
        output_path: Path to save GraphML file
        alpha: Significance threshold for edge weight calculation

    Edge attributes:
        - weight: -log10(q_value or p_value), higher = more significant
        - lag_steps: Lag in timesteps
        - lag_days: Lag in days
        - methods: Comma-separated agreeing methods
        - vote_count: Number of agreeing methods
        - all_significant: Boolean, all methods significant
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error("NetworkX not installed. Cannot export GraphML.")
        logger.info("Install with: pip install networkx")
        return

    if consensus_df.empty:
        logger.warning("No consensus edges to export")
        return

    G = nx.DiGraph()

    # Add nodes (unique variables)
    nodes = set(consensus_df["source"].unique()) | set(consensus_df["target"].unique())
    for node in nodes:
        G.add_node(node)

    # Add edges with attributes
    for _, row in consensus_df.iterrows():
        # Calculate edge weight: -log10(significance)
        if "best_q_value" in row and not pd.isna(row["best_q_value"]):
            sig_val = max(row["best_q_value"], 1e-10)  # Avoid log(0)
        elif "best_p_value" in row and not pd.isna(row["best_p_value"]):
            sig_val = max(row["best_p_value"], 1e-10)
        else:
            sig_val = alpha

        weight = -np.log10(sig_val)

        G.add_edge(
            row["source"],
            row["target"],
            weight=float(weight),
            lag_steps=int(row["lag_steps"]),
            lag_days=int(row["lag_days"]),
            methods=str(row["agreeing_methods"]),
            vote_count=int(row["vote_count"]),
            all_significant=bool(row["all_significant"]),
            best_q_value=float(row["best_q_value"])
            if not pd.isna(row.get("best_q_value"))
            else None,
            best_p_value=float(row["best_p_value"])
            if not pd.isna(row.get("best_p_value"))
            else None,
        )

    # Write GraphML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))

    logger.info(
        f"✅ Exported {len(G.edges())} consensus edges to GraphML: {output_path}"
    )
    logger.info(f"   Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")


def voting_matrix(consensus_df: pd.DataFrame) -> Dict:
    """
    Generate voting matrix showing which methods agree on each edge.

    Parameters:
        consensus_df (pd.DataFrame): Consensus results from detect_agreement()

    Returns:
        Dict: Voting matrix with method combinations
    """
    voting = {
        "all_three": int((consensus_df["vote_count"] == 3).sum()),
        "two_methods": int((consensus_df["vote_count"] == 2).sum()),
        "total_consensus": len(consensus_df),
    }

    # Count method combinations
    method_combos = {}
    for _, row in consensus_df.iterrows():
        methods = row["agreeing_methods"]
        method_combos[methods] = method_combos.get(methods, 0) + 1

    voting["method_combinations"] = method_combos

    return voting


def consensus_report(consensus_df: pd.DataFrame) -> str:
    """
    Generate human-readable consensus report.

    Parameters:
        consensus_df (pd.DataFrame): Consensus results

    Returns:
        str: Formatted report
    """
    if len(consensus_df) == 0:
        return "No consensus relationships detected."

    report = f"""
CONSENSUS CAUSAL DISCOVERY REPORT
{"=" * 70}

Total Consensus Edges Found: {len(consensus_df)}

Agreement Distribution:
  - Full agreement (3 methods):   {(consensus_df["vote_count"] == 3).sum()} edges
  - Partial agreement (2 methods): {(consensus_df["vote_count"] == 2).sum()} edges

All Methods Significant: {consensus_df["all_significant"].sum()} edges

Top 10 Most Robust Relationships:
{"-" * 70}
"""

    for idx, (_, row) in enumerate(consensus_df.head(10).iterrows()):
        q_str = (
            f", q={row['best_q_value']:.4f}"
            if not pd.isna(row.get("best_q_value"))
            else ""
        )
        p_str = (
            f", p={row['best_p_value']:.4f}"
            if not pd.isna(row.get("best_p_value"))
            else ""
        )

        report += f"""
{idx + 1}. {row["source"]} → {row["target"]}
   Lag: {row["lag_steps"]} steps ({row["lag_days"]} days) ±{row["lag_std_steps"]:.2f}
   Methods: {row["agreeing_methods"]} ({row["vote_count"]}/3)
   Significance: {row["n_significant"]}/3 methods{q_str}{p_str}
   All significant: {row["all_significant"]}
"""

    report += f"\n{'=' * 70}\n"

    return report


def merge_method_results(
    granger_df: Optional[pd.DataFrame] = None,
    te_df: Optional[pd.DataFrame] = None,
    pcmci_df: Optional[pd.DataFrame] = None,
    min_votes: int = 2,
    lag_tolerance_steps: int = 1,
    sampling_days: int = 1,
    alpha: float = 0.05,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Merge results from all three methods and run consensus detection.

    Convenience function that combines method results and produces consensus.

    Parameters:
        granger_df (Optional[pd.DataFrame]): Granger results
        te_df (Optional[pd.DataFrame]): Transfer Entropy results
        pcmci_df (Optional[pd.DataFrame]): PCMCI+ results
        min_votes (int): Minimum methods that must agree (default: 2)
        lag_tolerance_steps (int): Lag tolerance in timesteps (default: 1)
        sampling_days (int): Days per timestep (default: 5)
        alpha (float): Significance threshold (default: 0.05)
        output_dir (Optional[str]): If provided, save consensus CSV and GraphML

    Returns:
        Dict: Consensus results + report + paths
    """
    logger.info("\n" + "=" * 70)
    logger.info("CONSENSUS ANALYSIS: Merging All Methods")
    logger.info("=" * 70)

    consensus_df = detect_agreement(
        granger_df if granger_df is not None else pd.DataFrame(),
        te_df if te_df is not None else pd.DataFrame(),
        pcmci_df if pcmci_df is not None else pd.DataFrame(),
        lag_tolerance_steps=lag_tolerance_steps,
        min_votes=min_votes,
        alpha=alpha,
        sampling_days=sampling_days,
    )

    voting = voting_matrix(consensus_df) if len(consensus_df) > 0 else {}
    report = consensus_report(consensus_df)

    result = {
        "consensus_edges": consensus_df,
        "voting_matrix": voting,
        "report": report,
        "n_consensus_edges": len(consensus_df),
    }

    # Save outputs if directory provided
    if output_dir and len(consensus_df) > 0:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save consensus CSV
        csv_path = output_path / "consensus.csv"
        consensus_df.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved consensus CSV: {csv_path}")
        result["consensus_csv"] = str(csv_path)

        # Save GraphML
        graphml_path = output_path / "consensus_graph.graphml"
        export_graphml(consensus_df, str(graphml_path), alpha=alpha)
        result["consensus_graphml"] = str(graphml_path)

        # Save report
        report_path = output_path / "consensus_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"✅ Saved consensus report: {report_path}")
        result["consensus_report"] = str(report_path)

    return result
