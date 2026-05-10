"""
Lag-Stratified Analysis Utilities

Provides functions to analyze causal discovery results by lag categories:
- Short-lag (≤3 timesteps): Fast physiological responses
- Long-lag (≥10 timesteps): Seasonal/accumulation effects
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def stratify_results_by_lag(
    results_dict: Dict[str, pd.DataFrame],
    short_lag_max: int = 3,
    long_lag_min: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Stratify causal discovery results by lag categories.

    Parameters:
        results_dict: Dictionary with keys 'granger', 'transfer_entropy', 'pcmci'
        short_lag_max: Maximum lag for short-lag category (default: 3)
        long_lag_min: Minimum lag for long-lag category (default: 10)
        output_dir: Optional directory to save stratified results

    Returns:
        Dictionary with stratified results and summary statistics
    """
    logger.info("\n" + "=" * 70)
    logger.info("LAG-STRATIFIED ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Short-lag: ≤{short_lag_max} timesteps (~{short_lag_max * 5}-15 days)")
    logger.info(f"Long-lag: ≥{long_lag_min} timesteps (~{long_lag_min * 5}-60 days)")

    stratified = {
        "short_lag": {},
        "long_lag": {},
        "summary": {},
    }

    for method_name, df in results_dict.items():
        if df is None or len(df) == 0:
            logger.info(f"\n{method_name}: No results to stratify")
            continue

        # Identify lag column (check multiple possible names)
        lag_col = None
        for col in [
            "lag",
            "delay",
            "optimal_lag",
            "best_lag",
            "best_lag_days",
            "lag_steps",
        ]:
            if col in df.columns:
                lag_col = col
                break

        # If best_lag_days is found, convert to timesteps for comparison
        if lag_col == "best_lag_days":
            # Convert days to timesteps (assuming 5 days per timestep)
            # This is approximate - ideally we'd have the actual sampling_days
            df = df.copy()
            df["_lag_timesteps"] = df[lag_col] / 5.0
            lag_col = "_lag_timesteps"

        if lag_col is None:
            logger.warning(
                f"{method_name}: No lag column found (checked: lag, delay, optimal_lag, best_lag, best_lag_days, lag_steps), skipping"
            )
            continue

        # Ensure significance flag exists
        sig_col = "is_significant"
        if sig_col not in df.columns:
            if "significant" in df.columns:
                sig_col = "significant"
            elif "fdr_significant" in df.columns:
                sig_col = "fdr_significant"
            else:
                logger.warning(f"{method_name}: No significance flag found")
                continue

        # Filter to significant edges only
        sig_df = df[df[sig_col].astype(bool)].copy()

        if len(sig_df) == 0:
            logger.info(f"\n{method_name}: No significant edges")
            stratified["short_lag"][method_name] = pd.DataFrame()
            stratified["long_lag"][method_name] = pd.DataFrame()
            stratified["summary"][method_name] = {
                "total_significant": 0,
                "short_lag_count": 0,
                "long_lag_count": 0,
                "short_lag_percent": 0.0,
                "long_lag_percent": 0.0,
            }
            continue

        # Stratify by lag
        short_lag_df = sig_df[sig_df[lag_col] <= short_lag_max].copy()
        long_lag_df = sig_df[sig_df[lag_col] >= long_lag_min].copy()

        stratified["short_lag"][method_name] = short_lag_df
        stratified["long_lag"][method_name] = long_lag_df

        # Compute summary statistics
        summary = {
            "total_significant": len(sig_df),
            "short_lag_count": len(short_lag_df),
            "long_lag_count": len(long_lag_df),
            "short_lag_percent": (
                len(short_lag_df) / len(sig_df) * 100 if len(sig_df) > 0 else 0
            ),
            "long_lag_percent": (
                len(long_lag_df) / len(sig_df) * 100 if len(sig_df) > 0 else 0
            ),
        }

        # Add per-cause breakdown
        if "cause" in sig_df.columns:
            for cause_var in sig_df["cause"].unique():
                cause_df = sig_df[sig_df["cause"] == cause_var]
                cause_short = short_lag_df[short_lag_df["cause"] == cause_var]
                cause_long = long_lag_df[long_lag_df["cause"] == cause_var]

                summary[f"{cause_var}_total"] = len(cause_df)
                summary[f"{cause_var}_short"] = len(cause_short)
                summary[f"{cause_var}_long"] = len(cause_long)

        stratified["summary"][method_name] = summary

        logger.info(f"\n{method_name}:")
        logger.info(f"  Total significant edges: {summary['total_significant']}")
        logger.info(
            f"  Short-lag (≤{short_lag_max}): {summary['short_lag_count']} "
            f"({summary['short_lag_percent']:.1f}%)"
        )
        logger.info(
            f"  Long-lag (≥{long_lag_min}): {summary['long_lag_count']} "
            f"({summary['long_lag_percent']:.1f}%)"
        )

        # Show per-cause breakdown
        if "cause" in sig_df.columns:
            for cause_var in sig_df["cause"].unique():
                logger.info(
                    f"    {cause_var}: "
                    f"{summary.get(f'{cause_var}_short', 0)} short, "
                    f"{summary.get(f'{cause_var}_long', 0)} long"
                )

    # Save stratified results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for stratum in ["short_lag", "long_lag"]:
            for method_name, df in stratified[stratum].items():
                if len(df) > 0:
                    filename = output_dir / f"{stratum}_{method_name}.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Saved: {filename}")

        # Save summary statistics
        if stratified["summary"]:
            summary_df = pd.DataFrame(stratified["summary"]).T
            summary_file = output_dir / "lag_stratification_summary.csv"
            summary_df.to_csv(summary_file)
            logger.info(f"Saved: {summary_file}")
        else:
            # Create empty summary if no results
            summary_file = output_dir / "lag_stratification_summary.csv"
            pd.DataFrame().to_csv(summary_file)
            logger.warning(
                f"No summary data to save, created empty file: {summary_file}"
            )

    return stratified


def compute_lag_agreement(
    stratified_results: Dict,
    min_methods: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Compute method agreement within each lag stratum.

    Parameters:
        stratified_results: Output from stratify_results_by_lag
        min_methods: Minimum number of methods for agreement

    Returns:
        Dictionary with agreement statistics per stratum
    """
    logger.info("\n" + "=" * 70)
    logger.info("LAG-STRATIFIED METHOD AGREEMENT")
    logger.info("=" * 70)

    agreement = {}

    for stratum in ["short_lag", "long_lag"]:
        logger.info(f"\n{stratum.upper().replace('_', ' ')}:")

        # Collect all edges from each method
        method_edges = {}
        for method_name, df in stratified_results[stratum].items():
            if len(df) == 0:
                continue

            # Create edge identifiers
            if "cause" in df.columns and "effect" in df.columns:
                if "unit_id" in df.columns:
                    edges = set(
                        df.apply(
                            lambda r: (r["cause"], r["effect"], r["unit_id"]), axis=1
                        )
                    )
                else:
                    edges = set(df.apply(lambda r: (r["cause"], r["effect"]), axis=1))
                method_edges[method_name] = edges

        if len(method_edges) == 0:
            logger.info("  No method results available")
            agreement[stratum] = pd.DataFrame()
            continue

        # Find agreements
        all_edges = set()
        for edges in method_edges.values():
            all_edges.update(edges)

        agreement_list = []
        for edge in all_edges:
            methods_supporting = [
                method for method, edges in method_edges.items() if edge in edges
            ]
            n_methods = len(methods_supporting)

            if n_methods >= min_methods:
                if len(edge) == 3:  # Has unit_id
                    cause, effect, unit_id = edge
                    agreement_list.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "unit_id": unit_id,
                            "n_methods": n_methods,
                            "methods": ",".join(methods_supporting),
                        }
                    )
                else:  # No unit_id
                    cause, effect = edge
                    agreement_list.append(
                        {
                            "cause": cause,
                            "effect": effect,
                            "n_methods": n_methods,
                            "methods": ",".join(methods_supporting),
                        }
                    )

        agreement_df = pd.DataFrame(agreement_list)
        agreement[stratum] = agreement_df

        if len(agreement_df) > 0:
            logger.info(
                f"  Edges with ≥{min_methods} method agreement: {len(agreement_df)}"
            )
            for n in sorted(agreement_df["n_methods"].unique(), reverse=True):
                count = (agreement_df["n_methods"] == n).sum()
                logger.info(f"    {n} methods: {count} edges")

            # Show per-cause breakdown
            if "cause" in agreement_df.columns:
                for cause in agreement_df["cause"].unique():
                    cause_count = (agreement_df["cause"] == cause).sum()
                    logger.info(f"      {cause}: {cause_count} edges")
        else:
            logger.info(f"  No edges with ≥{min_methods} method agreement")

    return agreement


def generate_lag_report(
    results_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    short_lag_max: int = 3,
    long_lag_min: int = 10,
) -> None:
    """
    Generate comprehensive lag-stratified analysis report.

    Parameters:
        results_dict: Dictionary with causal discovery results
        output_dir: Directory to save report
        short_lag_max: Maximum lag for short-lag category
        long_lag_min: Minimum lag for long-lag category
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stratify results
    stratified = stratify_results_by_lag(
        results_dict,
        short_lag_max=short_lag_max,
        long_lag_min=long_lag_min,
        output_dir=output_dir,
    )

    # Compute agreement
    agreement = compute_lag_agreement(stratified, min_methods=2)

    # Save agreement results
    for stratum, df in agreement.items():
        if len(df) > 0:
            filename = output_dir / f"{stratum}_agreement.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved: {filename}")

    # Generate text report
    report_file = output_dir / "lag_stratification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LAG-STRATIFIED CAUSAL DISCOVERY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Short-lag category: ≤{short_lag_max} timesteps (~{short_lag_max * 5}-15 days)\n"
        )
        f.write(
            f"Long-lag category: ≥{long_lag_min} timesteps (~{long_lag_min * 5}-60 days)\n\n"
        )

        f.write("INDIVIDUAL METHOD RESULTS\n")
        f.write("-" * 70 + "\n\n")

        for method_name, summary in stratified["summary"].items():
            f.write(f"{method_name}:\n")
            f.write(f"  Total significant edges: {summary['total_significant']}\n")
            f.write(
                f"  Short-lag: {summary['short_lag_count']} "
                f"({summary['short_lag_percent']:.1f}%)\n"
            )
            f.write(
                f"  Long-lag: {summary['long_lag_count']} "
                f"({summary['long_lag_percent']:.1f}%)\n"
            )

            # Per-cause breakdown
            causes = [
                k.replace("_total", "") for k in summary.keys() if k.endswith("_total")
            ]
            for cause in causes:
                f.write(
                    f"    {cause}: "
                    f"{summary.get(f'{cause}_short', 0)} short, "
                    f"{summary.get(f'{cause}_long', 0)} long\n"
                )
            f.write("\n")

        f.write("\nMETHOD AGREEMENT\n")
        f.write("-" * 70 + "\n\n")

        for stratum, df in agreement.items():
            f.write(f"{stratum.upper().replace('_', ' ')}:\n")
            if len(df) > 0:
                f.write(f"  Total consensus edges: {len(df)}\n")
                for n in sorted(df["n_methods"].unique(), reverse=True):
                    count = (df["n_methods"] == n).sum()
                    f.write(f"    {n} methods: {count} edges\n")

                # Per-cause breakdown
                if "cause" in df.columns:
                    for cause in df["cause"].unique():
                        cause_count = (df["cause"] == cause).sum()
                        f.write(f"      {cause}: {cause_count} edges\n")
            else:
                f.write("  No consensus edges\n")
            f.write("\n")

    logger.info(f"\nSaved lag stratification report: {report_file}")
