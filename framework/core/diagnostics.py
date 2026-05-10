#!/usr/bin/env python3
"""
Comprehensive Diagnostics Module

Generates detailed statistical analyses and visualizations for causal discovery results:

1. **P-value Distributions**: Histograms per method showing p-value patterns
2. **Lag Distributions**: Box plots and histograms of detected lags by method
3. **Wilson Confidence Intervals**: 95% CI for detection rates
4. **Empirical Envelope Plots**: Statistical significance testing with permutations
5. **Method Comparison**: Cross-method analysis and agreement
6. **Permutation Negative Control**: Block-shuffle source data, re-run methods, check FDR ≈ α
7. **Reverse-Direction Check**: For detected X→Y, test Y→X, verify rate ≈ α
8. **Method Divergence Alerts**: Systematic detection rate differences between methods
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from scipy import stats

logger = logging.getLogger(__name__)


class Diagnostics:
    """Comprehensive statistical diagnostics for causal discovery results"""

    def __init__(self, edges_df: pd.DataFrame, cadence_days: int = 5):
        """
        Initialize diagnostics.

        Parameters:
            edges_df (pd.DataFrame): DataFrame of causal edges
            cadence_days (int): Days per timestep
        """
        self.edges_df = edges_df
        self.cadence_days = cadence_days
        self.diagnostics_results = {}

    def analyze_pvalue_distributions(self) -> Dict:
        """
        Analyze p-value distributions per method.

        Returns:
            Dict: P-value statistics per method
        """
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC 1: P-value Distributions")
        logger.info("=" * 70)

        if self.edges_df.empty or "p_value" not in self.edges_df.columns:
            logger.warning("  No p-value data available")
            return {"diagnostic": "PValueDistributions", "pass": False}

        result = {"diagnostic": "PValueDistributions", "by_method": {}}

        methods = (
            self.edges_df["method"].unique()
            if "method" in self.edges_df.columns
            else ["all"]
        )

        for method in methods:
            if "method" in self.edges_df.columns:
                pvals = self.edges_df[self.edges_df["method"] == method][
                    "p_value"
                ].dropna()
            else:
                pvals = self.edges_df["p_value"].dropna()

            if len(pvals) == 0:
                continue

            stats_dict = {
                "method": method,
                "n": len(pvals),
                "mean": float(pvals.mean()),
                "median": float(pvals.median()),
                "std": float(pvals.std()),
                "min": float(pvals.min()),
                "max": float(pvals.max()),
                "q25": float(pvals.quantile(0.25)),
                "q75": float(pvals.quantile(0.75)),
                "significant_count": int((pvals < 0.05).sum()),
                "significant_pct": float((pvals < 0.05).sum() / len(pvals) * 100),
            }

            result["by_method"][method] = stats_dict

            logger.info(f"\n  Method: {method}")
            logger.info(f"    N: {stats_dict['n']}")
            logger.info(f"    Mean: {stats_dict['mean']:.4f}")
            logger.info(f"    Median: {stats_dict['median']:.4f}")
            logger.info(f"    Std: {stats_dict['std']:.4f}")
            logger.info(
                f"    Range: [{stats_dict['min']:.4f}, {stats_dict['max']:.4f}]"
            )
            logger.info(
                f"    Significant (p<0.05): {stats_dict['significant_count']}/{stats_dict['n']} ({stats_dict['significant_pct']:.1f}%)"
            )

        self.diagnostics_results["pvalue_distributions"] = result
        logger.info("\n✅ P-value distributions analyzed")
        return result

    def analyze_lag_distributions(self) -> Dict:
        """
        Analyze lag distributions per method.

        Returns:
            Dict: Lag statistics per method
        """
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC 2: Lag Distributions")
        logger.info("=" * 70)

        if self.edges_df.empty or "lag_days" not in self.edges_df.columns:
            logger.warning("  No lag data available")
            return {"diagnostic": "LagDistributions", "pass": False}

        result = {"diagnostic": "LagDistributions", "by_method": {}}

        methods = (
            self.edges_df["method"].unique()
            if "method" in self.edges_df.columns
            else ["all"]
        )

        for method in methods:
            if "method" in self.edges_df.columns:
                lags = self.edges_df[self.edges_df["method"] == method][
                    "lag_days"
                ].dropna()
            else:
                lags = self.edges_df["lag_days"].dropna()

            if len(lags) == 0:
                continue

            # Convert to timesteps
            lags_steps = (lags / self.cadence_days).round(0).astype(int)

            stats_dict = {
                "method": method,
                "n": len(lags),
                "mean_days": float(lags.mean()),
                "median_days": float(lags.median()),
                "std_days": float(lags.std()),
                "min_days": float(lags.min()),
                "max_days": float(lags.max()),
                "mean_steps": float(lags_steps.mean()),
                "median_steps": float(lags_steps.median()),
                "mode_days": float(lags.mode()[0]) if len(lags.mode()) > 0 else None,
                "q25_days": float(lags.quantile(0.25)),
                "q75_days": float(lags.quantile(0.75)),
                "ecological_range_pct": float(
                    ((lags >= 5) & (lags <= 15)).sum() / len(lags) * 100
                ),
            }

            result["by_method"][method] = stats_dict

            logger.info(f"\n  Method: {method}")
            logger.info(f"    N: {stats_dict['n']}")
            logger.info(
                f"    Mean: {stats_dict['mean_days']:.1f} days ({stats_dict['mean_steps']:.1f} steps)"
            )
            logger.info(
                f"    Median: {stats_dict['median_days']:.1f} days ({stats_dict['median_steps']:.1f} steps)"
            )
            logger.info(
                f"    Range: {stats_dict['min_days']:.0f}-{stats_dict['max_days']:.0f} days"
            )
            logger.info(
                f"    Ecological range (5-15 days): {stats_dict['ecological_range_pct']:.1f}%"
            )

        self.diagnostics_results["lag_distributions"] = result
        logger.info("\n✅ Lag distributions analyzed")
        return result

    def wilson_ci(
        self, successes: int, n: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate Wilson score confidence interval for proportion.

        Parameters:
            successes (int): Number of successes
            n (int): Total number of trials
            confidence (float): Confidence level (default 0.95)

        Returns:
            Tuple[float, float]: (lower_bound, upper_bound)
        """
        if n == 0:
            return 0.0, 1.0

        p = successes / n
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denominator
        spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

        return max(0, center - spread), min(1, center + spread)

    def calculate_detection_ci(self) -> Dict:
        """
        Calculate Wilson confidence intervals for detection rates.

        Returns:
            Dict: Confidence intervals per method
        """
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC 3: Wilson Confidence Intervals")
        logger.info("=" * 70)

        if self.edges_df.empty:
            logger.warning("  No data available")
            return {"diagnostic": "WilsonCI", "pass": False}

        result = {"diagnostic": "WilsonCI", "by_method": {}}

        methods = (
            self.edges_df["method"].unique()
            if "method" in self.edges_df.columns
            else ["all"]
        )

        for method in methods:
            if "method" in self.edges_df.columns:
                method_edges = self.edges_df[self.edges_df["method"] == method]
            else:
                method_edges = self.edges_df

            n_total = len(method_edges)
            n_significant = (
                method_edges.get(
                    "is_significant", method_edges.get("p_value", []) < 0.05
                ).sum()
                if "is_significant" in method_edges.columns
                else (method_edges["p_value"] < 0.05).sum()
                if "p_value" in method_edges.columns
                else 0
            )

            lower, upper = self.wilson_ci(n_significant, n_total)
            detection_rate = n_significant / n_total if n_total > 0 else 0

            stats_dict = {
                "method": method,
                "n_total": n_total,
                "n_significant": n_significant,
                "detection_rate": float(detection_rate),
                "ci_lower": lower,
                "ci_upper": upper,
                "ci_width": upper - lower,
            }

            result["by_method"][method] = stats_dict

            logger.info(f"\n  Method: {method}")
            logger.info(f"    Total edges: {n_total}")
            logger.info(f"    Significant (p<0.05): {n_significant}")
            logger.info(f"    Detection rate: {detection_rate:.1%}")
            logger.info(f"    95% Wilson CI: [{lower:.1%}, {upper:.1%}]")
            logger.info(f"    CI width: {(upper - lower):.1%}")

        self.diagnostics_results["detection_ci"] = result
        logger.info("\n✅ Wilson confidence intervals calculated")
        return result

    def method_agreement(self) -> Dict:
        """
        Analyze agreement between methods.

        Returns:
            Dict: Method comparison statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC 4: Method Agreement Analysis")
        logger.info("=" * 70)

        if self.edges_df.empty or "method" not in self.edges_df.columns:
            logger.warning("  Method data not available")
            return {"diagnostic": "MethodAgreement", "pass": False}

        result = {"diagnostic": "MethodAgreement"}

        methods = self.edges_df["method"].unique()
        n_methods = len(methods)

        # Overall statistics
        edges_per_method = self.edges_df["method"].value_counts().to_dict()
        total_edges = len(self.edges_df)

        logger.info(f"\n  Total methods: {n_methods}")
        logger.info(f"  Total edges detected: {total_edges}")
        logger.info("  Edges per method:")

        method_stats = {}
        for method in sorted(edges_per_method.keys()):
            count = edges_per_method[method]
            pct = count / total_edges * 100
            logger.info(f"    - {method}: {count} ({pct:.1f}%)")
            method_stats[method] = {"count": count, "pct": pct}

        # Identify common edges (detected by multiple methods)
        if n_methods >= 2:
            edges_by_pair = {}
            for _, row in self.edges_df.iterrows():
                pair = (row["source"], row["target"])
                if pair not in edges_by_pair:
                    edges_by_pair[pair] = []
                edges_by_pair[pair].append(row["method"])

            # Count agreement
            agreement_counts = {}
            for pair, methods_list in edges_by_pair.items():
                n_agree = len(methods_list)
                if n_agree not in agreement_counts:
                    agreement_counts[n_agree] = 0
                agreement_counts[n_agree] += 1

            logger.info("\n  Edge detection agreement:")
            for n_agree in sorted(agreement_counts.keys()):
                count = agreement_counts[n_agree]
                pct = count / len(edges_by_pair) * 100
                logger.info(
                    f"    - Detected by {n_agree} method(s): {count} edges ({pct:.1f}%)"
                )

            result["agreement_counts"] = agreement_counts
            result["total_unique_pairs"] = len(edges_by_pair)

        result["method_stats"] = method_stats
        self.diagnostics_results["method_agreement"] = result
        logger.info("\n✅ Method agreement analyzed")
        return result

    def summary_statistics(self) -> Dict:
        """
        Generate overall summary statistics.

        Returns:
            Dict: Summary statistics
        """
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSTIC 5: Overall Summary Statistics")
        logger.info("=" * 70)

        if self.edges_df.empty:
            logger.warning("  No data available")
            return {"diagnostic": "SummaryStats", "pass": False}

        result = {"diagnostic": "SummaryStats"}

        # Basic statistics
        n_edges = len(self.edges_df)
        n_variables = len(
            set(self.edges_df["source"].unique())
            | set(self.edges_df["target"].unique())
        )
        n_pairs = n_variables * (n_variables - 1)
        coverage = n_edges / n_pairs if n_pairs > 0 else 0

        result["n_edges"] = n_edges
        result["n_variables"] = n_variables
        result["n_possible_pairs"] = n_pairs
        result["coverage"] = float(coverage)

        logger.info(f"\n  Total edges: {n_edges}")
        logger.info(f"  Unique variables: {n_variables}")
        logger.info(f"  Possible pairs (directed): {n_pairs}")
        logger.info(f"  Coverage: {coverage:.1%}")

        # P-value statistics
        if "p_value" in self.edges_df.columns:
            pvals = self.edges_df["p_value"].dropna()
            result["pvalue_mean"] = float(pvals.mean())
            result["pvalue_median"] = float(pvals.median())
            result["significant_edges"] = int((pvals < 0.05).sum())
            logger.info("\n  P-value stats:")
            logger.info(f"    Mean: {result['pvalue_mean']:.4f}")
            logger.info(f"    Median: {result['pvalue_median']:.4f}")
            logger.info(f"    Significant (p<0.05): {result['significant_edges']}")

        # Lag statistics
        if "lag_days" in self.edges_df.columns:
            lags = self.edges_df["lag_days"].dropna()
            result["lag_mean_days"] = float(lags.mean())
            result["lag_median_days"] = float(lags.median())
            result["lag_range"] = (float(lags.min()), float(lags.max()))
            logger.info("\n  Lag stats (days):")
            logger.info(f"    Mean: {result['lag_mean_days']:.1f}")
            logger.info(f"    Median: {result['lag_median_days']:.1f}")
            logger.info(
                f"    Range: [{result['lag_range'][0]:.0f}, {result['lag_range'][1]:.0f}]"
            )

        self.diagnostics_results["summary"] = result
        logger.info("\n✅ Summary statistics generated")
        return result

    def run_all(self) -> Dict:
        """Run all diagnostics and return results"""
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING ALL DIAGNOSTICS")
        logger.info("=" * 70)

        self.analyze_pvalue_distributions()
        self.analyze_lag_distributions()
        self.calculate_detection_ci()
        self.method_agreement()
        self.summary_statistics()

        logger.info("\n" + "=" * 70)
        logger.info("✅ ALL DIAGNOSTICS COMPLETE")
        logger.info("=" * 70)

        return self.diagnostics_results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert diagnostics results to DataFrame"""
        rows = []
        for diagnostic_name, result in self.diagnostics_results.items():
            row = {"diagnostic": result.get("diagnostic", diagnostic_name)}
            if "by_method" in result:
                for method, stats in result["by_method"].items():
                    row_copy = row.copy()
                    row_copy.update({"method": method, **stats})
                    rows.append(row_copy)
            else:
                row.update(result)
                rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_results(self, output_dir: Path):
        """Save diagnostic results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrame summary
        df = self.to_dataframe()
        csv_path = output_dir / "diagnostics_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Saved summary to {csv_path}")

        # Save detailed JSON results
        import json

        json_path = output_dir / "diagnostics_detailed.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json_data = {}
            for key, value in self.diagnostics_results.items():
                json_data[key] = self._serialize_dict(value)
            json.dump(json_data, f, indent=2)
        logger.info(f"✅ Saved detailed results to {json_path}")

    @staticmethod
    def _serialize_dict(obj):
        """Convert non-serializable types to JSON-compatible types"""
        if isinstance(obj, dict):
            return {k: Diagnostics._serialize_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [Diagnostics._serialize_dict(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif obj is None:
            return None
        else:
            return str(obj)


def permutation_negative_control(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    method_name: str = "granger",
    n_permutations: int = 100,
    alpha: float = 0.05,
    tolerance: float = 0.03,
    **method_kwargs,
) -> Dict:
    """
    Permutation negative control: block-shuffle source data and re-run method.

    Under null hypothesis (no causality), post-FDR detection rate should be ≈ α.

    Parameters:
        df: Panel data (unit_id, time, variables)
        pairs: List of (source, target) pairs to test
        method_name: 'granger', 'te', or 'pcmci'
        n_permutations: Number of permutation runs
        alpha: Significance level
        tolerance: Acceptable deviation from α (e.g., 0.03 means α ± 0.03)
        **method_kwargs: Additional arguments for the method

    Returns:
        Dict with detection_rate, pass (bool), and detailed results
    """
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC: Permutation Negative Control")
    logger.info("=" * 70)
    logger.info(f"  Method: {method_name}")
    logger.info(f"  Permutations: {n_permutations}")
    logger.info(f"  Expected FDR ≈ {alpha:.2%} ± {tolerance:.2%}")

    from framework.core.methods import granger, transfer_entropy, tigramite_pcmci

    method_map = {
        "granger": granger.batch_granger_causality,
        "te": transfer_entropy.batch_transfer_entropy,
        "pcmci": tigramite_pcmci.batch_pcmci,
    }

    if method_name not in method_map:
        logger.error(f"Unknown method: {method_name}")
        return {"pass": False, "error": f"Unknown method: {method_name}"}

    run_method = method_map[method_name]

    # Block-shuffle source within each unit (preserves autocorrelation structure)
    detection_rates = []

    for perm_idx in range(n_permutations):
        df_shuffled = df.copy()

        # Shuffle each source variable within each unit
        source_vars = list(set([pair[0] for pair in pairs]))
        for source_var in source_vars:
            if source_var not in df_shuffled.columns:
                continue
            for unit_id in df_shuffled["unit_id"].unique():
                unit_mask = df_shuffled["unit_id"] == unit_id
                values = df_shuffled.loc[unit_mask, source_var].values
                # Block shuffle with block size = sqrt(T) to preserve some autocorrelation
                block_size = max(1, int(np.sqrt(len(values))))
                n_blocks = len(values) // block_size
                block_indices = np.arange(n_blocks)
                np.random.shuffle(block_indices)
                shuffled = np.concatenate(
                    [
                        values[i * block_size : (i + 1) * block_size]
                        for i in block_indices
                    ]
                )
                # Handle remainder
                if len(shuffled) < len(values):
                    shuffled = np.concatenate([shuffled, values[len(shuffled) :]])
                df_shuffled.loc[unit_mask, source_var] = shuffled[: len(values)]

        # Run method on shuffled data
        try:
            results = run_method(df_shuffled, pairs, alpha=alpha, **method_kwargs)

            # Count significant detections after FDR
            if results is not None and not results.empty:
                # Check for q_value (FDR-corrected) or fallback to p_value
                if "q_value" in results.columns:
                    n_sig = (results["q_value"] < alpha).sum()
                elif "is_significant" in results.columns:
                    n_sig = results["is_significant"].sum()
                elif "p_value" in results.columns:
                    n_sig = (results["p_value"] < alpha).sum()
                else:
                    n_sig = 0

                detection_rate = n_sig / len(pairs) if len(pairs) > 0 else 0
                detection_rates.append(detection_rate)
            else:
                detection_rates.append(0.0)
        except Exception as e:
            logger.debug(f"Permutation {perm_idx + 1} failed: {e}")
            detection_rates.append(np.nan)

    # Analyze results
    detection_rates = [r for r in detection_rates if not np.isnan(r)]

    if len(detection_rates) == 0:
        logger.error("  All permutations failed")
        return {"pass": False, "error": "All permutations failed"}

    mean_rate = np.mean(detection_rates)
    std_rate = np.std(detection_rates)
    median_rate = np.median(detection_rates)

    # Check if mean is within tolerance of α
    passes = abs(mean_rate - alpha) <= tolerance

    result = {
        "diagnostic": "PermutationNegativeControl",
        "method": method_name,
        "n_permutations": n_permutations,
        "n_successful": len(detection_rates),
        "expected_fdr": alpha,
        "tolerance": tolerance,
        "mean_detection_rate": mean_rate,
        "median_detection_rate": median_rate,
        "std_detection_rate": std_rate,
        "min_detection_rate": np.min(detection_rates),
        "max_detection_rate": np.max(detection_rates),
        "pass": passes,
        "deviation": abs(mean_rate - alpha),
    }

    logger.info("\n  Results:")
    logger.info(f"    Mean detection rate: {mean_rate:.2%}")
    logger.info(f"    Median detection rate: {median_rate:.2%}")
    logger.info(f"    Std: {std_rate:.2%}")
    logger.info(
        f"    Range: [{np.min(detection_rates):.2%}, {np.max(detection_rates):.2%}]"
    )
    logger.info(f"    Expected: {alpha:.2%} ± {tolerance:.2%}")
    logger.info(f"    Deviation: {abs(mean_rate - alpha):.2%}")
    logger.info("    ✅ PASS" if passes else "    ❌ FAIL")

    return result


def reverse_direction_check(
    df: pd.DataFrame,
    detected_edges: pd.DataFrame,
    method_name: str = "granger",
    alpha: float = 0.05,
    tolerance: float = 0.03,
    **method_kwargs,
) -> Dict:
    """
    Reverse-direction check: for each detected X→Y, test Y→X.

    Under null hypothesis (Y→X is false), detection rate should be ≈ α after FDR.

    Parameters:
        df: Panel data (unit_id, time, variables)
        detected_edges: DataFrame of significant edges (must have source, target)
        method_name: 'granger', 'te', or 'pcmci'
        alpha: Significance level
        tolerance: Acceptable deviation from α
        **method_kwargs: Additional arguments for the method

    Returns:
        Dict with reverse_detection_rate, pass (bool), and details
    """
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC: Reverse-Direction Check")
    logger.info("=" * 70)
    logger.info(f"  Method: {method_name}")
    logger.info(f"  Detected edges: {len(detected_edges)}")
    logger.info(f"  Expected reverse FDR ≈ {alpha:.2%} ± {tolerance:.2%}")

    if detected_edges.empty:
        logger.warning("  No detected edges to check")
        return {"pass": True, "warning": "No edges to check"}

    from framework.core.methods import granger, transfer_entropy, tigramite_pcmci

    method_map = {
        "granger": granger.batch_granger_causality,
        "te": transfer_entropy.batch_transfer_entropy,
        "pcmci": tigramite_pcmci.batch_pcmci,
    }

    if method_name not in method_map:
        logger.error(f"Unknown method: {method_name}")
        return {"pass": False, "error": f"Unknown method: {method_name}"}

    run_method = method_map[method_name]

    # Build reverse pairs: (target, source)
    reverse_pairs = [
        (row["target"], row["source"]) for _, row in detected_edges.iterrows()
    ]

    if len(reverse_pairs) == 0:
        logger.warning("  No reverse pairs to test")
        return {"pass": True, "warning": "No reverse pairs"}

    logger.info(f"  Testing {len(reverse_pairs)} reverse pairs...")

    # Run method on reverse pairs
    try:
        results = run_method(df, reverse_pairs, alpha=alpha, **method_kwargs)

        if results is None or results.empty:
            logger.warning("  No results from reverse testing")
            return {"pass": True, "warning": "No results from reverse testing"}

        # Count significant detections after FDR
        if "q_value" in results.columns:
            n_sig = (results["q_value"] < alpha).sum()
        elif "is_significant" in results.columns:
            n_sig = results["is_significant"].sum()
        elif "p_value" in results.columns:
            n_sig = (results["p_value"] < alpha).sum()
        else:
            n_sig = 0

        reverse_rate = n_sig / len(reverse_pairs)
        passes = abs(reverse_rate - alpha) <= tolerance

        result = {
            "diagnostic": "ReverseDirectionCheck",
            "method": method_name,
            "n_detected_edges": len(detected_edges),
            "n_reverse_pairs": len(reverse_pairs),
            "n_reverse_significant": n_sig,
            "reverse_detection_rate": reverse_rate,
            "expected_fdr": alpha,
            "tolerance": tolerance,
            "pass": passes,
            "deviation": abs(reverse_rate - alpha),
        }

        logger.info("\n  Results:")
        logger.info(f"    Reverse significant: {n_sig}/{len(reverse_pairs)}")
        logger.info(f"    Reverse detection rate: {reverse_rate:.2%}")
        logger.info(f"    Expected: {alpha:.2%} ± {tolerance:.2%}")
        logger.info(f"    Deviation: {abs(reverse_rate - alpha):.2%}")
        logger.info("    ✅ PASS" if passes else "    ❌ FAIL")

        return result

    except Exception as e:
        logger.error(f"Reverse direction check failed: {e}")
        return {"pass": False, "error": str(e)}


def method_divergence_alert(
    results_dict: Dict[str, pd.DataFrame],
    alpha: float = 0.05,
    divergence_threshold: float = 0.20,
) -> Dict:
    """
    Check for systematic detection rate differences between methods.

    Alert if one method's detection rate is systematically higher than others
    by more than divergence_threshold.

    Parameters:
        results_dict: Dict of method_name -> results DataFrame
        alpha: Significance level
        divergence_threshold: Relative difference threshold (e.g., 0.20 = 20%)

    Returns:
        Dict with alerts and detection rates per method
    """
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC: Method Divergence Alert")
    logger.info("=" * 70)

    if len(results_dict) < 2:
        logger.info("  Only one method present, no divergence check needed")
        return {"pass": True, "warning": "Need at least 2 methods"}

    # Calculate detection rates per method
    detection_rates = {}
    total_tests = {}

    for method, results in results_dict.items():
        if results is None or results.empty:
            continue

        n_tests = len(results)

        # Count significant detections
        if "q_value" in results.columns:
            n_sig = (results["q_value"] < alpha).sum()
        elif "is_significant" in results.columns:
            n_sig = results["is_significant"].sum()
        elif "p_value" in results.columns:
            n_sig = (results["p_value"] < alpha).sum()
        else:
            n_sig = 0

        detection_rates[method] = n_sig / n_tests if n_tests > 0 else 0
        total_tests[method] = n_tests

    if len(detection_rates) < 2:
        logger.info("  Insufficient methods with results")
        return {"pass": True, "warning": "Need at least 2 methods with results"}

    # Check for divergence
    rates = list(detection_rates.values())
    mean_rate = np.mean(rates)
    max_rate = np.max(rates)
    min_rate = np.min(rates)

    # Relative divergence from mean
    divergences = {
        method: abs(rate - mean_rate) / mean_rate if mean_rate > 0 else 0
        for method, rate in detection_rates.items()
    }

    max_divergence_method = max(divergences, key=divergences.get)
    max_divergence = divergences[max_divergence_method]

    has_alert = max_divergence > divergence_threshold

    result = {
        "diagnostic": "MethodDivergenceAlert",
        "n_methods": len(detection_rates),
        "detection_rates": detection_rates,
        "total_tests": total_tests,
        "mean_detection_rate": mean_rate,
        "min_detection_rate": min_rate,
        "max_detection_rate": max_rate,
        "divergence_threshold": divergence_threshold,
        "max_divergence": max_divergence,
        "max_divergence_method": max_divergence_method,
        "has_alert": has_alert,
        "pass": not has_alert,
    }

    logger.info("\n  Detection rates by method:")
    for method, rate in sorted(detection_rates.items()):
        div = divergences[method]
        logger.info(
            f"    {method}: {rate:.2%} (n={total_tests[method]}, divergence={div:.1%})"
        )

    logger.info(f"\n  Mean detection rate: {mean_rate:.2%}")
    logger.info(f"  Range: [{min_rate:.2%}, {max_rate:.2%}]")
    logger.info(f"  Max divergence: {max_divergence:.1%} ({max_divergence_method})")
    logger.info(f"  Threshold: {divergence_threshold:.1%}")

    if has_alert:
        logger.warning(
            f"  ⚠️  ALERT: {max_divergence_method} diverges by {max_divergence:.1%}"
        )
    else:
        logger.info("  ✅ No significant divergence")

    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Create sample edges
    sample_edges = pd.DataFrame(
        [
            {
                "source": "RR",
                "target": "NDVI",
                "method": "Granger",
                "lag_days": 5,
                "p_value": 0.03,
                "is_significant": 1,
            },
            {
                "source": "TG",
                "target": "NDVI",
                "method": "TransferEntropy",
                "lag_days": 10,
                "p_value": 0.05,
                "is_significant": 1,
            },
            {
                "source": "NDVI",
                "target": "RR",
                "method": "PCMCI+",
                "lag_days": 15,
                "p_value": 0.02,
                "is_significant": 1,
            },
            {
                "source": "TG",
                "target": "RR",
                "method": "Granger",
                "lag_days": 0,
                "p_value": 0.001,
                "is_significant": 1,
            },
            {
                "source": "PP",
                "target": "TG",
                "method": "TransferEntropy",
                "lag_days": 5,
                "p_value": 0.08,
                "is_significant": 0,
            },
        ]
    )

    # Run diagnostics
    diag = Diagnostics(sample_edges, cadence_days=5)
    results = diag.run_all()

    print("\n✅ Diagnostics completed!")
