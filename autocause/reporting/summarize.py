"""
Report Summarization and Literature Alignment

Generates comprehensive analysis reports comparing framework results
against published baselines (Papagiannopoulou et al. 2017, Nature Climate Change).

Reference Baseline:
    - Detection rate: 61% of grid cells show water-to-NDVI causality
    - Dominant lags: 1-12 weeks (7-84 days)
    - Significant in: Mediterranean, Central Europe, Sahel
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)

# Reference baseline from literature
BASELINE = {
    "detection_rate": 0.61,  # 61% of grid cells
    "lag_range_days": (7, 84),  # 1-12 weeks
    "lag_range_weeks": (1, 12),
    "significant_regions": ["Mediterranean", "Central Europe", "Sahel"],
    "paper": "Papagiannopoulou et al. (2017), Nature Climate Change",
}


def compute_detection_statistics(
    results_df: pd.DataFrame, alpha: float = 0.05, min_lag: int = 1, max_lag: int = 12
) -> Dict:
    """
    Compute causal discovery statistics.

    Parameters:
        results_df (pd.DataFrame): Causal results with 'p_value', 'delay'/'lag' columns
        alpha (float): Significance threshold
        min_lag (int): Minimum lag to count (weeks)
        max_lag (int): Maximum lag to count (weeks)

    Returns:
        Dict: Statistics including detection rate, lag distribution
    """
    if results_df is None or len(results_df) == 0:
        return {
            "n_total": 0,
            "n_significant": 0,
            "detection_rate": 0,
            "mean_lag": np.nan,
            "median_lag": np.nan,
            "lags_in_range": 0,
            "pct_in_range": 0,
        }

    # Count significant results
    sig_mask = results_df.get("is_significant", results_df["p_value"] < alpha)
    n_sig = sig_mask.sum()
    n_total = len(results_df)

    # Filter for lag range
    lag_col = (
        results_df.columns.get("delay") if "delay" in results_df.columns else "lag"
    )
    if lag_col not in results_df.columns:
        lag_col = "delay" if "delay" in results_df.columns else "lag"

    lags = results_df[lag_col].dropna()
    lags_in_range = ((lags >= min_lag) & (lags <= max_lag) & sig_mask).sum()

    stats_dict = {
        "n_total": n_total,
        "n_significant": int(n_sig),
        "detection_rate": n_sig / n_total if n_total > 0 else 0,
        "mean_lag": lags[sig_mask].mean() if n_sig > 0 else np.nan,
        "median_lag": lags[sig_mask].median() if n_sig > 0 else np.nan,
        "std_lag": lags[sig_mask].std() if n_sig > 0 else np.nan,
        "lags_in_range": int(lags_in_range),
        "pct_in_range": (lags_in_range / n_sig * 100) if n_sig > 0 else 0,
    }

    logger.info(
        f"Detection Stats: {n_sig}/{n_total} significant ({100 * stats_dict['detection_rate']:.1f}%)"
    )
    logger.info(
        f"Lag stats: mean={stats_dict['mean_lag']:.1f}, median={stats_dict['median_lag']:.1f}"
    )

    return stats_dict


def compare_with_baseline(
    results_dict: Dict[str, pd.DataFrame],
    experiment_name: str = "Unknown",
) -> pd.DataFrame:
    """
    Compare framework results against Papagiannopoulou et al. (2017) baseline.

    Parameters:
        results_dict (Dict[str, pd.DataFrame]): Results by method
        experiment_name (str): Experiment name for reporting

    Returns:
        pd.DataFrame: Comparison table
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Literature Alignment: {experiment_name}")
    logger.info(f"Baseline: {BASELINE['paper']}")
    logger.info(f"{'=' * 70}")

    comparison_rows = []

    for method, results_df in results_dict.items():
        if results_df is None or len(results_df) == 0:
            logger.warning(f"No results for {method}")
            continue

        stats_dict = compute_detection_statistics(results_df)

        # Compare against baseline
        detection_diff = stats_dict["detection_rate"] - BASELINE["detection_rate"]
        in_lag_range = stats_dict["pct_in_range"]

        comparison_rows.append(
            {
                "method": method,
                "n_edges": stats_dict["n_total"],
                "n_significant": stats_dict["n_significant"],
                "detection_rate": stats_dict["detection_rate"],
                "baseline_rate": BASELINE["detection_rate"],
                "detection_diff": detection_diff,
                "mean_lag": stats_dict["mean_lag"],
                "baseline_lag_center": (
                    BASELINE["lag_range_weeks"][0] + BASELINE["lag_range_weeks"][1]
                )
                / 2,
                "pct_in_baseline_range": in_lag_range,
                "n_in_lag_range": stats_dict["lags_in_range"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    logger.info(f"\n{comparison_df.to_string()}")

    return comparison_df


def statistical_test_chi_square(
    observed: int, expected_rate: float, n_trials: int
) -> Tuple[float, float]:
    """
    Chi-square test: observed vs expected number of detections.

    Parameters:
        observed (int): Observed number of significant relationships
        expected_rate (float): Expected rate (from baseline)
        n_trials (int): Total number of tested pairs

    Returns:
        Tuple[float, float]: (chi2_statistic, p_value)
    """
    expected = expected_rate * n_trials
    chi2 = ((observed - expected) ** 2) / expected
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value


def statistical_test_binomial(
    observed: int, n_trials: int, baseline_rate: float
) -> Tuple[float, float]:
    """
    Binomial test: observed detections vs baseline probability.

    Parameters:
        observed (int): Observed number of significant relationships
        n_trials (int): Total number of tested pairs
        baseline_rate (float): Baseline probability (e.g., 0.61)

    Returns:
        Tuple[float, float]: (binomial_probability, p_value)
    """
    p_value = stats.binom_test(
        observed, n_trials, baseline_rate, alternative="two-sided"
    )
    prob = observed / n_trials if n_trials > 0 else 0

    return prob, p_value


def generate_summary_report(
    results_dict: Dict[str, pd.DataFrame],
    experiment_name: str = "Experiment",
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive summary report.

    Parameters:
        results_dict (Dict[str, pd.DataFrame]): Results by method
        experiment_name (str): Experiment name
        output_path (Optional[Path]): Save report path

    Returns:
        str: Formatted report text
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Generating Summary Report: {experiment_name}")
    logger.info(f"{'=' * 70}")

    # Get comparison table
    comparison_df = compare_with_baseline(results_dict, experiment_name)

    if comparison_df is None or len(comparison_df) == 0:
        logger.warning("No data for summary report")
        return ""

    # Build report
    report = f"""
{"=" * 70}
CAUSAL DISCOVERY ANALYSIS REPORT
{"=" * 70}

Experiment: {experiment_name}
Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

LITERATURE BASELINE
{"-" * 70}
Reference: {BASELINE["paper"]}
  - Detection Rate: {100 * BASELINE["detection_rate"]:.1f}%
  - Lag Range: {BASELINE["lag_range_weeks"][0]}-{BASELINE["lag_range_weeks"][1]} weeks ({BASELINE["lag_range_days"][0]}-{BASELINE["lag_range_days"][1]} days)
  - Significant Regions: {", ".join(BASELINE["significant_regions"])}

METHOD COMPARISON
{"-" * 70}
"""

    # Add comparison table
    report += comparison_df.to_string(index=False) + "\n\n"

    # Statistical tests
    report += f"""
STATISTICAL TESTS
{"-" * 70}
"""

    for _, row in comparison_df.iterrows():
        method = row["method"]
        observed = row["n_significant"]
        n_trials = row["n_edges"]

        chi2, p_chi2 = statistical_test_chi_square(
            observed, BASELINE["detection_rate"], n_trials
        )
        prob, p_binom = statistical_test_binomial(
            observed, n_trials, BASELINE["detection_rate"]
        )

        report += f"""
{method}:
  Chi-square test: χ² = {chi2:.4f}, p-value = {p_chi2:.4f}
  Binomial test: p = {prob:.4f}, p-value = {p_binom:.4f}
"""

    # Conclusions
    report += f"""
{"-" * 70}
CONCLUSIONS
{"-" * 70}

Summary of findings:
"""

    best_method = comparison_df.loc[comparison_df["n_significant"].idxmax()]
    report += f"""
  - Best performing method: {best_method["method"]} ({int(best_method["n_significant"])} significant edges)
  - Average detection rate: {100 * comparison_df["detection_rate"].mean():.1f}% (vs baseline {100 * BASELINE["detection_rate"]:.1f}%)
  - Methods exceeding baseline: {(comparison_df["detection_rate"] > BASELINE["detection_rate"]).sum()}/{len(comparison_df)}
"""

    report += f"\n{'=' * 70}\n"

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved: {output_path}")

    return report


def generate_latex_table(
    comparison_df: pd.DataFrame, output_path: Optional[Path] = None
) -> str:
    """
    Generate publication-ready LaTeX table.

    Parameters:
        comparison_df (pd.DataFrame): Comparison results
        output_path (Optional[Path]): Save path

    Returns:
        str: LaTeX table code
    """
    # Select columns for table
    display_df = comparison_df[
        [
            "method",
            "n_edges",
            "n_significant",
            "detection_rate",
            "mean_lag",
            "pct_in_baseline_range",
        ]
    ].copy()

    # Rename for readability
    display_df.columns = [
        "Method",
        "Total Pairs",
        "Significant",
        "Detection Rate",
        "Mean Lag (weeks)",
        "In Baseline Range (%)",
    ]

    # Format
    display_df["Detection Rate"] = display_df["Detection Rate"].apply(
        lambda x: f"{100 * x:.1f}%"
    )
    display_df["Mean Lag (weeks)"] = display_df["Mean Lag (weeks)"].apply(
        lambda x: f"{x:.1f}"
    )
    display_df["In Baseline Range (%)"] = display_df["In Baseline Range (%)"].apply(
        lambda x: f"{x:.1f}"
    )

    # Generate LaTeX
    latex_table = display_df.to_latex(index=False, escape=False)

    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        logger.info(f"LaTeX table saved: {output_path}")

    return latex_table
