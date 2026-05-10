"""
Plain-Text Causal Summary Generator

Generates human-readable, publication-ready summaries of causal relationships
in a standardized format: "X → Y (lag=N, p<threshold, strength=S)".

Key Features:
- Standardized formatting for reproducibility
- Strength metrics from multiple methods
- Effect size classification (weak/moderate/strong)
- Aggregation across units for panel data
- Compatible with academic writing and documentation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def classify_strength(value: float, metric: str = "correlation") -> str:
    """
    Classify effect strength using standard thresholds.

    Parameters:
        value (float): Strength metric value (absolute)
        metric (str): Type of metric ('correlation', 'granger_f', 'te_bits', 'pcmci_val')

    Returns:
        str: Classification ('weak', 'moderate', 'strong', 'very_strong')
    """
    value = abs(value)

    if metric == "correlation":
        # Cohen's thresholds
        if value < 0.3:
            return "weak"
        elif value < 0.5:
            return "moderate"
        elif value < 0.7:
            return "strong"
        else:
            return "very_strong"

    elif metric == "granger_f":
        # F-statistic thresholds (rough guidelines)
        if value < 5:
            return "weak"
        elif value < 10:
            return "moderate"
        elif value < 20:
            return "strong"
        else:
            return "very_strong"

    elif metric == "te_bits":
        # Transfer entropy in bits
        if value < 0.1:
            return "weak"
        elif value < 0.5:
            return "moderate"
        elif value < 1.0:
            return "strong"
        else:
            return "very_strong"

    elif metric == "pcmci_val":
        # Partial correlation values
        if value < 0.2:
            return "weak"
        elif value < 0.4:
            return "moderate"
        elif value < 0.6:
            return "strong"
        else:
            return "very_strong"

    else:
        # Default: treat as correlation
        if value < 0.3:
            return "weak"
        elif value < 0.5:
            return "moderate"
        else:
            return "strong"


def format_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Format p-value in standard notation.

    Parameters:
        p_value (float): P-value to format
        alpha (float): Significance threshold

    Returns:
        str: Formatted p-value (e.g., "p<0.001", "p=0.023", "p>0.05")
    """
    if p_value < 0.001:
        return "p<0.001"
    elif p_value < 0.01:
        return "p<0.01"
    elif p_value < alpha:
        return f"p<{alpha}"
    elif p_value < 0.1:
        return f"p={p_value:.3f}"
    else:
        return f"p>{alpha}"


def generate_causal_statement(
    source: str,
    target: str,
    lag_days: float,
    p_value: float,
    strength: Optional[float] = None,
    strength_metric: str = "correlation",
    n_units: Optional[int] = None,
    n_significant: Optional[int] = None,
    method: Optional[str] = None,
    confidence_level: Optional[str] = None,
) -> str:
    """
    Generate a standardized causal statement.

    Format: "X → Y (lag=N days, p<threshold, strength=S [classification])"

    Parameters:
        source (str): Source variable name
        target (str): Target variable name
        lag_days (float): Time lag in days
        p_value (float): Statistical significance
        strength (float): Effect strength (optional)
        strength_metric (str): Type of strength metric
        n_units (int): Total number of units tested (panel data)
        n_significant (int): Number of units with significant effect
        method (str): Causal method name (optional)
        confidence_level (str): Pre-classified confidence ('high', 'medium', 'low')

    Returns:
        str: Formatted causal statement

    Examples:
        >>> generate_causal_statement("RR", "NDVI", 35, 0.0001, 0.65, "correlation")
        'RR → NDVI (lag=35 days, p<0.001, strength=0.65 [strong])'

        >>> generate_causal_statement("TG", "NDVI", 30, 0.015, n_units=125, n_significant=54)
        'TG → NDVI (lag=30 days, p<0.05, significant in 54/125 units [43%])'
    """
    # Basic structure
    statement = f"{source} → {target} (lag={lag_days:.0f} days"

    # Add p-value
    p_str = format_p_value(p_value)
    statement += f", {p_str}"

    # Add strength if provided
    if strength is not None:
        classification = classify_strength(strength, strength_metric)
        statement += f", strength={strength:.2f} [{classification}]"

    # Add panel data statistics
    if n_units is not None and n_significant is not None:
        pct = 100 * n_significant / n_units
        statement += f", significant in {n_significant}/{n_units} units [{pct:.0f}%]"

    # Add method if provided
    if method:
        statement += f", method={method}"

    # Add confidence level if provided
    if confidence_level:
        statement += f", confidence={confidence_level}"

    statement += ")"

    return statement


def summarize_consensus_edges(
    consensus_df: pd.DataFrame,
    alpha: float = 0.05,
    include_method_details: bool = True,
    sort_by: str = "best_p_value",
) -> List[str]:
    """
    Generate plain-text summaries for all consensus edges.

    Parameters:
        consensus_df (pd.DataFrame): Consensus edges with columns:
            - source, target, lag_days, best_p_value, vote_count, n_significant, etc.
        alpha (float): Significance threshold
        include_method_details (bool): Include agreeing methods
        sort_by (str): Column to sort by ('best_p_value', 'vote_count', 'n_significant')

    Returns:
        List[str]: List of formatted causal statements
    """
    if consensus_df is None or len(consensus_df) == 0:
        logger.warning("No consensus edges to summarize")
        return []

    # Sort edges
    df_sorted = consensus_df.sort_values(by=sort_by).copy()

    statements = []

    for _, row in df_sorted.iterrows():
        # Extract confidence level from vote count
        if row["vote_count"] == 3:
            confidence = "high"
        elif row["vote_count"] == 2:
            confidence = "medium"
        else:
            confidence = "low"

        # Generate statement
        statement = generate_causal_statement(
            source=row["source"],
            target=row["target"],
            lag_days=row.get(
                "lag_days", row.get("lag_steps", 0) * 5
            ),  # Convert steps to days if needed
            p_value=row["best_p_value"],
            n_units=row.get("n_significant"),  # May need to infer total from context
            n_significant=row.get("n_significant"),
            confidence_level=confidence,
        )

        # Add method agreement details
        if include_method_details and "agreeing_methods" in row:
            methods = row["agreeing_methods"]
            statement += f"\n  Methods: {methods}"

        # Add vote breakdown
        if "vote_count" in row:
            statement += f"\n  Votes: {row['vote_count']}/3 methods agree"

        statements.append(statement)

    return statements


def summarize_method_results(
    results_df: pd.DataFrame,
    method_name: str,
    alpha: float = 0.05,
    top_n: int = 10,
    include_strength: bool = True,
) -> List[str]:
    """
    Generate plain-text summaries for top results from a single method.

    Parameters:
        results_df (pd.DataFrame): Method results
        method_name (str): Name of causal method
        alpha (float): Significance threshold
        top_n (int): Number of top edges to summarize
        include_strength (bool): Include strength metrics if available

    Returns:
        List[str]: List of formatted causal statements
    """
    if results_df is None or len(results_df) == 0:
        logger.warning(f"No results to summarize for {method_name}")
        return []

    # Detect p-value column
    p_col = None
    for col in ["p_value", "best_p_value", "pvalue"]:
        if col in results_df.columns:
            p_col = col
            break

    if p_col is None:
        logger.warning(f"No p-value column found in {method_name} results")
        return [f"{method_name}: No p-value column found (cannot assess significance)"]

    # Filter significant results
    sig_mask = results_df.get("is_significant", results_df[p_col] < alpha)
    df_sig = results_df[sig_mask].copy()

    if len(df_sig) == 0:
        return [
            f"{method_name}: No significant causal relationships detected (α={alpha})"
        ]

    # Sort by p-value
    df_sig = df_sig.sort_values(p_col).head(top_n)

    statements = [f"{method_name} Top {len(df_sig)} Results:"]

    for i, (_, row) in enumerate(df_sig.iterrows(), 1):
        # Detect lag column
        lag_days = None
        for col in ["lag_days", "best_lag_days", "lag", "best_lag"]:
            if col in row and pd.notna(row[col]):
                lag_days = row[col]
                if "days" not in col:  # Convert steps to days
                    lag_days *= 5
                break

        if lag_days is None:
            lag_days = 0

        # Detect strength metric
        strength = None
        strength_metric = "correlation"

        if include_strength:
            if "granger_beta_std" in row and pd.notna(row["granger_beta_std"]):
                strength = row["granger_beta_std"]
                strength_metric = "granger_f"
            elif "te_bits" in row and pd.notna(row["te_bits"]):
                strength = row["te_bits"]
                strength_metric = "te_bits"
            elif "val_matrix" in row and pd.notna(row["val_matrix"]):
                strength = row["val_matrix"]
                strength_metric = "pcmci_val"

        statement = generate_causal_statement(
            source=row["source"] if "source" in row else row.get("cause", "?"),
            target=row["target"] if "target" in row else row.get("effect", "?"),
            lag_days=lag_days,
            p_value=row[p_col],
            strength=strength,
            strength_metric=strength_metric,
            method=method_name,
        )

        statements.append(f"{i}. {statement}")

    return statements


def generate_full_summary_report(
    consensus_df: pd.DataFrame,
    results_dict: Dict[str, pd.DataFrame],
    output_path: Optional[Path] = None,
    experiment_name: str = "Causal Discovery Analysis",
    alpha: float = 0.05,
    top_n_per_method: int = 5,
) -> str:
    """
    Generate a comprehensive plain-text summary report.

    Includes:
    - Consensus edges (high-confidence findings)
    - Top results per method
    - Summary statistics
    - Interpretation guidelines

    Parameters:
        consensus_df (pd.DataFrame): Consensus edges
        results_dict (Dict): Results per method {method_name: results_df}
        output_path (Path): Path to save report (optional)
        experiment_name (str): Experiment name
        alpha (float): Significance threshold
        top_n_per_method (int): Number of top results per method

    Returns:
        str: Full report text
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{experiment_name.upper()}")
    lines.append("PLAIN-TEXT CAUSAL SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    # Consensus section
    lines.append("CONSENSUS FINDINGS (Multi-Method Agreement)")
    lines.append("-" * 80)

    if consensus_df is not None and len(consensus_df) > 0:
        consensus_statements = summarize_consensus_edges(
            consensus_df,
            alpha=alpha,
            include_method_details=True,
        )
        lines.extend(consensus_statements)
    else:
        lines.append("No consensus edges detected.")

    lines.append("")
    lines.append("")

    # Method-specific sections
    lines.append("METHOD-SPECIFIC RESULTS")
    lines.append("-" * 80)

    for method_name, results_df in results_dict.items():
        lines.append("")
        method_statements = summarize_method_results(
            results_df,
            method_name=method_name,
            alpha=alpha,
            top_n=top_n_per_method,
            include_strength=True,
        )
        lines.extend(method_statements)

    lines.append("")
    lines.append("")

    # Summary statistics
    lines.append("SUMMARY STATISTICS")
    lines.append("-" * 80)

    if consensus_df is not None and len(consensus_df) > 0:
        n_consensus = len(consensus_df)
        n_3way = len(consensus_df[consensus_df["vote_count"] == 3])
        n_2way = len(consensus_df[consensus_df["vote_count"] == 2])
        n_1way = len(consensus_df[consensus_df["vote_count"] == 1])

        lines.append(f"Total consensus edges: {n_consensus}")
        lines.append(f"  - 3-method agreement: {n_3way} (high confidence)")
        lines.append(f"  - 2-method agreement: {n_2way} (medium confidence)")
        lines.append(f"  - 1-method only: {n_1way} (low confidence)")
        lines.append("")

    for method_name, results_df in results_dict.items():
        if results_df is not None and len(results_df) > 0:
            # Detect p-value column
            p_col = None
            for col in ["p_value", "best_p_value", "pvalue"]:
                if col in results_df.columns:
                    p_col = col
                    break

            if p_col is None:
                continue

            n_total = len(results_df)
            sig_mask = results_df.get("is_significant", results_df[p_col] < alpha)
            n_sig = sig_mask.sum()
            detection_rate = 100 * n_sig / n_total if n_total > 0 else 0

            lines.append(f"{method_name}:")
            lines.append(f"  - Total tests: {n_total}")
            lines.append(f"  - Significant: {n_sig} ({detection_rate:.1f}%)")

    lines.append("")
    lines.append("")

    # Interpretation guide
    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 80)
    lines.append("Confidence Levels:")
    lines.append("  - HIGH: 3/3 methods agree (strongest evidence)")
    lines.append("  - MEDIUM: 2/3 methods agree (reliable evidence)")
    lines.append("  - LOW: 1/3 methods only (exploratory)")
    lines.append("")
    lines.append("Strength Classifications:")
    lines.append("  - WEAK: Small but detectable effect")
    lines.append("  - MODERATE: Clearly observable effect")
    lines.append("  - STRONG: Dominant causal influence")
    lines.append("  - VERY_STRONG: Overwhelming causal effect")
    lines.append("")
    lines.append("Statistical Significance:")
    lines.append(f"  - α = {alpha} (standard threshold)")
    lines.append("  - p<0.001: Very strong evidence")
    lines.append("  - p<0.01: Strong evidence")
    lines.append("  - p<0.05: Moderate evidence")
    lines.append("")

    lines.append("=" * 80)

    # Combine into single string
    report_text = "\n".join(lines)

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)
        logger.info(f"Summary report saved to: {output_path}")

    return report_text
