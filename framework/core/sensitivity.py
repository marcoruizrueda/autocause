"""
Sensitivity Analysis & Statistical Enhancements

Implements:
1. Seasonality controls for Granger causality
2. Placebo/permutation tests for false positive rate estimation
3. Bootstrap confidence intervals for lag estimates
4. Mediterranean short-lag validation (τ_max=3)

Author: Enhanced Analysis Framework
Date: October 30, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


def add_seasonal_terms(
    df: pd.DataFrame,
    date_column: Optional[str] = None,
    period: int = 365,
    n_harmonics: int = 2,
) -> pd.DataFrame:
    """
    Add seasonal harmonic terms to dataframe for Granger causality.

    Uses Fourier terms: sin(2πkt/T) and cos(2πkt/T) for k=1,...,n_harmonics

    Parameters:
        df: Input dataframe with time series
        date_column: Name of date column (if None, uses index)
        period: Seasonal period in days (default: 365 for annual)
        n_harmonics: Number of harmonic pairs to include (default: 2)

    Returns:
        DataFrame with added seasonal terms (sin1, cos1, sin2, cos2, ...)
    """
    df_out = df.copy()

    # Get time index
    if date_column and date_column in df.columns:
        time_idx = pd.to_datetime(df[date_column])
    else:
        time_idx = pd.to_datetime(df.index)

    # Convert to day-of-year
    t = time_idx.dayofyear.values

    # Add harmonic terms
    for k in range(1, n_harmonics + 1):
        freq = 2 * np.pi * k / period
        df_out[f"sin{k}"] = np.sin(freq * t)
        df_out[f"cos{k}"] = np.cos(freq * t)
        logger.info(f"Added seasonal harmonic {k}: sin{k}, cos{k}")

    return df_out


def run_granger_with_without_seasonals(
    df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    maxlag: int = 12,
    alpha: float = 0.05,
    controls: Optional[List[str]] = None,
    granger_func: Optional[Callable] = None,
    **granger_kwargs,
) -> Dict:
    """
    Run Granger causality with and without seasonal controls.

    Compares results to assess sensitivity to seasonal confounding.

    Parameters:
        df: Input dataframe
        cause_var: Cause variable name
        effect_var: Effect variable name
        maxlag: Maximum lag to test
        alpha: Significance level
        controls: Additional control variables
        granger_func: Granger causality function to use
        **granger_kwargs: Additional arguments to pass to granger_func

    Returns:
        Dict with results from both runs and comparison metrics
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SEASONALITY SENSITIVITY: {cause_var} → {effect_var}")
    logger.info(f"{'=' * 70}")

    if granger_func is None:
        from framework.core.methods.granger import run_granger_causality

        granger_func = run_granger_causality

    # Run WITHOUT seasonal controls
    logger.info("\n--- WITHOUT Seasonal Controls ---")
    result_without = granger_func(
        df=df,
        cause_var=cause_var,
        effect_var=effect_var,
        maxlag=maxlag,
        alpha=alpha,
        controls=controls,
        verbose=False,
        **granger_kwargs,
    )

    # Add seasonal terms
    df_seasonal = add_seasonal_terms(df, n_harmonics=2)

    # Add seasonal terms to controls
    seasonal_cols = [
        c for c in df_seasonal.columns if c.startswith("sin") or c.startswith("cos")
    ]
    controls_with_seasonal = (controls or []) + seasonal_cols

    # Run WITH seasonal controls
    logger.info("\n--- WITH Seasonal Controls ---")
    result_with = granger_func(
        df=df_seasonal,
        cause_var=cause_var,
        effect_var=effect_var,
        maxlag=maxlag,
        alpha=alpha,
        controls=controls_with_seasonal,
        verbose=False,
        **granger_kwargs,
    )

    # Compare results
    comparison = {
        "without_seasonals": {
            "is_causal": result_without.get("is_causal", False),
            "best_lag": result_without.get("best_lag", None),
            "best_p_value": result_without.get("best_p_value", None),
            "best_q_value": result_without.get("best_q_value", None),
        },
        "with_seasonals": {
            "is_causal": result_with.get("is_causal", False),
            "best_lag": result_with.get("best_lag", None),
            "best_p_value": result_with.get("best_p_value", None),
            "best_q_value": result_with.get("best_q_value", None),
        },
        "agreement": result_without.get("is_causal", False)
        == result_with.get("is_causal", False),
        "lag_shift": None,
        "significance_robust": False,
    }

    # Calculate lag shift if both detected
    if result_without.get("best_lag") and result_with.get("best_lag"):
        comparison["lag_shift"] = abs(
            result_without["best_lag"] - result_with["best_lag"]
        )

    # Check if significance is robust
    if result_without.get("is_causal") and result_with.get("is_causal"):
        comparison["significance_robust"] = True

    logger.info(f"\nComparison:")
    logger.info(f"  Agreement: {comparison['agreement']}")
    logger.info(f"  Lag shift: {comparison['lag_shift']}")
    logger.info(f"  Robust to seasonals: {comparison['significance_robust']}")

    return {
        "without_seasonals": result_without,
        "with_seasonals": result_with,
        "comparison": comparison,
    }


def permutation_test(
    df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    method_func: Callable,
    n_permutations: int = 100,
    seed: int = 42,
    **method_kwargs,
) -> Dict:
    """
    Run permutation/placebo test to estimate false positive rate.

    Randomly shuffles the effect variable to break temporal structure
    and reruns causal test to estimate null distribution.

    Parameters:
        df: Input dataframe
        cause_var: Cause variable name
        effect_var: Effect variable name (will be shuffled)
        method_func: Causal inference function to test
        n_permutations: Number of random shuffles
        seed: Random seed for reproducibility
        **method_kwargs: Arguments to pass to method_func

    Returns:
        Dict with observed result, null distribution, and p-value
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PERMUTATION TEST: {cause_var} → {effect_var}")
    logger.info(f"  Permutations: {n_permutations}")
    logger.info(f"{'=' * 70}")

    np.random.seed(seed)

    # Get observed result (real data)
    logger.info("\nRunning on OBSERVED data...")
    observed_result = method_func(
        df=df,
        cause_var=cause_var,
        effect_var=effect_var,
        verbose=False,
        **method_kwargs,
    )

    observed_significant = observed_result.get(
        "is_causal", False
    ) or observed_result.get("is_significant", False)
    observed_statistic = observed_result.get("best_p_value", 1.0)

    logger.info(
        f"  Observed: significant={observed_significant}, p={observed_statistic:.4f}"
    )

    # Run permutations
    logger.info(f"\nRunning {n_permutations} PERMUTATIONS...")
    null_significant = []
    null_statistics = []

    for i in range(n_permutations):
        # Shuffle effect variable (breaks temporal structure)
        df_shuffled = df.copy()
        df_shuffled[effect_var] = np.random.permutation(df[effect_var].values)

        try:
            perm_result = method_func(
                df=df_shuffled,
                cause_var=cause_var,
                effect_var=effect_var,
                verbose=False,
                **method_kwargs,
            )

            perm_sig = perm_result.get("is_causal", False) or perm_result.get(
                "is_significant", False
            )
            perm_stat = perm_result.get("best_p_value", 1.0)

            null_significant.append(perm_sig)
            null_statistics.append(perm_stat)

        except Exception as e:
            logger.warning(f"Permutation {i + 1} failed: {e}")
            null_significant.append(False)
            null_statistics.append(1.0)

    # Calculate empirical FPR
    fpr = np.mean(null_significant)

    # Calculate permutation p-value (how extreme is observed statistic?)
    # Count how many permutations had lower p-value than observed
    if observed_statistic is not None:
        perm_p_value = np.mean(
            [s <= observed_statistic for s in null_statistics if s is not None]
        )
    else:
        perm_p_value = None

    logger.info(f"\nPermutation Test Results:")
    logger.info(
        f"  Empirical FPR: {fpr:.2%} ({sum(null_significant)}/{n_permutations})"
    )
    logger.info(
        f"  Permutation p-value: {perm_p_value:.4f}"
        if perm_p_value
        else "  Permutation p-value: N/A"
    )

    return {
        "observed": {
            "significant": observed_significant,
            "statistic": observed_statistic,
            "result": observed_result,
        },
        "null": {
            "n_permutations": n_permutations,
            "significant_count": sum(null_significant),
            "fpr": fpr,
            "statistics": null_statistics,
            "p_value": perm_p_value,
        },
        "interpretation": {
            "is_robust": perm_p_value < 0.05 if perm_p_value else False,
            "fpr_acceptable": fpr < 0.1,  # FPR should be below nominal alpha
        },
    }


def bootstrap_lag_ci(
    df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    method_func: Callable,
    n_bootstrap: int = 100,
    ci_level: float = 0.95,
    seed: int = 42,
    **method_kwargs,
) -> Dict:
    """
    Bootstrap confidence intervals for lag estimates.

    Resamples data with replacement and re-estimates lag to get CI.

    Parameters:
        df: Input dataframe
        cause_var: Cause variable name
        effect_var: Effect variable name
        method_func: Causal inference function
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed
        **method_kwargs: Arguments to pass to method_func

    Returns:
        Dict with point estimate, CI bounds, and bootstrap distribution
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"BOOTSTRAP LAG CI: {cause_var} → {effect_var}")
    logger.info(f"  Samples: {n_bootstrap}, CI: {ci_level * 100:.0f}%")
    logger.info(f"{'=' * 70}")

    np.random.seed(seed)

    # Point estimate (original data)
    logger.info("\nComputing point estimate...")
    point_result = method_func(
        df=df,
        cause_var=cause_var,
        effect_var=effect_var,
        verbose=False,
        **method_kwargs,
    )

    point_lag = point_result.get("best_lag") or point_result.get("lag_steps")
    point_sig = point_result.get("is_causal", False) or point_result.get(
        "is_significant", False
    )

    logger.info(f"  Point estimate: lag={point_lag}, significant={point_sig}")

    if not point_sig or point_lag is None:
        logger.warning(
            "Point estimate not significant or lag unavailable. Skipping bootstrap."
        )
        return {
            "point_estimate": point_lag,
            "ci_lower": None,
            "ci_upper": None,
            "bootstrap_distribution": [],
            "ci_level": ci_level,
            "note": "Skipped - point estimate not significant",
        }

    # Bootstrap resampling
    logger.info(f"\nBootstrapping {n_bootstrap} samples...")
    bootstrap_lags = []
    bootstrap_significant = []

    n = len(df)

    for i in range(n_bootstrap):
        # Resample with replacement (block bootstrap to preserve temporal structure)
        # Use block size of sqrt(n) as heuristic
        block_size = max(1, int(np.sqrt(n)))
        n_blocks = int(np.ceil(n / block_size))

        # Generate random block starts
        block_starts = np.random.randint(0, n - block_size + 1, size=n_blocks)

        # Build resampled indices
        resampled_indices = []
        for start in block_starts:
            resampled_indices.extend(range(start, min(start + block_size, n)))
        resampled_indices = resampled_indices[:n]  # Trim to original length

        df_boot = df.iloc[resampled_indices].reset_index(drop=True)

        try:
            boot_result = method_func(
                df=df_boot,
                cause_var=cause_var,
                effect_var=effect_var,
                verbose=False,
                **method_kwargs,
            )

            boot_lag = boot_result.get("best_lag") or boot_result.get("lag_steps")
            boot_sig = boot_result.get("is_causal", False) or boot_result.get(
                "is_significant", False
            )

            if boot_sig and boot_lag is not None:
                bootstrap_lags.append(boot_lag)
                bootstrap_significant.append(True)
            else:
                bootstrap_significant.append(False)

        except Exception as e:
            logger.debug(f"Bootstrap {i + 1} failed: {e}")
            bootstrap_significant.append(False)

    # Calculate CI from bootstrap distribution
    if len(bootstrap_lags) > 0:
        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_lags, lower_percentile)
        ci_upper = np.percentile(bootstrap_lags, upper_percentile)

        logger.info(f"\nBootstrap Results:")
        logger.info(f"  Valid samples: {len(bootstrap_lags)}/{n_bootstrap}")
        logger.info(
            f"  Lag CI ({ci_level * 100:.0f}%): [{ci_lower:.1f}, {ci_upper:.1f}]"
        )
        logger.info(f"  Point estimate: {point_lag}")
        logger.info(f"  Bootstrap mean: {np.mean(bootstrap_lags):.1f}")
        logger.info(f"  Bootstrap std: {np.std(bootstrap_lags):.1f}")
    else:
        logger.warning("No valid bootstrap samples - cannot compute CI")
        ci_lower, ci_upper = None, None

    return {
        "point_estimate": point_lag,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "bootstrap_distribution": bootstrap_lags,
        "n_valid_samples": len(bootstrap_lags),
        "n_total_samples": n_bootstrap,
        "bootstrap_mean": float(np.mean(bootstrap_lags)) if bootstrap_lags else None,
        "bootstrap_std": float(np.std(bootstrap_lags)) if bootstrap_lags else None,
    }


def run_short_lag_validation(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    method_func: Callable,
    tau_max_short: int = 3,
    tau_max_long: int = 12,
    alpha: float = 0.05,
    **method_kwargs,
) -> Dict:
    """
    Validate short-lag signals (e.g., Mediterranean 5-10 day findings).

    Runs analysis with restricted tau_max to check if short lags
    are artifacts of long tau search or genuine signals.

    Parameters:
        df: Input dataframe
        pairs: List of (cause, effect) tuples to test
        method_func: Causal inference function
        tau_max_short: Short lag maximum (default: 3 steps = 15 days)
        tau_max_long: Long lag maximum (default: 12 steps = 60 days)
        alpha: Significance level
        **method_kwargs: Arguments to pass to method_func

    Returns:
        Dict with results from both tau_max settings and comparison
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"SHORT-LAG VALIDATION")
    logger.info(f"  τ_max short: {tau_max_short} steps")
    logger.info(f"  τ_max long: {tau_max_long} steps")
    logger.info(f"{'=' * 70}")

    results = {
        "tau_max_short": tau_max_short,
        "tau_max_long": tau_max_long,
        "pairs": {},
    }

    for cause, effect in pairs:
        logger.info(f"\n--- Testing: {cause} → {effect} ---")

        # Run with SHORT tau_max
        logger.info(f"Running with τ_max={tau_max_short}...")
        result_short = method_func(
            df=df,
            cause_var=cause,
            effect_var=effect,
            maxlag=tau_max_short,
            alpha=alpha,
            verbose=False,
            **method_kwargs,
        )

        # Run with LONG tau_max
        logger.info(f"Running with τ_max={tau_max_long}...")
        result_long = method_func(
            df=df,
            cause_var=cause,
            effect_var=effect,
            maxlag=tau_max_long,
            alpha=alpha,
            verbose=False,
            **method_kwargs,
        )

        # Compare
        short_lag = result_short.get("best_lag") or result_short.get("lag_steps")
        short_sig = result_short.get("is_causal", False) or result_short.get(
            "is_significant", False
        )

        long_lag = result_long.get("best_lag") or result_long.get("lag_steps")
        long_sig = result_long.get("is_causal", False) or result_long.get(
            "is_significant", False
        )

        # Check if short lag is preserved in long search
        short_lag_preserved = False
        if short_sig and long_sig and short_lag and long_lag:
            short_lag_preserved = (long_lag <= tau_max_short) or (
                abs(long_lag - short_lag) <= 1
            )

        pair_key = f"{cause}->{effect}"
        results["pairs"][pair_key] = {
            "short_tau": {
                "lag": short_lag,
                "significant": short_sig,
                "p_value": result_short.get("best_p_value"),
            },
            "long_tau": {
                "lag": long_lag,
                "significant": long_sig,
                "p_value": result_long.get("best_p_value"),
            },
            "short_lag_preserved": short_lag_preserved,
            "interpretation": "SHORT LAG VALIDATED"
            if short_lag_preserved
            else "ARTIFACT (not preserved)",
        }

        logger.info(f"  Short τ: lag={short_lag}, sig={short_sig}")
        logger.info(f"  Long τ: lag={long_lag}, sig={long_sig}")
        logger.info(f"  → {results['pairs'][pair_key]['interpretation']}")

    # Summary
    n_validated = sum(1 for p in results["pairs"].values() if p["short_lag_preserved"])
    n_total = len(results["pairs"])

    results["summary"] = {
        "n_pairs_tested": n_total,
        "n_short_lags_validated": n_validated,
        "validation_rate": n_validated / n_total if n_total > 0 else 0,
    }

    logger.info(f"\n{'=' * 70}")
    logger.info(f"SHORT-LAG VALIDATION SUMMARY")
    logger.info(f"  Validated: {n_validated}/{n_total} pairs")
    logger.info(f"  Rate: {results['summary']['validation_rate']:.1%}")
    logger.info(f"{'=' * 70}")

    return results


def save_sensitivity_results(
    results: Dict, output_dir: Path, filename: str = "sensitivity_analysis.json"
):
    """Save sensitivity analysis results to JSON."""
    import json

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results_clean = convert_types(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_clean, f, indent=2)

    logger.info(f"✓ Saved sensitivity results: {output_path}")
    return output_path
