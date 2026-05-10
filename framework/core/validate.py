"""
Validation Module: Temporal Causal Validation

Provides temporal validation methods for causal discovery including:
- Cross-validation for edge stability across temporal splits
- Block bootstrap confidence intervals
- Surrogate data tests for null hypothesis testing
- CUSUM structural stability detection

Dataset-agnostic: Works with any time series DataFrame with numeric columns.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


# ============================================================================
# Temporal Causal Validation
# ============================================================================


def temporal_cross_validation(
    data: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    method_fn: Callable,
    n_splits: int = 3,
    test_size: Optional[int] = None,
    min_train_size: Optional[int] = None,
    **method_kwargs,
) -> Dict:
    """
    Perform time-split cross-validation for causal edges.

    Validates that causal relationships are stable across temporal splits,
    not just artifacts of a specific time period.

    Parameters:
        data: Time series DataFrame
        pairs: List of (source, target) tuples to test
        method_fn: Causal discovery function (must accept data, pairs, **kwargs)
        n_splits: Number of temporal folds
        test_size: Size of test set (None = auto)
        min_train_size: Minimum training set size (None = auto)
        **method_kwargs: Additional arguments for method_fn

    Returns:
        Dict with:
        - stable_edges: Edges appearing in ≥ (n_splits - 1) folds
        - edge_votes: Dict mapping edge -> number of folds detecting it
        - fold_results: Per-fold DataFrames
        - stability_score: Fraction of consistent edges

    Example:
        >>> from framework.core.methods import granger
        >>> stable = temporal_cross_validation(
        ...     data, pairs, granger.batch_granger_causality,
        ...     n_splits=3, alpha=0.05
        ... )
        >>> print(f"Stable edges: {len(stable['stable_edges'])}")
    """
    if len(data) < n_splits * 30:
        logger.warning(
            f"Data length ({len(data)}) may be too short for "
            f"{n_splits}-fold cross-validation"
        )

    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=test_size,
        max_train_size=None,
    )

    fold_results = []
    edge_votes = {}

    logger.info(f"Running {n_splits}-fold temporal cross-validation...")

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
        train_data = data.iloc[train_idx]

        logger.info(
            f"  Fold {fold_idx}/{n_splits}: "
            f"train=[{train_idx[0]}:{train_idx[-1]}], "
            f"test=[{test_idx[0]}:{test_idx[-1]}]"
        )

        try:
            # Run causal method on training fold
            fold_edges = method_fn(train_data, pairs, **method_kwargs)

            if fold_edges is not None and len(fold_edges) > 0:
                fold_results.append(fold_edges)

                # Count votes for each edge
                for _, row in fold_edges.iterrows():
                    if "is_significant" in row and row["is_significant"]:
                        edge = (row["source"], row["target"])
                        edge_votes[edge] = edge_votes.get(edge, 0) + 1

        except Exception as e:
            logger.warning(f"  Fold {fold_idx} failed: {e}")

    # Identify stable edges (appear in at least n_splits - 1 folds)
    min_votes = max(1, n_splits - 1)
    stable_edges = [edge for edge, votes in edge_votes.items() if votes >= min_votes]

    # Calculate stability score
    all_edges = set(edge_votes.keys())
    stability_score = len(stable_edges) / len(all_edges) if all_edges else 0.0

    logger.info(
        f"  ✅ Stable edges: {len(stable_edges)}/{len(all_edges)} "
        f"(stability={stability_score:.1%})"
    )

    return {
        "stable_edges": stable_edges,
        "edge_votes": edge_votes,
        "fold_results": fold_results,
        "stability_score": stability_score,
        "n_folds": len(fold_results),
        "min_votes_required": min_votes,
    }


def block_bootstrap_ci(
    data: pd.DataFrame,
    source: str,
    target: str,
    method_fn: Callable,
    n_bootstrap: int = 100,
    block_size: Optional[int] = None,
    confidence_level: float = 0.95,
    **method_kwargs,
) -> Dict:
    """
    Block bootstrap confidence intervals for causal effect size.

    Preserves autocorrelation structure by resampling in blocks rather
    than individual observations.

    Parameters:
        data: Time series DataFrame
        source: Source variable
        target: Target variable
        method_fn: Causal method returning dict with 'effect_size' key
        n_bootstrap: Number of bootstrap samples
        block_size: Block size (None = auto as sqrt(n))
        confidence_level: CI level (e.g., 0.95 = 95%)
        **method_kwargs: Additional method arguments

    Returns:
        Dict with:
        - point_estimate: Original effect size
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
        - bootstrap_distribution: Array of bootstrap effect sizes
        - se: Bootstrap standard error

    Example:
        >>> from framework.core.methods import granger
        >>> ci = block_bootstrap_ci(
        ...     data, 'var1', 'var2',
        ...     granger.run_granger_causality,
        ...     n_bootstrap=100, maxlag=5
        ... )
        >>> print(f"Effect: {ci['point_estimate']:.3f} "
        ...       f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]")
    """
    n = len(data)

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    # Compute point estimate on original data
    try:
        original_result = method_fn(
            data[[source, target]], source, target, **method_kwargs
        )
        point_estimate = original_result.get("effect_size", np.nan)
    except Exception as e:
        logger.error(f"Failed to compute point estimate: {e}")
        point_estimate = np.nan

    # Bootstrap
    bootstrap_effects = []

    logger.info(
        f"Block bootstrap: {source}→{target} "
        f"(n_bootstrap={n_bootstrap}, block_size={block_size})"
    )

    for i in range(n_bootstrap):
        # Block resample
        n_blocks = n // block_size
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)

        resampled_indices = []
        for block_idx in block_indices:
            start = block_idx * block_size
            end = min(start + block_size, n)
            resampled_indices.extend(range(start, end))

        # Trim to original length
        resampled_indices = resampled_indices[:n]
        bs_data = data.iloc[resampled_indices].reset_index(drop=True)

        # Run method on bootstrap sample
        try:
            bs_result = method_fn(
                bs_data[[source, target]], source, target, **method_kwargs
            )
            effect = bs_result.get("effect_size", np.nan)
            bootstrap_effects.append(effect)
        except Exception:
            bootstrap_effects.append(np.nan)

    # Remove NaN
    bootstrap_effects = np.array([e for e in bootstrap_effects if not np.isnan(e)])

    if len(bootstrap_effects) < 10:
        logger.warning("Too few successful bootstrap samples")
        return {
            "point_estimate": point_estimate,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "bootstrap_distribution": bootstrap_effects,
            "se": np.nan,
        }

    # Compute CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_effects, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)
    se = np.std(bootstrap_effects)

    logger.info(
        f"  ✅ Bootstrap CI ({confidence_level:.0%}): "
        f"[{ci_lower:.3f}, {ci_upper:.3f}], SE={se:.3f}"
    )

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_distribution": bootstrap_effects,
        "se": se,
        "n_successful": len(bootstrap_effects),
    }


def surrogate_test(
    data: pd.DataFrame,
    source: str,
    target: str,
    method_fn: Callable,
    n_surrogates: int = 100,
    surrogate_method: str = "phase_randomization",
    **method_kwargs,
) -> Dict:
    """
    Surrogate data test for null hypothesis of no causality.

    Generates surrogate time series that preserve autocorrelation
    structure but destroy causal relationships. If original effect
    is significantly larger than surrogate distribution, reject null.

    Parameters:
        data: Time series DataFrame
        source: Source variable
        target: Target variable
        method_fn: Causal method returning 'effect_size'
        n_surrogates: Number of surrogate series
        surrogate_method: 'phase_randomization', 'block_shuffle', or 'iaaft'
        **method_kwargs: Method arguments

    Returns:
        Dict with:
        - original_effect: Effect size on real data
        - surrogate_mean: Mean effect on surrogates
        - surrogate_std: Std of surrogate effects
        - p_value: Fraction of surrogates ≥ original
        - is_significant: Whether p < alpha
        - surrogate_distribution: Array of surrogate effects

    Example:
        >>> result = surrogate_test(
        ...     data, 'var1', 'var2',
        ...     granger.run_granger_causality,
        ...     n_surrogates=100, maxlag=5
        ... )
        >>> print(f"p-value: {result['p_value']:.4f}")
    """
    # Compute original effect
    try:
        original_result = method_fn(
            data[[source, target]], source, target, **method_kwargs
        )
        original_effect = original_result.get("effect_size", np.nan)
    except Exception as e:
        logger.error(f"Failed to compute original effect: {e}")
        return {"error": str(e)}

    # Generate surrogates and test
    surrogate_effects = []

    logger.info(
        f"Surrogate test: {source}→{target} "
        f"(n_surrogates={n_surrogates}, method={surrogate_method})"
    )

    for i in range(n_surrogates):
        # Generate surrogate source series
        if surrogate_method == "block_shuffle":
            surrogate_source = _block_shuffle(data[source].values)
        elif surrogate_method == "phase_randomization":
            surrogate_source = _phase_randomization(data[source].values)
        else:
            surrogate_source = _block_shuffle(data[source].values)

        # Create surrogate dataset
        surrogate_data = data[[source, target]].copy()
        surrogate_data[source] = surrogate_source

        # Run method on surrogate
        try:
            surrogate_result = method_fn(
                surrogate_data, source, target, **method_kwargs
            )
            effect = surrogate_result.get("effect_size", np.nan)
            surrogate_effects.append(effect)
        except Exception:
            surrogate_effects.append(np.nan)

    # Remove NaN
    surrogate_effects = np.array([e for e in surrogate_effects if not np.isnan(e)])

    if len(surrogate_effects) < 10:
        logger.warning("Too few successful surrogate tests")
        return {
            "original_effect": original_effect,
            "error": "Insufficient successful surrogates",
        }

    # Compute p-value: fraction of surrogates >= original
    if not np.isnan(original_effect):
        p_value = (surrogate_effects >= original_effect).sum() / len(surrogate_effects)
    else:
        p_value = np.nan

    logger.info(
        f"  ✅ Surrogate test: p={p_value:.4f}, "
        f"original={original_effect:.3f}, "
        f"surrogate_mean={surrogate_effects.mean():.3f}"
    )

    return {
        "original_effect": original_effect,
        "surrogate_mean": float(surrogate_effects.mean()),
        "surrogate_std": float(surrogate_effects.std()),
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "is_significant": p_value < method_kwargs.get("alpha", 0.05)
        if not np.isnan(p_value)
        else False,
        "surrogate_distribution": surrogate_effects,
        "n_successful": len(surrogate_effects),
    }


def _block_shuffle(series: np.ndarray, block_size: Optional[int] = None) -> np.ndarray:
    """Shuffle time series in blocks to preserve autocorrelation"""
    n = len(series)

    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))

    n_blocks = n // block_size
    block_indices = np.arange(n_blocks)
    np.random.shuffle(block_indices)

    shuffled = np.concatenate(
        [series[i * block_size : (i + 1) * block_size] for i in block_indices]
    )

    # Handle remainder
    if len(shuffled) < n:
        shuffled = np.concatenate([shuffled, series[len(shuffled) :]])

    return shuffled[:n]


def _phase_randomization(series: np.ndarray) -> np.ndarray:
    """
    Phase randomization surrogate (preserves power spectrum).

    Fourier transform, randomize phases, inverse transform.
    """
    n = len(series)

    # FFT
    fft = np.fft.rfft(series)

    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, len(fft))

    # Apply random phases
    fft_randomized = np.abs(fft) * np.exp(1j * phases)

    # Inverse FFT
    surrogate = np.fft.irfft(fft_randomized, n=n)

    return surrogate.real


def cusum_structural_stability(
    data: pd.DataFrame,
    source: str,
    target: str,
    method_fn: Callable,
    window_size: int = 50,
    step_size: int = 10,
    **method_kwargs,
) -> Dict:
    """
    CUSUM test for structural stability of causal relationship.

    Detects change points in causal effect size over time.

    Parameters:
        data: Time series DataFrame
        source: Source variable
        target: Target variable
        method_fn: Causal method
        window_size: Rolling window size
        step_size: Step between windows
        **method_kwargs: Method arguments

    Returns:
        Dict with:
        - cusum_statistic: CUSUM values over time
        - change_points: List of detected change points
        - is_stable: Whether relationship is stable
        - threshold: Critical value for CUSUM
    """
    n = len(data)

    if window_size > n:
        logger.warning(f"Window size ({window_size}) exceeds data length ({n})")
        return {"error": "Window too large"}

    # Compute effect size in rolling windows
    effects = []
    window_centers = []

    for start in range(0, n - window_size + 1, step_size):
        end = start + window_size
        window_data = data.iloc[start:end]

        try:
            result = method_fn(
                window_data[[source, target]], source, target, **method_kwargs
            )
            effect = result.get("effect_size", np.nan)
            effects.append(effect)
            window_centers.append((start + end) // 2)
        except Exception:
            effects.append(np.nan)
            window_centers.append((start + end) // 2)

    effects = np.array([e for e in effects if not np.isnan(e)])

    if len(effects) < 5:
        return {"error": "Too few valid windows"}

    # CUSUM computation
    mean_effect = effects.mean()
    cusum = np.cumsum(effects - mean_effect)

    # Detect change points (CUSUM exceeds threshold)
    threshold = 3 * effects.std()  # 3-sigma threshold
    change_points = np.where(np.abs(cusum) > threshold)[0]

    is_stable = len(change_points) == 0

    logger.info(
        f"CUSUM structural stability: {source}→{target} "
        f"({'stable' if is_stable else f'{len(change_points)} change points'})"
    )

    return {
        "cusum_statistic": cusum,
        "window_centers": window_centers[: len(cusum)],
        "change_points": change_points.tolist(),
        "is_stable": is_stable,
        "threshold": threshold,
        "mean_effect": mean_effect,
    }
