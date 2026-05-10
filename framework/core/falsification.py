"""
Falsification Tests for Causal Discovery

Implements statistical tests to verify that detected edges are not spurious.
True causal relationships should survive randomization that preserves data
structure but breaks causal relationships.

Tests:
1. Block Permutation: Preserves autocorrelation structure
2. IAAFT: Preserves power spectrum but randomizes phases
3. Season Swap: Randomizes seasonal labels
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, ifft

logger = logging.getLogger(__name__)


def block_permutation(
    x: np.ndarray,
    block_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Permute time series in blocks to preserve autocorrelation within blocks
    but break long-range causal relationships.

    This is appropriate for autoregressive data where short-term dependencies
    should be preserved, but long-range causality should be broken.

    Parameters
    ----------
    x : np.ndarray
        Time series to permute
    block_size : int, optional
        Block size (default: 2 * estimated AR order, min 10)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Block-permuted time series

    Examples
    --------
    >>> x = np.arange(20)
    >>> x_perm = block_permutation(x, block_size=5, seed=42)
    >>> len(x_perm)
    20
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(x)

    if block_size is None:
        # Estimate AR order using AIC
        from statsmodels.tsa.ar_model import AutoReg

        try:
            ar = AutoReg(x, lags=min(20, n // 10), trend="n").fit()
            ar_order = len(ar.params)
            block_size = max(10, 2 * ar_order)
        except Exception:
            # Default to sqrt(n) if AR estimation fails
            block_size = max(10, int(np.sqrt(n)))

    # Ensure block_size doesn't exceed data length
    block_size = min(block_size, n // 2)

    # Split into blocks
    n_blocks = n // block_size
    remainder = n % block_size

    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        blocks.append(x[start:end])

    if remainder > 0:
        blocks.append(x[-remainder:])

    # Randomly permute blocks
    np.random.shuffle(blocks)

    # Concatenate
    x_perm = np.concatenate(blocks)

    return x_perm


def iaaft(
    x: np.ndarray,
    max_iter: int = 1000,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogate.

    Generates surrogate data with same amplitude distribution and power spectrum
    as original, but randomized phases. This breaks nonlinear causal relationships
    while preserving linear autocorrelation structure.

    Parameters
    ----------
    x : np.ndarray
        Time series
    max_iter : int, default=1000
        Maximum iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    seed : int, optional
        Random seed

    Returns
    -------
    np.ndarray
        IAAFT surrogate

    References
    ----------
    Schreiber, T., & Schmitz, A. (1996). "Improved surrogate data for nonlinearity tests."
    Physical Review Letters, 77(4), 635.

    Examples
    --------
    >>> x = np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1
    >>> surrogate = iaaft(x, seed=42)
    >>> len(surrogate)
    100
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(x)

    # Sort original data to get amplitude distribution
    x_sorted = np.sort(x)

    # Fourier transform of original
    x_fft = fft(x)
    x_amp = np.abs(x_fft)

    # Generate initial surrogate: randomize phases
    random_phases = np.random.uniform(-np.pi, np.pi, n)
    surrogate_fft = x_amp * np.exp(1j * random_phases)
    surrogate = np.real(ifft(surrogate_fft))

    # Iterative amplitude adjustment
    for iteration in range(max_iter):
        # Amplitude adjustment: match amplitude distribution
        surrogate_ranks = stats.rankdata(surrogate, method="ordinal") - 1
        surrogate_adjusted = x_sorted[surrogate_ranks]

        # Spectrum adjustment: match power spectrum
        surrogate_fft = fft(surrogate_adjusted)
        surrogate_phases = np.angle(surrogate_fft)
        surrogate_fft = x_amp * np.exp(1j * surrogate_phases)
        surrogate_new = np.real(ifft(surrogate_fft))

        # Check convergence
        diff = np.max(np.abs(surrogate_new - surrogate))
        if diff < tolerance:
            break

        surrogate = surrogate_new

    # Final amplitude adjustment
    surrogate_ranks = stats.rankdata(surrogate, method="ordinal") - 1
    surrogate_final = x_sorted[surrogate_ranks]

    return surrogate_final


def season_label_swap(
    data: pd.DataFrame,
    date_col: str = "date",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Randomize seasonal labels while preserving within-season data structure.

    This breaks seasonal causal relationships (e.g., winter → spring) but
    preserves within-season patterns.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with date column
    date_col : str, default='date'
        Column name for date
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Data with permuted seasonal labels

    Examples
    --------
    >>> data = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100, freq='D'),
    ...     'x': np.random.randn(100)
    ... })
    >>> shuffled = season_label_swap(data, seed=42)
    >>> len(shuffled)
    100
    """
    if seed is not None:
        np.random.seed(seed)

    df = data.copy()

    # Extract month/DOY
    if date_col in df.columns:
        df["month"] = pd.to_datetime(df[date_col]).dt.month
        df["doy"] = pd.to_datetime(df[date_col]).dt.dayofyear
    else:
        raise ValueError(f"Date column '{date_col}' not found")

    # Permute month labels
    unique_months = df["month"].unique()
    perm_months = np.random.permutation(unique_months)
    month_map = dict(zip(unique_months, perm_months))

    df["month_shuffled"] = df["month"].map(month_map)

    # Permute DOY within months
    for month in unique_months:
        mask = df["month"] == month
        doys = df.loc[mask, "doy"].values
        df.loc[mask, "doy_shuffled"] = np.random.permutation(doys)

    return df


def test_edge_falsification(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    test_func: callable,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    falsification_type: str = "block_permutation",
    seed: Optional[int] = None,
    store_surrogates: bool = False,  # Memory optimization: store only summary stats
    **test_kwargs,
) -> Dict[str, any]:
    """
    Test if an edge survives falsification.

    A true causal edge should become non-significant when the source is
    randomized in a way that preserves data structure but breaks causality.

    Parameters
    ----------
    source : np.ndarray
        Source variable time series
    target : np.ndarray
        Target variable time series
    lag : int
        Lag to test
    test_func : callable
        Function that tests X → Y causality, returns p-value
        Signature: test_func(source, target, lag, **kwargs) -> float (p-value)
    n_surrogates : int, default=100
        Number of surrogates to generate
    alpha : float, default=0.05
        Significance level
    falsification_type : str, default='block_permutation'
        Type of falsification: 'block_permutation', 'iaaft', or 'season_swap'
    seed : int, optional
        Random seed
    **test_kwargs
        Additional arguments for test_func

    Returns
    -------
    Dict
        Results with keys:
        - original_pvalue: p-value for original data
        - surrogate_pvalues: p-values for all surrogates
        - fraction_significant: fraction of surrogates with p < alpha
        - passed: True if edge survives (most surrogates non-significant)
        - threshold: expected fraction if no causality (should be ≈ alpha)

    Examples
    --------
    >>> def simple_test(x, y, lag):
    ...     # Simple correlation test
    ...     from scipy.stats import pearsonr
    ...     return pearsonr(x[:-lag], y[lag:])[1]
    >>> x = np.random.randn(100)
    >>> y = x + np.random.randn(100) * 0.1  # Strong relationship
    >>> result = test_edge_falsification(x, y, lag=1, test_func=simple_test, n_surrogates=10)
    >>> 'passed' in result
    True
    """
    if seed is not None:
        np.random.seed(seed)

    # Test on original data
    try:
        original_pval = test_func(source, target, lag, **test_kwargs)
    except Exception as e:
        logger.warning(f"Original test failed: {e}")
        return {
            "original_pvalue": np.nan,
            "surrogate_pvalues": [],
            "fraction_significant": np.nan,
            "passed": False,
            "threshold": alpha,
            "error": str(e),
        }

    # Generate surrogates and test
    surrogate_pvals = []

    for i in range(n_surrogates):
        # Generate surrogate for source
        if falsification_type == "block_permutation":
            source_surr = block_permutation(source, seed=seed + i if seed else None)
        elif falsification_type == "iaaft":
            source_surr = iaaft(source, seed=seed + i if seed else None)
        else:
            raise ValueError(f"Unknown falsification type: {falsification_type}")

        # Test on surrogate
        try:
            surr_pval = test_func(source_surr, target, lag, **test_kwargs)
            surrogate_pvals.append(surr_pval)
        except Exception as e:
            logger.debug(f"Surrogate {i} test failed: {e}")
            continue

    if len(surrogate_pvals) == 0:
        return {
            "original_pvalue": original_pval,
            "surrogate_pvalues": [],
            "fraction_significant": np.nan,
            "passed": False,
            "threshold": alpha,
            "error": "All surrogate tests failed",
        }

    # Compute fraction of surrogates that are significant
    surrogate_pvals = np.array(surrogate_pvals)
    fraction_sig = np.mean(surrogate_pvals < alpha)

    # Statistical test: Under null hypothesis (no causality), fraction_sig should follow
    # Binomial(n_surrogates, alpha). We test if observed fraction is significantly higher.
    # Use one-sided binomial test: H0: p = alpha, H1: p > alpha
    from scipy.stats import binom

    n_sig = int(fraction_sig * len(surrogate_pvals))

    # P-value for observing n_sig or more significant surrogates under null
    binom_pval = 1 - binom.cdf(n_sig - 1, len(surrogate_pvals), alpha)

    # Pass if:
    # 1. Original is significant (p < alpha)
    # 2. Surrogates are NOT significantly different from random (binom_pval > alpha)
    #    i.e., the fraction of significant surrogates is consistent with chance (≈ alpha)
    passed = (original_pval < alpha) and (binom_pval > alpha)

    result = {
        "original_pvalue": original_pval,
        "fraction_significant": fraction_sig,
        "binomial_pvalue": binom_pval,
        "passed": passed,
        "threshold": alpha,
        "n_surrogates": len(surrogate_pvals),
    }

    # Memory optimization: only store full surrogate arrays if requested
    if store_surrogates:
        result["surrogate_pvalues"] = surrogate_pvals.tolist()
    else:
        # Store only summary statistics (5 floats vs 200+ floats)
        result["surrogate_summary"] = {
            "mean": float(np.mean(surrogate_pvals)),
            "std": float(np.std(surrogate_pvals)),
            "min": float(np.min(surrogate_pvals)),
            "max": float(np.max(surrogate_pvals)),
            "median": float(np.median(surrogate_pvals)),
        }

    return result


def run_falsification_battery(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    test_func: callable,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    store_surrogates: bool = False,  # Memory optimization
    **test_kwargs,
) -> Dict[str, Dict]:
    """
    Run full battery of falsification tests.

    Parameters
    ----------
    source, target : np.ndarray
        Time series
    lag : int
        Lag to test
    test_func : callable
        Causality test function
    n_surrogates : int, default=100
        Number of surrogates per test
    alpha : float, default=0.05
        Significance level
    seed : int, optional
        Random seed
    **test_kwargs
        Passed to test_func

    Returns
    -------
    Dict[str, Dict]
        Results for each falsification test:
        - block_permutation: results
        - iaaft: results

    Examples
    --------
    >>> def simple_test(x, y, lag):
    ...     from scipy.stats import pearsonr
    ...     return pearsonr(x[:-lag], y[lag:])[1]
    >>> x = np.random.randn(100)
    >>> y = x + np.random.randn(100) * 0.1
    >>> results = run_falsification_battery(x, y, lag=1, test_func=simple_test, n_surrogates=10)
    >>> 'block_permutation' in results
    True
    >>> 'iaaft' in results
    True
    """
    results = {}

    # Block permutation
    logger.info("Running block permutation test...")
    results["block_permutation"] = test_edge_falsification(
        source,
        target,
        lag,
        test_func,
        n_surrogates=n_surrogates,
        alpha=alpha,
        falsification_type="block_permutation",
        seed=seed,
        store_surrogates=store_surrogates,  # Pass through memory optimization
        **test_kwargs,
    )

    # IAAFT
    logger.info("Running IAAFT test...")
    results["iaaft"] = test_edge_falsification(
        source,
        target,
        lag,
        test_func,
        n_surrogates=n_surrogates,
        alpha=alpha,
        falsification_type="iaaft",
        seed=seed + 1000 if seed else None,
        store_surrogates=store_surrogates,  # Pass through memory optimization
        **test_kwargs,
    )

    return results


def summarize_falsification_results(results: Dict[str, Dict]) -> Dict[str, any]:
    """
    Summarize falsification test results.

    Parameters
    ----------
    results : Dict[str, Dict]
        Results from run_falsification_battery

    Returns
    -------
    Dict
        Summary with:
        - tests_passed: number of tests passed
        - tests_total: total tests run
        - pass_rate: fraction passed
        - overall_pass: True if ≥2 tests passed
        - details: individual test results
    """
    tests_passed = sum(1 for r in results.values() if r.get("passed", False))
    tests_total = len(results)

    return {
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "pass_rate": tests_passed / tests_total if tests_total > 0 else 0,
        "overall_pass": tests_passed >= 2,  # Require ≥2 tests for Tier-1
        "details": results,
    }
