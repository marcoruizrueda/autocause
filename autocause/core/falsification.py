"""
Falsification Tests for Causal Discovery

Surrogate sensitivity checks for detected lagged associations.

An observed test score is ranked against scores from randomized source series.
Block permutation and IAAFT are approximate null models for time series;
passing is not proof of a direct causal edge or post-selection error control.

Tests:
1. Block Permutation: Retains ordering within blocks, changes their boundaries
2. IAAFT: Retains amplitude distribution and approximates the power spectrum
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
    while disrupting alignment with another series.

    Block boundaries alter autocorrelation. This is an approximate null model
    whose suitability depends on stationarity, memory length, and block size.

    Parameters
    ----------
    x : np.ndarray
        Time series to permute
    block_size : int, optional
        Block size (default: twice a fitted fixed AR order, min 10;
        capped at half the series length)
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
    rng = np.random.default_rng(seed)

    n = len(x)

    if block_size is None:
        # Heuristic fixed AR order; no AIC lag selection is performed here.
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
    rng.shuffle(blocks)

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

    Generates surrogate data with the same amplitude distribution and an
    approximately matching power spectrum, disrupting alignment with another
    series. This does not selectively distinguish nonlinear from linear causes.

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
    rng = np.random.default_rng(seed)

    n = len(x)

    # Sort original data to get amplitude distribution
    x_sorted = np.sort(x)

    # Fourier transform of original
    x_fft = fft(x)
    x_amp = np.abs(x_fft)

    # Generate initial surrogate: randomize phases
    random_phases = rng.uniform(-np.pi, np.pi, n)
    surrogate_fft = x_amp * np.exp(1j * random_phases)
    surrogate = np.real(ifft(surrogate_fft))

    # Iterative amplitude adjustment
    for iteration in range(max_iter):
        # Amplitude adjustment: match amplitude distribution
        # Ordinal ranks are integral, but recent SciPy returns a float dtype.
        surrogate_ranks = stats.rankdata(surrogate, method="ordinal").astype(np.intp) - 1
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
    surrogate_ranks = stats.rankdata(surrogate, method="ordinal").astype(np.intp) - 1
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

    Rank the original callback p-value against the surrogate p-values, with
    the finite Monte Carlo correction. Smaller callback p-values are stronger
    evidence. The callback must use the same model and sample size on each draw.
    Validity depends on the surrogate null assumptions; these checks do not
    establish a direct causal link or correct for discovery on the same data.

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
        Type of check: 'block_permutation' or 'iaaft'
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
        - empirical_pvalue: corrected lower-tail rank of original p-value
        - passed: original and empirical p-values both below alpha
        - status: ok or failed (failed/incomplete simulations never pass)
        - threshold: alpha

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
    from .surrogate_inference import monte_carlo_pvalue

    if not isinstance(n_surrogates, (int, np.integer)) or n_surrogates < 1:
        raise ValueError("n_surrogates must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if falsification_type not in {"block_permutation", "iaaft"}:
        raise ValueError(f"Unknown falsification type: {falsification_type}")

    original_pval = np.nan
    surrogate_pvals = []

    def failure(error):
        return {
            "original_pvalue": original_pval,
            "empirical_pvalue": np.nan,
            "fraction_significant": np.nan,
            "passed": False, "threshold": alpha, "status": "failed",
            "error": str(error), "n_surrogates": len(surrogate_pvals),
            "n_surrogates_requested": n_surrogates,
        }

    def checked_pvalue(x):
        value = float(test_func(x, target, lag, **test_kwargs))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("test_func must return a finite p-value in [0, 1]")
        return value

    try:
        original_pval = checked_pvalue(source)
    except Exception as exc:
        logger.warning(f"Original test failed: {exc}")
        return failure(exc)

    rng = np.random.default_rng(seed)
    generator = block_permutation if falsification_type == "block_permutation" else iaaft
    for i in range(n_surrogates):
        try:
            surrogate_seed = int(rng.integers(0, 2**32))
            source_surr = generator(source, seed=surrogate_seed)
            surrogate_pvals.append(checked_pvalue(source_surr))
        except Exception as exc:
            # Dropping failed draws can bias the tail probability. Fail closed.
            logger.warning(f"Surrogate {i} test failed: {exc}")
            return failure(exc)

    surrogate_pvals = np.asarray(surrogate_pvals)
    # The callback returns a p-value: smaller values indicate stronger evidence.
    # With the same fitted model/df on every draw this ranks the test statistic.
    empirical_pval = monte_carlo_pvalue(original_pval, surrogate_pvals, alternative="less")
    result = {
        "original_pvalue": original_pval,
        "empirical_pvalue": empirical_pval,
        "fraction_significant": float(np.mean(surrogate_pvals < alpha)),
        "passed": bool(original_pval < alpha and empirical_pval < alpha),
        "threshold": alpha, "status": "ok",
        "n_surrogates": n_surrogates,
        "n_surrogates_requested": n_surrogates,
        "p_value_resolution": 1 / (n_surrogates + 1),
        "null_model": falsification_type,
        "interpretation": "Approximate surrogate sensitivity check, not proof of direct causality",
    }
    if store_surrogates:
        result["surrogate_pvalues"] = surrogate_pvals.tolist()
    else:
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
        seed=seed + 1000 if seed is not None else None,
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
