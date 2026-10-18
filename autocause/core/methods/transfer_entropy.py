"""
Transfer Entropy (TE) Analysis

Implements information-theoretic causal inference using Transfer Entropy.
Prefers continuous CMI/kNN estimator (tigramite CMIknn) with surrogate testing,
and supports an explicitly selected discrete CMI estimator or MI proxy.
An estimator failure is reported rather than mixing estimators in one null test.
Source shuffling is an approximate sensitivity check: it is not a conditional
randomization test given all other variables. Pairwise TE may reflect indirect
paths or common causes and does not by itself identify direct causal edges.

References:
    - Schreiber, T. (2000). "Measuring Information Transfer". Phys. Rev. Lett., 85(2), 461
    - Kraskov et al. (2004). "Estimating mutual information". Phys. Rev. E, 69(6), 066138
    - tigramite CMIknn: https://jakobrunge.github.io/tigramite/
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

# Optional tigramite availability flag (lazy import later)
try:
    import importlib.util

    tigramite_spec = importlib.util.find_spec("tigramite")
    TIGRAMITE_AVAILABLE = tigramite_spec is not None
except Exception:  # pragma: no cover
    TIGRAMITE_AVAILABLE = False


def standardize_array(arr: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Standardize array to zero mean and unit variance (z-score normalization).

    Required for CMIknn to avoid numerical issues with KNN distance computation.

    Parameters:
        arr (np.ndarray): Input array
        epsilon (float): Small constant to avoid division by zero

    Returns:
        np.ndarray: Standardized array
    """
    mean = np.mean(arr)
    std = np.std(arr)
    if std < epsilon:
        # Constant series: return zeros (no information)
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mean) / (std + epsilon)


def discretize_series(
    series: np.ndarray, bins: int = 5, method: str = "quantile"
) -> np.ndarray:
    """
    Discretize a continuous time series into symbolic states.

    Parameters:
        series (np.ndarray): Continuous time series
        bins (int): Number of bins/states
        method (str): "quantile" (equal probability) or "uniform" (equal width)

    Returns:
        np.ndarray: Discretized integer-encoded series
    """
    series = series.flatten()

    # Handle cases where all values are identical
    if len(np.unique(series)) < bins:
        logger.warning(
            f"Series has fewer than {bins} unique values. Using all unique values."
        )
        return pd.factorize(series)[0]

    if method == "quantile":
        # Equal probability binning (quantile-based)
        discretized = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    else:
        # Equal width binning
        discretized = pd.cut(series, bins=bins, labels=False, duplicates="drop")

    # Handle both Series and ndarray returns
    if isinstance(discretized, np.ndarray):
        return discretized.astype(int)
    else:
        return discretized.values.astype(int)


def compute_mi(x: np.ndarray, y: np.ndarray, bins: int = 5) -> float:
    """
    Compute Mutual Information between two discretized time series.

    Parameters:
        x (np.ndarray): First discrete series
        y (np.ndarray): Second discrete series
        bins (int): Number of bins (for continuous data)

    Returns:
        float: Mutual Information (nats)
    """
    # Discretize if continuous
    if x.dtype in [np.float64, np.float32]:
        x = discretize_series(x, bins=bins)
    if y.dtype in [np.float64, np.float32]:
        y = discretize_series(y, bins=bins)

    # Compute joint and marginal entropies
    xy = np.column_stack([x, y])

    # Joint probability
    joint = np.unique(xy, axis=0, return_counts=True)[1] / len(xy)
    h_xy = -np.sum(joint * np.log(joint + 1e-10))

    # Marginal probabilities
    px = np.unique(x, return_counts=True)[1] / len(x)
    py = np.unique(y, return_counts=True)[1] / len(y)
    h_x = -np.sum(px * np.log(px + 1e-10))
    h_y = -np.sum(py * np.log(py + 1e-10))

    # MI = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    return max(0, mi)  # Ensure non-negative


def transfer_entropy_discrete(
    source: np.ndarray,
    target: np.ndarray,
    delay: int = 1,
    history_len: int = 1,
    bins: int = 5,
) -> float:
    """
    Compute Transfer Entropy using discretized series.

    Parameters:
        source (np.ndarray): Source time series
        target (np.ndarray): Target time series
        delay (int): Time delay (lag)
        history_len (int): History length for conditioning
        bins (int): Number of discretization bins

    Returns:
        float: Transfer Entropy (nats)
    """
    source_disc = discretize_series(source, bins=bins)
    target_disc = discretize_series(target, bins=bins)
    rows = _lagged_arrays(source_disc, target_disc, delay, history_len)

    def entropy(values):
        if len(values) == 0:
            return 0.0
        _, counts = np.unique(values.T, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log(probabilities)))

    # I(target_current; source_lag | target_history), in nats.
    z = rows[2:]
    return max(0.0, entropy(np.vstack([rows[0], z]))
               + entropy(np.vstack([rows[1], z])) - entropy(z) - entropy(rows))


def _lagged_arrays(source, target, delay, history_len):
    """Align all rows at t >= max(delay, history_len), without truncation shifts."""
    if not isinstance(delay, (int, np.integer)) or delay < 1:
        raise ValueError("delay must be a positive integer")
    if not isinstance(history_len, (int, np.integer)) or history_len < 0:
        raise ValueError("history_len must be a nonnegative integer")
    source, target = np.asarray(source), np.asarray(target)
    if source.ndim != 1 or target.ndim != 1 or len(source) != len(target):
        raise ValueError("source and target must be aligned one-dimensional arrays")
    start, n = max(delay, history_len), len(target)
    if n <= start:
        raise ValueError("Insufficient observations for the requested history and delay")
    return np.vstack([target[start:], source[start-delay:n-delay]]
                     + [target[start-k:n-k] for k in range(1, history_len+1)])


def transfer_entropy_mi_proxy(
    source: np.ndarray, target: np.ndarray, delay: int = 1
) -> float:
    """
    Compute Transfer Entropy using MI-based proxy (when pyinform unavailable).

    Simple proxy: lagged mutual information I(Source_t-delay; Target_t).

    Parameters:
        source (np.ndarray): Source time series
        target (np.ndarray): Target time series
        delay (int): Time delay

    Returns:
        float: TE proxy (normalized to 0-1)
    """
    source = source.flatten()
    target = target.flatten()

    if len(source) < delay + 1 or len(target) < delay + 1:
        return 0.0

    # Align: source at t-delay, target at t
    source_lagged = source[:-delay] if delay > 0 else source
    target_current = target[delay:] if delay > 0 else target

    # Discretize and compute MI
    te_proxy = compute_mi(source_lagged, target_current, bins=5)

    # Normalize to 0-1 range
    max_te = np.log(5)  # Max entropy for 5 bins
    te_norm = te_proxy / max_te if max_te > 0 else 0

    return float(np.clip(te_norm, 0, 1))


def run_transfer_entropy(
    df: pd.DataFrame,
    source_var: str,
    target_var: str,
    delay: int = 1,
    method: str = "cmiknn",
    k_neighbors: int = 5,
    history_len: int = 1,
    n_surrogates: int = 200,
    block_shuffle: bool = True,
    block_length: int = None,
    bins: int = 5,
    verbose: bool = True,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
) -> Dict:
    """
    Compute Transfer Entropy from source to target.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        source_var (str): Source variable name
        target_var (str): Target variable name
        delay (int): Time delay (lag)
        method (str): "cmiknn" | "discrete" | "proxy"
        k_neighbors (int): For CMIknn
        history_len (int): History length of target for conditioning
        n_surrogates (int): Surrogate permutations for p-value
        bins (int): Number of bins for discrete estimator (default: 5)
        verbose (bool): Print results
        alpha (float): Significance threshold before multiple-testing correction
        seed (int): Local surrogate random seed; does not reset global RNG

    Returns:
        Dict: TE value, p-value, significance
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Transfer Entropy: {source_var} → {target_var} (delay={delay})")
    logger.info(f"{'=' * 70}")

    data = df[[source_var, target_var]].dropna()
    source = data[source_var].values
    target = data[target_var].values

    if len(source) < delay + 2:
        logger.error(f"Insufficient data ({len(source)} obs) for delay={delay}")
        return {
            "source": source_var,
            "target": target_var,
            "delay": delay,
            "te": np.nan,
            "p_value": np.nan,
            "is_significant": False,
            "method": "TransferEntropy",
        }

    from ..surrogate_inference import monte_carlo_pvalue

    if not isinstance(n_surrogates, (int, np.integer)) or n_surrogates < 1:
        raise ValueError("n_surrogates must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if method not in {"cmiknn", "discrete", "proxy"}:
        raise ValueError("method must be cmiknn, discrete, or proxy")
    # Validate and align before computing observed or surrogate statistics.
    aligned = _lagged_arrays(source, target, delay, history_len)
    if method == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn
        effective_k = min(k_neighbors, max(2, aligned.shape[1] // 10))

    def estimate(src):
        if method == "cmiknn":
            rows = _lagged_arrays(src, target, delay, history_len)
            rows = np.vstack([standardize_array(row) for row in rows])
            # Tigramite labels every conditioning row 2, not 2, 3, ... .
            xyz = np.array([0, 1] + [2] * history_len)
            estimator = CMIknn(significance=None, knn=effective_k)
            return float(estimator.get_dependence_measure(rows, xyz) / np.log(2))
        if method == "discrete":
            return float(transfer_entropy_discrete(
                src, target, delay=delay, history_len=history_len, bins=bins
            ) / np.log(2))
        # The explicit MI proxy is normalized by log(5); report its lagged
        # mutual information in bits, consistently with the other estimators.
        return float(transfer_entropy_mi_proxy(src, target, delay=delay) * np.log2(5))

    # Use exactly the same estimator for the observed data and every null draw.
    te_bits = estimate(source)
    rng = np.random.default_rng(seed)
    blen = block_length if block_length is not None else max(2 * delay, 5)
    if not isinstance(blen, (int, np.integer)) or blen < 1:
        raise ValueError("block_length must be a positive integer")
    te_surrogates = []
    for _ in range(n_surrogates):
        if block_shuffle and blen > 1:
            blocks = [source[i:i+blen] for i in range(0, len(source), blen)]
            source_shuffled = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        else:
            source_shuffled = rng.permutation(source)
        te_surrogates.append(estimate(source_shuffled))
    te_surrogates = np.asarray(te_surrogates)
    p_value = monte_carlo_pvalue(te_bits, te_surrogates)
    is_significant = p_value < alpha

    result = {
        "source": source_var,
        "target": target_var,
        "delay": delay,
        "te_bits": te_bits,
        "estimator": method,
        "n_surrogates": n_surrogates,
        "p_value_resolution": 1 / (n_surrogates + 1),
        "null_model": "source_block_permutation" if block_shuffle else "source_permutation",
        "te_mean_surrogate": te_surrogates.mean(),
        "te_std_surrogate": te_surrogates.std(),
        "p_value": p_value,
        "is_significant": is_significant,
        "n_observations": len(data),
        "method": "TransferEntropy",
    }

    if verbose:
        logger.info(
            f"TE(bits) = {te_bits:.6f}, p-value = {p_value:.4f}, significant = {is_significant}"
        )
        logger.info(f"Delay: {delay} steps ({delay * 5} days)")

    return result


def batch_transfer_entropy(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    delays: List[int] = None,
    alpha: float = 0.05,
    apply_global_fdr: bool = False,
    reduce_to_best_per_pair: bool = True,
    enforce_antisymmetry: bool = True,
    method: str = "cmiknn",
) -> pd.DataFrame:
    """
    Compute Transfer Entropy for multiple pairs and delays.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        variable_pairs (List[Tuple[str, str]]): List of (source, target) pairs
        delays (List[int]): List of delays to test (default: [1, 2, 3])
        alpha (float): Significance threshold

    Returns:
        pd.DataFrame: Results for all pairs and delays
    """
    if delays is None:
        delays = [1, 2, 3]

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Computing TE for {len(variable_pairs)} pairs, {len(delays)} delays")
    logger.info(f"{'=' * 70}")

    results = []
    for source, target in variable_pairs:
        for delay in delays:
            try:
                result = run_transfer_entropy(
                    df, source, target, delay=delay, method=method, verbose=False, alpha=alpha
                )
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "delay": delay,
                        "te_bits": result.get("te_bits", np.nan),
                        "p_value": result.get("p_value", np.nan),
                        "significant": result.get("is_significant", False),
                        "n_obs": result.get("n_observations", np.nan),
                        "estimator": result.get("estimator", method),
                        "n_surrogates": result.get("n_surrogates", np.nan),
                        "p_value_resolution": result.get("p_value_resolution", np.nan),
                        "null_model": result.get("null_model", ""),
                    }
                )
            except Exception as e:
                logger.error(f"Failed for {source} → {target} (delay={delay}): {e}")
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "delay": delay,
                        "te": np.nan,
                        "p_value": np.nan,
                        "significant": False,
                        "n_obs": np.nan,
                    }
                )

    results_df = pd.DataFrame(results)

    # Add is_significant as alias for backward compatibility with consensus module
    results_df["is_significant"] = results_df["significant"]

    # Apply within-pair BH-FDR across lags
    try:
        from ..multiple_testing import apply_fdr_to_dataframe

        results_df["q_value"] = np.nan
        results_df["fdr_significant"] = False
        for (s, t), sub in results_df.groupby(["source", "target"], as_index=False):
            tmp = apply_fdr_to_dataframe(
                sub.rename(columns={"p_value": "p_value"}), p_col="p_value", alpha=alpha
            )
            results_df.loc[sub.index, "q_value"] = tmp["q_value"].values
            results_df.loc[sub.index, "fdr_significant"] = tmp["significant"].values
        # Update significant by q-value when available
        results_df["significant"] = results_df["fdr_significant"].astype(bool)
        results_df["is_significant"] = results_df["significant"]
    except Exception as e:
        logger.warning(f"FDR correction failed: {e}")

    # Optionally apply global FDR across all pairs (and delays)
    if apply_global_fdr and not results_df.empty and "p_value" in results_df.columns:
        try:
            from ..multiple_testing import apply_fdr_to_dataframe

            tmp = apply_fdr_to_dataframe(
                results_df.rename(columns={"p_value": "p_value"}),
                p_col="p_value",
                alpha=alpha,
            )
            results_df["q_value"] = tmp["q_value"]
            results_df["significant"] = tmp["significant"].astype(bool)
            results_df["is_significant"] = results_df["significant"]
        except Exception as e:
            logger.warning(f"Global FDR correction failed: {e}")

    # Optionally reduce to the best (most significant) delay per pair
    if reduce_to_best_per_pair and not results_df.empty:
        best_indices = []
        # Prefer q_value when present, else p_value; among equals, prefer shortest delay
        for (s, t), sub in results_df.groupby(["source", "target"], as_index=False):
            # Keep only significant rows if any
            sig = sub[sub.get("significant", False).astype(bool)]
            if not sig.empty:
                if "q_value" in sig.columns and sig["q_value"].notna().any():
                    sig_nonan = sig.copy()
                    sig_nonan["_key"] = list(
                        zip(sig_nonan["q_value"].fillna(np.inf), sig_nonan["delay"])
                    )
                    idx = sig_nonan["_key"].idxmin()
                else:
                    sig_nonan = sig.copy()
                    sig_nonan["_key"] = list(
                        zip(sig_nonan["p_value"].fillna(np.inf), sig_nonan["delay"])
                    )
                    idx = sig_nonan["_key"].idxmin()
                best_indices.append(idx)
            else:
                # No significant delays — keep the row with lowest p-value
                # so non-significant pairs are retained in the output
                nonsig = sub.copy()
                nonsig["_key"] = list(
                    zip(nonsig["p_value"].fillna(np.inf), nonsig["delay"])
                )
                idx = nonsig["_key"].idxmin()
                best_indices.append(idx)
        if best_indices:
            results_df = results_df.loc[best_indices].copy()

    # Optionally enforce antisymmetry: if both directions between a pair are significant,
    # keep only the direction with stronger evidence (lower q, else higher TE).
    # IMPORTANT: Non-significant rows are ALWAYS retained in the output (marked as non-significant).
    # Only the losing direction (if both are significant) is marked as non-significant.
    if enforce_antisymmetry and not results_df.empty:
        results_df["_undir_key"] = results_df.apply(
            lambda r: tuple(sorted([r["source"], r["target"]])), axis=1
        )
        for _, sub in results_df.groupby("_undir_key", as_index=False):
            sig = sub[sub.get("significant", False).astype(bool)]
            if len(sig) <= 1:
                # 0 or 1 significant direction — no conflict, keep as-is
                continue
            # Two directions significant: choose winner, mark loser as non-significant
            if "q_value" in sig.columns and sig["q_value"].notna().any():
                best_idx = sig["q_value"].idxmin()
            else:
                best_idx = sig["te_bits"].idxmax()
            loser_indices = [i for i in sig.index if i != best_idx]
            results_df.loc[loser_indices, "significant"] = False
            results_df.loc[loser_indices, "is_significant"] = False
        # Clean up temp column
        if "_undir_key" in results_df.columns:
            results_df = results_df.drop(columns=["_undir_key"], errors="ignore")

    # Sort by TE (highest first)
    sort_col = "te_bits" if "te_bits" in results_df.columns else "p_value"
    results_df = results_df.sort_values(sort_col, ascending=False)

    logger.info(
        f"Completed. Found {results_df['significant'].sum()} significant relationships (α={alpha})"
    )

    return results_df
