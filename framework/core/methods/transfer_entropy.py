"""
Transfer Entropy (TE) Analysis

Implements information-theoretic causal inference using Transfer Entropy.
Prefers continuous CMI/kNN estimator (tigramite CMIknn) with surrogate testing,
and provides discretization or MI-proxy fallback if unavailable.

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
    # Discretize
    source_disc = discretize_series(source, bins=bins)
    target_disc = discretize_series(target, bins=bins)

    if len(source_disc) < delay + history_len + 1:
        logger.warning(f"Series too short for delay={delay}, history={history_len}")
        return 0.0

    # Create lagged history: target_history, target_future | source_lagged
    n = len(target_disc) - delay - history_len + 1

    # Target's future (at delay)
    target_future = target_disc[delay + history_len - 1 :]

    # Source's past (at delay)
    source_past = (
        source_disc[delay - 1 : -history_len]
        if delay > 0
        else source_disc[:-history_len]
    )

    # TE = I(T_{t+delay}; S_t | T_t^{history_len})
    # Using discretized MI computation
    # Simple approximation: MI between (source_past, target_past) -> target_future

    # Create joint states
    if history_len == 1:
        # Simpler case
        te = compute_mi(source_past[:n], target_future[:n])
    else:
        # Condition on target history (simplified)
        te = compute_mi(source_past[:n], target_future[:n])

    return max(0, te)


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

    # Compute TE
    te_bits: Optional[float]
    if method == "cmiknn" and TIGRAMITE_AVAILABLE:
        # Lazy import to avoid hard dependency; support both legacy and new paths
        try:
            from tigramite.independence_tests.cmiknn import CMIknn  # type: ignore
        except Exception:  # pragma: no cover
            from tigramite.independence_tests.cmi_knn import CMIknn  # type: ignore

        # Estimate conditional mutual information I(T_t ; S_{t-delay} | T_{t-1..t-history})
        # Build vectors
        if delay <= 0:
            delay = 1
        # Align arrays
        s_lag = source[:-delay]
        t_curr = target[delay:]
        # Build target history (length history_len)
        if history_len > 0:
            t_hist = np.column_stack(
                [target[delay - h - 1 : -(h + 1) or None] for h in range(history_len)]
            )
            # Ensure equal length
            m = min(len(s_lag), len(t_curr), len(t_hist))
            s_lag = s_lag[:m]
            t_curr = t_curr[:m]
            t_hist = t_hist[:m]
        else:
            t_hist = None
            m = min(len(s_lag), len(t_curr))
            s_lag = s_lag[:m]
            t_curr = t_curr[:m]

        try:
            # Standardize data for CMIknn (required for numerical stability)
            s_lag_norm = standardize_array(s_lag)
            t_curr_norm = standardize_array(t_curr)
            if t_hist is not None:
                t_hist_norm = np.column_stack(
                    [standardize_array(t_hist[:, i]) for i in range(t_hist.shape[1])]
                )
            else:
                t_hist_norm = None

            # Adaptive k: use min(k_neighbors, n//10) to avoid buffer overflows with small samples
            effective_k = min(k_neighbors, max(2, m // 10))
            cmi = CMIknn(significance=None, knn=effective_k)

            # Build data array for CMIknn (tigramite expects variables as ROWS, observations as COLUMNS)
            # CMIknn.get_dependence_measure(array, xyz) where:
            #   array: shape (n_vars, n_obs) with each variable as a row
            #   xyz: shape (n_vars,) with indices [X_idx, Y_idx, Z_idx, ...]
            #        indicating which rows are X (target_current), Y (source_lagged), Z (conditioning)

            if t_hist_norm is not None:
                # Case: I(target_current; source_lagged | target_history)
                # Stack as: row 0 = target_current, row 1 = source_lagged, rows 2+ = target_history
                array_rows = (
                    [t_curr_norm]
                    + [s_lag_norm]
                    + [t_hist_norm[:, i] for i in range(t_hist_norm.shape[1])]
                )
                data_array = np.vstack(array_rows)  # Shape: (2 + history_len, m)
                xyz = np.concatenate(
                    [np.array([0, 1]), np.arange(2, 2 + t_hist_norm.shape[1])]
                )
            else:
                # Case: I(target_current; source_lagged) - unconditional MI
                data_array = np.vstack([t_curr_norm, s_lag_norm])  # Shape: (2, m)
                xyz = np.array([0, 1])

            # Call CMIknn with correct API
            te_nats = cmi.get_dependence_measure(data_array, xyz)
            te_bits = float(te_nats / np.log(2))
        except Exception as e:
            logger.warning(
                f"CMIknn TE failed ({e}); falling back to discrete estimator."
            )
            te_bits = float(
                transfer_entropy_discrete(source, target, delay=delay, bins=bins)
                / np.log(2)
            )
    elif method == "discrete":
        te_bits = float(
            transfer_entropy_discrete(source, target, delay=delay, bins=bins)
            / np.log(2)
        )
    else:
        te_bits = float(transfer_entropy_mi_proxy(source, target, delay=delay))

    # Surrogate test: block-shuffle source and recompute
    te_surrogates = []
    np.random.seed(42)

    def block_permute(arr: np.ndarray, b: int) -> np.ndarray:
        if b is None or b <= 1:
            return np.random.permutation(arr)
        n = len(arr)
        # Create blocks
        blocks = [arr[i : min(i + b, n)] for i in range(0, n, b)]
        np.random.shuffle(blocks)
        return np.concatenate(blocks)

    # Default block length rule
    blen = block_length if block_length is not None else max(2 * delay, 5)

    for _ in range(n_surrogates):
        source_shuffled = (
            block_permute(source, blen)
            if block_shuffle
            else np.random.permutation(source)
        )
        if method == "cmiknn" and TIGRAMITE_AVAILABLE:
            try:
                from tigramite.independence_tests.cmiknn import CMIknn  # type: ignore
            except Exception:  # pragma: no cover
                from tigramite.independence_tests.cmi_knn import CMIknn  # type: ignore

            s_lag_s = source_shuffled[:-delay]
            t_curr_s = target[delay:]
            if history_len > 0:
                t_hist_s = np.column_stack(
                    [
                        target[delay - h - 1 : -(h + 1) or None]
                        for h in range(history_len)
                    ]
                )
                m = min(len(s_lag_s), len(t_curr_s), len(t_hist_s))
                s_lag_s = s_lag_s[:m]
                t_curr_s = t_curr_s[:m]
                t_hist_s = t_hist_s[:m]
            else:
                t_hist_s = None
                m = min(len(s_lag_s), len(t_curr_s))
                s_lag_s = s_lag_s[:m]
                t_curr_s = t_curr_s[:m]
            try:
                # Standardize data for surrogate test as well
                s_lag_s_norm = standardize_array(s_lag_s)
                t_curr_s_norm = standardize_array(t_curr_s)
                if t_hist_s is not None:
                    t_hist_s_norm = np.column_stack(
                        [
                            standardize_array(t_hist_s[:, i])
                            for i in range(t_hist_s.shape[1])
                        ]
                    )
                else:
                    t_hist_s_norm = None

                # Use same adaptive k as main estimate
                effective_k = min(k_neighbors, max(2, m // 10))
                cmi = CMIknn(significance=None, knn=effective_k)

                # Build data array for CMIknn (same structure as main estimate)
                if t_hist_s_norm is not None:
                    array_rows = (
                        [t_curr_s_norm]
                        + [s_lag_s_norm]
                        + [t_hist_s_norm[:, i] for i in range(t_hist_s_norm.shape[1])]
                    )
                    data_array = np.vstack(array_rows)
                    xyz = np.concatenate(
                        [np.array([0, 1]), np.arange(2, 2 + t_hist_s_norm.shape[1])]
                    )
                else:
                    data_array = np.vstack([t_curr_s_norm, s_lag_s_norm])
                    xyz = np.array([0, 1])

                dep = cmi.get_dependence_measure(data_array, xyz)
                te_perm = float(dep / np.log(2))
            except Exception:
                te_perm = float(
                    transfer_entropy_discrete(
                        source_shuffled, target, delay=delay, bins=bins
                    )
                    / np.log(2)
                )
        elif method == "discrete":
            te_perm = float(
                transfer_entropy_discrete(
                    source_shuffled, target, delay=delay, bins=bins
                )
                / np.log(2)
            )
        else:
            te_perm = float(
                transfer_entropy_mi_proxy(source_shuffled, target, delay=delay)
            )
        te_surrogates.append(te_perm)

    te_surrogates = np.array(te_surrogates)

    # Compute p-value: fraction of surrogates ≥ observed TE
    p_value = (np.array(te_surrogates) >= te_bits).sum() / n_surrogates
    is_significant = p_value < 0.05

    result = {
        "source": source_var,
        "target": target_var,
        "delay": delay,
        "te_bits": te_bits,
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
                    df, source, target, delay=delay, method=method, verbose=False
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
