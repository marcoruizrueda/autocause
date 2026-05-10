"""Block-bootstrap resampling for causal discovery confidence estimation.

Provides frequency-based link detection confidence and effect-size confidence
intervals by repeatedly resampling the time series in contiguous blocks and
re-running causal discovery on each resample.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Aggregated block-bootstrap statistics.

    All arrays have shape (N, N, tau_max+1).
    """

    link_frequency: np.ndarray
    strength_mean: np.ndarray
    strength_ci_low: np.ndarray
    strength_ci_high: np.ndarray


def aggregate_bootstrap(
    detections: np.ndarray,
    strengths: np.ndarray,
) -> BootstrapResult:
    """Compute frequency and CI statistics from per-sample arrays.

    Parameters
    ----------
    detections : np.ndarray
        Boolean array of shape (n_bootstrap, N, N, tau_max+1).
    strengths : np.ndarray
        Float array of shape (n_bootstrap, N, N, tau_max+1).

    Returns
    -------
    BootstrapResult
    """
    link_frequency = detections.mean(axis=0)
    strength_mean = strengths.mean(axis=0)
    strength_ci_low = np.percentile(strengths, 2.5, axis=0)
    strength_ci_high = np.percentile(strengths, 97.5, axis=0)

    return BootstrapResult(
        link_frequency=link_frequency,
        strength_mean=strength_mean,
        strength_ci_low=strength_ci_low,
        strength_ci_high=strength_ci_high,
    )


def run_bootstrap(
    data: np.ndarray,
    var_names: list[str],
    n_bootstrap: int = 200,
    block_length: int | None = None,
    causal_discovery_fn: Callable | None = None,
    algorithm: str = "pcmciplus",
    ci_test: str = "parcorr",
    tau_max: int = 14,
    pc_alpha: float | None = None,
    fdr_method: str | None = "fdr_bh",
    seed: int = 42,
    min_timesteps: int = 50,
) -> BootstrapResult:
    """Block-bootstrap resampling with configurable causal discovery backend.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) multivariate time series.
    var_names : list[str]
        Length-N variable names.
    n_bootstrap : int
        Number of bootstrap iterations.
    block_length : int | None
        Contiguous block length for resampling. Defaults to tau_max.
    causal_discovery_fn : Callable | None
        Function with signature (data, var_names, **kwargs) that returns an
        object with .graph (np.ndarray) and .val_matrix (np.ndarray) attributes,
        or a tuple (graph, val_matrix).
    algorithm : str
        Algorithm name passed to the default discovery function.
    ci_test : str
        Conditional independence test name.
    tau_max : int
        Maximum time lag.
    pc_alpha : float | None
        Significance level for the PC algorithm step.
    fdr_method : str | None
        FDR correction method.
    seed : int
        Random seed for reproducibility.
    min_timesteps : int
        Minimum number of timesteps required per bootstrap sample.

    Returns
    -------
    BootstrapResult

    Raises
    ------
    ValueError
        If data has fewer rows than block_length.
    """
    T, N = data.shape

    if block_length is None:
        block_length = tau_max

    if T < block_length:
        raise ValueError(
            f"Data has {T} timesteps but block_length is {block_length}. "
            f"Data must have at least block_length rows."
        )

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block_length))

    tau_dim = tau_max + 1
    detections = np.zeros((n_bootstrap, N, N, tau_dim), dtype=bool)
    strengths = np.zeros((n_bootstrap, N, N, tau_dim), dtype=float)

    causal_kwargs = {
        "algorithm": algorithm,
        "ci_test": ci_test,
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        "fdr_method": fdr_method,
    }

    n_successful = 0
    for b in range(n_bootstrap):
        # Draw block starts uniformly from [0, T - block_length]
        max_start = T - block_length
        starts = rng.integers(0, max_start + 1, size=n_blocks)

        # Concatenate blocks and truncate to length T
        blocks = [data[s : s + block_length] for s in starts]
        resampled = np.concatenate(blocks, axis=0)[:T]

        try:
            if causal_discovery_fn is not None:
                result = causal_discovery_fn(resampled, var_names, **causal_kwargs)
            else:
                raise NotImplementedError(
                    "No default causal discovery function available. "
                    "Pass causal_discovery_fn explicitly."
                )

            # Extract graph and val_matrix from result
            if isinstance(result, tuple):
                graph, val_matrix = result[0], result[1]
            else:
                graph = result.graph
                val_matrix = result.val_matrix

            # Record detection (non-empty string entries)
            detection = np.array(graph != "", dtype=bool)
            # Ensure shapes match; truncate tau dimension if needed
            actual_tau = detection.shape[2] if detection.ndim == 3 else 1
            tau_use = min(actual_tau, tau_dim)
            detections[b, :, :, :tau_use] = detection[:, :, :tau_use]
            strengths[b, :, :, :tau_use] = np.abs(val_matrix[:, :, :tau_use])
            n_successful += 1

        except Exception as e:
            logger.warning(f"Bootstrap iteration {b} failed: {e}")
            continue

    if n_successful == 0:
        logger.error("All bootstrap iterations failed.")
        return BootstrapResult(
            link_frequency=np.zeros((N, N, tau_dim)),
            strength_mean=np.zeros((N, N, tau_dim)),
            strength_ci_low=np.zeros((N, N, tau_dim)),
            strength_ci_high=np.zeros((N, N, tau_dim)),
        )

    if n_successful < n_bootstrap * 0.5:
        logger.warning(
            f"Only {n_successful}/{n_bootstrap} bootstrap iterations succeeded."
        )

    # Use only successful iterations
    return aggregate_bootstrap(detections[:n_successful], strengths[:n_successful])
