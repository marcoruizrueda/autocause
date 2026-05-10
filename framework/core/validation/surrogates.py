"""Surrogate data generation and false positive rate estimation.

Provides shuffle surrogates (destroying temporal structure while preserving
marginal distributions) and phase surrogates (preserving power spectra while
randomizing phase relationships) for empirical significance testing.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def generate_shuffle_surrogates(data: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each column, preserving marginal distributions.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) multivariate time series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shuffled array of the same shape as data.
    """
    rng = np.random.default_rng(seed)
    surrogate = data.copy()
    for col in range(surrogate.shape[1]):
        rng.shuffle(surrogate[:, col])
    return surrogate


def generate_phase_surrogates(data: np.ndarray, seed: int) -> np.ndarray:
    """Phase-randomize each column via FFT, preserving power spectra.

    For each column, compute the FFT, randomize the phases uniformly while
    maintaining conjugate symmetry (so the inverse FFT yields real values),
    then transform back.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) multivariate time series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Phase-randomized array of the same shape as data.
    """
    rng = np.random.default_rng(seed)
    T, N = data.shape
    surrogate = np.zeros_like(data)

    for col in range(N):
        x = data[:, col]
        ft = np.fft.rfft(x)
        amplitudes = np.abs(ft)

        # Generate random phases; keep DC (index 0) and Nyquist (last if T even)
        n_freq = len(ft)
        random_phases = rng.uniform(0, 2 * np.pi, size=n_freq)
        random_phases[0] = 0.0
        if T % 2 == 0:
            random_phases[-1] = 0.0

        ft_surrogate = amplitudes * np.exp(1j * random_phases)
        surrogate[:, col] = np.fft.irfft(ft_surrogate, n=T)

    return surrogate


def compute_fpr(
    data: np.ndarray,
    var_names: list[str],
    n_surrogates: int = 100,
    seed: int = 42,
    causal_discovery_fn: Callable | None = None,
    **causal_kwargs,
) -> dict:
    """Compute empirical false positive rate from surrogate analysis.

    Runs causal discovery on shuffle and phase surrogates. Since surrogates
    destroy causal relationships, any detected links are false positives.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) original time series.
    var_names : list[str]
        Variable names.
    n_surrogates : int
        Number of surrogate datasets per type.
    seed : int
        Base random seed.
    causal_discovery_fn : Callable | None
        Function with signature (data, var_names, **kwargs) returning a result
        with a .graph attribute or a tuple (graph, val_matrix).
    **causal_kwargs
        Additional keyword arguments passed to the discovery function.

    Returns
    -------
    dict
        Keys: fpr_shuffle, fpr_phase, fpr_combined, n_surrogates,
        links_per_surrogate_shuffle, links_per_surrogate_phase,
        total_possible_links.
    """
    T, N = data.shape
    tau_max = causal_kwargs.get("tau_max", 14)
    # Total possible links: all (i, j, tau) positions excluding self at lag 0
    total_possible_links = N * N * (tau_max + 1) - N

    if total_possible_links <= 0:
        total_possible_links = 1  # Avoid division by zero for trivial cases

    links_shuffle: list[int] = []
    links_phase: list[int] = []

    for i in range(n_surrogates):
        # Shuffle surrogate
        surr_shuffle = generate_shuffle_surrogates(data, seed=seed + i)
        n_links_shuffle = _count_links(
            surr_shuffle, var_names, causal_discovery_fn, causal_kwargs
        )
        links_shuffle.append(n_links_shuffle)

        # Phase surrogate
        surr_phase = generate_phase_surrogates(data, seed=seed + n_surrogates + i)
        n_links_phase = _count_links(
            surr_phase, var_names, causal_discovery_fn, causal_kwargs
        )
        links_phase.append(n_links_phase)

    fpr_shuffle = float(np.mean(links_shuffle)) / total_possible_links
    fpr_phase = float(np.mean(links_phase)) / total_possible_links
    fpr_combined = (fpr_shuffle + fpr_phase) / 2.0

    return {
        "fpr_shuffle": fpr_shuffle,
        "fpr_phase": fpr_phase,
        "fpr_combined": fpr_combined,
        "n_surrogates": n_surrogates,
        "links_per_surrogate_shuffle": links_shuffle,
        "links_per_surrogate_phase": links_phase,
        "total_possible_links": total_possible_links,
    }


def _count_links(
    surrogate_data: np.ndarray,
    var_names: list[str],
    causal_discovery_fn: Callable | None,
    causal_kwargs: dict,
) -> int:
    """Run causal discovery on surrogate data and count detected links."""
    if causal_discovery_fn is None:
        raise NotImplementedError(
            "No default causal discovery function available. "
            "Pass causal_discovery_fn explicitly."
        )

    try:
        result = causal_discovery_fn(surrogate_data, var_names, **causal_kwargs)
        if isinstance(result, tuple):
            graph = result[0]
        else:
            graph = result.graph

        # Count non-empty entries, excluding self-links at lag 0
        N = graph.shape[0]
        count = 0
        for i in range(N):
            for j in range(N):
                for tau in range(graph.shape[2]):
                    if tau == 0 and i == j:
                        continue
                    if graph[i, j, tau] != "" and str(graph[i, j, tau]).strip() != "":
                        count += 1
        return count

    except Exception as e:
        logger.warning(f"Surrogate causal discovery failed: {e}")
        return 0
