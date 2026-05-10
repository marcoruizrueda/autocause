"""Split-half stability analysis for causal discovery results.

Assesses temporal stability by splitting the time series at its midpoint,
running causal discovery independently on each half, and comparing the
resulting link sets via Jaccard similarity.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Parameters
    ----------
    set_a : set
        First set of elements.
    set_b : set
        Second set of elements.

    Returns
    -------
    float
        Jaccard index in [0, 1]. Returns 1.0 if both sets are empty.
    """
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def run_split_half_stability(
    data: np.ndarray,
    var_names: list[str],
    min_timesteps: int = 50,
    causal_discovery_fn: Callable | None = None,
    **causal_kwargs,
) -> dict:
    """Split data at midpoint, run causal discovery on each half, compare.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) multivariate time series.
    var_names : list[str]
        Variable names.
    min_timesteps : int
        Minimum number of timesteps required in each half.
    causal_discovery_fn : Callable | None
        Function with signature (data, var_names, **kwargs) returning a result
        with a .graph attribute or a tuple (graph, val_matrix).
    **causal_kwargs
        Additional keyword arguments passed to the discovery function.

    Returns
    -------
    dict
        Keys: jaccard, graph_first, graph_second, links_first_only,
        links_second_only, links_both, n_first, n_second.

    Raises
    ------
    ValueError
        If either half has fewer rows than min_timesteps.
    """
    T = data.shape[0]
    midpoint = T // 2

    n_first = midpoint
    n_second = T - midpoint

    if n_first < min_timesteps:
        raise ValueError(
            f"First half has {n_first} timesteps, which is below the "
            f"minimum of {min_timesteps}."
        )
    if n_second < min_timesteps:
        raise ValueError(
            f"Second half has {n_second} timesteps, which is below the "
            f"minimum of {min_timesteps}."
        )

    if causal_discovery_fn is None:
        raise NotImplementedError(
            "No default causal discovery function available. "
            "Pass causal_discovery_fn explicitly."
        )

    data_first = data[:midpoint]
    data_second = data[midpoint:]

    # Run causal discovery on each half
    result_first = causal_discovery_fn(data_first, var_names, **causal_kwargs)
    result_second = causal_discovery_fn(data_second, var_names, **causal_kwargs)

    graph_first = (
        result_first[0] if isinstance(result_first, tuple) else result_first.graph
    )
    graph_second = (
        result_second[0] if isinstance(result_second, tuple) else result_second.graph
    )

    # Extract link sets as (i, j, tau) tuples
    links_first = _extract_link_set(graph_first)
    links_second = _extract_link_set(graph_second)

    links_both = links_first & links_second
    links_first_only = links_first - links_second
    links_second_only = links_second - links_first

    jaccard = compute_jaccard(links_first, links_second)

    return {
        "jaccard": jaccard,
        "graph_first": graph_first,
        "graph_second": graph_second,
        "links_first_only": links_first_only,
        "links_second_only": links_second_only,
        "links_both": links_both,
        "n_first": n_first,
        "n_second": n_second,
    }


def _extract_link_set(graph: np.ndarray) -> set:
    """Extract the set of detected link positions from a graph array.

    Parameters
    ----------
    graph : np.ndarray
        String array of shape (N, N, tau_max+1).

    Returns
    -------
    set
        Set of (i, j, tau) tuples where a link is detected.
    """
    links = set()
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            for tau in range(graph.shape[2]):
                entry = graph[i, j, tau]
                if entry != "" and str(entry).strip() != "":
                    links.add((i, j, tau))
    return links
