"""Cross-site comparison and consensus graph computation.

Provides pairwise Jaccard similarity between site-level causal graphs and
group-level consensus graphs that retain only links detected in a sufficient
fraction of sites within each group.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_pairwise_jaccard(graphs: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity matrix across units.

    Parameters
    ----------
    graphs : dict[str, np.ndarray]
        Mapping from unit ID to graph array of shape (N, N, tau_max+1)
        with string entries. Non-empty entries indicate detected links.

    Returns
    -------
    pd.DataFrame
        Square symmetric matrix indexed by unit IDs, with diagonal = 1.0
        and off-diagonal values in [0, 1].
    """
    unit_ids = list(graphs.keys())
    n_units = len(unit_ids)

    # Pre-compute link sets for each unit
    link_sets: dict[str, set] = {}
    for uid in unit_ids:
        link_sets[uid] = _extract_link_set(graphs[uid])

    # Compute pairwise Jaccard
    matrix = np.ones((n_units, n_units), dtype=float)
    for i in range(n_units):
        for j in range(i + 1, n_units):
            set_a = link_sets[unit_ids[i]]
            set_b = link_sets[unit_ids[j]]
            if len(set_a) == 0 and len(set_b) == 0:
                jaccard = 1.0
            else:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                jaccard = intersection / union
            matrix[i, j] = jaccard
            matrix[j, i] = jaccard

    return pd.DataFrame(matrix, index=unit_ids, columns=unit_ids)


def compute_consensus_graph(
    graphs: dict[str, np.ndarray],
    groups: dict[str, list[str]],
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Compute per-group consensus graphs retaining links above threshold.

    For each group, computes the fraction of units detecting each link.
    Links meeting or exceeding the threshold are marked "-->"; absent
    links are marked with an empty string.

    Parameters
    ----------
    graphs : dict[str, np.ndarray]
        Unit ID -> graph array of shape (N, N, tau_max+1) with string entries.
    groups : dict[str, list[str]]
        Group label -> list of unit IDs belonging to that group.
    threshold : float
        Minimum detection fraction to retain a link in the consensus.

    Returns
    -------
    dict[str, np.ndarray]
        Group label -> consensus graph array with entries "-->" or "".

    Notes
    -----
    Groups with fewer than 2 available units (units present in `graphs`)
    are skipped with a logged warning.
    """
    consensus: dict[str, np.ndarray] = {}

    for group_label, unit_ids in groups.items():
        # Filter to units that are actually available in graphs
        available = [uid for uid in unit_ids if uid in graphs]

        if len(available) < 2:
            logger.warning(
                f"Group '{group_label}' has {len(available)} available unit(s) "
                f"(need >= 2). Skipping."
            )
            continue

        # Determine shape from first available graph
        shape = graphs[available[0]].shape
        n_units = len(available)

        # Count detections at each position
        detection_count = np.zeros(shape, dtype=float)
        for uid in available:
            g = graphs[uid]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for tau in range(shape[2]):
                        entry = g[i, j, tau]
                        if entry != "" and str(entry).strip() != "":
                            detection_count[i, j, tau] += 1.0

        # Compute fraction and apply threshold
        detection_fraction = detection_count / n_units
        consensus_graph = np.full(shape, "", dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for tau in range(shape[2]):
                    if detection_fraction[i, j, tau] >= threshold:
                        consensus_graph[i, j, tau] = "-->"
                    else:
                        consensus_graph[i, j, tau] = ""

        consensus[group_label] = consensus_graph

    return consensus


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
