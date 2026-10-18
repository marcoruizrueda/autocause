"""Additional evaluation diagnostics for causal discovery.

These capabilities extend graph-recovery scoring with metrics that go beyond
adjacency F1 and SHD. They are computed automatically by the workflow whenever a
reference graph is supplied, and each one degrades gracefully when its
precondition is not met, so a run never fails because an extra diagnostic could
not be computed.

The capabilities are:

  * ``chance_baseline``: the expected precision and recall of a random graph,
    following the negative-control argument of Petersen (UAI 2025). It depends
    only on the reference graph, so it is always available.
  * ``method_agreement``: pairwise Jaccard agreement between the edge sets of the
    discovery methods, which quantifies how independent the method votes are.
  * ``adjustment_identification_distance``: the parent and ancestor adjustment
    identification distances of Henckel, Wuertzen and Weichwald (UAI 2024),
    computed through the optional ``gadjid`` package when the reference is a DAG.
  * ``structural_intervention_distance``: the SID of Peters and Buehlmann (2015),
    reported as the parent-adjustment special case of the AID when available.

References are listed in the project bibliography. The functions take the same
edge-set and variable-name inputs already used by ``graph_metrics``.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

Edge = Tuple[str, str]
EdgeSet = Set[Edge]


def _undirected(edges: EdgeSet) -> Set[frozenset]:
    """Undirected pairs, dropping self-loops."""
    return {frozenset(e) for e in edges if e[0] != e[1]}


# ---------------------------------------------------------------------------
# Chance baseline (always available)
# ---------------------------------------------------------------------------


def chance_baseline(true_edges: EdgeSet, var_names: Sequence[str]) -> Dict[str, float]:
    """Expected adjacency precision and recall under random guessing.

    With ``m_true`` reference adjacencies over ``m_max = d(d-1)/2`` possible
    undirected pairs on ``d`` variables, a graph whose edges are placed at random
    recovers true adjacencies in proportion to their prevalence. The expected
    precision is ``m_true / m_max`` and does not depend on how many edges the
    estimate contains; the expected recall of an estimate with ``m_est`` edges is
    ``m_est / m_max`` (Petersen, UAI 2025). Edges are counted on the undirected
    skeleton, matching ``binary_metrics_undirected``.

    Returns the prevalence ``m_true / m_max`` as ``expected_precision`` together
    with the counts, so a caller can form the expected recall for any ``m_est``.
    """
    d = len(var_names)
    if d < 2:
        return {
            "d": d,
            "m_true": 0,
            "m_max": 0,
            "expected_precision": float("nan"),
        }
    m_max = d * (d - 1) // 2
    m_true = len(_undirected(true_edges))
    return {
        "d": int(d),
        "m_true": int(m_true),
        "m_max": int(m_max),
        "expected_precision": m_true / m_max,
    }


def chance_expected_recall(m_est: int, var_names: Sequence[str]) -> float:
    """Expected recall of a random estimate with ``m_est`` edges."""
    d = len(var_names)
    if d < 2:
        return float("nan")
    m_max = d * (d - 1) // 2
    return m_est / m_max


# ---------------------------------------------------------------------------
# Pairwise method agreement (Jaccard)
# ---------------------------------------------------------------------------


def method_agreement(
    discovered_by_method: Dict[str, EdgeSet],
) -> Dict[str, float]:
    """Pairwise Jaccard agreement between method edge sets, on the skeleton.

    For each pair of methods the Jaccard index is the size of the intersection of
    their undirected edge sets over the size of the union. High agreement between
    two methods means their votes are not independent, which qualifies how a
    method-count consensus should be read. Granger causality and transfer entropy
    coincide for Gaussian variables (Barnett, Barrett and Seth, 2009), so a high
    value between them is expected and reduces the effective number of distinct
    votes. Methods with empty edge sets are reported with Jaccard zero.
    """
    methods = sorted(discovered_by_method)
    out: Dict[str, float] = {}
    for a, b in combinations(methods, 2):
        ea = _undirected(discovered_by_method.get(a, set()))
        eb = _undirected(discovered_by_method.get(b, set()))
        union = ea | eb
        jac = len(ea & eb) / len(union) if union else 0.0
        out[f"{a}__{b}"] = jac
    return out


# ---------------------------------------------------------------------------
# Adjustment identification distance (optional, needs gadjid)
# ---------------------------------------------------------------------------


def _edges_to_adjacency(edges: EdgeSet, var_names: Sequence[str]):
    """Directed adjacency matrix (numpy) with order given by var_names."""
    import numpy as np

    idx = {v: i for i, v in enumerate(var_names)}
    d = len(var_names)
    mat = np.zeros((d, d), dtype=np.int8)
    for s, t in edges:
        if s in idx and t in idx and s != t:
            mat[idx[s], idx[t]] = 1
    return mat


def _has_two_cycle(mat) -> bool:
    """True if the directed graph contains a mutual pair i->j and j->i."""
    import numpy as np

    a = (np.asarray(mat) != 0).astype(np.int64)
    return bool(np.any((a == 1) & (a.T == 1)))


def _is_dag(mat) -> bool:
    import numpy as np

    a = (np.asarray(mat) != 0).astype(np.int64)
    reach = a.copy()
    power = a.copy()
    for _ in range(a.shape[0]):
        power = (power @ a > 0).astype(np.int64)
        reach = ((reach + power) > 0).astype(np.int64)
    return bool(np.trace(reach) == 0)


def adjustment_identification_distance(
    discovered: EdgeSet,
    true_edges: EdgeSet,
    var_names: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Parent and ancestor AID between the estimated and reference graphs.

    AID (Henckel, Wuertzen and Weichwald, UAI 2024) counts the ordered variable
    pairs for which the estimated graph implies a wrong adjustment set relative to
    the reference. The parent-adjustment case equals the structural intervention
    distance of Peters and Buehlmann (2015). The distance is defined when the
    reference is a directed acyclic graph. When the reference contains a cycle,
    for example a lag-1 two-cycle after collapsing to a summary graph, or when the
    optional ``gadjid`` package is not installed, the function returns ``None``
    values with a status string rather than raising.
    """
    status = "ok"
    result: Dict[str, Optional[float]] = {
        "parent_aid": None,
        "ancestor_aid": None,
        "sid": None,
        "status": status,
    }

    ref_mat = _edges_to_adjacency(true_edges, var_names)
    est_mat = _edges_to_adjacency(discovered, var_names)

    if not _is_dag(ref_mat):
        result["status"] = "reference_not_dag"
        return result

    # gadjid requires simple graphs, so a two-cycle in either graph (which a
    # summary graph can contain, e.g. lag-1 mutual causation) is not scored.
    if _has_two_cycle(ref_mat) or _has_two_cycle(est_mat):
        result["status"] = "non_simple_graph"
        return result

    try:
        import gadjid  # type: ignore
    except ImportError:
        result["status"] = "gadjid_not_installed"
        return result

    try:
        parent = gadjid.parent_aid(
            ref_mat, est_mat, edge_direction="from row to column"
        )
        ancestor = gadjid.ancestor_aid(
            ref_mat, est_mat, edge_direction="from row to column"
        )
        # gadjid returns (normalised_distance, mistake_count); keep the count.
        result["parent_aid"] = (
            float(parent[1]) if isinstance(parent, tuple) else float(parent)
        )
        result["ancestor_aid"] = (
            float(ancestor[1]) if isinstance(ancestor, tuple) else float(ancestor)
        )
        result["sid"] = result["parent_aid"]  # SID is the parent-adjustment case
        result["status"] = "ok"
    except BaseException as exc:  # noqa: BLE001  gadjid raises a Rust panic
        # The gadjid backend can raise a PanicException that does not subclass
        # Exception, so the broad catch keeps a diagnostic from failing the run.
        logger.warning("AID computation skipped: %s", exc)
        result["status"] = "not_computed"
    return result


# ---------------------------------------------------------------------------
# Convenience: compute all extras for one run
# ---------------------------------------------------------------------------


def compute_evaluation_extras(
    discovered_by_method: Dict[str, EdgeSet],
    true_edges: EdgeSet,
    var_names: Sequence[str],
) -> Dict[str, object]:
    """Compute every extra diagnostic for one dataset, with graceful fallbacks.

    Returns a nested dictionary suitable for serialisation alongside the standard
    graph-recovery metrics. Nothing here runs causal discovery; it reads only the
    edge sets a run already produced and the reference graph.
    """
    extras: Dict[str, object] = {
        "chance_baseline": chance_baseline(true_edges, var_names),
        "method_agreement_jaccard": method_agreement(discovered_by_method),
        "adjustment_identification_distance": {
            method: adjustment_identification_distance(edges, true_edges, var_names)
            for method, edges in discovered_by_method.items()
        },
    }
    return extras
