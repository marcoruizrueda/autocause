"""Tests for the reviewer-requested evaluation diagnostics.

Covers the chance baseline, pairwise method agreement, and the adjustment
identification distance. The AID test is skipped when the optional gadjid
package is absent, so the suite passes in a minimal environment while still
exercising the metric where the dependency is installed.
"""
import pytest

from autocause.core.evaluation_extras import (
    adjustment_identification_distance,
    chance_baseline,
    chance_expected_recall,
    method_agreement,
)

VARS = ["X0", "X1", "X2", "X3", "X4", "X5"]
TRUE = {
    ("X0", "X1"), ("X1", "X2"), ("X1", "X5"), ("X0", "X4"),
    ("X2", "X3"), ("X2", "X4"), ("X3", "X4"), ("X3", "X5"), ("X1", "X4"),
}


def test_chance_precision_is_prevalence():
    cb = chance_baseline(TRUE, VARS)
    assert cb["d"] == 6
    assert cb["m_max"] == 15  # 6 * 5 / 2
    assert cb["m_true"] == 9
    assert cb["expected_precision"] == pytest.approx(9 / 15)


def test_chance_precision_ignores_self_loops_and_direction():
    # A self-loop and a reversed duplicate must not inflate m_true.
    edges = {("X0", "X0"), ("X0", "X1"), ("X1", "X0")}
    cb = chance_baseline(edges, ["X0", "X1", "X2"])
    assert cb["m_true"] == 1  # single undirected pair {X0, X1}
    assert cb["m_max"] == 3


def test_chance_expected_recall_scales_with_edges():
    assert chance_expected_recall(0, VARS) == pytest.approx(0.0)
    assert chance_expected_recall(15, VARS) == pytest.approx(1.0)
    assert chance_expected_recall(3, VARS) == pytest.approx(3 / 15)


def test_method_agreement_identical_sets_is_one():
    disc = {
        "granger": {("X0", "X1"), ("X1", "X2")},
        "transfer_entropy": {("X0", "X1"), ("X1", "X2")},
        "pcmci": {("X0", "X1")},
    }
    agr = method_agreement(disc)
    assert agr["granger__transfer_entropy"] == pytest.approx(1.0)
    assert agr["granger__pcmci"] == pytest.approx(0.5)


def test_method_agreement_handles_empty_sets():
    agr = method_agreement({"a": set(), "b": set()})
    assert agr["a__b"] == pytest.approx(0.0)


def test_aid_reference_not_dag_is_flagged():
    # A two-cycle reference is not a DAG; AID must decline rather than raise.
    cyclic = {("X0", "X1"), ("X1", "X0")}
    res = adjustment_identification_distance({("X0", "X1")}, cyclic, ["X0", "X1"])
    assert res["status"] == "reference_not_dag"
    assert res["parent_aid"] is None


def test_aid_on_dag_when_gadjid_available():
    pytest.importorskip("gadjid")
    true = {("X0", "X1"), ("X1", "X2")}  # DAG
    exact = adjustment_identification_distance(true, true, ["X0", "X1", "X2"])
    assert exact["status"] == "ok"
    assert exact["parent_aid"] == pytest.approx(0.0)
    # Missing an edge should raise the distance above zero.
    partial = adjustment_identification_distance(
        {("X0", "X1")}, true, ["X0", "X1", "X2"]
    )
    assert partial["status"] == "ok"
    assert partial["parent_aid"] > 0
