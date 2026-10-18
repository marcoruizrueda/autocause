"""Regressions found by executing and inspecting the quickstart workflow."""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "statistics",
    [
        {"p_value": 0.5},
        {"p_value": 0.001, "q_value": 0.2},
        {"p_value": np.nan, "q_value": np.nan, "is_significant": True},
        {"p_value": -np.inf},
    ],
)
def test_consensus_rejects_failed_or_nonsignificant_tests(statistics):
    from autocause.core.methods.consensus import detect_agreement

    granger = pd.DataFrame([{"source": "X", "target": "Y", "lag": 1, "p_value": 0.03}])
    te = pd.DataFrame([{"source": "X", "target": "Y", "delay": 1, **statistics}])
    result = detect_agreement(granger, te, pd.DataFrame(), min_votes=2)
    # p=.03 is significant but outside the existing p<.02 singleton exception.
    assert result.empty


def test_consensus_preserves_zero_p_values_and_valid_votes():
    from autocause.core.methods.consensus import detect_agreement

    granger = pd.DataFrame([{"source": "X", "target": "Y", "lag": 1, "p_value": 0.03}])
    te = pd.DataFrame([{"source": "X", "target": "Y", "delay": 1, "p_value": 0.0, "q_value": 0.0}])
    result = detect_agreement(granger, te, pd.DataFrame(), min_votes=2)
    assert len(result) == 1
    assert result.iloc[0].vote_count == 2
    assert result.iloc[0].best_p_value == 0.0
    assert result.iloc[0].best_q_value == 0.0


def test_undirected_pcmci_adjacency_does_not_vote_for_a_direction():
    from autocause.core.methods.consensus import detect_agreement

    granger = pd.DataFrame([{"source": "X", "target": "Y", "lag": 1, "p_value": 0.03}])
    pcmci = pd.DataFrame([{
        "source": "X", "target": "Y", "best_lag": 1,
        "best_p_value": 0.001, "causal": True, "is_directed": False,
    }])
    assert detect_agreement(granger, pd.DataFrame(), pcmci, min_votes=2).empty


@pytest.mark.parametrize(
    "contemp_p,lagged_p,expected_lag,expected_directed",
    [(0.02, 1e-8, 1, True), (1e-8, 0.02, 0, False)],
)
def test_pcmci_summary_fields_describe_the_same_edge(
    monkeypatch, contemp_p, lagged_p, expected_lag, expected_directed
):
    from autocause.core.methods import tigramite_pcmci as pcmci

    def fake_algorithm(data, **kwargs):
        names = list(data.columns)
        i, j = names.index("X"), names.index("Y")
        graph = np.full((2, 2, 2), "", dtype="<U3")
        p_values = np.ones((2, 2, 2))
        graph[i, j, 0] = graph[j, i, 0] = "o-o"
        p_values[i, j, 0] = p_values[j, i, 0] = contemp_p
        graph[i, j, 1] = "-->"
        p_values[i, j, 1] = lagged_p
        return {"graph": graph, "significances": p_values}

    monkeypatch.setattr(pcmci, "run_pcmci_algorithm", fake_algorithm)
    df = pd.DataFrame(np.random.default_rng(1).normal(size=(100, 2)), columns=["X", "Y"])
    result = pcmci.batch_pcmci(df, [("X", "Y"), ("Y", "X")], tau_max=1)
    row = result[(result.source == "X") & (result.target == "Y")].iloc[0]
    assert row.best_lag == expected_lag
    assert row.best_p_value == min(contemp_p, lagged_p)
    assert bool(row.is_directed) == expected_directed
    assert bool(row.is_contemporaneous) == (expected_lag == 0)


def test_workflow_falsification_callback_uses_actual_granger_p_value(monkeypatch, tmp_path):
    from autocause.core import falsification
    from autocause.core.methods import visualize_results
    from autocause.core.run_workflow import run_causal_discovery_workflow

    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    y = np.r_[0, x[:-1]] + rng.normal(scale=0.1, size=200)
    df = pd.DataFrame({"X": x, "Y": y})
    observed_p_values = []

    def capture_battery(source, target, lag, test_func, **kwargs):
        observed_p_values.append(test_func(source, target, lag))
        return {"block_permutation": {"passed": False}, "iaaft": {"passed": False}}

    monkeypatch.setattr(falsification, "run_falsification_battery", capture_battery)
    monkeypatch.setattr(visualize_results, "visualize_all_results", lambda *args: None)
    workflow = run_causal_discovery_workflow(
        data_df=df,
        output_dir=tmp_path,
        tau_max=1,
        pairs=[("X", "Y")],
        enable_preprocessing=False,
        enable_distribution_tests=False,
        enable_strength_analysis=True,
        enable_temporal_validation=False,
        enable_tracking=False,
        enable_consensus=True,
        method_config={
            "granger": {"enabled": True},
            "transfer_entropy": {"enabled": False},
            "pcmci": {"enabled": False},
        },
    )
    assert observed_p_values, "The workflow must actually call the falsification test"
    assert all(np.isfinite(p) and p < 0.001 for p in observed_p_values)
    assert not workflow['results']['granger'].empty
    assert workflow['n_edges']['granger'] == len(workflow['results']['granger'])
