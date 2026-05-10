"""Tests for temporal ICP stability (single-unit time series)."""

import numpy as np
import pandas as pd
import pytest

from framework.core.icp_stability import (
    create_temporal_environments,
    test_edge_stability_temporal,
    test_consensus_stability_temporal,
)


@pytest.fixture
def stable_data():
    """Create synthetic data with a stable causal relationship."""
    np.random.seed(42)
    T = 600
    x = np.cumsum(np.random.randn(T) * 0.5)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * x[t - 1] + np.random.randn() * 0.3
    return pd.DataFrame(
        {"X": x, "Y": y},
        index=pd.date_range("2020-01-01", periods=T, freq="D"),
    )


@pytest.fixture
def unstable_data():
    """Create data with a regime-switching (unstable) relationship."""
    np.random.seed(42)
    T = 600
    x = np.cumsum(np.random.randn(T) * 0.5)
    y = np.zeros(T)
    for t in range(1, T):
        if t < T // 2:
            y[t] = 0.8 * x[t - 1] + np.random.randn() * 0.2
        else:
            y[t] = -0.3 * x[t - 1] + np.random.randn() * 0.2
    return pd.DataFrame(
        {"X": x, "Y": y},
        index=pd.date_range("2020-01-01", periods=T, freq="D"),
    )


def test_create_temporal_environments(stable_data):
    envs = create_temporal_environments(stable_data, n_blocks=3)
    assert len(envs.unique()) == 3
    assert len(envs) == len(stable_data)


def test_stable_edge_detected(stable_data):
    result = test_edge_stability_temporal(
        stable_data, source_col="X", target_col="Y", lag=1, n_blocks=3
    )
    assert result["n_environments"] >= 2
    assert result["stable"] == True
    # Coefficients should all be near 0.5
    for env_r in result["environment_results"]:
        assert 0.3 < env_r["coefficient"] < 0.7


def test_unstable_edge_detected(unstable_data):
    result = test_edge_stability_temporal(
        unstable_data, source_col="X", target_col="Y", lag=1, n_blocks=3
    )
    assert result["n_environments"] >= 2
    assert result["stable"] == False


def test_consensus_stability_temporal_integration(stable_data, unstable_data):
    # Combine both into one DataFrame
    df = stable_data.copy()
    df["Y_unstable"] = unstable_data["Y"].values

    consensus_df = pd.DataFrame(
        {
            "source": ["X", "X"],
            "target": ["Y", "Y_unstable"],
            "lag_steps": [1, 1],
            "vote_count": [3, 2],
        }
    )

    result = test_consensus_stability_temporal(df, consensus_df, n_blocks=3)
    assert result.iloc[0]["icp_stable"] == True
    assert result.iloc[1]["icp_stable"] == False
    assert "icp_n_environments" in result.columns
    assert "icp_test_statistic" in result.columns
