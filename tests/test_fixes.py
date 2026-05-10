"""Tests for the Issue fixes: lagged correlations, paradigm diversity, tau_max multi-var."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def causal_data():
    """Synthetic data: X causes Y at lag 3, Z is independent noise."""
    np.random.seed(42)
    T = 500
    x = np.random.randn(T)
    z = np.random.randn(T)
    y = np.zeros(T)
    for t in range(3, T):
        y[t] = 0.7 * x[t - 3] + 0.3 * np.random.randn()
    return pd.DataFrame(
        {"X": x, "Y": y, "Z": z},
        index=pd.date_range("2020-01-01", periods=T, freq="D"),
    )


class TestLaggedCorrelation:
    """Test Issue 8 fix: lagged cross-correlation baseline."""

    def test_detects_correct_lag(self, causal_data):
        from framework.core.methods.correlation import batch_lagged_correlation

        results = batch_lagged_correlation(
            causal_data,
            variable_pairs=[("X", "Y"), ("Z", "Y")],
            tau_max=10,
            alpha=0.05,
        )

        assert len(results) > 0
        assert "lag" in results.columns
        assert "pearson_r" in results.columns
        assert "is_significant" in results.columns
        assert "is_best_lag" in results.columns

        # X→Y should have best lag near 3
        xy_best = results[
            (results["source"] == "X")
            & (results["target"] == "Y")
            & results["is_best_lag"]
        ]
        assert len(xy_best) == 1
        assert abs(xy_best.iloc[0]["lag"] - 3) <= 1  # Within 1 step of true lag

        # X→Y at lag 3 should be significant
        xy_lag3 = results[
            (results["source"] == "X")
            & (results["target"] == "Y")
            & (results["lag"] == 3)
        ]
        assert len(xy_lag3) == 1
        assert xy_lag3.iloc[0]["is_significant"] == True
        assert abs(xy_lag3.iloc[0]["pearson_r"]) > 0.5

        # Z→Y should have weak correlations (noise)
        zy_best = results[
            (results["source"] == "Z")
            & (results["target"] == "Y")
            & results["is_best_lag"]
        ]
        assert len(zy_best) == 1
        assert abs(zy_best.iloc[0]["pearson_r"]) < 0.2

    def test_respects_tau_max(self, causal_data):
        from framework.core.methods.correlation import batch_lagged_correlation

        results = batch_lagged_correlation(
            causal_data,
            variable_pairs=[("X", "Y")],
            tau_max=5,
        )
        assert results["lag"].max() <= 5

    def test_handles_missing_data(self):
        """Lagged correlation should handle NaN values gracefully."""
        from framework.core.methods.correlation import batch_lagged_correlation

        np.random.seed(42)
        T = 200
        df = pd.DataFrame(
            {
                "X": np.random.randn(T),
                "Y": np.random.randn(T),
            },
            index=pd.date_range("2020-01-01", periods=T, freq="D"),
        )
        # Introduce 20% NaN
        df.loc[df.sample(frac=0.2).index, "X"] = np.nan

        results = batch_lagged_correlation(df, [("X", "Y")], tau_max=5)
        assert len(results) > 0
        assert results["n_obs"].min() > 0


class TestParadigmDiversity:
    """Test Issue 2 fix: paradigm-weighted consensus."""

    def test_paradigm_mapping(self):
        from framework.core.methods.consensus import compute_paradigm_diversity

        # Same paradigm: should be 1
        assert compute_paradigm_diversity(["Granger", "VARLiNGAM"]) == 1

        # Two paradigms
        assert compute_paradigm_diversity(["Granger", "PCMCI+"]) == 2
        assert compute_paradigm_diversity(["Granger", "TransferEntropy"]) == 2

        # Three paradigms (maximum useful diversity)
        assert compute_paradigm_diversity(["Granger", "TransferEntropy", "PCMCI+"]) == 3

        # All four
        assert (
            compute_paradigm_diversity(["Granger", "TransferEntropy", "PCMCI+", "RF"])
            == 4
        )

    def test_consensus_includes_paradigm_diversity(self, causal_data):
        """Consensus output should include paradigm_diversity column."""
        from framework.core.methods.consensus import detect_agreement

        # Create fake method results that agree on X→Y
        granger_df = pd.DataFrame(
            [
                {
                    "source": "X",
                    "target": "Y",
                    "lag_steps": 3,
                    "p_value": 0.001,
                    "is_significant": True,
                }
            ]
        )
        te_df = pd.DataFrame(
            [
                {
                    "source": "X",
                    "target": "Y",
                    "lag_steps": 3,
                    "p_value": 0.01,
                    "is_significant": True,
                }
            ]
        )
        pcmci_df = pd.DataFrame(
            [
                {
                    "source": "X",
                    "target": "Y",
                    "lag_steps": 3,
                    "p_value": 0.005,
                    "is_significant": True,
                }
            ]
        )

        consensus = detect_agreement(granger_df, te_df, pcmci_df, min_votes=2)

        assert len(consensus) > 0
        assert "paradigm_diversity" in consensus.columns
        # All three methods from different paradigms → diversity = 3
        assert consensus.iloc[0]["paradigm_diversity"] == 3


class TestTauMaxMultiVariable:
    """Test Issue 6 fix: tau_max estimated across all variables."""

    def test_uses_max_across_pairs(self):
        """tau_max should reflect the longest memory in the system."""
        from framework.core.tau_max_estimation import estimate_tau_max_scientific

        np.random.seed(42)
        T = 500

        # Variable with short memory (AR(1) with low persistence)
        x_short = np.zeros(T)
        for t in range(1, T):
            x_short[t] = 0.3 * x_short[t - 1] + np.random.randn()

        # Variable with long memory (AR(1) with high persistence)
        x_long = np.zeros(T)
        for t in range(1, T):
            x_long[t] = 0.95 * x_long[t - 1] + np.random.randn()

        # Estimate for short-memory pair
        result_short = estimate_tau_max_scientific(
            pd.Series(x_short),
            pd.Series(x_short),
            sampling_interval_days=1,
            domain_max_days=90,
        )

        # Estimate for long-memory pair
        result_long = estimate_tau_max_scientific(
            pd.Series(x_long),
            pd.Series(x_long),
            sampling_interval_days=1,
            domain_max_days=90,
        )

        # Long memory should give larger tau_max
        assert result_long["tau_max"] >= result_short["tau_max"]


class TestDifferencingWarning:
    """Test Issue 1 fix: differencing semantics warning."""

    def test_warning_emitted_on_differencing(self):
        """Preprocessing should warn when differencing changes causal semantics."""
        from framework.core.preprocessing import TimeSeriesPreprocessor

        np.random.seed(42)
        T = 200
        # Create non-stationary data (random walk)
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))
        df = pd.DataFrame(
            {"X": x, "Y": y}, index=pd.date_range("2020-01-01", periods=T, freq="D")
        )

        preprocessor = TimeSeriesPreprocessor(
            stationarity_test="adf",
            normalize=False,
            outlier_method="none",
        )
        result_df, report = preprocessor.preprocess(df, verbose=False)

        # Check that differencing was applied and warning is in quality flags
        has_diff_warning = any(
            "DIFFERENCING_SEMANTICS" in f for f in report.quality_flags
        )
        has_diff_transform = any(
            report.transformations.get(col, {}).get("differenced", 0) > 0
            for col in ["X", "Y"]
        )

        # At least one variable should be differenced (random walk is non-stationary)
        assert has_diff_transform, "Random walk should trigger differencing"
        assert has_diff_warning, "Differencing should emit semantics warning"
