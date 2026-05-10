"""
Scientific validation of AutoCause causal discovery methods.

Tests construct stationary VAR(1) processes with known causal graphs
and verify that each method's output is consistent with ground truth.

Methods tested:
- VAR-based Granger causality (bivariate, regression-based)
- Transfer entropy via CMIknn (nonparametric, information-theoretic)
- PCMCI+ with ParCorr (constraint-based, momentary conditional independence)
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path


def make_cause(n=500, lag=2, strength=0.6, seed=42):
    """Stationary X causes Y at a known lag."""
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.normal(0, 1)
    for t in range(lag, n):
        y[t] = strength * x[t - lag] + 0.3 * y[t - 1] + rng.normal(0, 0.5)
    return pd.DataFrame(
        {"X": x, "Y": y}, index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


def make_strong_cause(n=800, lag=2, seed=42):
    """Strong causal signal for PCMCI+ (needs more power)."""
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(0, 1)
    for t in range(lag, n):
        y[t] = 0.9 * x[t - lag] + 0.1 * y[t - 1] + rng.normal(0, 0.2)
    return pd.DataFrame(
        {"X": x, "Y": y}, index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


def make_independent(n=500, seed=42):
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.5 * x[t - 1] + rng.normal(0, 1)
        y[t] = 0.5 * y[t - 1] + rng.normal(0, 1)
    return pd.DataFrame(
        {"X": x, "Y": y}, index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


def make_chain(n=800, seed=42):
    """A → B → C chain (no direct A → C)."""
    rng = np.random.default_rng(seed)
    a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(2, n):
        a[t] = 0.3 * a[t - 1] + rng.normal(0, 1)
        b[t] = 0.8 * a[t - 1] + 0.1 * b[t - 1] + rng.normal(0, 0.3)
        c[t] = 0.8 * b[t - 1] + 0.1 * c[t - 1] + rng.normal(0, 0.3)
    return pd.DataFrame(
        {"A": a, "B": b, "C": c}, index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


class TestGranger:
    def test_detects_true_cause(self):
        from framework.core.methods import granger

        r = granger.batch_granger_causality(
            make_cause(), [("X", "Y")], maxlag=5, alpha=0.05
        )
        assert r is not None and r["significant"].any()

    def test_no_false_positive(self):
        from framework.core.methods import granger

        r = granger.batch_granger_causality(
            make_independent(), [("X", "Y")], maxlag=5, alpha=0.05
        )
        if r is not None and len(r) > 0:
            assert not r["significant"].any()

    def test_lag_recovery(self):
        from framework.core.methods import granger

        r = granger.batch_granger_causality(
            make_cause(lag=3, strength=0.7), [("X", "Y")], maxlag=8
        )
        best = r.loc[r["best_p_value"].idxmin()]
        assert abs(best["best_lag"] - 3) <= 1

    def test_columns(self):
        from framework.core.methods import granger

        r = granger.batch_granger_causality(make_cause(n=300), [("X", "Y")], maxlag=5)
        for c in ["cause", "effect", "best_lag", "best_p_value", "significant"]:
            assert c in r.columns


class TestTransferEntropy:
    def test_detects_true_cause(self):
        from framework.core.methods.transfer_entropy import batch_transfer_entropy

        r = batch_transfer_entropy(
            make_cause(), [("X", "Y")], delays=[1, 2, 3], alpha=0.05
        )
        assert r is not None and r["significant"].any()

    def test_columns(self):
        from framework.core.methods.transfer_entropy import batch_transfer_entropy

        r = batch_transfer_entropy(make_cause(n=300), [("X", "Y")], delays=[1, 2, 3])
        for c in ["source", "target", "delay", "p_value", "significant"]:
            assert c in r.columns


class TestPCMCI:
    def test_detects_link_bivariate(self):
        """PCMCI+ should detect a link between X and Y (direction may flip in bivariate case)."""
        from framework.core.methods.tigramite_pcmci import batch_pcmci

        r = batch_pcmci(
            make_strong_cause(), [("X", "Y"), ("Y", "X")], tau_max=5, alpha=0.05
        )
        assert r["is_significant"].any(), "PCMCI+ should detect a link between X and Y"

    def test_chain_screens_off(self):
        """A→B→C: conditioning on B should screen off A from C."""
        from framework.core.methods.tigramite_pcmci import batch_pcmci

        r = batch_pcmci(
            make_chain(),
            [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"), ("A", "C"), ("C", "A")],
            tau_max=3,
            alpha=0.05,
        )
        sig = r[r["is_significant"]]
        ab = sig[
            ((sig["source"] == "A") & (sig["target"] == "B"))
            | ((sig["source"] == "B") & (sig["target"] == "A"))
        ]
        bc = sig[
            ((sig["source"] == "B") & (sig["target"] == "C"))
            | ((sig["source"] == "C") & (sig["target"] == "B"))
        ]
        ac = sig[
            ((sig["source"] == "A") & (sig["target"] == "C"))
            | ((sig["source"] == "C") & (sig["target"] == "A"))
        ]
        assert len(ab) > 0, "Should detect A↔B link"
        assert len(bc) > 0, "Should detect B↔C link"
        assert len(ac) == 0, "Should NOT detect A↔C (screened off by B)"


class TestTauMax:
    def test_positive(self):
        from framework.core.tau_max_estimation import estimate_tau_max_scientific

        df = make_cause()
        assert estimate_tau_max_scientific(df["X"], df["Y"], 1, 30)["tau_max"] > 0

    def test_bounded(self):
        from framework.core.tau_max_estimation import estimate_tau_max_scientific

        df = make_cause()
        assert estimate_tau_max_scientific(df["X"], df["Y"], 1, 10)["tau_max"] <= 10


class TestWorkflow:
    def test_granger_only(self):
        from framework.core.run_workflow import run_causal_discovery_workflow

        with tempfile.TemporaryDirectory() as d:
            run_causal_discovery_workflow(
                data_df=make_cause(n=300),
                output_dir=Path(d),
                target_var="Y",
                alpha=0.05,
                sampling_days=1,
                enable_preprocessing=False,
                enable_distribution_tests=False,
                enable_strength_analysis=False,
                enable_temporal_validation=False,
                enable_tracking=False,
                enable_consensus=False,
                method_config={
                    "granger": {"enabled": True},
                    "transfer_entropy": {"enabled": False},
                    "pcmci": {"enabled": False},
                },
            )
            assert (
                Path(d) / "method" / "granger" / "1-raw" / "results_granger.csv"
            ).exists()


def make_nongaussian_cause(n=500, lag=1, strength=0.7, seed=42):
    """Stationary X causes Y with non-Gaussian (t-distributed) noise.
    VAR-LiNGAM requires non-Gaussianity for identifiability."""
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.standard_t(5)
    for t in range(lag, n):
        y[t] = strength * x[t - lag] + 0.2 * y[t - 1] + rng.standard_t(5) * 0.3
    return pd.DataFrame(
        {"X": x, "Y": y}, index=pd.date_range("2020-01-01", periods=n, freq="D")
    )


class TestVARLiNGAM:
    def test_detects_true_cause(self):
        """VAR-LiNGAM should detect X→Y with non-Gaussian noise."""
        from framework.core.methods.varlingam import batch_varlingam

        r = batch_varlingam(make_nongaussian_cause(), [("X", "Y")], lags=3)
        assert r is not None and len(r) > 0
        sig = r[r["is_significant"]]
        assert len(sig) > 0, "VAR-LiNGAM should detect X→Y"

    def test_coefficient_sign(self):
        """Recovered coefficient should be positive (true strength=0.7)."""
        from framework.core.methods.varlingam import batch_varlingam

        r = batch_varlingam(make_nongaussian_cause(), [("X", "Y")], lags=3)
        sig = r[r["is_significant"] & (r["source"] == "X") & (r["target"] == "Y")]
        if len(sig) > 0:
            best = sig.loc[sig["abs_coefficient"].idxmax()]
            assert best["coefficient"] > 0, (
                f"Expected positive, got {best['coefficient']}"
            )

    def test_output_columns(self):
        from framework.core.methods.varlingam import batch_varlingam

        r = batch_varlingam(make_nongaussian_cause(n=300), [("X", "Y")], lags=2)
        for c in ["source", "target", "lag", "coefficient", "is_significant", "method"]:
            assert c in r.columns, f"Missing: {c}"
        assert (r["method"] == "VAR-LiNGAM").all()
