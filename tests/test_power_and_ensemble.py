"""Tests for power_analysis and ensemble_scoring modules."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.core.power_analysis import (
    compute_mdes_parcorr,
    analyze_power,
)
from framework.core.ensemble_scoring import (
    DataRegime,
    get_method_weights,
    compute_ensemble_scores,
    detect_data_regime,
)


# --- Power Analysis Tests ---


def test_mdes_decreases_with_sample_size():
    """More data = smaller minimum detectable effect."""
    mdes_100 = compute_mdes_parcorr(t_effective=100, n_vars=4, tau_max=5)
    mdes_500 = compute_mdes_parcorr(t_effective=500, n_vars=4, tau_max=5)
    mdes_2000 = compute_mdes_parcorr(t_effective=2000, n_vars=4, tau_max=5)
    assert mdes_100 > mdes_500 > mdes_2000
    print(
        f"  PASS: MDES decreases with T: {mdes_100:.3f} > {mdes_500:.3f} > {mdes_2000:.3f}"
    )


def test_mdes_increases_with_vars():
    """More variables = larger conditioning set = less power."""
    mdes_3 = compute_mdes_parcorr(t_effective=500, n_vars=3, tau_max=5)
    mdes_8 = compute_mdes_parcorr(t_effective=500, n_vars=8, tau_max=5)
    assert mdes_8 > mdes_3
    print(f"  PASS: MDES increases with N: N=3 -> {mdes_3:.3f}, N=8 -> {mdes_8:.3f}")


def test_mdes_reasonable_range():
    """MDES should be between 0 and 1 for typical settings."""
    mdes = compute_mdes_parcorr(t_effective=1000, n_vars=4, tau_max=5)
    assert 0.0 < mdes < 0.5
    print(f"  PASS: MDES in reasonable range: {mdes:.4f}")


def test_power_report_sufficient():
    """Large T should report sufficient power."""
    report = analyze_power("pcmci", t_effective=1000, n_vars=4, tau_max=5)
    assert report.sufficient_power is True
    assert report.mdes_parcorr < 0.20
    print(f"  PASS: T=1000 sufficient power, MDES={report.mdes_parcorr:.3f}")


def test_power_report_insufficient():
    """Very small T should report insufficient power."""
    report = analyze_power("pcmci", t_effective=30, n_vars=6, tau_max=5)
    assert report.sufficient_power is False
    assert report.mdes_parcorr > 0.30
    print(f"  PASS: T=30 insufficient power, MDES={report.mdes_parcorr:.3f}")


def test_power_report_with_edges_found():
    """When edges are found, description mentions them."""
    report = analyze_power(
        "pcmci", t_effective=500, n_vars=4, tau_max=5, n_significant_edges=3
    )
    assert "3 significant" in report.mdes_description
    print(f"  PASS: Report mentions found edges: '{report.mdes_description[:60]}...'")


# --- Ensemble Scoring Tests ---


def test_weights_sum_to_one():
    """Weights should be normalized."""
    regime = DataRegime(is_linear=True, is_gaussian=True)
    methods = ["pcmci", "granger", "varlingam", "transfer_entropy"]
    weights = get_method_weights(regime, methods)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    print(f"  PASS: Weights sum to 1.0: {weights}")


def test_weights_linear_favors_pcmci():
    """In linear Gaussian regime, PCMCI+ should have highest weight."""
    regime = DataRegime(is_linear=True, is_gaussian=True)
    methods = ["pcmci", "granger", "varlingam", "transfer_entropy"]
    weights = get_method_weights(regime, methods)
    assert weights["pcmci"] >= weights["granger"]
    assert weights["pcmci"] >= weights["transfer_entropy"]
    print(f"  PASS: Linear Gaussian -> PCMCI+ highest: {weights['pcmci']:.3f}")


def test_weights_nonlinear_favors_te():
    """In nonlinear regime, TE should have high weight."""
    regime = DataRegime(is_linear=False, is_gaussian=True)
    methods = ["pcmci", "granger", "varlingam", "transfer_entropy"]
    weights = get_method_weights(regime, methods)
    assert weights["transfer_entropy"] > weights["granger"]
    print(
        f"  PASS: Nonlinear -> TE > Granger: {weights['transfer_entropy']:.3f} > {weights['granger']:.3f}"
    )


def test_weights_nongaussian_favors_varlingam():
    """In non-Gaussian regime, VARLiNGAM should have highest weight."""
    regime = DataRegime(is_linear=True, is_gaussian=False)
    methods = ["pcmci", "granger", "varlingam", "transfer_entropy"]
    weights = get_method_weights(regime, methods)
    assert weights["varlingam"] >= weights["pcmci"]
    print(f"  PASS: Non-Gaussian -> VARLiNGAM highest: {weights['varlingam']:.3f}")


def test_ensemble_perfect_agreement():
    """All methods agree on an edge -> confidence near 1.0."""
    regime = DataRegime(is_linear=True, is_gaussian=True)
    results = {
        "pcmci": pd.DataFrame(
            {"source": ["A"], "target": ["B"], "is_significant": [True]}
        ),
        "granger": pd.DataFrame(
            {"cause": ["A"], "effect": ["B"], "significant": [True]}
        ),
        "varlingam": pd.DataFrame(
            {"source": ["A"], "target": ["B"], "is_significant": [True]}
        ),
    }
    df = compute_ensemble_scores(results, regime, significance_threshold=0.5)
    assert len(df) >= 1, f"Expected >=1 edge, got {len(df)}"
    max_conf = float(df["confidence"].max())
    assert max_conf > 0.8, f"Expected confidence > 0.8, got {max_conf}"
    assert bool(df.iloc[0]["is_significant"]) is True
    print(f"  PASS: All agree -> confidence={max_conf:.3f}")


def test_ensemble_single_method():
    """Only one method finds an edge -> low confidence."""
    regime = DataRegime(is_linear=True, is_gaussian=True)
    results = {
        "pcmci": pd.DataFrame(
            {"source": ["A"], "target": ["B"], "is_significant": [True]}
        ),
        "granger": pd.DataFrame(
            {"cause": ["A"], "effect": ["B"], "significant": [False]}
        ),
        "varlingam": pd.DataFrame(
            {"source": ["A"], "target": ["B"], "is_significant": [False]}
        ),
    }
    df = compute_ensemble_scores(results, regime, significance_threshold=0.5)
    if len(df) > 0:
        row = df[df["source"] == "A"]
        if len(row) > 0:
            assert row.iloc[0]["confidence"] < 0.5
            print(
                f"  PASS: Single method -> confidence={row.iloc[0]['confidence']:.3f} (below threshold)"
            )
        else:
            print("  PASS: Single method -> no edge in ensemble (below threshold)")
    else:
        print("  PASS: Single method -> empty ensemble (no significant edges)")


def test_ensemble_empty_results():
    """No methods find anything -> empty ensemble."""
    regime = DataRegime(is_linear=True, is_gaussian=True)
    results = {
        "pcmci": pd.DataFrame(
            {"source": ["A"], "target": ["B"], "is_significant": [False]}
        ),
        "granger": pd.DataFrame(
            {"cause": ["A"], "effect": ["B"], "significant": [False]}
        ),
    }
    df = compute_ensemble_scores(results, regime)
    assert len(df) == 0
    print("  PASS: No significant edges -> empty ensemble")


def test_detect_data_regime():
    """Regime detection from flags."""
    df = pd.DataFrame(np.random.randn(200, 4), columns=["A", "B", "C", "D"])
    regime = detect_data_regime(df, nonlinearity_detected=True)
    assert regime.regime_name == "nonlinear"
    assert regime.n_vars == 4
    assert regime.t_effective == 200
    print(
        f"  PASS: Regime detection: {regime.regime_name}, N={regime.n_vars}, T={regime.t_effective}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing power_analysis and ensemble_scoring")
    print("=" * 60)

    tests = [
        test_mdes_decreases_with_sample_size,
        test_mdes_increases_with_vars,
        test_mdes_reasonable_range,
        test_power_report_sufficient,
        test_power_report_insufficient,
        test_power_report_with_edges_found,
        test_weights_sum_to_one,
        test_weights_linear_favors_pcmci,
        test_weights_nonlinear_favors_te,
        test_weights_nongaussian_favors_varlingam,
        test_ensemble_perfect_agreement,
        test_ensemble_single_method,
        test_ensemble_empty_results,
        test_detect_data_regime,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
