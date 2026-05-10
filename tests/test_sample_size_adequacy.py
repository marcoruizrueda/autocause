"""Tests for sample_size_adequacy module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.core.sample_size_adequacy import (
    assess_method_adequacy,
    assess_sample_size,
    compute_effective_sample_size,
    suggest_ci_test_for_sample_size,
    _parcorr_minimum,
    _cmiknn_minimum,
    _gpdc_minimum,
    _granger_minimum,
    _varlingam_minimum,
    _robust_parcorr_minimum,
)


def test_effective_sample_size_no_missing():
    """T_eff = T - tau_max when no missing values."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((500, 4)),
        columns=["X1", "X2", "X3", "X4"],
    )
    t_eff = compute_effective_sample_size(df, tau_max=10)
    assert t_eff == 490, f"Expected 490, got {t_eff}"
    print(f"  PASS: T_eff={t_eff} (T=500, tau_max=10, no missing)")


def test_effective_sample_size_with_missing():
    """T_eff accounts for rows with NaN."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((200, 3))
    # Introduce 20% missing in first column
    mask = rng.random(200) < 0.2
    data[mask, 0] = np.nan
    df = pd.DataFrame(data, columns=["A", "B", "C"])

    t_eff = compute_effective_sample_size(df, tau_max=5)
    complete_rows = df.notna().all(axis=1).sum()
    expected = complete_rows - 5
    assert t_eff == expected, f"Expected {expected}, got {t_eff}"
    print(f"  PASS: T_eff={t_eff} (T=200, ~20% missing, tau_max=5)")


def test_effective_sample_size_all_missing():
    """Edge case: all rows have at least one NaN."""
    df = pd.DataFrame({"A": [np.nan] * 50, "B": np.arange(50, dtype=float)})
    t_eff = compute_effective_sample_size(df, tau_max=3)
    assert t_eff == 0, f"Expected 0, got {t_eff}"
    print(f"  PASS: T_eff={t_eff} (all rows have NaN)")


def test_minimum_requirements_ordering():
    """ParCorr < RobustParCorr < CMIknn < GPDC for same N, tau_max."""
    n_vars, tau_max = 5, 10
    parcorr = _parcorr_minimum(n_vars, tau_max)
    robust = _robust_parcorr_minimum(n_vars, tau_max)
    cmiknn = _cmiknn_minimum(n_vars, tau_max)
    gpdc = _gpdc_minimum(n_vars, tau_max)

    assert parcorr <= robust, f"ParCorr ({parcorr}) > RobustParCorr ({robust})"
    assert robust <= cmiknn, f"RobustParCorr ({robust}) > CMIknn ({cmiknn})"
    assert cmiknn <= gpdc, f"CMIknn ({cmiknn}) > GPDC ({gpdc})"
    print(
        f"  PASS: Ordering correct: ParCorr={parcorr} <= Robust={robust} "
        f"<= CMIknn={cmiknn} <= GPDC={gpdc}"
    )


def test_minimum_requirements_scale_with_vars():
    """More variables require more data."""
    tau_max = 5
    for method_func in [_parcorr_minimum, _granger_minimum, _varlingam_minimum]:
        t_4 = method_func(4, tau_max)
        t_8 = method_func(8, tau_max)
        assert t_8 >= t_4, f"{method_func.__name__}: N=8 ({t_8}) < N=4 ({t_4})"
    print("  PASS: All methods require more data with more variables")


def test_assess_method_adequate():
    """Large dataset is adequate for all methods."""
    assessment = assess_method_adequacy(
        "parcorr", t_effective=1000, n_vars=4, tau_max=5
    )
    assert assessment.adequate is True
    assert assessment.score == 1.0
    assert assessment.recommendation == ""
    print(f"  PASS: ParCorr adequate with T_eff=1000 (T_min={assessment.t_required})")


def test_assess_method_inadequate():
    """Tiny dataset is inadequate for CMIknn."""
    assessment = assess_method_adequacy("cmiknn", t_effective=50, n_vars=6, tau_max=10)
    assert assessment.adequate is False
    assert assessment.score < 0.7
    assert (
        "unreliable" in assessment.reason.lower()
        or "insufficient" in assessment.reason.lower()
        or "<" in assessment.reason
    )
    assert assessment.recommendation != ""
    print(
        f"  PASS: CMIknn inadequate with T_eff=50 "
        f"(T_min={assessment.t_required}, score={assessment.score:.2f})"
    )


def test_assess_method_marginal():
    """Marginal case: between 70% and 100% of minimum."""
    # Find a case where T_eff is ~80% of minimum
    t_min = _parcorr_minimum(4, 5)
    t_eff = int(t_min * 0.8)
    assessment = assess_method_adequacy(
        "parcorr", t_effective=t_eff, n_vars=4, tau_max=5
    )
    assert assessment.adequate is True  # marginal but acceptable
    assert 0.7 <= assessment.score < 1.0
    assert "marginal" in assessment.reason.lower()
    print(
        f"  PASS: ParCorr marginal with T_eff={t_eff} "
        f"(T_min={t_min}, score={assessment.score:.2f})"
    )


def test_full_report_adequate():
    """Full report on adequate dataset."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((1000, 4)),
        columns=["X1", "X2", "X3", "X4"],
    )
    report = assess_sample_size(df, tau_max=5)
    assert report.overall_adequate is True
    assert report.t_effective == 995
    assert report.n_variables == 4
    assert report.missing_fraction == 0.0
    assert len(report.recommended_methods) > 0
    assert len(report.warnings) == 0
    print(
        f"  PASS: Full report adequate. T_eff={report.t_effective}, "
        f"recommended={report.recommended_methods}"
    )


def test_full_report_short_series():
    """Full report on short series emits warnings."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((80, 6)),
        columns=[f"X{i}" for i in range(6)],
    )
    report = assess_sample_size(df, tau_max=10)
    assert report.t_effective == 70
    assert len(report.warnings) > 0
    # Some methods should be inadequate
    inadequate = [m for m, a in report.method_assessments.items() if not a.adequate]
    assert len(inadequate) > 0, "Expected some methods to be inadequate"
    print(
        f"  PASS: Short series report. T_eff={report.t_effective}, "
        f"inadequate={inadequate}, warnings={len(report.warnings)}"
    )


def test_full_report_serialization():
    """Report serializes to dict correctly."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((200, 3)), columns=["A", "B", "C"])
    report = assess_sample_size(df, tau_max=5)
    d = report.to_dict()
    assert "n_variables" in d
    assert "method_assessments" in d
    assert "warnings" in d
    assert isinstance(d["method_assessments"], dict)
    for method_name, method_dict in d["method_assessments"].items():
        assert "adequate" in method_dict
        assert "score" in method_dict
        assert "t_required" in method_dict
    print(f"  PASS: Serialization OK. Keys: {list(d.keys())}")


def test_suggest_ci_test_nonlinear_adequate():
    """Nonlinear data with enough samples gets CMIknn."""
    result = suggest_ci_test_for_sample_size(
        t_effective=1000, n_vars=4, tau_max=5, is_nonlinear=True
    )
    assert result == "cmiknn"
    print(f"  PASS: Nonlinear + adequate -> {result}")


def test_suggest_ci_test_nonlinear_short():
    """Nonlinear data with short series falls back to RobustParCorr."""
    result = suggest_ci_test_for_sample_size(
        t_effective=80, n_vars=4, tau_max=5, is_nonlinear=True
    )
    assert result in ("robust_parcorr", "parcorr")
    print(f"  PASS: Nonlinear + short -> {result}")


def test_suggest_ci_test_linear():
    """Linear data gets RobustParCorr or ParCorr."""
    result = suggest_ci_test_for_sample_size(
        t_effective=500, n_vars=4, tau_max=5, is_nonlinear=False
    )
    assert result in ("robust_parcorr", "parcorr")
    print(f"  PASS: Linear + adequate -> {result}")


def test_unknown_method():
    """Unknown method returns adequate by default."""
    assessment = assess_method_adequacy(
        "unknown_method", t_effective=10, n_vars=2, tau_max=3
    )
    assert assessment.adequate is True
    assert assessment.score == 1.0
    print("  PASS: Unknown method -> adequate (no requirement defined)")


def test_tau_max_zero():
    """tau_max=0 edge case."""
    t_eff = compute_effective_sample_size(
        pd.DataFrame(np.ones((100, 2)), columns=["A", "B"]), tau_max=0
    )
    assert t_eff == 100
    assessment = assess_method_adequacy("parcorr", t_effective=100, n_vars=2, tau_max=0)
    assert assessment.adequate is True
    print("  PASS: tau_max=0 -> T_eff=100, adequate")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing sample_size_adequacy module")
    print("=" * 60)

    tests = [
        test_effective_sample_size_no_missing,
        test_effective_sample_size_with_missing,
        test_effective_sample_size_all_missing,
        test_minimum_requirements_ordering,
        test_minimum_requirements_scale_with_vars,
        test_assess_method_adequate,
        test_assess_method_inadequate,
        test_assess_method_marginal,
        test_full_report_adequate,
        test_full_report_short_series,
        test_full_report_serialization,
        test_suggest_ci_test_nonlinear_adequate,
        test_suggest_ci_test_nonlinear_short,
        test_suggest_ci_test_linear,
        test_unknown_method,
        test_tau_max_zero,
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
