"""Sample size adequacy diagnostics for causal discovery methods.

Provides pre-discovery checks that assess whether the available data
(effective sample size after accounting for missing values, tau_max,
and conditioning set size) is sufficient for each causal discovery method.

This addresses a recurrent gap in tigramite (GitHub issue #482, #32):
users with short time series (T=60-300) receive no guidance on whether
PCMCI results are statistically reliable. AutoCause fills this gap by
computing method-specific adequacy scores and emitting actionable warnings.

Theory:
    - ParCorr requires T_eff > 3*(p+1) per conditioning dimension, where
      p = max conditioning set size (typically N*tau_max in worst case,
      but PCMCI's PC-stable phase prunes this).
    - CMIknn requires T_eff > k * 2^d for reliable k-NN MI estimation,
      where d = conditioning set dimension and k = n_neighbors (default 10).
    - GPDC requires T_eff > 200 for GP regression to converge reliably.
    - VARLiNGAM requires T_eff > N * (tau_max + 1) * 5 for stable ICA.
    - Granger (VAR F-test) requires T_eff > N * tau_max * 3.

References:
    - Runge et al. (2019). Detecting and quantifying causal associations
      in large nonlinear time series datasets. Science Advances.
    - Frenzel & Pompe (2007). Partial mutual information for coupling
      analysis of multivariate time series. PRL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MethodAdequacy:
    """Adequacy assessment for a single method."""

    method: str
    adequate: bool
    score: float  # 0.0 (inadequate) to 1.0 (fully adequate)
    t_effective: int
    t_required: int
    reason: str
    recommendation: str = ""


@dataclass
class SampleSizeReport:
    """Complete sample size adequacy report for a dataset."""

    n_variables: int
    n_observations: int
    t_effective: int
    tau_max: int
    missing_fraction: float
    method_assessments: Dict[str, MethodAdequacy] = field(default_factory=dict)
    recommended_methods: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    overall_adequate: bool = True

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON output."""
        return {
            "n_variables": self.n_variables,
            "n_observations": self.n_observations,
            "t_effective": self.t_effective,
            "tau_max": self.tau_max,
            "missing_fraction": round(self.missing_fraction, 4),
            "overall_adequate": self.overall_adequate,
            "recommended_methods": self.recommended_methods,
            "warnings": self.warnings,
            "method_assessments": {
                k: {
                    "adequate": v.adequate,
                    "score": round(v.score, 3),
                    "t_effective": v.t_effective,
                    "t_required": v.t_required,
                    "reason": v.reason,
                    "recommendation": v.recommendation,
                }
                for k, v in self.method_assessments.items()
            },
        }


def compute_effective_sample_size(
    df: pd.DataFrame,
    tau_max: int,
) -> int:
    """Compute effective sample size accounting for lags and missing values.

    The effective T for PCMCI is T - tau_max (observations lost to lagging).
    Additionally, rows with any NaN reduce the usable sample further because
    tigramite masks entire time slices where missing values occur.

    Parameters
    ----------
    df : pd.DataFrame
        Multivariate time series (numeric columns only).
    tau_max : int
        Maximum lag used in causal discovery.

    Returns
    -------
    int
        Effective sample size available for conditional independence testing.
    """
    numeric_df = df.select_dtypes(include=["number"])

    # Rows where ALL variables are non-NaN (tigramite's default masking)
    complete_rows = numeric_df.notna().all(axis=1).sum()

    # Subtract tau_max (observations consumed by lagging)
    t_eff = max(0, complete_rows - tau_max)

    return int(t_eff)


def _parcorr_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for ParCorr to be reliable.

    ParCorr estimates partial correlations via OLS regression. The conditioning
    set in PCMCI grows up to ~N*tau_max in the worst case, but PC-stable
    pruning keeps it manageable. Conservative estimate: need at least
    3 observations per parameter in the largest regression.

    Practical minimum: max(50, 3 * N * min(tau_max, 3) + 10).
    The min(tau_max, 3) reflects that PC-stable rarely conditions on more
    than 3 lags per variable after pruning.
    """
    max_cond_dim = n_vars * min(tau_max, 3)
    return max(50, 3 * max_cond_dim + 10)


def _robust_parcorr_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for RobustParCorr.

    RobustParCorr applies rank-based normal score transformation before
    OLS. The transformation itself requires sufficient samples for stable
    rank estimation. Slightly higher requirement than ParCorr.
    """
    return int(_parcorr_minimum(n_vars, tau_max) * 1.2)


def _cmiknn_minimum(n_vars: int, tau_max: int, k: int = 10) -> int:
    """Minimum T for CMIknn to produce reliable estimates.

    CMIknn estimates conditional mutual information via k-NN distances.
    The estimator's bias and variance depend on T relative to the
    conditioning set dimension d. Rule of thumb from Frenzel & Pompe (2007):
    need T >> k * 2^(d/2) for the k-NN estimator to converge.

    Conservative: max(200, k * ceil(2^(d/2))) where d = typical conditioning
    set size after PC-stable pruning (~min(N, 4) variables * 1 lag).
    """
    d = min(n_vars, 4) * min(tau_max, 2)
    t_min = max(200, int(k * np.ceil(2 ** (d / 2))))
    # Cap at a reasonable maximum (beyond 2000, CMIknn is always fine)
    return min(t_min, 2000)


def _gpdc_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for GPDC (Gaussian Process Distance Correlation).

    GPDC fits a GP regression for each conditional independence test.
    GP regression requires sufficient data for kernel hyperparameter
    optimization. Empirically, T < 200 leads to unreliable GP fits,
    and T < 500 shows high variance in the distance correlation statistic.
    """
    return max(500, 10 * n_vars * min(tau_max, 3))


def _granger_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for VAR-based Granger causality.

    VAR(p) with N variables has N*p + 1 parameters per equation.
    Need at least 3 observations per parameter for stable F-test.
    """
    n_params = n_vars * tau_max + 1
    return max(30, 3 * n_params + 10)


def _varlingam_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for VARLiNGAM.

    VARLiNGAM estimates a VAR model then applies ICA to residuals.
    ICA requires sufficient samples for stable independent component
    estimation. Rule of thumb: T > 5 * N * (tau_max + 1).
    """
    return max(100, 5 * n_vars * (tau_max + 1))


def _transfer_entropy_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for transfer entropy (CMIknn-based surrogates).

    Transfer entropy uses CMIknn internally plus surrogate testing
    (typically 100-200 surrogates). Each surrogate evaluation needs
    the same T, so the data requirement matches CMIknn.
    """
    return _cmiknn_minimum(n_vars, tau_max, k=10)


def _lpcmci_minimum(n_vars: int, tau_max: int) -> int:
    """Minimum T for LPCMCI (latent-variable PCMCI).

    LPCMCI performs more conditional independence tests than PCMCI+
    due to the FCI-style orientation rules. Requires ~50% more data
    than standard PCMCI+ with the same CI test.
    """
    return int(_parcorr_minimum(n_vars, tau_max) * 1.5)


# Registry of method-specific minimum functions
_METHOD_MINIMUMS = {
    "parcorr": _parcorr_minimum,
    "robust_parcorr": _robust_parcorr_minimum,
    "cmiknn": _cmiknn_minimum,
    "gpdc": _gpdc_minimum,
    "granger": _granger_minimum,
    "varlingam": _varlingam_minimum,
    "transfer_entropy": _transfer_entropy_minimum,
    "lpcmci": _lpcmci_minimum,
}


def assess_method_adequacy(
    method: str,
    t_effective: int,
    n_vars: int,
    tau_max: int,
) -> MethodAdequacy:
    """Assess whether effective sample size is adequate for a given method.

    Parameters
    ----------
    method : str
        Method identifier (parcorr, cmiknn, gpdc, granger, varlingam, etc.)
    t_effective : int
        Effective sample size (after accounting for lags and missing values).
    n_vars : int
        Number of variables in the dataset.
    tau_max : int
        Maximum lag.

    Returns
    -------
    MethodAdequacy
        Assessment with adequacy flag, score, and recommendation.
    """
    method_lower = method.lower().replace("-", "_").replace(" ", "_")

    # Map PCMCI variants to their CI test
    if method_lower in ("pcmci", "pcmci+", "pcmciplus"):
        method_lower = "parcorr"  # default CI test

    min_func = _METHOD_MINIMUMS.get(method_lower)
    if min_func is None:
        return MethodAdequacy(
            method=method,
            adequate=True,
            score=1.0,
            t_effective=t_effective,
            t_required=0,
            reason="No minimum requirement defined for this method.",
        )

    t_required = min_func(n_vars, tau_max)

    # Score: linear ramp from 0 at t_required/2 to 1.0 at t_required
    if t_effective >= t_required:
        score = 1.0
        adequate = True
        reason = (
            f"T_eff={t_effective} >= T_min={t_required} (N={n_vars}, tau_max={tau_max})"
        )
        recommendation = ""
    elif t_effective >= t_required * 0.7:
        score = t_effective / t_required
        adequate = True  # marginal but acceptable
        reason = (
            f"T_eff={t_effective} is marginal (70-100% of T_min={t_required}). "
            f"Results may have elevated false positive/negative rates."
        )
        recommendation = "Consider reducing tau_max or number of variables if possible."
    else:
        score = max(0.0, t_effective / t_required)
        adequate = False
        reason = (
            f"T_eff={t_effective} < 70% of T_min={t_required} "
            f"(N={n_vars}, tau_max={tau_max}). "
            f"Results are unreliable for {method}."
        )
        if method_lower in ("cmiknn", "gpdc"):
            recommendation = (
                f"Switch to ParCorr (requires T_min={_parcorr_minimum(n_vars, tau_max)}) "
                f"or reduce tau_max."
            )
        elif method_lower == "lpcmci":
            recommendation = (
                f"Use standard PCMCI+ instead "
                f"(requires T_min={_parcorr_minimum(n_vars, tau_max)})."
            )
        else:
            recommendation = (
                f"Reduce tau_max to {max(1, int(tau_max * score))} "
                f"or acquire more data."
            )

    return MethodAdequacy(
        method=method,
        adequate=adequate,
        score=score,
        t_effective=t_effective,
        t_required=t_required,
        reason=reason,
        recommendation=recommendation,
    )


def assess_sample_size(
    df: pd.DataFrame,
    tau_max: int,
    methods: Optional[List[str]] = None,
) -> SampleSizeReport:
    """Run full sample size adequacy assessment for a dataset.

    This is the main entry point. Call before running causal discovery
    to get warnings and method recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Multivariate time series (numeric columns).
    tau_max : int
        Maximum lag to be used in discovery.
    methods : list of str, optional
        Methods to assess. If None, assesses all registered methods.

    Returns
    -------
    SampleSizeReport
        Complete adequacy report with per-method assessments.
    """
    numeric_df = df.select_dtypes(include=["number"])
    n_vars = len(numeric_df.columns)
    n_obs = len(numeric_df)

    # Missing fraction
    total_cells = n_obs * n_vars
    missing_cells = numeric_df.isna().sum().sum()
    missing_fraction = missing_cells / total_cells if total_cells > 0 else 0.0

    # Effective sample size
    t_eff = compute_effective_sample_size(df, tau_max)

    if methods is None:
        methods = list(_METHOD_MINIMUMS.keys())

    # Assess each method
    assessments: Dict[str, MethodAdequacy] = {}
    for method in methods:
        assessment = assess_method_adequacy(method, t_eff, n_vars, tau_max)
        assessments[method] = assessment

    # Determine recommended methods (adequate ones, sorted by score)
    recommended = [m for m, a in assessments.items() if a.adequate]
    # Sort by score descending (prefer methods with higher adequacy margin)
    recommended.sort(key=lambda m: assessments[m].score, reverse=True)

    # Generate warnings
    warnings = []
    inadequate = [m for m, a in assessments.items() if not a.adequate]

    if t_eff < 30:
        warnings.append(
            f"CRITICAL: T_eff={t_eff} is below absolute minimum (30) for any "
            f"causal discovery method. Results will be unreliable."
        )
    elif t_eff < 50:
        warnings.append(
            f"WARNING: T_eff={t_eff} is very low. Only ParCorr with reduced "
            f"tau_max may produce meaningful results."
        )

    if missing_fraction > 0.3:
        warnings.append(
            f"WARNING: {missing_fraction:.0%} missing values. Effective sample "
            f"size is substantially reduced."
        )

    if inadequate:
        warnings.append(
            f"Methods with insufficient data: {', '.join(inadequate)}. "
            f"Consider using: {', '.join(recommended[:3])}."
        )

    if tau_max > t_eff * 0.25:
        warnings.append(
            f"WARNING: tau_max={tau_max} consumes >{25}% of effective observations. "
            f"Consider reducing tau_max to {max(1, int(t_eff * 0.15))}."
        )

    overall_adequate = len(recommended) > 0

    report = SampleSizeReport(
        n_variables=n_vars,
        n_observations=n_obs,
        t_effective=t_eff,
        tau_max=tau_max,
        missing_fraction=missing_fraction,
        method_assessments=assessments,
        recommended_methods=recommended,
        warnings=warnings,
        overall_adequate=overall_adequate,
    )

    # Log the report
    if warnings:
        for w in warnings:
            logger.warning(w)
    else:
        logger.info(
            f"Sample size adequate: T_eff={t_eff}, N={n_vars}, tau_max={tau_max}. "
            f"All {len(recommended)} methods viable."
        )

    return report


def suggest_ci_test_for_sample_size(
    t_effective: int,
    n_vars: int,
    tau_max: int,
    is_nonlinear: bool = False,
) -> str:
    """Suggest the best CI test given sample size constraints.

    When data is nonlinear but T is too small for CMIknn, falls back
    to RobustParCorr (which handles non-Gaussian marginals via rank
    transformation but assumes monotonic relationships).

    Parameters
    ----------
    t_effective : int
        Effective sample size.
    n_vars : int
        Number of variables.
    tau_max : int
        Maximum lag.
    is_nonlinear : bool
        Whether nonlinearity was detected in the data.

    Returns
    -------
    str
        Recommended CI test name.
    """
    cmiknn_min = _cmiknn_minimum(n_vars, tau_max)
    parcorr_min = _parcorr_minimum(n_vars, tau_max)
    robust_min = _robust_parcorr_minimum(n_vars, tau_max)

    if is_nonlinear:
        if t_effective >= cmiknn_min:
            return "cmiknn"
        elif t_effective >= robust_min:
            logger.warning(
                f"Nonlinear data detected but T_eff={t_effective} < CMIknn minimum "
                f"({cmiknn_min}). Falling back to RobustParCorr."
            )
            return "robust_parcorr"
        else:
            logger.warning(
                f"Nonlinear data with very short series (T_eff={t_effective}). "
                f"Using ParCorr as last resort; results may miss nonlinear effects."
            )
            return "parcorr"
    else:
        if t_effective >= robust_min:
            return "robust_parcorr"
        elif t_effective >= parcorr_min:
            return "parcorr"
        else:
            logger.warning(
                f"T_eff={t_effective} is below ParCorr minimum ({parcorr_min}). "
                f"Consider reducing tau_max."
            )
            return "parcorr"
