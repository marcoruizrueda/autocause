"""Minimum Detectable Effect Size (MDES) and power analysis for causal discovery.

When a method finds no significant edges, this module computes what effect
size the method could have detected given the available data. This provides
actionable information: "We found nothing, but effects above X would have
been detected with 80% power."

This addresses a fundamental trustworthiness gap: tigramite reports nothing
when PCMCI+ finds no edges, leaving users uncertain whether the absence
reflects true independence or insufficient statistical power.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PowerReport:
    """Power analysis results for a causal discovery run."""

    method: str
    t_effective: int
    n_vars: int
    tau_max: int
    alpha: float
    power: float
    mdes_parcorr: float  # Minimum detectable partial correlation
    mdes_description: str
    sufficient_power: bool  # Whether T is enough for moderate effects

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "t_effective": self.t_effective,
            "n_vars": self.n_vars,
            "tau_max": self.tau_max,
            "alpha": self.alpha,
            "power": self.power,
            "mdes_parcorr": round(self.mdes_parcorr, 4),
            "mdes_description": self.mdes_description,
            "sufficient_power": self.sufficient_power,
        }


def compute_mdes_parcorr(
    t_effective: int,
    n_vars: int,
    tau_max: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Compute minimum detectable partial correlation for ParCorr-based PCMCI+.

    Uses the approximation from Cohen (1988) for partial correlation tests:
        r_min = sqrt(F_crit / (F_crit + df_residual))
    where df_residual = T_eff - p - 1, p = conditioning set size.

    For PCMCI+, the effective conditioning set after PC-stable pruning is
    typically ~min(N, 4) * min(tau_max, 3) variables.

    Parameters
    ----------
    t_effective : int
        Effective sample size after accounting for lags and missing values.
    n_vars : int
        Number of variables.
    tau_max : int
        Maximum lag.
    alpha : float
        Significance level (two-sided).
    power : float
        Desired statistical power (probability of detecting a true effect).

    Returns
    -------
    float
        Minimum detectable absolute partial correlation (0 to 1).
    """
    from scipy import stats as sp_stats

    # Effective conditioning set size (after PC-stable pruning)
    p = min(n_vars, 4) * min(tau_max, 3)

    # Degrees of freedom for the partial correlation test
    df = max(1, t_effective - p - 2)

    # Critical t-value for two-sided test at alpha
    t_crit = sp_stats.t.ppf(1 - alpha / 2, df)

    # For power = 0.80, we need the non-centrality parameter such that
    # P(|T| > t_crit | ncp) = power
    # Using the approximation: ncp ≈ t_crit + z_power * sqrt(1 + t_crit^2 / (2*df))
    z_power = sp_stats.norm.ppf(power)

    # Non-centrality parameter needed for desired power
    ncp = t_crit + z_power

    # Convert ncp to partial correlation: r = ncp / sqrt(ncp^2 + df)
    r_min = ncp / np.sqrt(ncp**2 + df)

    return float(min(r_min, 1.0))


def compute_mdes_cmiknn(
    t_effective: int,
    n_vars: int,
    tau_max: int,
    k: int = 10,
) -> float:
    """Estimate minimum detectable CMI for k-NN based estimator.

    CMIknn's detection threshold depends on the bias-variance tradeoff
    of the k-NN MI estimator. The variance scales as ~1/(k*T) and the
    bias scales with the conditioning dimension d.

    Returns an approximate minimum detectable CMI in nats.
    """
    d = min(n_vars, 4) * min(tau_max, 2)

    # Variance of the k-NN CMI estimator (Frenzel & Pompe 2007 approximation)
    # Var(CMI) ≈ (1/k + 1/T) * (1 + d/4)
    var_cmi = (1.0 / k + 1.0 / t_effective) * (1 + d / 4.0)

    # For significance at alpha=0.05 with surrogate testing (~200 surrogates),
    # need CMI > ~2 * std(CMI) above the null
    mdes = 2.0 * np.sqrt(var_cmi)

    return float(mdes)


def analyze_power(
    method: str,
    t_effective: int,
    n_vars: int,
    tau_max: int,
    alpha: float = 0.05,
    power: float = 0.80,
    n_significant_edges: int = 0,
) -> PowerReport:
    """Run power analysis for a completed causal discovery run.

    Call this after a method returns results. If no edges were found,
    the MDES tells the user what could have been detected.

    Parameters
    ----------
    method : str
        Method name (pcmci, granger, etc.)
    t_effective : int
        Effective sample size used.
    n_vars : int
        Number of variables.
    tau_max : int
        Maximum lag used.
    alpha : float
        Significance level used.
    power : float
        Target power for MDES computation.
    n_significant_edges : int
        Number of significant edges found by the method.

    Returns
    -------
    PowerReport
        Power analysis with MDES and interpretation.
    """
    try:
        mdes = compute_mdes_parcorr(t_effective, n_vars, tau_max, alpha, power)
    except Exception:
        mdes = 1.0  # Fallback: cannot compute

    # Interpret the MDES
    if mdes < 0.10:
        desc = (
            f"High power: can detect weak effects (|r| >= {mdes:.3f}). "
            f"Absence of edges likely reflects true independence."
        )
        sufficient = True
    elif mdes < 0.20:
        desc = (
            f"Moderate power: can detect medium effects (|r| >= {mdes:.3f}). "
            f"Weak effects (|r| < {mdes:.2f}) may be missed."
        )
        sufficient = True
    elif mdes < 0.40:
        desc = (
            f"Low power: only strong effects detectable (|r| >= {mdes:.3f}). "
            f"Consider increasing T or reducing tau_max/N."
        )
        sufficient = False
    else:
        desc = (
            f"Insufficient power: MDES={mdes:.3f} is very high. "
            f"Only very strong effects would be detected. "
            f"Results are unreliable; increase sample size."
        )
        sufficient = False

    if n_significant_edges > 0:
        desc = (
            f"Found {n_significant_edges} significant edge(s). "
            f"MDES={mdes:.3f} (effects below this may still exist but are undetected)."
        )

    return PowerReport(
        method=method,
        t_effective=t_effective,
        n_vars=n_vars,
        tau_max=tau_max,
        alpha=alpha,
        power=power,
        mdes_parcorr=mdes,
        mdes_description=desc,
        sufficient_power=sufficient,
    )
