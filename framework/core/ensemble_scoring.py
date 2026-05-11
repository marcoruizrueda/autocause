"""Confidence-weighted ensemble scoring for multi-method causal discovery.

Aggregates edges discovered by multiple methods into a single confidence
score per edge, weighted by each method's expected reliability given the
detected data properties (linearity, Gaussianity, sample size).

This is the key differentiator over running methods independently:
an edge found by 3/5 methods with high weights is more trustworthy than
an edge found by 1 method, regardless of its p-value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataRegime:
    """Detected data properties that determine method weights."""

    is_linear: bool = True
    is_gaussian: bool = True
    has_latent_confounders: bool = False
    t_effective: int = 500
    n_vars: int = 4

    @property
    def regime_name(self) -> str:
        if not self.is_linear:
            return "nonlinear"
        elif not self.is_gaussian:
            return "linear_nongaussian"
        elif self.has_latent_confounders:
            return "linear_confounded"
        else:
            return "linear_gaussian"


# Method weights per data regime.
# Higher weight = more reliable for that regime.
# Weights are normalized to sum to 1 within each regime.
_RAW_WEIGHTS = {
    "linear_gaussian": {
        "pcmci": 1.0,
        "lpcmci": 0.8,
        "granger": 0.9,
        "varlingam": 0.7,
        "transfer_entropy": 0.4,
        "predictive_baseline": 0.2,
        "correlation": 0.1,
    },
    "linear_nongaussian": {
        "pcmci": 0.8,
        "lpcmci": 0.7,
        "granger": 0.7,
        "varlingam": 1.0,  # VARLiNGAM excels with non-Gaussian (ICA identifiability)
        "transfer_entropy": 0.5,
        "predictive_baseline": 0.2,
        "correlation": 0.1,
    },
    "nonlinear": {
        "pcmci": 0.6,  # CMIknn selected but may lack power
        "lpcmci": 0.5,
        "granger": 0.2,  # Linear method on nonlinear data
        "varlingam": 0.7,  # Picks up linear shadow of nonlinear effects
        "transfer_entropy": 0.9,  # Information-theoretic, handles nonlinearity
        "predictive_baseline": 0.4,
        "correlation": 0.1,
    },
    "linear_confounded": {
        "pcmci": 0.5,
        "lpcmci": 1.0,  # Designed for latent confounders
        "granger": 0.4,
        "varlingam": 0.3,
        "transfer_entropy": 0.5,
        "predictive_baseline": 0.2,
        "correlation": 0.1,
    },
}


def get_method_weights(regime: DataRegime, methods: List[str]) -> Dict[str, float]:
    """Get normalized weights for available methods given the data regime.

    Parameters
    ----------
    regime : DataRegime
        Detected data properties.
    methods : list of str
        Methods that were actually run.

    Returns
    -------
    dict
        Method name -> normalized weight (sums to 1).
    """
    raw = _RAW_WEIGHTS.get(regime.regime_name, _RAW_WEIGHTS["linear_gaussian"])

    # Filter to available methods and normalize
    available = {m: raw.get(m, 0.3) for m in methods if m in raw}
    if not available:
        # Fallback: equal weights
        return {m: 1.0 / len(methods) for m in methods}

    total = sum(available.values())
    if total == 0:
        return {m: 1.0 / len(available) for m in available}

    return {m: w / total for m, w in available.items()}


@dataclass
class EnsembleEdge:
    """A single edge with ensemble confidence score."""

    source: str
    target: str
    confidence: float  # 0 to 1
    n_methods_found: int
    n_methods_total: int
    contributing_methods: List[str] = field(default_factory=list)
    weighted_score: float = 0.0
    is_significant: bool = False


def compute_ensemble_scores(
    results_dict: Dict[str, pd.DataFrame],
    regime: DataRegime,
    significance_threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute confidence-weighted ensemble scores for all edges.

    For each (source, target) pair found by any method, computes:
        confidence = Σ_m (w_m × I[method_m found this edge significant])

    An edge is "ensemble-significant" if confidence >= significance_threshold.

    Parameters
    ----------
    results_dict : dict
        Method name -> DataFrame with columns including source, target, is_significant.
    regime : DataRegime
        Detected data properties for weight selection.
    significance_threshold : float
        Minimum confidence for an edge to be considered significant (default 0.5).

    Returns
    -------
    pd.DataFrame
        One row per unique edge with confidence scores and contributing methods.
    """
    from framework.core.graph_metrics import METHOD_COLUMNS

    methods_run = [
        m
        for m in results_dict
        if results_dict[m] is not None
        and isinstance(results_dict[m], pd.DataFrame)
        and len(results_dict[m]) > 0
    ]

    if not methods_run:
        return pd.DataFrame()

    weights = get_method_weights(regime, methods_run)
    logger.info(
        f"Ensemble weights ({regime.regime_name}): "
        + ", ".join(
            f"{m}={w:.2f}" for m, w in sorted(weights.items(), key=lambda x: -x[1])
        )
    )

    # Collect all significant edges from all methods
    edge_votes: Dict[Tuple[str, str], Dict[str, float]] = {}

    for method, df in results_dict.items():
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            continue
        if method not in weights:
            continue

        cols = METHOD_COLUMNS.get(method, {})
        src_col = cols.get("src", "source")
        tgt_col = cols.get("tgt", "target")
        sig_col = cols.get("sig", "is_significant")

        if (
            sig_col not in df.columns
            or src_col not in df.columns
            or tgt_col not in df.columns
        ):
            continue

        sig_rows = df[df[sig_col]]
        for _, row in sig_rows.iterrows():
            edge = (row[src_col], row[tgt_col])
            if edge not in edge_votes:
                edge_votes[edge] = {}
            edge_votes[edge][method] = weights[method]

    # Compute ensemble scores
    rows = []
    for (src, tgt), method_weights in edge_votes.items():
        confidence = sum(method_weights.values())
        n_found = len(method_weights)
        contributing = sorted(method_weights.keys())

        rows.append(
            {
                "source": src,
                "target": tgt,
                "confidence": min(confidence, 1.0),
                "n_methods_found": n_found,
                "n_methods_total": len(methods_run),
                "contributing_methods": ", ".join(contributing),
                "weighted_score": confidence,
                "is_significant": confidence >= significance_threshold,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = (
        pd.DataFrame(rows)
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )
    n_sig = df_out["is_significant"].sum()
    logger.info(
        f"Ensemble: {len(df_out)} unique edges, {n_sig} significant "
        f"(threshold={significance_threshold:.2f})"
    )
    return df_out


def detect_data_regime(
    df: pd.DataFrame,
    nonlinearity_detected: bool = False,
    nongaussian_detected: bool = False,
    confounders_suspected: bool = False,
) -> DataRegime:
    """Create a DataRegime from detection results.

    This is a thin wrapper that packages the outputs of the existing
    diagnostics (RESET test, Shapiro-Wilk, causal-audit) into a DataRegime.

    Parameters
    ----------
    df : pd.DataFrame
        The data (used for T and N).
    nonlinearity_detected : bool
        Whether the RESET/dcor test detected nonlinearity.
    nongaussian_detected : bool
        Whether Shapiro-Wilk rejected Gaussianity.
    confounders_suspected : bool
        Whether causal-audit flagged confounding risk.

    Returns
    -------
    DataRegime
    """
    numeric_df = df.select_dtypes(include=["number"])
    return DataRegime(
        is_linear=not nonlinearity_detected,
        is_gaussian=not nongaussian_detected,
        has_latent_confounders=confounders_suspected,
        t_effective=len(numeric_df),
        n_vars=len(numeric_df.columns),
    )
