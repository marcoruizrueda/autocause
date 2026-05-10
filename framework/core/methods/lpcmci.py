"""
LPCMCI — Latent PCMCI for causal discovery with latent confounders.

Extends PCMCI+ to handle hidden (unobserved) common causes by outputting
a Partial Ancestral Graph (PAG) instead of a CPDAG.  Edge types in the
PAG encode what can be inferred about directionality and latent
confounding:

    -->   directed (cause at tail, effect at head)
    <--   reverse directed
    o->   possibly directed or confounded
    o-o   undetermined
    x-x   conflicting information

When no latent confounders exist, LPCMCI reduces to PCMCI+.

References:
    Gerhardus, A. & Runge, J. (2020). "High-recall causal discovery for
    autocorrelated time series with latent confounders". NeurIPS 33.

Library: tigramite (https://github.com/jakobrunge/tigramite)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from tigramite.lpcmci import LPCMCI

    LPCMCI_AVAILABLE = True
except ImportError:
    LPCMCI_AVAILABLE = False
    logger.warning("LPCMCI not available. Install tigramite ≥5.2")


def batch_lpcmci(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    tau_max: int = 12,
    test_method: str = "auto",
    alpha: float = 0.05,
    sampling_days: float = 1.0,
) -> pd.DataFrame:
    """
    Run LPCMCI on the full dataset and extract results for requested pairs.

    LPCMCI handles latent confounders by outputting a PAG.  Edges marked
    ``-->`` are identified as direct causal links; ``o->`` edges indicate
    possible causation that cannot be fully oriented due to potential
    hidden variables.

    Parameters:
        df: Multivariate time series (DatetimeIndex, numeric columns).
        variable_pairs: List of (source, target) pairs to report.
        tau_max: Maximum lag to test.
        test_method: "auto" (default), "parcorr", "robust_parcorr", or "cmiknn".
        alpha: Significance threshold for the PC-stable condition selection.
        sampling_days: Days per timestep for lag conversion.

    Returns:
        DataFrame with columns: source, target, best_lag, best_p_value,
        is_significant, edge_type, method.
    """
    if not LPCMCI_AVAILABLE:
        logger.error("LPCMCI not available (tigramite ≥5.2 required)")
        return pd.DataFrame()

    from framework.core.methods.tigramite_pcmci import (
        prepare_data_for_tigramite,
        select_ci_test,
        extract_causal_edges,
    )

    # Collect all variables
    all_vars = sorted(set(v for pair in variable_pairs for v in pair))
    var_cols = [v for v in all_vars if v in df.columns]
    data_subset = df[var_cols].copy()

    # Handle missing values
    n_missing = data_subset.isna().sum().sum()
    if n_missing > 0:
        logger.info(f"Interpolating {n_missing} missing values for LPCMCI")
        data_subset = data_subset.interpolate(method="linear", limit_direction="both")
        data_subset = data_subset.ffill().bfill()

    if len(data_subset) < tau_max + 5:
        logger.error(f"Insufficient data ({len(data_subset)}) for tau_max={tau_max}")
        return _empty_results(variable_pairs)

    logger.info(
        f"Running LPCMCI (tau_max={tau_max}, vars={len(var_cols)}, "
        f"T={len(data_subset)}, test={test_method})"
    )

    # Select CI test
    if test_method == "auto":
        test, actual_method = select_ci_test(data_subset, method="auto", verbose=False)
    else:
        from framework.core.methods.tigramite_pcmci import _make_ci_test

        test = _make_ci_test(test_method, verbose=False)
        actual_method = test_method

    try:
        tigramite_df = prepare_data_for_tigramite(data_subset, standardize=True)
        lpcmci_obj = LPCMCI(dataframe=tigramite_df, cond_ind_test=test, verbosity=0)
        results = lpcmci_obj.run_lpcmci(tau_max=tau_max, pc_alpha=alpha)

        graph = results.get("graph")
        p_matrix = results.get("p_matrix")

        if graph is None or p_matrix is None:
            logger.warning("LPCMCI returned empty results")
            return _empty_results(variable_pairs)

        # Extract edges using the shared extractor (handles deduplication)
        all_edges = extract_causal_edges(graph, p_matrix, var_cols, alpha=alpha)

        # Build per-pair results
        rows = []
        for source, target in variable_pairs:
            pair_edges = all_edges[
                (all_edges["source"] == source)
                & (all_edges["target"] == target)
                & (all_edges["is_significant"])
            ]
            if len(pair_edges) > 0:
                best = pair_edges.loc[pair_edges["p_value"].idxmin()]
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "best_lag": best["lag"],
                        "best_lag_days": best["lag"] * sampling_days,
                        "best_p_value": best["p_value"],
                        "is_significant": True,
                        "edge_type": best.get("edge_type", "lagged"),
                        "n_observations": len(data_subset),
                        "test_method": actual_method,
                        "method": "LPCMCI",
                    }
                )
            else:
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "best_lag": np.nan,
                        "best_lag_days": np.nan,
                        "best_p_value": np.nan,
                        "is_significant": False,
                        "edge_type": "",
                        "n_observations": len(data_subset),
                        "test_method": actual_method,
                        "method": "LPCMCI",
                    }
                )

        results_df = pd.DataFrame(rows)
        n_sig = results_df["is_significant"].sum()
        logger.info(
            f"LPCMCI: {n_sig} significant edges for {len(variable_pairs)} pairs"
        )
        return results_df.sort_values("best_p_value")

    except Exception as e:
        logger.error(f"LPCMCI failed: {e}")
        return _empty_results(variable_pairs)


def _empty_results(variable_pairs):
    """Return empty results for all pairs."""
    rows = []
    for source, target in variable_pairs:
        rows.append(
            {
                "source": source,
                "target": target,
                "best_lag": np.nan,
                "best_p_value": np.nan,
                "is_significant": False,
                "method": "LPCMCI",
            }
        )
    return pd.DataFrame(rows)
