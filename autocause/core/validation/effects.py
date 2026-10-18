"""Effect size extraction and driver ranking.

Extracts effect sizes from causal discovery results and ranks parent variables
by their influence on a specified target variable.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_effects(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    var_names: list[str],
    bootstrap_result=None,
) -> pd.DataFrame:
    """Extract effect sizes for all detected links.

    Parameters
    ----------
    graph : np.ndarray
        String array of shape (N, N, tau_max+1). Non-empty entries indicate
        detected links.
    val_matrix : np.ndarray
        MCI test-statistic matrix of the same shape as graph.
    var_names : list[str]
        Variable names of length N.
    bootstrap_result : BootstrapResult | None
        Optional bootstrap result providing confidence intervals.
        If provided, ci_low and ci_high are populated from the bootstrap
        strength percentiles.

    Returns
    -------
    pd.DataFrame
        Columns: parent, target, lag, link_type, effect_size, ci_low, ci_high.
    """
    N = len(var_names)
    tau_dim = graph.shape[2] if graph.ndim == 3 else 1

    rows: list[dict] = []
    for i in range(N):
        for j in range(N):
            for tau in range(tau_dim):
                link_str = graph[i, j, tau]
                if not link_str or str(link_str).strip() == "":
                    continue

                effect_size = float(np.abs(val_matrix[i, j, tau]))

                # Tigramite convention: graph[i, j, tau] = link from j(t-tau) to i(t)
                parent_name = var_names[j]
                target_name = var_names[i]

                # Determine link type category
                if parent_name == target_name:
                    link_type = "autoregressive"
                elif tau == 0:
                    link_type = "contemporaneous"
                else:
                    link_type = "lagged_directed"

                # Confidence intervals from bootstrap if available
                ci_low = np.nan
                ci_high = np.nan
                if bootstrap_result is not None:
                    try:
                        ci_low = float(bootstrap_result.strength_ci_low[i, j, tau])
                        ci_high = float(bootstrap_result.strength_ci_high[i, j, tau])
                    except (IndexError, AttributeError):
                        pass

                rows.append(
                    {
                        "parent": parent_name,
                        "target": target_name,
                        "lag": tau,
                        "link_type": link_type,
                        "effect_size": effect_size,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    columns = [
        "parent",
        "target",
        "lag",
        "link_type",
        "effect_size",
        "ci_low",
        "ci_high",
    ]
    return pd.DataFrame(rows, columns=columns)


def rank_drivers(
    effects_df: pd.DataFrame,
    target_var: str,
    exclude_autoregressive: bool = True,
) -> pd.DataFrame:
    """Rank parent variables by effect size on a specified target.

    Parameters
    ----------
    effects_df : pd.DataFrame
        Output from extract_effects().
    target_var : str
        Target variable to rank drivers for.
    exclude_autoregressive : bool
        If True (default), exclude links where parent == target.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted DataFrame with a link_category column,
        ordered by descending effect_size.
    """
    # Filter to target variable
    df = effects_df[effects_df["target"] == target_var].copy()

    # Exclude autoregressive links if requested
    if exclude_autoregressive:
        df = df[df["parent"] != target_var].copy()

    # Add link_category column (same as link_type for consistency)
    if "link_type" in df.columns:
        df["link_category"] = df["link_type"]
    else:
        df["link_category"] = "unknown"

    # Sort by descending effect size
    df = df.sort_values("effect_size", ascending=False).reset_index(drop=True)

    return df
