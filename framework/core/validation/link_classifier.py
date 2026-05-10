"""Link classification for causal discovery results.

Categorizes detected causal links into three mutually exclusive categories:
autoregressive, contemporaneous, and lagged directed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedLinks:
    """Container for categorized causal links."""

    lagged_directed: pd.DataFrame
    contemporaneous: pd.DataFrame
    autoregressive: pd.DataFrame
    all_links: pd.DataFrame


def classify_links(
    graph: np.ndarray,
    var_names: list[str],
    val_matrix: np.ndarray | None = None,
) -> ClassifiedLinks:
    """Categorize every detected link into exactly one category.

    Parameters
    ----------
    graph : np.ndarray
        String array of shape (N, N, tau_max+1). Non-empty entries indicate
        detected links.
    var_names : list[str]
        Variable names of length N.
    val_matrix : np.ndarray | None
        Optional MCI test-statistic matrix of the same shape as graph.
        Used to populate the effect_size column.

    Returns
    -------
    ClassifiedLinks
        Three DataFrames (one per category) plus the combined table.
        Each DataFrame has columns: parent, target, lag, link_type, effect_size.
    """
    n_vars = len(var_names)
    tau_max_plus_one = graph.shape[2] if graph.ndim == 3 else 1

    rows_lagged: list[dict] = []
    rows_contemp: list[dict] = []
    rows_auto: list[dict] = []

    for i in range(n_vars):
        for j in range(n_vars):
            for tau in range(tau_max_plus_one):
                link_str = graph[i, j, tau]
                if link_str == "" or link_str == "":
                    continue
                # Skip empty string entries
                if not link_str or str(link_str).strip() == "":
                    continue

                effect = (
                    float(np.abs(val_matrix[i, j, tau]))
                    if val_matrix is not None
                    else 0.0
                )

                # Tigramite convention: graph[i, j, tau] = link from j(t-tau) to i(t)
                # So parent = var_names[j], target = var_names[i]
                parent_name = var_names[j]
                target_name = var_names[i]

                row = {
                    "parent": parent_name,
                    "target": target_name,
                    "lag": tau,
                    "link_type": str(link_str),
                    "effect_size": effect,
                }

                # Classification priority: autoregressive > contemporaneous > lagged
                if parent_name == target_name:
                    rows_auto.append(row)
                elif tau == 0:
                    rows_contemp.append(row)
                else:
                    rows_lagged.append(row)

    columns = ["parent", "target", "lag", "link_type", "effect_size"]
    df_lagged = pd.DataFrame(rows_lagged, columns=columns)
    df_contemp = pd.DataFrame(rows_contemp, columns=columns)
    df_auto = pd.DataFrame(rows_auto, columns=columns)
    df_all = pd.concat([df_lagged, df_contemp, df_auto], ignore_index=True)

    return ClassifiedLinks(
        lagged_directed=df_lagged,
        contemporaneous=df_contemp,
        autoregressive=df_auto,
        all_links=df_all,
    )
