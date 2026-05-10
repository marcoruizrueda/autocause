"""
Multiple testing corrections utilities.

Implements Benjamini–Hochberg (BH/FDR) and basic wrappers.
"""

from typing import List, Dict
import numpy as np
import pandas as pd


def fdr_bh(p_values: List[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Benjamini–Hochberg FDR control. Returns q-values and significance mask."""
    p = np.asarray([pv if pv is not None else np.nan for pv in p_values], dtype=float)
    n_total = len(p)

    # Only process non-NaN p-values
    valid_mask = ~np.isnan(p)
    n_valid = np.sum(valid_mask)

    if n_valid == 0:
        return {
            "q": np.full_like(p, np.nan),
            "significant": np.zeros_like(p, dtype=bool),
        }

    # Extract valid p-values and their original indices
    p_valid = p[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Rank valid p-values
    order = np.argsort(p_valid)
    ranks = np.arange(1, n_valid + 1)

    # Compute BH q-values for valid p-values only
    q_valid = p_valid * n_valid / ranks

    # Enforce monotonicity (q-values should decrease as p-values increase)
    q_sorted = np.minimum.accumulate(q_valid[order][::-1])[::-1]
    q_bh_valid = np.empty_like(q_valid)
    q_bh_valid[order] = q_sorted

    # Ensure q >= p elementwise (guard)
    q_bh_valid = np.maximum(q_bh_valid, p_valid)

    # Create full-size arrays with NaN for invalid entries
    q_bh_full = np.full(n_total, np.nan)
    q_bh_full[valid_indices] = q_bh_valid

    significant_full = np.zeros(n_total, dtype=bool)
    significant_full[valid_indices] = q_bh_valid <= alpha

    return {"q": q_bh_full, "significant": significant_full}


def apply_fdr_to_dataframe(
    df: pd.DataFrame, p_col: str = "p_value", alpha: float = 0.05
) -> pd.DataFrame:
    """Append q_value and significant columns using BH to a DataFrame."""
    if p_col not in df.columns or len(df) == 0:
        return df
    res = fdr_bh(df[p_col].tolist(), alpha=alpha)
    df = df.copy()
    df["q_value"] = res["q"]
    df["significant"] = res["significant"]
    return df
