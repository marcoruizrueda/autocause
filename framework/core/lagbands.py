"""
Lag-Band Consensus Module

Reduces sensitivity to exact lag values by grouping lags into bands.
This is scientifically motivated: a 1-step difference (e.g., 35 vs 36 days)
should not break consensus when the underlying mechanism operates on
similar timescales.

Bands:
- Fast: 0-3 steps (immediate response, < 2 weeks for 5-day sampling)
- Mid: 4-9 steps (short-term integration, 2-7 weeks)
- Long: 10+ steps (long-term memory, > 7 weeks)
"""

from typing import Dict, Literal

import pandas as pd


LagBand = Literal["fast", "mid", "long"]


def get_lag_band(lag: int, fast_max: int = 3, mid_max: int = 9) -> LagBand:
    """
    Classify lag into fast/mid/long band.

    Parameters
    ----------
    lag : int
        Lag in steps
    fast_max : int, default=3
        Maximum lag for fast band (inclusive)
    mid_max : int, default=9
        Maximum lag for mid band (inclusive)

    Returns
    -------
    LagBand
        'fast', 'mid', or 'long'

    Examples
    --------
    >>> get_lag_band(2)
    'fast'
    >>> get_lag_band(5)
    'mid'
    >>> get_lag_band(12)
    'long'
    """
    if lag <= fast_max:
        return "fast"
    elif lag <= mid_max:
        return "mid"
    else:
        return "long"


def match_within_band(lag1: int, lag2: int, **kwargs) -> bool:
    """
    Check if two lags are in the same band.

    Parameters
    ----------
    lag1, lag2 : int
        Lags to compare
    **kwargs
        Passed to get_lag_band (fast_max, mid_max)

    Returns
    -------
    bool
        True if both lags in same band

    Examples
    --------
    >>> match_within_band(2, 3)  # both fast
    True
    >>> match_within_band(2, 5)  # fast vs mid
    False
    >>> match_within_band(35, 40)  # both long (steps 7, 8 for 5-day sampling)
    True
    """
    return get_lag_band(lag1, **kwargs) == get_lag_band(lag2, **kwargs)


def annotate_consensus_with_bands(
    consensus_df: pd.DataFrame,
    lag_col: str = "lag_steps",
    band_col: str = "lag_band",
    **kwargs,
) -> pd.DataFrame:
    """
    Add lag band annotation to consensus dataframe.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Consensus results with lag column
    lag_col : str, default='lag_steps'
        Column name containing lag values
    band_col : str, default='lag_band'
        Column name for new band annotation
    **kwargs
        Passed to get_lag_band

    Returns
    -------
    pd.DataFrame
        Consensus dataframe with added band column
    """
    df = consensus_df.copy()
    df[band_col] = df[lag_col].apply(lambda x: get_lag_band(x, **kwargs))
    return df


def compute_band_consensus(
    results_dict: Dict[str, pd.DataFrame],
    source_col: str = "source",
    target_col: str = "target",
    lag_col: str = "lag",
    pval_col: str = "p_value",
    alpha: float = 0.05,
    min_votes: int = 2,
    **band_kwargs,
) -> pd.DataFrame:
    """
    Compute consensus based on lag bands instead of exact lags.

    This is more robust than exact lag matching. Methods may detect the same
    causal relationship at slightly different lags (e.g., 35 vs 40 days) due to:
    - Different estimation procedures
    - Model specification differences
    - Numerical optimization variations

    Grouping into bands prevents breaking consensus over minor lag differences.

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        Dictionary mapping method names to results dataframes
    source_col, target_col : str
        Column names for source and target variables
    lag_col : str
        Column name for lag values
    pval_col : str
        Column name for p-values
    alpha : float, default=0.05
        Significance threshold
    min_votes : int, default=2
        Minimum methods required for consensus
    **band_kwargs
        Passed to get_lag_band (fast_max, mid_max)

    Returns
    -------
    pd.DataFrame
        Band-based consensus results with columns:
        - source, target: variable names
        - lag_band: fast/mid/long
        - vote_count: number of methods agreeing
        - agreeing_methods: comma-separated method names
        - representative_lag: median lag within band
        - lag_range: (min, max) lags in band
        - best_p_value: minimum p-value across methods
        - n_significant: number of units/instances significant

    Examples
    --------
    >>> results = {
    ...     'Granger': pd.DataFrame({
    ...         'source': ['RR', 'TG'],
    ...         'target': ['NDVI', 'NDVI'],
    ...         'lag': [7, 6],
    ...         'p_value': [0.001, 0.01]
    ...     }),
    ...     'PCMCI+': pd.DataFrame({
    ...         'source': ['RR', 'TG'],
    ...         'target': ['NDVI', 'NDVI'],
    ...         'lag': [8, 5],
    ...         'p_value': [0.002, 0.02]
    ...     })
    ... }
    >>> consensus = compute_band_consensus(results, min_votes=2)
    >>> consensus[['source', 'target', 'lag_band', 'vote_count']]
    """
    # Combine all significant edges with band annotation
    all_edges = []

    for method_name, df in results_dict.items():
        if df is None or len(df) == 0:
            continue

        # Find p-value column
        pval_column = None
        for col_name in [pval_col, "best_p_value", "pvalue", "q_value"]:
            if col_name in df.columns:
                pval_column = col_name
                break

        if pval_column is None:
            continue

        # Filter significant edges
        sig_df = df[df[pval_column] < alpha].copy()

        if len(sig_df) == 0:
            continue

        # Add band annotation
        sig_df["lag_band"] = sig_df[lag_col].apply(
            lambda x: get_lag_band(x, **band_kwargs)
        )
        sig_df["method"] = method_name
        sig_df["pval"] = sig_df[pval_column]

        # Keep needed columns
        keep_cols = [source_col, target_col, lag_col, "lag_band", "method", "pval"]
        optional_cols = ["unit_id", "n_obs"]
        for col in optional_cols:
            if col in sig_df.columns:
                keep_cols.append(col)

        all_edges.append(sig_df[keep_cols])

    if len(all_edges) == 0:
        return pd.DataFrame()

    # Combine all edges
    combined = pd.concat(all_edges, ignore_index=True)

    # Group by (source, target, lag_band) and count votes
    group_cols = [source_col, target_col, "lag_band"]

    consensus_records = []

    for (src, tgt, band), group in combined.groupby(group_cols):
        vote_count = group["method"].nunique()

        if vote_count < min_votes:
            continue

        methods = sorted(group["method"].unique())

        consensus_records.append(
            {
                "source": src,
                "target": tgt,
                "lag_band": band,
                "vote_count": vote_count,
                "agreeing_methods": ",".join(methods),
                "representative_lag": int(group[lag_col].median()),
                "lag_min": int(group[lag_col].min()),
                "lag_max": int(group[lag_col].max()),
                "lag_std": float(group[lag_col].std()),
                "best_p_value": float(group["pval"].min()),
                "mean_p_value": float(group["pval"].mean()),
                "n_detections": len(group),
            }
        )

    if len(consensus_records) == 0:
        return pd.DataFrame()

    consensus_df = pd.DataFrame(consensus_records)

    # Sort by vote count (descending), then p-value (ascending)
    consensus_df = consensus_df.sort_values(
        ["vote_count", "best_p_value"], ascending=[False, True]
    ).reset_index(drop=True)

    return consensus_df


def get_band_description(band: LagBand, sampling_days: int = 1) -> str:
    """
    Get human-readable description of lag band.

    Parameters
    ----------
    band : LagBand
        'fast', 'mid', or 'long'
    sampling_days : int, default=1
        Temporal sampling interval in days

    Returns
    -------
    str
        Description of lag band timescale

    Examples
    --------
    >>> get_band_description('fast', sampling_days=5)
    'Fast response (0-15 days, immediate)'
    >>> get_band_description('mid', sampling_days=1)
    'Mid-term (4-9 days, short memory)'
    """
    band_ranges = {
        "fast": (0, 3),
        "mid": (4, 9),
        "long": (10, float("inf")),
    }

    band_labels = {
        "fast": "immediate",
        "mid": "short memory",
        "long": "long memory",
    }

    min_step, max_step = band_ranges[band]
    label = band_labels[band]

    min_days = min_step * sampling_days
    max_days = "∞" if max_step == float("inf") else max_step * sampling_days

    day_range = (
        f"{min_days}-{max_days} days" if max_days != "∞" else f">{min_days} days"
    )

    return f"{band.capitalize()} response ({day_range}, {label})"


def compare_exact_vs_band_consensus(
    exact_consensus: pd.DataFrame,
    band_consensus: pd.DataFrame,
) -> Dict[str, int]:
    """
    Compare exact lag matching vs. band matching consensus.

    Parameters
    ----------
    exact_consensus : pd.DataFrame
        Consensus using exact lag matching
    band_consensus : pd.DataFrame
        Consensus using lag band matching

    Returns
    -------
    Dict[str, int]
        Statistics comparing the two approaches:
        - exact_edges: number of edges with exact consensus
        - band_edges: number of edges with band consensus
        - gained_edges: edges only in band consensus
        - lost_edges: edges only in exact consensus
    """
    n_exact = len(exact_consensus)
    n_band = len(band_consensus)

    # For simple comparison, count unique (source, target) pairs
    exact_pairs = set(zip(exact_consensus["source"], exact_consensus["target"]))

    if "lag_band" in band_consensus.columns:
        # Band consensus may have multiple bands for same pair
        band_pairs = set(zip(band_consensus["source"], band_consensus["target"]))
    else:
        band_pairs = set(zip(band_consensus["source"], band_consensus["target"]))

    gained = len(band_pairs - exact_pairs)
    lost = len(exact_pairs - band_pairs)

    return {
        "exact_edges": n_exact,
        "band_edges": n_band,
        "gained_edges": gained,
        "lost_edges": lost,
        "net_change": gained - lost,
    }
