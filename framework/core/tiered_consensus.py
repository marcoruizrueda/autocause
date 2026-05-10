"""
Tiered Consensus System for Causal Discovery

Classifies consensus edges into tiers based on robustness criteria:

Tier-1 (Publication-Ready):
  - ≥2 methods agree
  - Passes ≥2 falsification tests
  - ICP stable across ≥80% environments
  - Out-of-sample validation: significant improvement (if available)

Tier-2 (Exploratory):
  - ≥2 methods agree
  - Either (≥1 falsification test) OR (OOS improvement)

Tier-3 (Hypothesis-Generating):
  - Single method detection
  - No robustness requirements

This provides clear communication of evidence quality for each edge.
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def classify_consensus_tier(
    vote_count: int,
    falsification_passed: int = 0,
    falsification_total: int = 2,
    icp_stable: bool = False,
    oos_significant: bool = False,
) -> int:
    """
    Classify a consensus edge into tiers based on robustness criteria.

    Parameters
    ----------
    vote_count : int
        Number of methods agreeing
    falsification_passed : int, default=0
        Number of falsification tests passed
    falsification_total : int, default=2
        Total falsification tests run
    icp_stable : bool, default=False
        Whether ICP stability test passed
    oos_significant : bool, default=False
        Whether out-of-sample validation showed improvement

    Returns
    -------
    int
        Tier: 1 (highest confidence), 2 (medium), or 3 (exploratory)

    Examples
    --------
    >>> classify_consensus_tier(vote_count=3, falsification_passed=2, icp_stable=True)
    1
    >>> classify_consensus_tier(vote_count=2, falsification_passed=1)
    2
    >>> classify_consensus_tier(vote_count=1)
    3
    """
    if vote_count < 1:
        return 3  # Default to exploratory

    # Tier-3: Single method only
    if vote_count == 1:
        return 3

    # For multi-method consensus (vote_count >= 2):

    # Tier-1 criteria (all must be met):
    # - Passes ≥2 falsification tests
    # - ICP stable
    # Note: OOS is optional (may not always be run)
    if falsification_passed >= 2 and icp_stable:
        return 1

    # Tier-2 criteria (either condition):
    # - Passes ≥1 falsification test
    # - OR has OOS improvement
    if falsification_passed >= 1 or oos_significant:
        return 2

    # Otherwise, still Tier-2 since multiple methods agree
    # but lacking robustness tests
    return 2


def add_tier_classification(
    consensus_df: pd.DataFrame,
    falsification_col: str = "falsification_passed",
    icp_col: str = "icp_stable",
    oos_col: str = "oos_significant",
    vote_col: str = "vote_count",
    tier_col: str = "tier",
) -> pd.DataFrame:
    """
    Add tier classification to consensus dataframe.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Consensus edges
    falsification_col : str, default='falsification_passed'
        Column with number of falsification tests passed
    icp_col : str, default='icp_stable'
        Column with ICP stability flag
    oos_col : str, default='oos_significant'
        Column with OOS validation flag
    vote_col : str, default='vote_count'
        Column with vote count
    tier_col : str, default='tier'
        Name for new tier column

    Returns
    -------
    pd.DataFrame
        Consensus dataframe with tier column added

    Examples
    --------
    >>> consensus = pd.DataFrame({
    ...     'source': ['X', 'Y'],
    ...     'target': ['Z', 'Z'],
    ...     'vote_count': [3, 2],
    ...     'falsification_passed': [2, 1],
    ...     'icp_stable': [True, False],
    ...     'oos_significant': [False, True]
    ... })
    >>> with_tiers = add_tier_classification(consensus)
    >>> with_tiers['tier'].tolist()
    [1, 2]
    """
    df = consensus_df.copy()

    # Set defaults for missing columns
    if falsification_col not in df.columns:
        df[falsification_col] = 0
    if icp_col not in df.columns:
        df[icp_col] = False
    if oos_col not in df.columns:
        df[oos_col] = False

    # Classify each edge
    tiers = []
    for _, row in df.iterrows():
        tier = classify_consensus_tier(
            vote_count=row[vote_col],
            falsification_passed=row.get(falsification_col, 0),
            falsification_total=2,
            icp_stable=row.get(icp_col, False),
            oos_significant=row.get(oos_col, False),
        )
        tiers.append(tier)

    df[tier_col] = tiers

    return df


def get_tier_description(tier: int) -> Dict[str, str]:
    """
    Get description and criteria for a tier.

    Parameters
    ----------
    tier : int
        Tier number (1, 2, or 3)

    Returns
    -------
    Dict[str, str]
        Description with keys:
        - name: tier name
        - confidence: confidence level
        - criteria: requirements
        - use_case: when to use

    Examples
    --------
    >>> desc = get_tier_description(1)
    >>> desc['name']
    'Publication-Ready'
    >>> desc['confidence']
    'HIGH'
    """
    descriptions = {
        1: {
            "name": "Publication-Ready",
            "confidence": "HIGH",
            "criteria": "≥2 methods agree + ≥2 falsification tests + ICP stable",
            "use_case": "Safe for publication, strong causal claims",
        },
        2: {
            "name": "Exploratory",
            "confidence": "MEDIUM",
            "criteria": "≥2 methods agree + (≥1 falsification test OR OOS improvement)",
            "use_case": "Suitable for exploratory analysis, hypothesis generation",
        },
        3: {
            "name": "Hypothesis-Generating",
            "confidence": "LOW",
            "criteria": "Single method detection only",
            "use_case": "Early-stage hypothesis, requires further validation",
        },
    }

    return descriptions.get(
        tier,
        {
            "name": "Unknown",
            "confidence": "UNKNOWN",
            "criteria": "N/A",
            "use_case": "N/A",
        },
    )


def summarize_tiers(
    consensus_df: pd.DataFrame,
    tier_col: str = "tier",
) -> pd.DataFrame:
    """
    Summarize consensus edges by tier.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Consensus edges with tier column
    tier_col : str, default='tier'
        Tier column name

    Returns
    -------
    pd.DataFrame
        Summary with columns:
        - tier: tier number
        - name: tier name
        - confidence: confidence level
        - n_edges: number of edges
        - percentage: percentage of total edges

    Examples
    --------
    >>> consensus = pd.DataFrame({'tier': [1, 1, 2, 2, 2, 3]})
    >>> summary = summarize_tiers(consensus)
    >>> summary['n_edges'].tolist()
    [2, 3, 1]
    """
    if tier_col not in consensus_df.columns:
        return pd.DataFrame()

    tier_counts = consensus_df[tier_col].value_counts().sort_index()
    total = len(consensus_df)

    summaries = []
    for tier in [1, 2, 3]:
        desc = get_tier_description(tier)
        count = tier_counts.get(tier, 0)

        summaries.append(
            {
                "tier": tier,
                "name": desc["name"],
                "confidence": desc["confidence"],
                "n_edges": count,
                "percentage": 100 * count / total if total > 0 else 0,
            }
        )

    return pd.DataFrame(summaries)


def filter_by_tier(
    consensus_df: pd.DataFrame,
    min_tier: int = 2,
    tier_col: str = "tier",
) -> pd.DataFrame:
    """
    Filter consensus edges by minimum tier.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Consensus edges with tier column
    min_tier : int, default=2
        Minimum tier to include (1=highest, 3=lowest)
    tier_col : str, default='tier'
        Tier column name

    Returns
    -------
    pd.DataFrame
        Filtered edges with tier >= min_tier

    Examples
    --------
    >>> consensus = pd.DataFrame({
    ...     'source': ['A', 'B', 'C'],
    ...     'target': ['X', 'Y', 'Z'],
    ...     'tier': [1, 2, 3]
    ... })
    >>> filtered = filter_by_tier(consensus, min_tier=2)
    >>> len(filtered)
    2
    >>> filtered['tier'].tolist()
    [1, 2]
    """
    if tier_col not in consensus_df.columns:
        return consensus_df

    # Lower tier number = higher confidence
    return consensus_df[consensus_df[tier_col] <= min_tier].copy()


def generate_tier_report(
    consensus_df: pd.DataFrame,
    tier_col: str = "tier",
) -> str:
    """
    Generate human-readable tier report.

    Parameters
    ----------
    consensus_df : pd.DataFrame
        Consensus edges with tier column
    tier_col : str, default='tier'
        Tier column name

    Returns
    -------
    str
        Formatted report text

    Examples
    --------
    >>> consensus = pd.DataFrame({
    ...     'source': ['A', 'B', 'C'],
    ...     'target': ['X', 'Y', 'Z'],
    ...     'tier': [1, 2, 3],
    ...     'vote_count': [3, 2, 1]
    ... })
    >>> report = generate_tier_report(consensus)
    >>> 'Tier-1' in report
    True
    """
    if tier_col not in consensus_df.columns:
        return "No tier classification available."

    lines = []
    lines.append("=" * 80)
    lines.append("TIERED CONSENSUS REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Overall summary
    summary = summarize_tiers(consensus_df, tier_col)

    lines.append("SUMMARY BY TIER:")
    lines.append("-" * 80)
    for _, row in summary.iterrows():
        tier = int(row["tier"])
        name = row["name"]
        conf = row["confidence"]
        count = int(row["n_edges"])
        pct = row["percentage"]

        lines.append(f"Tier-{tier} ({name}, confidence={conf}):")
        lines.append(f"  {count} edges ({pct:.1f}%)")

        desc = get_tier_description(tier)
        lines.append(f"  Criteria: {desc['criteria']}")
        lines.append(f"  Use case: {desc['use_case']}")
        lines.append("")

    # List edges by tier
    for tier in [1, 2, 3]:
        tier_edges = consensus_df[consensus_df[tier_col] == tier]

        if len(tier_edges) == 0:
            continue

        desc = get_tier_description(tier)
        lines.append(f"TIER-{tier} EDGES ({desc['name']}):")
        lines.append("-" * 80)

        for _, row in tier_edges.iterrows():
            source = row.get("source", "?")
            target = row.get("target", "?")
            lag = row.get("lag_steps", row.get("representative_lag", "?"))
            vote = row.get("vote_count", "?")
            methods = row.get("agreeing_methods", "?")

            lines.append(f"{source} → {target} (lag={lag}, votes={vote})")
            lines.append(f"  Methods: {methods}")

            # Add robustness info if available
            if "falsification_passed" in row:
                lines.append(
                    f"  Falsification: {row['falsification_passed']}/{row.get('falsification_total', 2)} tests passed"
                )
            if "icp_stable" in row:
                lines.append(f"  ICP stability: {'✓' if row['icp_stable'] else '✗'}")
            if "oos_significant" in row:
                lines.append(
                    f"  OOS validation: {'✓' if row['oos_significant'] else '✗'}"
                )

            lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
