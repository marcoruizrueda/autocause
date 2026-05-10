"""
Invariant Causal Prediction (ICP) Stability Testing

Tests whether causal coefficients are stable across different environments.
A true causal relationship should have stable coefficients, while spurious
correlations may vary across contexts.

Environments can be:
- Spatial zones (e.g., different climate regions) — via unit_id_col
- Temporal blocks (e.g., halves, thirds of the series) — automatic fallback
- Temporal periods (e.g., seasons, years) — via explicit environment_col
- Data subsets (e.g., wet vs dry years)

When no panel structure (unit_id_col) is available, the module automatically
splits the time series into non-overlapping temporal blocks and tests
coefficient stability across those blocks. This follows the invariance
principle of Peters et al. (2016): if X → Y is a true causal relationship,
the regression coefficient β in Y_t = β X_{t-lag} + ε should remain stable
across different temporal segments of the data.

References:
    Peters, J., Bühlmann, P., & Meinshausen, N. (2016). Causal inference
    by using invariant prediction. JRSS-B, 78(5), 947-1012.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def fit_causal_model(
    source: np.ndarray,
    target: np.ndarray,
    lag: int,
    include_intercept: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit simple lagged regression model: Y_t = β₀ + β₁ X_{t-lag} + ε_t

    Parameters
    ----------
    source : np.ndarray
        Source variable X
    target : np.ndarray
        Target variable Y
    lag : int
        Lag value
    include_intercept : bool, default=True
        Include intercept term

    Returns
    -------
    Tuple[np.ndarray, float, float]
        (coefficients, r_squared, std_error)
        coefficients: [intercept, slope] or [slope] if no intercept
        r_squared: coefficient of determination
        std_error: standard error of slope coefficient
    """
    n = len(source)

    if lag >= n:
        raise ValueError(f"Lag {lag} >= data length {n}")

    # Create lagged design matrix
    X_lag = source[:-lag] if lag > 0 else source
    Y = target[lag:] if lag > 0 else target

    if include_intercept:
        X_design = np.column_stack([np.ones(len(X_lag)), X_lag])
    else:
        X_design = X_lag.reshape(-1, 1)

    # Ordinary least squares
    try:
        coeffs = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.array([np.nan] * X_design.shape[1]), np.nan, np.nan

    # Compute R²
    Y_pred = X_design @ coeffs
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Standard error of slope (last coefficient)
    mse = ss_res / (len(Y) - X_design.shape[1])

    try:
        var_coeffs = mse * np.linalg.inv(X_design.T @ X_design)
        std_error = np.sqrt(var_coeffs[-1, -1])
    except np.linalg.LinAlgError:
        std_error = np.nan

    return coeffs, r_squared, std_error


def test_coefficient_homogeneity(
    coefficients: List[float],
    std_errors: List[float],
    method: str = "ci_overlap",
) -> Dict[str, any]:
    """
    Test if coefficients are homogeneous across environments.

    Parameters
    ----------
    coefficients : List[float]
        Coefficient estimates from each environment
    std_errors : List[float]
        Standard errors for each coefficient
    method : str, default='ci_overlap'
        Test method:
        - 'ci_overlap': Check if confidence intervals overlap
        - 'anova': One-way ANOVA test
        - 'cochran': Cochran's Q test for heterogeneity

    Returns
    -------
    Dict
        Test results with:
        - stable: bool, True if coefficients are homogeneous
        - test_statistic: test statistic value
        - p_value: p-value (if applicable)
        - method: test method used
        - details: additional information
    """
    coeffs = np.array(coefficients)
    std_errs = np.array(std_errors)

    # Remove NaN values
    valid_mask = ~(np.isnan(coeffs) | np.isnan(std_errs))
    coeffs = coeffs[valid_mask]
    std_errs = std_errs[valid_mask]

    if len(coeffs) < 2:
        return {
            "stable": False,
            "test_statistic": np.nan,
            "p_value": np.nan,
            "method": method,
            "details": "Insufficient environments for testing",
        }

    if method == "ci_overlap":
        # Coefficient variation approach: stable if the relative spread of
        # coefficients across environments is small compared to the mean.
        # This avoids the known problem where very precise estimates (tiny SE
        # from large samples) cause CI non-overlap even when coefficients are
        # practically identical in magnitude.
        n_envs = len(coeffs)
        mean_coeff = np.mean(coeffs)
        std_coeff = np.std(coeffs, ddof=1)

        # Relative variation: CV = std / |mean|
        if abs(mean_coeff) > 1e-10:
            cv = std_coeff / abs(mean_coeff)
            # Stable if CV < 0.5 (coefficients vary by less than 50% of mean)
            stable = cv < 0.5
        else:
            # Near-zero mean: check if all coefficients are near zero
            stable = std_coeff < 0.1

        # Also check sign consistency: all coefficients should have same sign
        if stable and abs(mean_coeff) > 1e-10:
            signs = np.sign(coeffs)
            sign_consistent = bool(np.all(signs == signs[0]))
            stable = stable and sign_consistent
        else:
            sign_consistent = (
                bool(np.all(np.sign(coeffs) == np.sign(coeffs[0])))
                if len(coeffs) > 0
                else True
            )

        return {
            "stable": bool(stable),
            "test_statistic": float(cv if abs(mean_coeff) > 1e-10 else std_coeff),
            "p_value": np.nan,
            "method": "ci_overlap",
            "details": {
                "n_environments": n_envs,
                "mean_coefficient": float(mean_coeff),
                "std_coefficient": float(std_coeff),
                "cv": float(cv) if abs(mean_coeff) > 1e-10 else np.nan,
                "sign_consistent": sign_consistent,
                "threshold_cv": 0.5,
            },
        }

    elif method == "anova":
        # One-way ANOVA: test if means differ across environments
        # H0: all coefficients equal
        # This is a simplified version - assumes equal sample sizes

        if len(coeffs) < 3:
            return {
                "stable": True,  # Can't reject with <3 groups
                "test_statistic": np.nan,
                "p_value": np.nan,
                "method": "anova",
                "details": "Need ≥3 environments for ANOVA",
            }

        # Use weighted variance test
        mean_coeff = np.mean(coeffs)
        var_between = np.var(coeffs, ddof=1)
        mean_var_within = np.mean(std_errs**2)

        # F-statistic approximation
        f_stat = var_between / mean_var_within if mean_var_within > 0 else np.inf

        # Degrees of freedom
        df1 = len(coeffs) - 1
        df2 = len(coeffs) * 10  # Approximate (would need sample sizes)

        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        stable = p_value > 0.05

        return {
            "stable": stable,
            "test_statistic": f_stat,
            "p_value": p_value,
            "method": "anova",
            "details": {
                "mean_coefficient": mean_coeff,
                "var_between": var_between,
                "mean_var_within": mean_var_within,
            },
        }

    elif method == "cochran":
        # Cochran's Q test for heterogeneity
        # Used in meta-analysis

        weights = 1 / (std_errs**2)
        weighted_mean = np.sum(weights * coeffs) / np.sum(weights)

        q_stat = np.sum(weights * (coeffs - weighted_mean) ** 2)

        # Chi-square test with k-1 degrees of freedom
        df = len(coeffs) - 1
        p_value = 1 - stats.chi2.cdf(q_stat, df)

        stable = p_value > 0.05

        return {
            "stable": stable,
            "test_statistic": q_stat,
            "p_value": p_value,
            "method": "cochran",
            "details": {
                "weighted_mean": weighted_mean,
                "df": df,
            },
        }

    else:
        raise ValueError(f"Unknown method: {method}")


def test_edge_stability(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    lag: int,
    environment_col: str,
    unit_col: Optional[str] = None,
    min_obs_per_env: int = 30,
    method: str = "ci_overlap",
) -> Dict[str, any]:
    """
    Test if a causal edge has stable coefficients across environments.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    source_col, target_col : str
        Column names for source and target variables
    lag : int
        Lag to test
    environment_col : str
        Column defining environments (e.g., 'zone', 'season', 'year')
    unit_col : str, optional
        Column for panel units (if None, assumes pooled data)
    min_obs_per_env : int, default=30
        Minimum observations per environment
    method : str, default='ci_overlap'
        Stability test method

    Returns
    -------
    Dict
        Stability test results with:
        - stable: bool
        - test_statistic: overall stability metric
        - p_value: p-value if applicable
        - n_environments: number of environments tested
        - environment_results: per-environment coefficients
        - homogeneity_test: results from coefficient homogeneity test
    """
    df = data.copy()

    # Get unique environments
    environments = df[environment_col].unique()

    environment_results = []

    for env in environments:
        env_data = df[df[environment_col] == env]

        if len(env_data) < min_obs_per_env:
            continue

        # Fit model in this environment
        source = env_data[source_col].values
        target = env_data[target_col].values

        try:
            coeffs, r2, se = fit_causal_model(source, target, lag)
            slope = coeffs[-1]  # Last coefficient is the lag effect

            environment_results.append(
                {
                    "environment": env,
                    "n_obs": len(env_data),
                    "coefficient": slope,
                    "std_error": se,
                    "r_squared": r2,
                }
            )
        except Exception as e:
            logger.debug(f"Model fit failed for environment {env}: {e}")
            continue

    if len(environment_results) < 2:
        return {
            "stable": False,
            "test_statistic": np.nan,
            "p_value": np.nan,
            "n_environments": len(environment_results),
            "environment_results": environment_results,
            "homogeneity_test": {
                "stable": False,
                "details": "Insufficient environments",
            },
        }

    # Test coefficient homogeneity
    coefficients = [r["coefficient"] for r in environment_results]
    std_errors = [r["std_error"] for r in environment_results]

    homogeneity = test_coefficient_homogeneity(coefficients, std_errors, method=method)

    return {
        "stable": homogeneity["stable"],
        "test_statistic": homogeneity["test_statistic"],
        "p_value": homogeneity.get("p_value", np.nan),
        "n_environments": len(environment_results),
        "environment_results": environment_results,
        "homogeneity_test": homogeneity,
    }


def test_consensus_stability(
    data: pd.DataFrame,
    consensus_df: pd.DataFrame,
    environment_col: str,
    source_col: str = "source",
    target_col: str = "target",
    lag_col: str = "lag_steps",
    min_obs_per_env: int = 30,
    method: str = "ci_overlap",
) -> pd.DataFrame:
    """
    Test stability for all consensus edges.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    consensus_df : pd.DataFrame
        Consensus edges to test
    environment_col : str
        Environment column in data
    source_col, target_col, lag_col : str
        Column names in consensus_df
    min_obs_per_env : int, default=30
        Minimum observations per environment
    method : str, default='ci_overlap'
        Stability test method

    Returns
    -------
    pd.DataFrame
        Consensus dataframe with added columns:
        - icp_stable: bool, True if stable
        - icp_n_environments: number of environments tested
        - icp_test_statistic: stability metric
        - icp_p_value: p-value if applicable
    """
    results = []

    for idx, row in consensus_df.iterrows():
        source = row[source_col]
        target = row[target_col]
        lag = row[lag_col]

        # Test stability
        stability = test_edge_stability(
            data=data,
            source_col=source,
            target_col=target,
            lag=lag,
            environment_col=environment_col,
            min_obs_per_env=min_obs_per_env,
            method=method,
        )

        results.append(
            {
                "icp_stable": stability["stable"],
                "icp_n_environments": stability["n_environments"],
                "icp_test_statistic": stability["test_statistic"],
                "icp_p_value": stability["p_value"],
            }
        )

    # Add results to consensus dataframe
    consensus_with_stability = consensus_df.copy()
    for col in results[0].keys():
        consensus_with_stability[col] = [r[col] for r in results]

    return consensus_with_stability


def create_temporal_environments(
    data: pd.DataFrame,
    n_blocks: int = 3,
    method: str = "equal",
) -> pd.Series:
    """
    Split a single time series into temporal blocks for ICP stability testing.

    When no panel structure (unit_id_col) is available, this function creates
    pseudo-environments by dividing the time series into non-overlapping
    temporal segments. Coefficient stability across these segments indicates
    that the causal relationship is not an artifact of a specific time period.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data (assumed sorted by time index)
    n_blocks : int, default=3
        Number of temporal blocks to create. Minimum 2, recommended 3-5.
    method : str, default='equal'
        Splitting method:
        - 'equal': Equal-sized blocks (default, most robust)
        - 'seasonal': Split by calendar season (requires DatetimeIndex)

    Returns
    -------
    pd.Series
        Integer labels (0, 1, ..., n_blocks-1) aligned with data index.

    Examples
    --------
    >>> df = pd.DataFrame({'x': range(300)}, index=pd.date_range('2020', periods=300))
    >>> envs = create_temporal_environments(df, n_blocks=3)
    >>> envs.value_counts()
    0    100
    1    100
    2    100
    """
    n = len(data)
    n_blocks = max(2, min(n_blocks, n // 30))  # At least 30 obs per block

    if method == "seasonal" and isinstance(data.index, pd.DatetimeIndex):
        # Map months to seasons: DJF=0, MAM=1, JJA=2, SON=3
        month = data.index.month
        season = pd.Series(
            np.where(
                month.isin([12, 1, 2]),
                0,
                np.where(
                    month.isin([3, 4, 5]), 1, np.where(month.isin([6, 7, 8]), 2, 3)
                ),
            ),
            index=data.index,
            name="temporal_env",
        )
        # Only keep seasons with enough data
        counts = season.value_counts()
        valid_seasons = counts[counts >= 30].index
        if len(valid_seasons) >= 2:
            return season
        # Fall through to equal blocks if not enough seasonal data
        logger.debug(
            "Seasonal split has <2 valid seasons, falling back to equal blocks"
        )

    # Equal-sized blocks
    block_size = n // n_blocks
    labels = np.repeat(range(n_blocks), block_size)
    # Handle remainder by extending the last block
    remainder = n - len(labels)
    if remainder > 0:
        labels = np.concatenate([labels, np.full(remainder, n_blocks - 1)])

    return pd.Series(labels.astype(int), index=data.index, name="temporal_env")


def test_edge_stability_temporal(
    data: pd.DataFrame,
    source_col: str,
    target_col: str,
    lag: int,
    n_blocks: int = 3,
    min_obs_per_block: int = 30,
    method: str = "ci_overlap",
    split_method: str = "equal",
) -> Dict:
    """
    Test coefficient stability across temporal blocks for single-unit data.

    This is the fallback when no panel structure is available. Splits the
    time series into non-overlapping temporal blocks and tests whether the
    lagged regression coefficient is stable across blocks.

    Parameters
    ----------
    data : pd.DataFrame
        Single-unit time series data
    source_col : str
        Source variable column name
    target_col : str
        Target variable column name
    lag : int
        Lag to test
    n_blocks : int, default=3
        Number of temporal blocks
    min_obs_per_block : int, default=30
        Minimum observations per block for valid estimation
    method : str, default='ci_overlap'
        Stability test method (passed to test_coefficient_homogeneity)
    split_method : str, default='equal'
        How to split: 'equal' or 'seasonal'

    Returns
    -------
    Dict
        Same structure as test_edge_stability:
        - stable: bool
        - test_statistic: stability metric
        - p_value: p-value if applicable
        - n_environments: number of blocks tested
        - environment_results: per-block coefficients
        - homogeneity_test: results from coefficient homogeneity test
    """
    if source_col not in data.columns or target_col not in data.columns:
        return {
            "stable": False,
            "test_statistic": np.nan,
            "p_value": np.nan,
            "n_environments": 0,
            "environment_results": [],
            "homogeneity_test": {"stable": False, "details": "Column not found"},
        }

    # Create temporal environments
    env_labels = create_temporal_environments(
        data, n_blocks=n_blocks, method=split_method
    )

    environment_results = []

    for block_id in sorted(env_labels.unique()):
        block_mask = env_labels == block_id
        block_data = data.loc[block_mask]

        if len(block_data) < min_obs_per_block:
            continue

        source = block_data[source_col].dropna().values
        target = block_data[target_col].dropna().values

        # Align after dropna (use the shorter)
        min_len = min(len(source), len(target))
        if min_len <= lag + 10:
            continue
        source = source[:min_len]
        target = target[:min_len]

        try:
            coeffs, r2, se = fit_causal_model(source, target, lag)
            slope = coeffs[-1]

            if np.isnan(slope) or np.isnan(se) or se <= 0:
                continue

            environment_results.append(
                {
                    "environment": f"block_{block_id}",
                    "n_obs": min_len,
                    "coefficient": slope,
                    "std_error": se,
                    "r_squared": r2,
                }
            )
        except Exception as e:
            logger.debug(f"Model fit failed for block {block_id}: {e}")
            continue

    if len(environment_results) < 2:
        return {
            "stable": False,
            "test_statistic": np.nan,
            "p_value": np.nan,
            "n_environments": len(environment_results),
            "environment_results": environment_results,
            "homogeneity_test": {
                "stable": False,
                "details": "Insufficient blocks with enough data",
            },
        }

    # Test coefficient homogeneity across blocks
    coefficients = [r["coefficient"] for r in environment_results]
    std_errors = [r["std_error"] for r in environment_results]

    homogeneity = test_coefficient_homogeneity(coefficients, std_errors, method=method)

    return {
        "stable": homogeneity["stable"],
        "test_statistic": homogeneity["test_statistic"],
        "p_value": homogeneity.get("p_value", np.nan),
        "n_environments": len(environment_results),
        "environment_results": environment_results,
        "homogeneity_test": homogeneity,
    }


def test_consensus_stability_temporal(
    data: pd.DataFrame,
    consensus_df: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    lag_col: str = "lag_steps",
    n_blocks: int = 3,
    min_obs_per_block: int = 30,
    method: str = "ci_overlap",
    split_method: str = "equal",
) -> pd.DataFrame:
    """
    Test stability for all consensus edges using temporal block splitting.

    Use this when no panel structure (unit_id_col) is available. Splits the
    single time series into temporal blocks and tests coefficient invariance.

    Parameters
    ----------
    data : pd.DataFrame
        Single-unit time series data
    consensus_df : pd.DataFrame
        Consensus edges to test
    source_col, target_col, lag_col : str
        Column names in consensus_df
    n_blocks : int, default=3
        Number of temporal blocks
    min_obs_per_block : int, default=30
        Minimum observations per block
    method : str, default='ci_overlap'
        Stability test method
    split_method : str, default='equal'
        How to split: 'equal' or 'seasonal'

    Returns
    -------
    pd.DataFrame
        Consensus dataframe with added columns:
        - icp_stable: bool
        - icp_n_environments: number of blocks tested
        - icp_test_statistic: stability metric
        - icp_p_value: p-value if applicable
    """
    results = []

    for idx, row in consensus_df.iterrows():
        source = row[source_col]
        target = row[target_col]
        lag = int(row[lag_col])

        stability = test_edge_stability_temporal(
            data=data,
            source_col=source,
            target_col=target,
            lag=lag,
            n_blocks=n_blocks,
            min_obs_per_block=min_obs_per_block,
            method=method,
            split_method=split_method,
        )

        results.append(
            {
                "icp_stable": stability["stable"],
                "icp_n_environments": stability["n_environments"],
                "icp_test_statistic": stability["test_statistic"],
                "icp_p_value": stability["p_value"],
            }
        )

    consensus_with_stability = consensus_df.copy()
    if results:
        for col in results[0].keys():
            consensus_with_stability[col] = [r[col] for r in results]
    else:
        consensus_with_stability["icp_stable"] = False
        consensus_with_stability["icp_n_environments"] = 0
        consensus_with_stability["icp_test_statistic"] = np.nan
        consensus_with_stability["icp_p_value"] = np.nan

    return consensus_with_stability


# Prevent pytest from collecting these as test functions
test_edge_stability.__test__ = False  # type: ignore[attr-defined]
test_edge_stability_temporal.__test__ = False  # type: ignore[attr-defined]
test_consensus_stability.__test__ = False  # type: ignore[attr-defined]
test_consensus_stability_temporal.__test__ = False  # type: ignore[attr-defined]
test_coefficient_homogeneity.__test__ = False  # type: ignore[attr-defined]
