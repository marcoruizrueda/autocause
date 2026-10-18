"""
Robust Correlation Analysis

Provides statistically robust correlation measures that are less sensitive to:
- Outliers (robust estimators)
- Non-linearity (distance correlation, mutual information)
- Non-normality (rank-based methods)

All methods are dataset-agnostic and work with arbitrary panel/time series data.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def distance_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Distance correlation: measures both linear and nonlinear dependence.

    Range: [0, 1], where 0 = independence, 1 = perfect dependence.
    Advantage: Detects all types of dependence, not just linear.

    Parameters
    ----------
    x, y : np.ndarray
        Variables (same length)

    Returns
    -------
    Tuple[float, float]
        (distance_correlation, distance_covariance)

    References
    ----------
    Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
    "Measuring and testing dependence by correlation of distances."
    The Annals of Statistics, 35(6), 2769-2794.

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x**2 + np.random.randn(100) * 0.1  # Nonlinear relationship
    >>> dcor, dcov = distance_correlation(x, y)
    >>> dcor > 0.5  # Should detect dependence
    True
    """
    n = len(x)

    if len(y) != n:
        raise ValueError("x and y must have same length")

    if n < 4:
        return np.nan, np.nan

    # Compute distance matrices
    a = squareform(pdist(x.reshape(-1, 1), metric="euclidean"))
    b = squareform(pdist(y.reshape(-1, 1), metric="euclidean"))

    # Double-center distance matrices
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    # Distance covariance
    dcov_sq = (A * B).sum() / (n * n)
    dcov = np.sqrt(np.abs(dcov_sq))

    # Distance variances
    dvar_x = (A * A).sum() / (n * n)
    dvar_y = (B * B).sum() / (n * n)

    # Distance correlation
    if dvar_x > 0 and dvar_y > 0:
        dcor = dcov / np.sqrt(dvar_x * dvar_y)
    else:
        dcor = 0.0

    return float(dcor), float(dcov)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Spearman rank correlation: robust to outliers and monotonic relationships.

    Range: [-1, 1], where 0 = no monotonic relationship.
    Advantage: Robust to outliers, detects monotonic (not just linear) relationships.

    Parameters
    ----------
    x, y : np.ndarray
        Variables

    Returns
    -------
    Tuple[float, float]
        (correlation, p_value)

    Examples
    --------
    >>> x = np.arange(100)
    >>> y = x**2  # Monotonic but not linear
    >>> corr, pval = spearman_correlation(x, y)
    >>> corr
    1.0
    """
    corr, pval = stats.spearmanr(x, y, nan_policy="omit")
    return float(corr), float(pval)


def kendall_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Kendall's tau: robust rank correlation for ordinal data.

    Range: [-1, 1].
    Advantage: More robust than Spearman for small samples, handles ties better.

    Parameters
    ----------
    x, y : np.ndarray
        Variables

    Returns
    -------
    Tuple[float, float]
        (correlation, p_value)

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> corr, pval = kendall_correlation(x, y)
    >>> corr
    1.0
    """
    corr, pval = stats.kendalltau(x, y, nan_policy="omit")
    return float(corr), float(pval)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation: standard linear correlation.

    Range: [-1, 1].
    Note: Not robust to outliers or nonlinearity. Included for comparison.

    Parameters
    ----------
    x, y : np.ndarray
        Variables

    Returns
    -------
    Tuple[float, float]
        (correlation, p_value)
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return np.nan, np.nan

    corr, pval = stats.pearsonr(x_clean, y_clean)
    return float(corr), float(pval)


def compute_all_correlations(
    x: np.ndarray,
    y: np.ndarray,
    lag: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute all correlation measures between two variables.

    Parameters
    ----------
    x, y : np.ndarray
        Variables
    lag : int, default=0
        Lag to apply (x[:-lag] vs y[lag:])

    Returns
    -------
    Dict[str, Dict[str, float]]
        Results for each method:
        - pearson: {correlation, p_value}
        - spearman: {correlation, p_value}
        - kendall: {correlation, p_value}
        - distance: {correlation, covariance}

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x + np.random.randn(100) * 0.5
    >>> results = compute_all_correlations(x, y)
    >>> 'pearson' in results
    True
    >>> 'distance' in results
    True
    """
    # Apply lag
    if lag > 0:
        x_lag = x[:-lag]
        y_lag = y[lag:]
    else:
        x_lag = x
        y_lag = y

    results = {}

    # Pearson
    try:
        corr, pval = pearson_correlation(x_lag, y_lag)
        results["pearson"] = {"correlation": corr, "p_value": pval}
    except Exception as e:
        logger.debug(f"Pearson failed: {e}")
        results["pearson"] = {"correlation": np.nan, "p_value": np.nan}

    # Spearman
    try:
        corr, pval = spearman_correlation(x_lag, y_lag)
        results["spearman"] = {"correlation": corr, "p_value": pval}
    except Exception as e:
        logger.debug(f"Spearman failed: {e}")
        results["spearman"] = {"correlation": np.nan, "p_value": np.nan}

    # Kendall
    try:
        corr, pval = kendall_correlation(x_lag, y_lag)
        results["kendall"] = {"correlation": corr, "p_value": pval}
    except Exception as e:
        logger.debug(f"Kendall failed: {e}")
        results["kendall"] = {"correlation": np.nan, "p_value": np.nan}

    # Distance correlation
    try:
        dcor, dcov = distance_correlation(x_lag, y_lag)
        results["distance"] = {"correlation": dcor, "covariance": dcov}
    except Exception as e:
        logger.debug(f"Distance correlation failed: {e}")
        results["distance"] = {"correlation": np.nan, "covariance": np.nan}

    return results


def correlation_matrix(
    data: pd.DataFrame,
    variables: Optional[list] = None,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute correlation matrix using robust method.

    Parameters
    ----------
    data : pd.DataFrame
        Data with variables
    variables : list, optional
        Variables to correlate (default: all numeric columns)
    method : str, default='spearman'
        Method: 'pearson', 'spearman', 'kendall', or 'distance'

    Returns
    -------
    pd.DataFrame
        Correlation matrix

    Examples
    --------
    >>> data = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
    >>> corr_mat = correlation_matrix(data, method='spearman')
    >>> corr_mat.shape
    (2, 2)
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()

    n_vars = len(variables)
    corr_mat = np.zeros((n_vars, n_vars))

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i == j:
                corr_mat[i, j] = 1.0
            elif i > j:
                corr_mat[i, j] = corr_mat[j, i]  # Symmetric
            else:
                x = data[var1].values
                y = data[var2].values

                # Remove NaN pairs
                mask = ~(np.isnan(x) | np.isnan(y))
                x_clean = x[mask]
                y_clean = y[mask]

                if len(x_clean) < 3:
                    corr_mat[i, j] = np.nan
                    continue

                if method == "pearson":
                    corr, _ = pearson_correlation(x_clean, y_clean)
                elif method == "spearman":
                    corr, _ = spearman_correlation(x_clean, y_clean)
                elif method == "kendall":
                    corr, _ = kendall_correlation(x_clean, y_clean)
                elif method == "distance":
                    corr, _ = distance_correlation(x_clean, y_clean)
                else:
                    raise ValueError(f"Unknown method: {method}")

                corr_mat[i, j] = corr

    return pd.DataFrame(corr_mat, index=variables, columns=variables)


def test_correlation_significance(
    correlation: float,
    n_obs: int,
    method: str = "spearman",
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Test significance of correlation coefficient.

    Parameters
    ----------
    correlation : float
        Correlation coefficient
    n_obs : int
        Number of observations
    method : str, default='spearman'
        Correlation method used
    alpha : float, default=0.05
        Significance level

    Returns
    -------
    Dict
        Test results:
        - correlation: coefficient
        - n_obs: sample size
        - t_statistic: t-test statistic
        - p_value: two-tailed p-value
        - significant: bool
        - confidence_interval: (lower, upper)

    Examples
    --------
    >>> test_correlation_significance(0.5, n_obs=100)
    {'correlation': 0.5, 'significant': True, ...}
    """
    if n_obs < 3:
        return {
            "correlation": correlation,
            "n_obs": n_obs,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "confidence_interval": (np.nan, np.nan),
        }

    # T-test for correlation
    # t = r * sqrt((n-2) / (1-r²))
    r = correlation

    if abs(r) >= 1.0:
        # Perfect correlation
        return {
            "correlation": r,
            "n_obs": n_obs,
            "t_statistic": np.inf if r > 0 else -np.inf,
            "p_value": 0.0,
            "significant": True,
            "confidence_interval": (r, r),
        }

    t_stat = r * np.sqrt((n_obs - 2) / (1 - r**2))
    df = n_obs - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Fisher z-transformation for CI
    z = np.arctanh(r)
    se_z = 1 / np.sqrt(n_obs - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    ci_lower = np.tanh(z - z_crit * se_z)
    ci_upper = np.tanh(z + z_crit * se_z)

    return {
        "correlation": r,
        "n_obs": n_obs,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
        "confidence_interval": (ci_lower, ci_upper),
    }


def partial_correlation(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    method: str = "spearman",
) -> Tuple[float, float]:
    """
    Compute partial correlation: correlation between x and y controlling for z.

    Parameters
    ----------
    x, y : np.ndarray
        Variables of interest (1D arrays)
    z : np.ndarray
        Conditioning variable(s) (1D or 2D array)
        If 2D, uses multiple regression residuals
    method : str, default='spearman'
        Correlation method

    Returns
    -------
    Tuple[float, float]
        (partial_correlation, p_value)

    Examples
    --------
    >>> x = np.random.randn(100)
    >>> z = np.random.randn(100)
    >>> y = x + z + np.random.randn(100) * 0.1  # y depends on both x and z
    >>> pcorr, pval = partial_correlation(x, y, z)
    >>> pcorr > 0  # Should still detect x->y after controlling for z
    True
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z)

    # Handle 2D z (multiple control variables)
    if z.ndim == 2:
        # Remove rows with any NaN
        mask = ~(np.isnan(x) | np.isnan(y) | np.any(np.isnan(z), axis=1))
        x = x[mask]
        y = y[mask]
        z = z[mask]

        if len(x) < z.shape[1] + 3:  # Need more samples than control variables
            return np.nan, np.nan

        # Use regression residuals for multiple control variables
        from sklearn.linear_model import LinearRegression

        reg_x = LinearRegression()
        reg_x.fit(z, x)
        residual_x = x - reg_x.predict(z)

        # Regress y on z
        reg_y = LinearRegression()
        reg_y.fit(z, y)
        residual_y = y - reg_y.predict(z)

        # Compute correlation of residuals
        if method == "pearson":
            return pearson_correlation(residual_x, residual_y)
        elif method == "spearman":
            return spearman_correlation(residual_x, residual_y)
        elif method == "kendall":
            return kendall_correlation(residual_x, residual_y)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Handle 1D z (single control variable) - original implementation
    z = z.flatten()
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x = x[mask]
    y = y[mask]
    z = z[mask]

    if len(x) < 4:
        return np.nan, np.nan

    # Compute correlations
    if method == "pearson":
        r_xy, _ = pearson_correlation(x, y)
        r_xz, _ = pearson_correlation(x, z)
        r_yz, _ = pearson_correlation(y, z)
    elif method == "spearman":
        r_xy, _ = spearman_correlation(x, y)
        r_xz, _ = spearman_correlation(x, z)
        r_yz, _ = spearman_correlation(y, z)
    elif method == "kendall":
        r_xy, _ = kendall_correlation(x, y)
        r_xz, _ = kendall_correlation(x, z)
        r_yz, _ = kendall_correlation(y, z)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Partial correlation formula
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

    if denominator == 0:
        return np.nan, np.nan

    r_xy_z = (r_xy - r_xz * r_yz) / denominator

    # Test significance
    n = len(x)
    test_result = test_correlation_significance(r_xy_z, n, method)

    return float(r_xy_z), test_result["p_value"]
