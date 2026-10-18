"""
Statistics module for causal discovery framework.

Implements stationarity testing, linearity assessment, and lag selection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import adfuller, kpss, acf

logger = logging.getLogger(__name__)


def test_stationarity(
    series: pd.Series,
    method: str = "adf",
    critical_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Test stationarity using ADF or KPSS test.

    Parameters
    ----------
    series : pd.Series
        Time series to test.
    method : str, default="adf"
        Test method: 'adf' (null=non-stationary) or 'kpss' (null=stationary).
    critical_level : float, default=0.05
        Significance level.

    Returns
    -------
    dict
        Test results with 'is_stationary' key.
    """

    series_clean = series.dropna()
    if len(series_clean) < 2:
        logger.warning(
            "Series too short for stationarity test. Assuming non-stationary."
        )
        return {
            "method": method,
            "is_stationary": False,
            "p_value": np.nan,
            "test_statistic": np.nan,
            "note": "Insufficient data",
        }

    try:
        if method.lower() == "adf":
            result = adfuller(series_clean, autolag="AIC", regression="c")
            p_value = result[1]
            is_stat = p_value < critical_level  # Reject null=non-stationary

            return {
                "method": "ADF",
                "is_stationary": is_stat,
                "p_value": p_value,
                "test_statistic": result[0],
                "critical_values": result[4],
                "interpretation": "Series is stationary"
                if is_stat
                else "Series is non-stationary",
            }

        elif method.lower() == "kpss":
            result = kpss(series_clean, regression="c", nlags="auto")
            p_value = result[1]
            is_stat = p_value > critical_level  # Fail to reject null=stationary

            return {
                "method": "KPSS",
                "is_stationary": is_stat,
                "p_value": p_value,
                "test_statistic": result[0],
                "critical_values": result[3],
                "interpretation": "Series is stationary"
                if is_stat
                else "Series is non-stationary",
            }

    except Exception as e:
        logger.error(f"Stationarity test failed: {e}")
        return {
            "method": method,
            "is_stationary": False,
            "p_value": np.nan,
            "test_statistic": np.nan,
            "note": str(e),
        }


def assess_linearity(
    x: pd.Series,
    y: pd.Series,
    pearson_r2_threshold: float = 0.8,
    mi_to_r2_ratio: float = 1.5,
    min_r2_threshold: float = 0.8,
) -> Dict[str, Any]:
    """
    Assess linearity between two series using multiple metrics.

    Parameters
    ----------
    x : pd.Series
        Independent variable.
    y : pd.Series
        Dependent variable.
    pearson_r2_threshold : float, default=0.8
        R² threshold for high linearity.
    mi_to_r2_ratio : float, default=1.5
        Threshold for MI/R² ratio (nonlinearity indicator).
    min_r2_threshold : float, default=0.8
        Minimum R² to consider MI/R² ratio.

    Returns
    -------
    dict
        Linearity assessment with 'is_linear' key.
    """

    # Combine and drop NaNs
    combined = pd.concat([x, y], axis=1).dropna()
    if len(combined) < 10:
        logger.warning("Insufficient data for linearity assessment.")
        return {
            "is_linear": None,
            "note": "Insufficient data",
        }

    x_vals = combined.iloc[:, 0].values
    y_vals = combined.iloc[:, 1].values

    # Pearson correlation R²
    try:
        corr, _ = pearsonr(x_vals, y_vals)
        r2 = corr**2
    except Exception:
        r2 = np.nan

    # Mutual information
    try:
        mi = mutual_info_regression(x_vals.reshape(-1, 1), y_vals, random_state=42)
        mi_value = mi[0]
    except Exception:
        mi_value = np.nan

    # Spearman correlation
    try:
        spear_corr, _ = spearmanr(x_vals, y_vals)
    except Exception:
        spear_corr = np.nan

    # Decision logic
    if np.isnan(r2):
        is_linear = None
    elif r2 >= pearson_r2_threshold:
        is_linear = True
    elif r2 >= min_r2_threshold and not np.isnan(mi_value) and mi_value > 0:
        ratio = mi_value / r2
        is_linear = ratio < mi_to_r2_ratio
    else:
        is_linear = False

    return {
        "is_linear": is_linear,
        "pearson_r2": r2,
        "spearman_corr": spear_corr,
        "mutual_info": mi_value,
        "mi_to_r2_ratio": mi_value / r2 if not np.isnan(r2) and r2 > 0 else np.nan,
    }


def suggest_max_lag(
    series: pd.Series,
    sampling_interval_days: float = 5,
    default_max_days: int = 90,
    use_acf: bool = True,
    acf_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Suggest maximum lag based on data characteristics and heuristics.

    Parameters
    ----------
    series : pd.Series
        Time series data.
    sampling_interval_days : float, default=5
        Sampling interval in days.
    default_max_days : int, default=90
        Default maximum lag in days.
    use_acf : bool, default=True
        Use ACF to refine max lag.
    acf_threshold : float, default=0.1
        Threshold for ACF decay.

    Returns
    -------
    dict
        Lag suggestion with 'max_lag_timesteps' and 'max_lag_days' keys.
    """

    series_clean = series.dropna()

    # Base calculation
    max_lag_timesteps = int(np.floor(default_max_days / sampling_interval_days))
    max_lag_timesteps = min(max_lag_timesteps, 18)  # Safety cap

    result = {
        "max_lag_timesteps": max_lag_timesteps,
        "max_lag_days": max_lag_timesteps * sampling_interval_days,
        "method": "default",
        "acf_analysis": None,
    }

    # Refine with ACF if requested
    if use_acf and len(series_clean) >= 50:
        try:
            acf_vals = acf(series_clean, nlags=40, fft=False)
            # Find lag where ACF drops below threshold
            cross = np.where(np.abs(acf_vals[1:]) < acf_threshold)[0]
            if len(cross) > 0:
                acf_cutoff_lags = int(cross[0]) + 1
                acf_max_lag_timesteps = min(acf_cutoff_lags, max_lag_timesteps)

                result["acf_analysis"] = {
                    "acf_cutoff_lags": acf_cutoff_lags,
                    "cutoff_days": acf_cutoff_lags * sampling_interval_days,
                }
                result["max_lag_timesteps"] = acf_max_lag_timesteps
                result["max_lag_days"] = acf_max_lag_timesteps * sampling_interval_days
                result["method"] = "acf_informed"
        except Exception as e:
            logger.warning(f"ACF analysis failed: {e}")

    logger.info(
        f"Suggested max lag: {result['max_lag_timesteps']} timesteps ({result['max_lag_days']} days)"
    )

    return result


def select_lag_via_aic(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 12,
) -> Dict[str, Any]:
    """
    Select optimal lag via AIC using univariate AR models.

    Parameters
    ----------
    x : pd.Series
        Independent variable.
    y : pd.Series
        Dependent variable.
    max_lag : int, default=12
        Maximum lag to test.

    Returns
    -------
    dict
        Best lag and AIC values.
    """

    try:
        from statsmodels.tsa.ar_model import AutoReg

        combined = pd.concat([x, y], axis=1).dropna()
        if len(combined) < 2 * max_lag:
            logger.warning("Insufficient data for lag selection.")
            return {"optimal_lag": 1, "aic_values": {}, "note": "Insufficient data"}

        y_vals = combined.iloc[:, 1]
        aic_values = {}

        for lag in range(1, max_lag + 1):
            try:
                model = AutoReg(y_vals, lags=lag, seasonal=False, trend="c")
                res = model.fit()
                aic_values[lag] = res.aic
            except Exception:
                aic_values[lag] = np.inf

        optimal_lag = min(aic_values, key=aic_values.get)

        return {
            "optimal_lag": optimal_lag,
            "aic_values": aic_values,
            "min_aic": aic_values[optimal_lag],
        }

    except ImportError:
        logger.warning("AutoReg not available. Returning default lag.")
        return {"optimal_lag": 1, "aic_values": {}, "note": "AutoReg not available"}
    except Exception as e:
        logger.error(f"Lag selection failed: {e}")
        return {"optimal_lag": 1, "aic_values": {}, "note": str(e)}


def apply_seasonal_detrending(
    series: pd.Series,
    period: int = 365,
) -> pd.Series:
    """
    Apply seasonal detrending to a time series.

    Parameters
    ----------
    series : pd.Series
        Input series.
    period : int, default=365
        Seasonal period.

    Returns
    -------
    pd.Series
        Detrended series.
    """

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        decomposition = seasonal_decompose(
            series, model="additive", period=period, extrapolate="fill_ea"
        )
        detrended = series - decomposition.seasonal - decomposition.trend

        return detrended

    except Exception as e:
        logger.warning(
            f"Seasonal detrending failed: {e}. Returning differenced series."
        )
        return series.diff().dropna()


def apply_differencing(
    series: pd.Series,
    order: int = 1,
) -> pd.Series:
    """
    Apply differencing to achieve stationarity.

    Parameters
    ----------
    series : pd.Series
        Input series.
    order : int, default=1
        Differencing order.

    Returns
    -------
    pd.Series
        Differenced series.
    """

    diff_series = series.copy()
    for _ in range(order):
        diff_series = diff_series.diff()

    return diff_series.dropna()
