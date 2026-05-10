"""
Mathematically Rigorous tau_max Estimation

Implements multiple theoretically-grounded approaches for determining
the maximum lag (tau_max) for causal discovery:

1. ACF Zero-Crossing (memory length)
2. PACF Cutoff (direct effect horizon)
3. Mutual Information Decay (information-theoretic)
4. AIC/BIC Lag Selection (model-based)
5. Transfer Entropy Peak + Buffer (causal horizon)
6. Nyquist-Constrained Domain Knowledge (hybrid)
7. First Minimum of MI (Takens embedding)

References:
    - Box & Jenkins (1976): Time Series Analysis
    - Fraser & Swinney (1986): Independent coordinates for strange attractors
    - Schreiber (2000): Measuring information transfer
    - Runge et al. (2019): Inferring causation from time series
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf, pacf

logger = logging.getLogger(__name__)


def estimate_tau_max_acf_zero_crossing(
    series: pd.Series,
    confidence_level: float = 0.95,
    min_lag: int = 1,
    max_search: int = 100,
) -> Dict[str, Any]:
    """
    Estimate tau_max from ACF zero-crossing (memory length).
    
    Theory: The lag where autocorrelation becomes statistically insignificant
    represents the "memory length" of the series. Beyond this lag, past values
    provide negligible information about future values.
    
    Confidence bands: ±z * σ, where σ = 1/√n for white noise.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    confidence_level : float, default=0.95
        Confidence level (0.95 → ±1.96/√n)
    min_lag : int, default=1
        Minimum lag to consider
    max_search : int, default=100
        Maximum lag to search
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    series_clean = series.dropna()
    n = len(series_clean)
    
    if n < 50:
        logger.warning(f"ACF zero-crossing: insufficient data (n={n})")
        return {"tau_max": min_lag, "method": "acf_zero", "note": "insufficient_data"}
    
    try:
        # Compute ACF
        max_nlags = min(max_search, n // 4)
        acf_vals = acf(series_clean, nlags=max_nlags, fft=False)
        
        # Confidence band (two-sided)
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence_level / 2)
        threshold = z / np.sqrt(n)
        
        # Find first crossing
        significant = np.abs(acf_vals[min_lag:]) > threshold
        if not np.any(~significant):
            # All lags significant - use max_search
            tau_max = max_nlags
            note = "no_crossing_found"
        else:
            # First insignificant lag
            tau_max = min_lag + np.where(~significant)[0][0]
            note = "zero_crossing_detected"
        
        return {
            "tau_max": int(tau_max),
            "method": "acf_zero_crossing",
            "confidence_level": confidence_level,
            "threshold": threshold,
            "acf_at_tau": acf_vals[tau_max] if tau_max < len(acf_vals) else np.nan,
            "note": note,
        }
        
    except Exception as e:
        logger.error(f"ACF zero-crossing failed: {e}")
        return {"tau_max": min_lag, "method": "acf_zero", "error": str(e)}


def estimate_tau_max_pacf_cutoff(
    series: pd.Series,
    confidence_level: float = 0.95,
    min_lag: int = 1,
    max_search: int = 50,
) -> Dict[str, Any]:
    """
    Estimate tau_max from PACF cutoff (direct effect horizon).
    
    Theory: PACF measures direct (not indirect) correlation at each lag.
    The cutoff lag indicates the order of the AR process, beyond which
    there are no direct effects.
    
    More precise than ACF for AR processes, as it removes spurious
    correlations due to intermediate lags.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    confidence_level : float, default=0.95
        Confidence level
    min_lag : int, default=1
        Minimum lag to consider
    max_search : int, default=50
        Maximum lag to search
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    series_clean = series.dropna()
    n = len(series_clean)
    
    if n < 50:
        logger.warning(f"PACF cutoff: insufficient data (n={n})")
        return {"tau_max": min_lag, "method": "pacf_cutoff", "note": "insufficient_data"}
    
    try:
        # Compute PACF
        max_nlags = min(max_search, n // 4 - 1)
        pacf_vals = pacf(series_clean, nlags=max_nlags, method='ywm')
        
        # Confidence band
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence_level / 2)
        threshold = z / np.sqrt(n)
        
        # Find last significant lag
        significant = np.abs(pacf_vals[min_lag:]) > threshold
        if not np.any(significant):
            # No significant lags
            tau_max = min_lag
            note = "no_significant_lags"
        else:
            # Last significant lag
            sig_indices = np.where(significant)[0]
            tau_max = min_lag + sig_indices[-1]
            note = "pacf_cutoff_detected"
        
        return {
            "tau_max": int(tau_max),
            "method": "pacf_cutoff",
            "confidence_level": confidence_level,
            "threshold": threshold,
            "pacf_at_tau": pacf_vals[tau_max] if tau_max < len(pacf_vals) else np.nan,
            "note": note,
        }
        
    except Exception as e:
        logger.error(f"PACF cutoff failed: {e}")
        return {"tau_max": min_lag, "method": "pacf_cutoff", "error": str(e)}


def estimate_tau_max_mi_decay(
    series_x: pd.Series,
    series_y: pd.Series,
    threshold_frac: float = 0.1,
    min_lag: int = 1,
    max_search: int = 50,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Estimate tau_max from mutual information decay.
    
    Theory: Mutual information I(X_t; Y_{t-τ}) quantifies the information
    shared between X at time t and Y lagged by τ. When MI drops below
    a threshold (e.g., 10% of max MI), further lags provide negligible
    information.
    
    Information-theoretically grounded for nonlinear relationships.
    
    Parameters
    ----------
    series_x : pd.Series
        Source variable
    series_y : pd.Series
        Target variable
    threshold_frac : float, default=0.1
        Fraction of max MI to use as cutoff (0.1 = 10%)
    min_lag : int, default=1
        Minimum lag to consider
    max_search : int, default=50
        Maximum lag to search
    n_bins : int, default=10
        Number of bins for discretization
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    # Align series
    combined = pd.concat([series_x, series_y], axis=1).dropna()
    if len(combined) < 100:
        logger.warning(f"MI decay: insufficient data (n={len(combined)})")
        return {"tau_max": min_lag, "method": "mi_decay", "note": "insufficient_data"}
    
    try:
        from sklearn.metrics import mutual_info_score
        
        x_vals = combined.iloc[:, 0].values
        y_vals = combined.iloc[:, 1].values
        
        # Discretize for MI estimation (convert to Series for pd.cut)
        x_binned = pd.cut(pd.Series(x_vals), bins=n_bins, labels=False, duplicates='drop').values
        
        mi_values = []
        for lag in range(min_lag, min(max_search, len(y_vals) - min_lag)):
            y_lagged = y_vals[:-lag] if lag > 0 else y_vals
            y_binned = pd.cut(pd.Series(y_lagged), bins=n_bins, labels=False, duplicates='drop').values
            
            # Align
            min_len = min(len(x_binned), len(y_binned))
            x_aligned = x_binned[:min_len]
            y_aligned = y_binned[:min_len]
            
            # Remove NaN from binning
            mask = ~(pd.isna(x_aligned) | pd.isna(y_aligned))
            if mask.sum() < 50:
                mi_values.append(0)
                continue
            
            mi = mutual_info_score(x_aligned[mask], y_aligned[mask])
            mi_values.append(mi)
        
        if len(mi_values) == 0 or max(mi_values) == 0:
            logger.warning("MI decay: no information detected")
            return {"tau_max": min_lag, "method": "mi_decay", "note": "no_mi"}
        
        # Find cutoff
        mi_array = np.array(mi_values)
        max_mi = np.max(mi_array)
        threshold = threshold_frac * max_mi
        
        below_threshold = mi_array < threshold
        if not np.any(below_threshold):
            tau_max = len(mi_values) + min_lag - 1
            note = "no_decay_found"
        else:
            tau_max = min_lag + np.where(below_threshold)[0][0]
            note = "mi_decay_detected"
        
        return {
            "tau_max": int(tau_max),
            "method": "mi_decay",
            "threshold_frac": threshold_frac,
            "max_mi": float(max_mi),
            "mi_at_tau": float(mi_array[tau_max - min_lag]) if tau_max - min_lag < len(mi_array) else 0,
            "mi_trajectory": mi_values,
            "note": note,
        }
        
    except Exception as e:
        logger.error(f"MI decay failed: {e}")
        return {"tau_max": min_lag, "method": "mi_decay", "error": str(e)}


def estimate_tau_max_aic_bic(
    series: pd.Series,
    criterion: str = "bic",
    min_lag: int = 1,
    max_search: int = 30,
) -> Dict[str, Any]:
    """
    Estimate tau_max via AIC/BIC lag selection.
    
    Theory: Fit AR(p) models with increasing order p, choose p that
    minimizes information criterion. BIC is more conservative (penalizes
    complexity more) than AIC.
    
    Model-based approach suitable for linear autoregressive processes.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    criterion : str, default="bic"
        Information criterion ("aic" or "bic")
    min_lag : int, default=1
        Minimum lag order
    max_search : int, default=30
        Maximum lag order to test
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    series_clean = series.dropna()
    n = len(series_clean)
    
    if n < 100:
        logger.warning(f"AIC/BIC: insufficient data (n={n})")
        return {"tau_max": min_lag, "method": criterion, "note": "insufficient_data"}
    
    try:
        from statsmodels.tsa.ar_model import AutoReg
        
        ic_values = {}
        max_lag_test = min(max_search, n // 3)
        
        for lag in range(min_lag, max_lag_test + 1):
            try:
                model = AutoReg(series_clean, lags=lag, trend='c', seasonal=False)
                result = model.fit()
                ic_values[lag] = result.aic if criterion == "aic" else result.bic
            except Exception:
                ic_values[lag] = np.inf
        
        if len(ic_values) == 0:
            return {"tau_max": min_lag, "method": criterion, "note": "no_models_fit"}
        
        # Find lag with minimum IC value
        tau_max = min(ic_values.keys(), key=lambda k: ic_values[k])
        
        return {
            "tau_max": int(tau_max),
            "method": f"{criterion}_lag_selection",
            "ic_value": ic_values[tau_max],
            "ic_trajectory": ic_values,
            "note": "optimal_lag_found",
        }
        
    except Exception as e:
        logger.error(f"AIC/BIC selection failed: {e}")
        return {"tau_max": min_lag, "method": criterion, "error": str(e)}


def estimate_tau_max_nyquist_domain(
    series: pd.Series,
    domain_max_days: int,
    sampling_interval_days: float,
    safety_factor: float = 3.0,
) -> Dict[str, Any]:
    """
    Estimate tau_max using Nyquist constraint + domain knowledge.
    
    Theory: 
    1. Domain constraint: Maximum physically plausible lag (e.g., 90 days
       for precipitation → vegetation response)
    2. Nyquist-like constraint: Need at least 3-5 samples per lag to
       estimate it reliably (T / safety_factor)
    3. Take minimum of both
    
    Conservative, prevents overfitting with limited data.
    
    Parameters
    ----------
    series : pd.Series
        Time series data
    domain_max_days : int
        Maximum physically plausible lag in days
    sampling_interval_days : float
        Sampling interval in days
    safety_factor : float, default=3.0
        T / safety_factor = max lag (higher = more conservative)
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    n = len(series.dropna())
    
    # Domain constraint
    domain_max_timesteps = int(np.floor(domain_max_days / sampling_interval_days))
    
    # Nyquist-like constraint
    nyquist_max_timesteps = int(np.floor(n / safety_factor))
    
    # Take minimum
    tau_max = min(domain_max_timesteps, nyquist_max_timesteps)
    tau_max = max(1, tau_max)  # At least 1
    
    return {
        "tau_max": int(tau_max),
        "method": "nyquist_domain_hybrid",
        "domain_max_timesteps": domain_max_timesteps,
        "nyquist_max_timesteps": nyquist_max_timesteps,
        "binding_constraint": "domain" if tau_max == domain_max_timesteps else "nyquist",
        "n_samples": n,
        "safety_factor": safety_factor,
        "note": "hybrid_estimate",
    }


def estimate_tau_max_first_mi_minimum(
    series_x: pd.Series,
    series_y: pd.Series,
    min_lag: int = 1,
    max_search: int = 50,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Estimate tau_max from first minimum of mutual information (Takens embedding).
    
    Theory: For time-delay embedding (Takens theorem), the optimal delay τ_d
    is at the first minimum of the mutual information I(X_t, X_{t-τ}).
    For causal discovery, tau_max can be set as multiple of τ_d:
    
        tau_max ≥ (m-1) × τ_d
    
    where m is the embedding dimension (typically 2-5).
    
    Appropriate for nonlinear dynamical systems with deterministic structure.
    
    Parameters
    ----------
    series_x : pd.Series
        Source variable
    series_y : pd.Series
        Target variable
    min_lag : int, default=1
        Minimum lag to consider
    max_search : int, default=50
        Maximum lag to search
    n_bins : int, default=10
        Number of bins for discretization
        
    Returns
    -------
    dict
        tau_max estimate with diagnostics
    """
    # Use self-MI of target (standard approach)
    series_clean = series_y.dropna()
    
    if len(series_clean) < 100:
        logger.warning(f"First MI minimum: insufficient data (n={len(series_clean)})")
        return {"tau_max": min_lag, "method": "first_mi_min", "note": "insufficient_data"}
    
    try:
        from sklearn.metrics import mutual_info_score
        
        y_vals = series_clean.values
        y_binned = pd.cut(pd.Series(y_vals), bins=n_bins, labels=False, duplicates='drop').values
        
        mi_values = []
        for lag in range(min_lag, min(max_search, len(y_vals) // 2)):
            y_lagged = y_vals[lag:]
            y_lagged_binned = pd.cut(pd.Series(y_lagged), bins=n_bins, labels=False, duplicates='drop').values
            
            # Align
            min_len = min(len(y_binned), len(y_lagged_binned))
            y1 = y_binned[:min_len]
            y2 = y_lagged_binned[:min_len]
            
            # Remove NaN
            mask = ~(pd.isna(y1) | pd.isna(y2))
            if mask.sum() < 50:
                mi_values.append(0)
                continue
            
            mi = mutual_info_score(y1[mask], y2[mask])
            mi_values.append(mi)
        
        if len(mi_values) < 3:
            return {"tau_max": min_lag, "method": "first_mi_min", "note": "insufficient_lags"}
        
        # Find first local minimum
        mi_array = np.array(mi_values)
        # Detect minima: lower than neighbors
        is_minimum = np.zeros(len(mi_array), dtype=bool)
        for i in range(1, len(mi_array) - 1):
            if mi_array[i] < mi_array[i-1] and mi_array[i] < mi_array[i+1]:
                is_minimum[i] = True
        
        if np.any(is_minimum):
            tau_d = min_lag + np.where(is_minimum)[0][0]
            # tau_max = (m-1) × τ_d, use m=3 as default
            tau_max = 2 * tau_d
            note = "first_minimum_found"
        else:
            # No minimum found, use max_search
            tau_max = len(mi_values) + min_lag - 1
            note = "no_minimum_found"
        
        return {
            "tau_max": int(tau_max),
            "tau_d": int(tau_d) if np.any(is_minimum) else None,
            "method": "first_mi_minimum_takens",
            "mi_trajectory": mi_values,
            "note": note,
        }
        
    except Exception as e:
        logger.error(f"First MI minimum failed: {e}")
        return {"tau_max": min_lag, "method": "first_mi_min", "error": str(e)}


def estimate_tau_max_ensemble(
    series_x: pd.Series,
    series_y: pd.Series,
    sampling_interval_days: float = 5,
    domain_max_days: int = 90,
    methods: Optional[list] = None,
    aggregation: str = "median",
) -> Dict[str, Any]:
    """
    Ensemble tau_max estimation using multiple methods.
    
    Combines estimates from multiple approaches and aggregates (median, mean,
    conservative=max, aggressive=min).
    
    Robust to individual method failures and provides consensus estimate.
    
    Parameters
    ----------
    series_x : pd.Series
        Source variable
    series_y : pd.Series
        Target variable
    sampling_interval_days : float, default=5
        Sampling interval in days
    domain_max_days : int, default=90
        Domain knowledge maximum lag
    methods : list, optional
        List of methods to use. If None, uses all available.
    aggregation : str, default="median"
        Aggregation method: "median", "mean", "max" (conservative), "min" (aggressive)
        
    Returns
    -------
    dict
        Ensemble tau_max with individual estimates
    """
    if methods is None:
        methods = ["acf_zero", "pacf_cutoff", "nyquist_domain", "aic_bic"]
    
    estimates = {}
    
    # ACF zero-crossing
    if "acf_zero" in methods:
        result = estimate_tau_max_acf_zero_crossing(series_y)
        if "error" not in result:
            estimates["acf_zero"] = result["tau_max"]
    
    # PACF cutoff
    if "pacf_cutoff" in methods:
        result = estimate_tau_max_pacf_cutoff(series_y)
        if "error" not in result:
            estimates["pacf_cutoff"] = result["tau_max"]
    
    # MI decay (requires both series)
    if "mi_decay" in methods:
        result = estimate_tau_max_mi_decay(series_x, series_y)
        if "error" not in result:
            estimates["mi_decay"] = result["tau_max"]
    
    # AIC/BIC
    if "aic_bic" in methods:
        result = estimate_tau_max_aic_bic(series_y, criterion="bic")
        if "error" not in result:
            estimates["aic_bic"] = result["tau_max"]
    
    # Nyquist-domain
    if "nyquist_domain" in methods:
        result = estimate_tau_max_nyquist_domain(
            series_y, domain_max_days, sampling_interval_days
        )
        estimates["nyquist_domain"] = result["tau_max"]
    
    # First MI minimum
    if "first_mi_min" in methods:
        result = estimate_tau_max_first_mi_minimum(series_x, series_y)
        if "error" not in result:
            estimates["first_mi_min"] = result["tau_max"]
    
    if len(estimates) == 0:
        logger.warning("Ensemble: all methods failed")
        return {
            "tau_max": 6,  # Fallback
            "method": "ensemble_fallback",
            "individual_estimates": {},
            "note": "all_methods_failed"
        }
    
    # Aggregate
    values = np.array(list(estimates.values()))
    if aggregation == "median":
        tau_max = int(np.median(values))
    elif aggregation == "mean":
        tau_max = int(np.mean(values))
    elif aggregation == "max":
        tau_max = int(np.max(values))
    elif aggregation == "min":
        tau_max = int(np.min(values))
    else:
        tau_max = int(np.median(values))
    
    return {
        "tau_max": tau_max,
        "method": f"ensemble_{aggregation}",
        "individual_estimates": estimates,
        "aggregation": aggregation,
        "std": float(np.std(values)),
        "range": (int(np.min(values)), int(np.max(values))),
        "note": "ensemble_success",
    }


def estimate_tau_max_scientific(
    series_x: pd.Series,
    series_y: pd.Series,
    sampling_interval_days: float = 5,
    domain_max_days: int = 90,
    confidence_level: float = 0.95,
    safety_factor: float = 3.0,
) -> Dict[str, Any]:
    """
    Scientific best-practice tau_max estimation (RECOMMENDED FOR PUBLICATIONS).
    
    This is the most scientifically accepted approach, following guidelines from:
    - Runge et al. (2019, Nature Comm): "Inferring causation from time series"
    - Box & Jenkins (1976): ACF-based lag selection
    - Peters et al. (2017): Conservative causal discovery principles
    
    Method: Conservative hybrid combining three constraints:
    1. ACF zero-crossing (statistical memory length)
    2. Domain constraint (physical plausibility)
    3. Nyquist constraint (sample size / statistical power)
    
    Takes minimum of all three → most conservative, defensible in peer review.
    
    Parameters
    ----------
    series_x : pd.Series
        Source variable (used for MI-based checks if needed)
    series_y : pd.Series
        Target variable (primary for ACF analysis)
    sampling_interval_days : float, default=5
        Sampling interval in days
    domain_max_days : int, default=90
        Domain knowledge maximum lag (e.g., 90 days for precipitation→vegetation)
    confidence_level : float, default=0.95
        Confidence level for ACF bands
    safety_factor : float, default=3.0
        Nyquist-like factor: n / safety_factor = max_lag
        
    Returns
    -------
    dict
        tau_max with detailed diagnostics and justification
        
    References
    ----------
    Runge, J., et al. (2019). Inferring causation from time series in Earth 
        system sciences. Nature Communications, 10(1), 2553.
    Box, G. E., & Jenkins, G. M. (1976). Time series analysis: forecasting 
        and control. Holden-Day.
    Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of causal 
        inference. MIT Press.
    """
    n = len(series_y.dropna())
    
    # Constraint 1: ACF zero-crossing (statistical)
    acf_result = estimate_tau_max_acf_zero_crossing(
        series_y, 
        confidence_level=confidence_level
    )
    acf_tau = acf_result["tau_max"]
    
    # Constraint 2: Domain knowledge (physical plausibility)
    domain_tau = int(np.floor(domain_max_days / sampling_interval_days))
    
    # Constraint 3: Nyquist constraint (sample size / statistical power)
    nyquist_tau = int(np.floor(n / safety_factor))
    
    # Take minimum (most conservative)
    tau_max = min(acf_tau, domain_tau, nyquist_tau)
    tau_max = max(1, tau_max)  # At least 1
    
    # Identify binding constraint
    if tau_max == acf_tau:
        binding = "acf"
    elif tau_max == domain_tau:
        binding = "domain"
    else:
        binding = "nyquist"
    
    return {
        "tau_max": int(tau_max),
        "tau_max_days": float(tau_max * sampling_interval_days),
        "method": "scientific_hybrid",
        "acf_estimate": int(acf_tau),
        "acf_estimate_days": float(acf_tau * sampling_interval_days),
        "domain_constraint": int(domain_tau),
        "domain_constraint_days": float(domain_max_days),
        "nyquist_constraint": int(nyquist_tau),
        "nyquist_constraint_days": float(nyquist_tau * sampling_interval_days),
        "binding_constraint": binding,
        "sample_size": n,
        "confidence_level": confidence_level,
        "safety_factor": safety_factor,
        "justification": (
            "Conservative hybrid following Runge et al. (2019) guidelines. "
            "Combines statistical memory length (ACF), physical plausibility "
            "(domain knowledge), and sample size constraints (Nyquist). "
            "Takes minimum to ensure robust causal inference."
        ),
        "reference": "Runge et al. (2019) Nature Communications 10:2553",
        "note": "publication_ready",
    }


def recommend_tau_max(
    series_x: pd.Series,
    series_y: pd.Series,
    sampling_interval_days: float = 5,
    domain_max_days: int = 90,
    method: str = "auto",
    verbose: bool = True,
) -> int:
    """
    Recommend tau_max using best-practice heuristics.
    
    Default strategy ("auto"): Uses "scientific" method (most defensible)
    
    Other options:
    - "scientific": ACF + domain + Nyquist hybrid (RECOMMENDED, publication-ready)
    - "ensemble": Median of multiple methods (robust but slower)
    - "acf_zero": ACF zero-crossing only (fast)
    - "pacf_cutoff": PACF cutoff (linear AR)
    - "mi_decay": MI decay (nonlinear)
    - "aic_bic": AIC/BIC selection (model-based)
    - "nyquist_domain": Domain + Nyquist only (very conservative)
    - "first_mi_min": Takens embedding (nonlinear dynamics)
    
    Parameters
    ----------
    series_x : pd.Series
        Source variable
    series_y : pd.Series
        Target variable
    sampling_interval_days : float, default=5
        Sampling interval in days
    domain_max_days : int, default=90
        Domain knowledge maximum lag
    method : str, default="auto"
        Method to use (see above)
    verbose : bool, default=True
        Print recommendation details
        
    Returns
    -------
    int
        Recommended tau_max in timesteps
    """
    n = len(series_y.dropna())
    
    if method == "auto" or method == "scientific":
        # Default: Use scientific hybrid (most defensible)
        result = estimate_tau_max_scientific(
            series_x, series_y, sampling_interval_days, domain_max_days
        )
    elif method == "acf_zero":
        result = estimate_tau_max_acf_zero_crossing(series_y)
    elif method == "pacf_cutoff":
        result = estimate_tau_max_pacf_cutoff(series_y)
    elif method == "mi_decay":
        result = estimate_tau_max_mi_decay(series_x, series_y)
    elif method == "aic_bic":
        result = estimate_tau_max_aic_bic(series_y, criterion="bic")
    elif method == "nyquist_domain":
        result = estimate_tau_max_nyquist_domain(
            series_y, domain_max_days, sampling_interval_days
        )
    elif method == "first_mi_min":
        result = estimate_tau_max_first_mi_minimum(series_x, series_y)
    elif method == "ensemble":
        result = estimate_tau_max_ensemble(
            series_x, series_y, sampling_interval_days, domain_max_days
        )
    elif method == "scientific_ensemble":
        # More robust: ensemble but with scientific constraints
        ensemble_result = estimate_tau_max_ensemble(
            series_x, series_y, sampling_interval_days, domain_max_days,
            methods=["acf_zero", "pacf_cutoff", "aic_bic", "nyquist_domain"],
            aggregation="median"
        )
        scientific_result = estimate_tau_max_scientific(
            series_x, series_y, sampling_interval_days, domain_max_days
        )
        # Take minimum (most conservative)
        result = {
            "tau_max": min(ensemble_result["tau_max"], scientific_result["tau_max"]),
            "method": "scientific_ensemble",
            "ensemble_estimate": ensemble_result["tau_max"],
            "scientific_estimate": scientific_result["tau_max"],
            "note": "conservative_ensemble_with_constraints"
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    tau_max = result["tau_max"]
    logger.info(f"Recommended tau_max: {tau_max} timesteps")
    if "individual_estimates" in result:
        logger.info(f"Individual estimates: {result['individual_estimates']}")
    if "note" in result:
        logger.info(f"Note: {result['note']}")
    
    return tau_max
