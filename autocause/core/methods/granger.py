"""
Granger Causality Analysis

Implements linear causal discovery using Vector Autoregression (VAR) and
Granger causality tests. Includes lag selection, pre-whitening, and
rigorous statistical testing with multiple testing correction.

References:
    - Granger, C. W. J. (1969). "Investigating causal relations by econometric models"
    - Toda & Yamamoto (1995). "Statistical inference on VAR with possibly integrated processes"
    - Statsmodels: https://www.statsmodels.org/stable/grangercausalitytests.html
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_var_model(
    df: pd.DataFrame, maxlags: Optional[int] = None, ic: str = "aic"
) -> Tuple[object, int]:
    """
    Fit Vector Autoregression (VAR) model and select optimal lag length.

    Parameters:
        df (pd.DataFrame): Multivariate time series (columns are variables, rows are time)
        maxlags (int, optional): Maximum lag to consider. If None, uses automatic selection
        ic (str): Information criterion for lag selection ("aic", "bic", "fpe", "hq")

    Returns:
        Tuple[VARResults, int]: Fitted VAR model results and optimal lag length
    """
    try:
        model = VAR(df)

        if maxlags is None:
            # Use automatic lag selection
            lag_results = model.select_lags(maxlags=min(20, len(df) // 4))
            optimal_lag = lag_results.aic if ic == "aic" else lag_results.bic
            logger.info(
                f"Automatic lag selection: optimal lag = {optimal_lag} ({ic.upper()})"
            )
        else:
            optimal_lag = maxlags
            logger.info(f"Fixed lag length: {optimal_lag}")

        results = model.fit(optimal_lag)
        logger.info(
            f"VAR({optimal_lag}) model fitted successfully. AIC={results.aic:.2f}, BIC={results.bic:.2f}"
        )

        return results, optimal_lag

    except Exception as e:
        logger.error(f"Error fitting VAR model: {e}")
        raise


def test_stationarity_and_difference(
    series: pd.Series, method: str = "adf", alpha: float = 0.05
) -> Tuple[bool, pd.Series]:
    """
    Test stationarity of a series and difference if needed.

    Parameters:
        series (pd.Series): Time series to test
        method (str): Test method ("adf" for Augmented Dickey-Fuller, "kpss")
        alpha (float): Significance level

    Returns:
        Tuple[bool, pd.Series]: (is_stationary, series_or_differenced_series)
    """
    # Check for constant values (zero variance) - cannot be tested with ADF
    series_clean = series.dropna()
    if len(series_clean) > 0 and np.var(series_clean) == 0:
        logger.warning(
            f"{series.name}: Zero variance (constant) - treating as non-stationary"
        )
        return False, series.diff().dropna()

    try:
        if method == "adf":
            adf_result = adfuller(series_clean, autolag="AIC")
            p_value = adf_result[1]
            is_stationary = p_value < alpha
            test_name = "ADF"
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'adf'.")

        if is_stationary:
            logger.info(
                f"{series.name}: {test_name} p-value={p_value:.4f} → STATIONARY"
            )
            return True, series
        else:
            logger.info(
                f"{series.name}: {test_name} p-value={p_value:.4f} → NON-STATIONARY, differencing..."
            )
            differenced = series.diff().dropna()
            return False, differenced

    except Exception as e:
        logger.error(f"Stationarity test failed for {series.name}: {e}")
        return False, series.diff().dropna()


def prewhiten_series(series: pd.Series, max_lag: int = 5) -> pd.Series:
    """
    Pre-whiten a time series by removing autocorrelation via VAR residuals.

    Parameters:
        series (pd.Series): Time series to pre-whiten
        max_lag (int): Maximum lag for VAR model

    Returns:
        pd.Series: Pre-whitened (residuals of VAR model)
    """
    try:
        # Fit VAR(1) to remove autocorrelation
        df_temp = series.to_frame()
        model = VAR(df_temp)
        results = model.fit(min(max_lag, 3))
        residuals = pd.Series(
            results.resid.flatten(), index=series.index[max_lag:], name=series.name
        )
        logger.info(f"Pre-whitened {series.name} (lag={min(max_lag, 3)})")
        return residuals
    except Exception as e:
        logger.warning(
            f"Pre-whitening failed for {series.name}: {e}. Using original series."
        )
        return series


def run_granger_causality(
    df: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    maxlag: int = 12,
    prewhiten: bool = True,
    alpha: float = 0.05,
    sampling_days: float = 1.0,
    controls: List[str] = None,
    verbose: bool = True,
    prefer_shorter_lags: bool = True,
    p_tolerance_factor: float = 2.0,
    # Heuristic gates (default OFF for pure library behavior)
    min_partial_r2: float = 0.0,
    enable_partial_r2_gating: bool = False,
    require_whiteness: bool = False,
    stability_check: bool = False,
    stability_tolerance: int = 1,
    within_pair_alpha_factor: float = 1.0,
    skip_stationarity: bool = False,  # Skip stationarity tests (for speed in falsification)
) -> Dict:
    """
    Run Granger causality test using multivariate VAR with optional controls.

    Tests H0: cause does not Granger-cause effect (given controls if provided).
    Computes per-lag p-values using VARResults.test_causality, applies BH-FDR across lags,
    computes partial R² effect size, and Ljung-Box residual whiteness diagnostic.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        cause_var (str): Name of potential cause variable
        effect_var (str): Name of effect variable
        maxlag (int): Maximum lag to test (default: 12 timesteps)
        prewhiten (bool): Whether to pre-whiten series (deprecated when using multivariate VAR)
        alpha (float): Significance level
        sampling_days (float): Days per timestep for lag conversion
        controls (List[str]): Optional control variables for conditional causality
        verbose (bool): Print detailed results

    Returns:
        Dict: Results with p/q per lag, best lag, effect size (partial R²), residual diagnostics
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Granger Causality: {cause_var} → {effect_var} | {controls or 'none'}")
    logger.info(f"{'=' * 70}")

    # Prepare multivariate data including controls (optional)
    cols = [cause_var, effect_var]
    if controls:
        cols.extend([c for c in controls if c not in cols and c in df.columns])

    # Only copy if we'll modify (dropna or stationarity checks)
    # For skip_stationarity with clean data, this avoids unnecessary copy
    data = df[cols].dropna()
    if not data.index.is_unique or not skip_stationarity:
        data = data.copy()  # Only copy when needed for modifications

    # Handle duplicate indices (common with irregular sampling or synthetic data)
    if not data.index.is_unique:
        logger.warning(
            f"Duplicate indices detected ({data.index.duplicated().sum()} rows). Keeping first occurrence."
        )
        data = data[~data.index.duplicated(keep="first")]

    if len(data) < maxlag + 10:
        logger.warning(
            f"Insufficient data ({len(data)} obs) for lag {maxlag}. Reducing maxlag."
        )
        maxlag = max(1, len(data) // 5)

    # Check and ensure stationarity (all variables) - skip if requested for speed
    differenced_cols: List[str] = []
    transformed_data = {}
    if skip_stationarity:
        # Skip stationarity checks - use data as-is (for surrogate tests)
        for col in data.columns:
            transformed_data[col] = data[col]
    else:
        for col in data.columns:
            is_stat, transformed_series = test_stationarity_and_difference(data[col])
            transformed_data[col] = transformed_series
            if not is_stat:
                logger.warning(f"{col} required differencing for stationarity")
                differenced_cols.append(col)

    # Align all series to common index (in case differencing shortened some)
    # Find common index across all transformed series
    common_index = transformed_data[cols[0]].index
    for col in cols[1:]:
        common_index = common_index.intersection(transformed_data[col].index)

    # Rebuild dataframe with aligned data and reset index to avoid duplicates
    data = pd.DataFrame({col: transformed_data[col].loc[common_index] for col in cols})
    data = data.reset_index(
        drop=True
    )  # Reset to integer index to avoid duplicate datetime issues

    # Standardize for comparability
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data), index=data.index, columns=data.columns
    )

    # Test causality per lag using VARResults.test_causality
    try:
        # Test H0: all coefficients of cause_var at given lag are zero in effect_var equation
        rows = []
        p_values = []
        test_statistics = []

        for lag in range(1, maxlag + 1):
            try:
                # Fit VAR with this specific lag order
                var_model_lag = VAR(data_scaled)
                var_res_lag = var_model_lag.fit(lag)

                # test_causality tests if cause_var→effect_var (coefficients of cause in effect equation)
                # caused="effect_var" tests if cause_var Granger-causes effect_var
                test_res = var_res_lag.test_causality(
                    caused=effect_var, causing=cause_var, kind="f", signif=alpha
                )
                f_stat = float(test_res.test_statistic)
                p_val = float(test_res.pvalue)
            except Exception as e:
                logger.debug(f"Causality test failed at lag {lag}: {e}")
                f_stat, p_val = np.nan, np.nan

            p_values.append(p_val)
            test_statistics.append(f_stat)
            rows.append({"lag": lag, "p_value": p_val, "f_stat": f_stat})

        # Apply within-pair FDR across lags
        q_values = [np.nan] * len(rows)
        try:
            from ..multiple_testing import apply_fdr_to_dataframe

            tmp = pd.DataFrame(rows)
            tmp = apply_fdr_to_dataframe(tmp, p_col="p_value", alpha=alpha)
            q_values = tmp["q_value"].tolist()
        except Exception as e:
            logger.debug(f"FDR within pair failed: {e}")

        # Pick best lag: among significant lags, choose the one with MINIMUM p-value
        # (strongest statistical evidence of causality at that specific lag)
        # But strongly prefer shorter lags to avoid spurious high-lag detections
        best_lag = np.nan
        best_p_value = np.nan
        best_q_value = np.nan

        # Determine significance across lags using per-lag p-values (optionally relaxed).
        # Rationale: we apply global FDR across pairs later; within-pair FDR can be conservative.
        # Allow tightening (factor < 1.0) or relaxing (> 1.0) of within-pair per-lag alpha
        local_alpha = float(alpha) * float(within_pair_alpha_factor)
        sig_mask = [(not np.isnan(p)) and (p < local_alpha) for p in p_values]

        if any(sig_mask):
            # Among significant lags, choose by minimum p; optionally prefer shorter lag within tolerance
            sig_indices = [i for i, is_sig in enumerate(sig_mask) if is_sig]
            sig_pvals = [(i, p_values[i]) for i in sig_indices]
            # Strict minimum
            min_idx, min_p = min(sig_pvals, key=lambda x: np.nan_to_num(x[1], nan=1.0))

            if prefer_shorter_lags:
                # Find smallest lag whose p is within tolerance of the best p
                # Use more aggressive lag preference: accept higher p-value if lag is much shorter
                viable = []
                for i, p in sig_pvals:
                    lag_ratio = (i + 1) / (min_idx + 1) if min_idx > 0 else 1.0
                    # Accept if: p is close to best OR lag is significantly shorter
                    # Penalize higher lags with exponential penalty
                    adjusted_threshold = min_p * max(
                        1.0, p_tolerance_factor * (lag_ratio**0.5)
                    )
                    if (not np.isnan(p)) and (p <= adjusted_threshold):
                        viable.append((i, p))

                if viable:
                    # Choose smallest lag among viable
                    best_idx = min(viable, key=lambda x: x[0])[0]
                else:
                    best_idx = min_idx
            else:
                best_idx = min_idx

            best_lag = best_idx + 1
            best_p_value = p_values[best_idx]
            best_q_value = q_values[best_idx]

        # Initial causal flag from significance
        is_causal = bool(any(sig_mask))

        # Compute partial R² effect size: fit nested models and compare
        partial_r2 = np.nan
        try:
            if not np.isnan(best_lag) and best_lag >= 1:
                # Full model: effect ~ lags(effect) + lags(cause) + lags(controls)
                var_full = VAR(data_scaled).fit(int(best_lag))

                # Restricted model: effect ~ lags(effect) + lags(controls) only
                # (exclude cause_var temporarily to compute ΔR²)
                cols_restricted = [c for c in data_scaled.columns if c != cause_var]
                if len(cols_restricted) > 0:
                    var_restr = VAR(data_scaled[cols_restricted]).fit(int(best_lag))

                    # R² for effect_var equation in full model
                    effect_idx_full = list(var_full.names).index(effect_var)
                    r2_full = 1.0 - (
                        var_full.resid[:, effect_idx_full].var()
                        / data_scaled[effect_var].var()
                    )

                    # R² for effect_var in restricted (no cause)
                    effect_idx_restr = list(var_restr.names).index(effect_var)
                    r2_restr = 1.0 - (
                        var_restr.resid[:, effect_idx_restr].var()
                        / data_scaled[effect_var].var()
                    )

                    partial_r2 = float(max(0, r2_full - r2_restr))
        except Exception as eff_e:
            logger.debug(
                f"Could not compute partial R² for {cause_var}->{effect_var}: {eff_e}"
            )

        # Residual diagnostics: Ljung-Box on effect_var residuals
        diag_resid_whiteness = True
        ljungbox_p = np.nan
        try:
            if not np.isnan(best_lag) and best_lag >= 1:
                var_diag = VAR(data_scaled).fit(int(best_lag))
                effect_idx_diag = list(var_diag.names).index(effect_var)
                resid_effect = var_diag.resid[:, effect_idx_diag]

                # Ljung-Box test for no autocorrelation up to lag=min(10, T/5)
                from statsmodels.stats.diagnostic import acorr_ljungbox

                lb_res = acorr_ljungbox(
                    resid_effect, lags=min(10, len(resid_effect) // 5), return_df=False
                )
                # lb_res is (lb_stat, p_values); take max p-value (least significant)
                ljungbox_p = float(np.max(lb_res[1])) if len(lb_res[1]) > 0 else np.nan
                diag_resid_whiteness = (
                    ljungbox_p > 0.05
                )  # fail to reject null → white noise
        except Exception as diag_e:
            logger.debug(f"Ljung-Box diagnostic failed: {diag_e}")

        # Optional split-half stability check to suppress spurious edges
        def _half_stability_ok() -> bool:
            if not stability_check:
                return True
            try:
                if np.isnan(best_lag) or best_lag < 1:
                    return False
                # Build contiguous halves
                n = len(data_scaled)
                if n < (2 * (int(best_lag) + 10)):
                    # Not enough data for a robust split-half test
                    return True
                mid = n // 2

                def _is_sig_on_segment(seg_df: pd.DataFrame) -> bool:
                    # Try within a small neighborhood of best_lag
                    for lag in range(
                        max(1, int(best_lag) - stability_tolerance),
                        int(best_lag) + stability_tolerance + 1,
                    ):
                        try:
                            var_model = VAR(seg_df)
                            var_res = var_model.fit(lag)
                            t_res = var_res.test_causality(
                                caused=effect_var,
                                causing=cause_var,
                                kind="f",
                                signif=alpha,
                            )
                            pval = float(t_res.pvalue)
                            if pval < alpha:
                                return True
                        except Exception:
                            continue
                    return False

                seg1 = data_scaled.iloc[:mid, :]
                seg2 = data_scaled.iloc[mid:, :]
                return _is_sig_on_segment(seg1) and _is_sig_on_segment(seg2)
            except Exception:
                return True

        # Apply gating criteria to improve precision
        if is_causal:
            if require_whiteness and not diag_resid_whiteness:
                is_causal = False
            if (
                enable_partial_r2_gating
                and (not np.isnan(min_partial_r2))
                and (min_partial_r2 > 0)
            ):
                try:
                    is_causal = is_causal and (
                        float(partial_r2) >= float(min_partial_r2)
                    )
                except Exception:
                    pass
            # Stability gating
            if is_causal and not _half_stability_ok():
                is_causal = False

        # Adjust reported lag if differencing was applied to either variable
        reported_best_lag = best_lag
        try:
            if not np.isnan(best_lag):
                if (cause_var in differenced_cols) or (effect_var in differenced_cols):
                    reported_best_lag = max(1, int(best_lag) - 1)
        except Exception:
            pass

        result = {
            "cause": cause_var,
            "effect": effect_var,
            "maxlag": maxlag,
            "maxlag_days": maxlag * sampling_days,
            "p_values": p_values,
            "q_values": q_values,
            "test_statistics": test_statistics,
            "best_lag": reported_best_lag,
            "best_lag_days": (reported_best_lag * sampling_days)
            if not np.isnan(reported_best_lag)
            else np.nan,
            "best_p_value": best_p_value,
            "best_q_value": best_q_value,
            "is_causal": is_causal,
            "partial_r2": partial_r2,
            "diag_resid_whiteness": diag_resid_whiteness,
            "ljungbox_p": ljungbox_p,
            "alpha": alpha,
            "n_observations": len(data_scaled),
            "method": "Granger",
            "controls": controls or [],
            "cadence": f"{sampling_days}-day cadence",
            "cadence_note": "Lags reported both in steps and days",
            "best_lag_raw": best_lag,
            "differenced_vars": differenced_cols,
        }

        if verbose:
            logger.info(
                f"Best lag: {reported_best_lag} steps ({reported_best_lag * sampling_days:.1f} days), p={best_p_value:.6f}, q={best_q_value:.6f}"
            )
            logger.info(
                f"Causality: {is_causal}, Partial R²={partial_r2:.4f}, Resid whiteness={diag_resid_whiteness}"
            )

        return result

    except Exception as e:
        logger.error(f"Granger causality test failed: {e}")
        return {
            "cause": cause_var,
            "effect": effect_var,
            "error": str(e),
            "is_causal": False,
            "method": "Granger",
        }


def run_bidirectional_granger(
    df: pd.DataFrame, var1: str, var2: str, maxlag: int = 12, alpha: float = 0.05
) -> Dict:
    """
    Run bidirectional Granger causality test (var1 → var2 AND var2 → var1).

    Parameters:
        df (pd.DataFrame): Multivariate time series
        var1 (str): First variable
        var2 (str): Second variable
        maxlag (int): Maximum lag
        alpha (float): Significance level

    Returns:
        Dict: Bidirectional causality results
    """
    logger.info(f"\nTesting bidirectional causality: {var1} ↔ {var2}")

    result_12 = run_granger_causality(
        df, var1, var2, maxlag, alpha=alpha, verbose=False
    )
    result_21 = run_granger_causality(
        df, var2, var1, maxlag, alpha=alpha, verbose=False
    )

    return {
        f"{var1}_to_{var2}": result_12,
        f"{var2}_to_{var1}": result_21,
        "bidirectional_coupling": result_12.get("is_causal", False)
        and result_21.get("is_causal", False),
    }


def batch_granger_causality(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    maxlag: int = 12,
    alpha: float = 0.05,
    sampling_days: float = 1.0,
    # Heuristic post-processing flags (default OFF)
    enable_mediator_pruning: bool = False,
    enable_transitive_reduction: bool = False,
    within_pair_alpha_factor: float = 1.0,
    enable_partial_r2_gating: bool = False,
    min_partial_r2: float = 0.0,
    require_whiteness: bool = False,
    stability_check: bool = False,
    # Conditional Granger options
    use_all_controls: bool = True,
    control_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run Granger causality tests for multiple variable pairs.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        variable_pairs (List[Tuple[str, str]]): List of (cause, effect) pairs
        maxlag (int): Maximum lag
        alpha (float): Significance level

    Returns:
        pd.DataFrame: Results for all pairs
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Running Granger causality for {len(variable_pairs)} pairs...")
    logger.info(f"{'=' * 70}")

    results = []

    # Determine candidate control columns once, if needed
    control_candidates: List[str] = []
    if use_all_controls or (control_columns is not None):
        if control_columns is not None:
            control_candidates = [c for c in control_columns if c in df.columns]
        else:
            # Default: all numeric columns (exclude time/unit identifiers)
            exclude_cols = {"time", "unit_id"}
            control_candidates = [
                c
                for c in df.select_dtypes(include=["number"]).columns
                if c not in exclude_cols
            ]
            # Only use controls when the system is small enough for stable VAR estimation.
            # With N variables and L lags the VAR has N²L parameters; when N > 10 the
            # model becomes unstable unless T is very large.
            if len(control_candidates) > 10:
                logger.info(
                    f"Disabling conditional Granger: {len(control_candidates)} variables "
                    f"exceeds the 10-variable limit for stable multivariate VAR"
                )
                control_candidates = []
    for cause, effect in variable_pairs:
        try:
            # Choose controls for this pair if requested
            controls_for_pair: Optional[List[str]] = None
            if control_candidates:
                controls_for_pair = [
                    c for c in control_candidates if c not in (cause, effect)
                ]

            result = run_granger_causality(
                df,
                cause,
                effect,
                maxlag,
                alpha=alpha,
                sampling_days=sampling_days,
                verbose=False,
                within_pair_alpha_factor=within_pair_alpha_factor,
                enable_partial_r2_gating=enable_partial_r2_gating,
                min_partial_r2=min_partial_r2,
                require_whiteness=require_whiteness,
                stability_check=stability_check,
                controls=controls_for_pair,
            )
            results.append(
                {
                    "cause": result.get("cause", cause),
                    "effect": result.get("effect", effect),
                    "best_lag": result.get("best_lag", np.nan),
                    "best_lag_days": result.get("best_lag_days", np.nan),
                    "best_p_value": result.get("best_p_value", np.nan),
                    "is_causal": result.get("is_causal", False),
                    "granger_beta_std": result.get("granger_beta_std", np.nan),
                    "n_obs": result.get("n_observations", np.nan),
                }
            )
        except Exception as e:
            logger.error(f"Failed for {cause} → {effect}: {e}")
            results.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "best_lag": np.nan,
                    "best_p_value": np.nan,
                    "is_causal": False,
                    "granger_beta_std": np.nan,
                    "n_obs": np.nan,
                }
            )

    results_df = pd.DataFrame(results)

    # Apply global FDR correction across ALL pairs using Benjamini-Hochberg
    try:
        from ..multiple_testing import apply_fdr_to_dataframe

        logger.info("Applying FDR correction across all pairs...")
        results_df = apply_fdr_to_dataframe(
            results_df, p_col="best_p_value", alpha=alpha
        )

        # Update is_causal based on q-values (FDR-corrected)
        if "q_value" in results_df.columns:
            results_df["is_causal"] = results_df["q_value"] < alpha
            logger.info(
                f"FDR correction: {results_df['is_causal'].sum()} relationships remain significant after correction"
            )
    except Exception as e:
        logger.warning(f"Global FDR correction failed: {e}")

    # Optional mediator (backdoor) check to prune indirect edges (opt-in)
    try:
        if enable_mediator_pruning and not results_df.empty:
            kept_indices = []
            cols_all = list(df.columns)
            # Snapshot of initially significant edges to propose mediators
            initial_sig_mask = results_df.get(
                "is_causal",
                pd.Series([False] * len(results_df), index=results_df.index),
            )
            initial_sig = results_df[initial_sig_mask]
            initial_edge_set = set(
                (r.get("cause", r.get("source")), r.get("effect", r.get("target")))
                for _, r in initial_sig.iterrows()
            )
            for idx, row in results_df.iterrows():
                # Only consider edges currently marked significant
                if not bool(row.get("is_causal", False)):
                    continue
                cause = row.get("cause") or row.get("source")
                effect = row.get("effect") or row.get("target")
                if cause not in cols_all or effect not in cols_all:
                    kept_indices.append(idx)
                    continue

                # Mediator pruning: only consider m where cause→m and m→effect are both detected
                mediators = [
                    m
                    for m in cols_all
                    if m not in (cause, effect)
                    and (cause, m) in initial_edge_set
                    and (m, effect) in initial_edge_set
                ]
                remains_significant = True
                if mediators:
                    for m in mediators:
                        try:
                            retest = run_granger_causality(
                                df,
                                cause,
                                effect,
                                maxlag=maxlag,
                                alpha=alpha,
                                controls=[m],
                                verbose=False,
                            )
                            if not retest.get("is_causal", False):
                                remains_significant = False
                                break
                        except Exception as _:
                            # If retest fails, keep original edge
                            pass
                # Additional robustness pruning for long-lag edges: test with any single control
                try:
                    if remains_significant:
                        best_l = int(row.get("best_lag", row.get("lag", np.nan)))
                        # Define a conservative threshold for "long" lags
                        long_lag_threshold = max(6, int(maxlag * 0.5))
                        if not np.isnan(best_l) and best_l >= long_lag_threshold:
                            for m in cols_all:
                                if m in (cause, effect):
                                    continue
                                try:
                                    retest_any = run_granger_causality(
                                        df,
                                        cause,
                                        effect,
                                        maxlag=maxlag,
                                        alpha=alpha,
                                        controls=[m],
                                        verbose=False,
                                    )
                                    if not retest_any.get("is_causal", False):
                                        remains_significant = False
                                        break
                                except Exception:
                                    continue
                except Exception:
                    pass
                if remains_significant:
                    kept_indices.append(idx)

            if kept_indices:
                results_df = results_df.loc[kept_indices].copy()

            # Recompute FDR on kept edges only
            try:
                results_df = apply_fdr_to_dataframe(
                    results_df, p_col="best_p_value", alpha=alpha
                )
                if "q_value" in results_df.columns:
                    results_df["is_causal"] = results_df["q_value"] < alpha
            except Exception:
                pass

            # Transitive reduction heuristic: drop edges explained by a two-step path (opt-in)
            try:
                sig_mask = results_df.get(
                    "is_causal",
                    pd.Series([False] * len(results_df), index=results_df.index),
                )
                sig_edges = results_df[sig_mask]
                edges_set = set(
                    (r["cause"], r["effect"]) for _, r in sig_edges.iterrows()
                )
                drop_pairs = set()
                for u, v in list(edges_set):
                    for x, y in list(edges_set):
                        if x == u and (x, y) != (u, v):
                            m = y
                            if (m, v) in edges_set:
                                drop_pairs.add((u, v))
                                break
                if enable_transitive_reduction and drop_pairs:
                    mask = [
                        (row["cause"], row["effect"]) not in drop_pairs
                        for _, row in results_df.iterrows()
                    ]
                    results_df = results_df.loc[mask].copy()
            except Exception as _:
                pass
    except Exception as e:
        logger.debug(f"Mediator pruning skipped due to error: {e}")

    # Sort by p-value (most significant first)
    if not results_df.empty and "best_p_value" in results_df.columns:
        results_df = results_df.sort_values("best_p_value")

    logger.info(
        f"\nCompleted. Found {results_df['is_causal'].sum()} significant causal relationships (α={alpha})"
    )

    return results_df
