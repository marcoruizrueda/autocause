#!/usr/bin/env python3
"""
Centralized Preprocessing Module for Causal Time Series Analysis

This module provides a comprehensive preprocessing pipeline ensuring all time series
are cleaned, stationary, normalized, and ready for causal discovery methods.

Key Features:
1. Advanced missing data interpolation (Gaussian Process + linear fallback)
2. Stationarity testing (ADF + KPSS dual tests)
3. Automatic differencing when non-stationary
4. Seasonal decomposition (STL) for detrending
5. HP filter for trend removal
6. Outlier detection and handling
7. Z-score normalization
8. Comprehensive transformation reporting

The module is dataset-agnostic and works with any pandas DataFrame containing
time series data.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
import json

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingReport:
    """Report of all preprocessing transformations applied"""

    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    variables: List[str]
    transformations: Dict[str, Dict] = field(default_factory=dict)
    stationarity_tests: Dict[str, Dict] = field(default_factory=dict)
    outliers_detected: Dict[str, int] = field(default_factory=dict)
    missing_data: Dict[str, Dict] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "variables": self.variables,
            "transformations": self.transformations,
            "stationarity_tests": self.stationarity_tests,
            "outliers_detected": self.outliers_detected,
            "missing_data": self.missing_data,
            "quality_flags": self.quality_flags,
        }

    def save(self, filepath: Path):
        """Save report to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Preprocessing report saved to {filepath}")


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessing pipeline for causal time series analysis.

    This class ensures all series are:
    - Clean (interpolated missing values, outliers handled)
    - Stationary (tested and transformed if needed)
    - Normalized (z-score standardization)
    - Detrended (optional: remove seasonality or HP trend)

    All operations are logged in a detailed PreprocessingReport.
    """

    def __init__(
        self,
        interpolation_method: Literal["gp", "linear", "spline"] = "gp",
        max_missing_frac: float = 0.2,
        min_valid_obs: int = 60,
        stationarity_test: Literal["adf", "kpss", "both"] = "both",
        stationarity_alpha: float = 0.05,
        normalize: bool = True,
        remove_seasonality: bool = False,
        seasonal_period: Optional[int] = None,
        hp_filter_lambda: Optional[float] = None,
        outlier_method: Literal["iqr", "zscore", "none"] = "iqr",
        outlier_threshold: float = 3.0,
        min_length: int = 30,
    ):
        """
        Initialize preprocessor.

        Parameters:
            interpolation_method: Method for missing data ('gp', 'linear', 'spline')
            max_missing_frac: Max fraction of missing data before rejection (0.2 = 20%)
                             For time series data or dense observations only.
            min_valid_obs: Minimum number of valid observations for panel data with
                          irregular sampling (e.g., satellite data). Default 60.
            stationarity_test: Which test to use ('adf', 'kpss', or 'both')
            stationarity_alpha: Significance level for stationarity tests
            normalize: Whether to apply z-score normalization
            remove_seasonality: Whether to remove seasonal component (STL)
            seasonal_period: Period for STL decomposition (auto-detect if None)
            hp_filter_lambda: Lambda for HP filter (None = no HP filtering)
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'none')
            outlier_threshold: Threshold for outlier detection
            min_length: Minimum series length after preprocessing
        """
        self.interpolation_method = interpolation_method
        self.max_missing_frac = max_missing_frac
        self.min_valid_obs = min_valid_obs
        self.stationarity_test = stationarity_test
        self.stationarity_alpha = stationarity_alpha
        self.normalize = normalize
        self.remove_seasonality = remove_seasonality
        self.seasonal_period = seasonal_period
        self.hp_filter_lambda = hp_filter_lambda
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.min_length = min_length

        self.report = None

    def preprocess(
        self,
        data: pd.DataFrame,
        metadata_cols: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, PreprocessingReport]:
        """
        Apply full preprocessing pipeline.

        Pipeline stages:
        1. Separate metadata from time series
        2. Check missing data fractions
        3. Interpolate missing values
        4. Detect and handle outliers
        5. Remove seasonality (if requested)
        6. Apply HP filter (if requested)
        7. Test and ensure stationarity
        8. Normalize (if requested)
        9. Recombine with metadata

        Parameters:
            data: Input DataFrame (can contain mixed data types)
            metadata_cols: List of columns to exclude from processing
            verbose: Whether to log detailed progress

        Returns:
            Tuple of (processed_df, PreprocessingReport)
        """
        if verbose:
            logger.info("=" * 70)
            logger.info("PREPROCESSING PIPELINE")
            logger.info("=" * 70)

        # Initialize report
        self.report = PreprocessingReport(
            original_shape=data.shape,
            final_shape=data.shape,  # Updated at end
            variables=[],
        )

        # Step 1: Separate metadata from time series
        if metadata_cols is None:
            metadata_cols = []

        metadata_cols = [c for c in metadata_cols if c in data.columns]

        if metadata_cols:
            metadata = data[metadata_cols].copy()
            ts_data = data.drop(columns=metadata_cols)
            if verbose:
                logger.info(f"Metadata columns: {metadata_cols}")
        else:
            metadata = None
            ts_data = data.copy()

        # Get numeric columns only
        numeric_cols = ts_data.select_dtypes(include=[np.number]).columns.tolist()
        self.report.variables = numeric_cols

        if verbose:
            logger.info(
                f"Processing {len(numeric_cols)} numeric variables: {numeric_cols}"
            )

        ts_data = ts_data[numeric_cols]

        # Step 2: Check missing data fractions
        # For panel data, check per-unit coverage; for time series, check global coverage
        if verbose:
            logger.info("\n[Stage 1/7] Checking missing data...")

        # Detect if this is panel data (has unit_id in metadata)
        is_panel = metadata is not None and "unit_id" in metadata.columns

        cols_to_drop = []

        for col in numeric_cols:
            n_missing = ts_data[col].isna().sum()
            frac_missing = n_missing / len(ts_data)

            # For panel data, check per-unit coverage
            if is_panel:
                # Check median valid observation count across units (vectorized)
                temp_df = pd.DataFrame(
                    {"unit_id": metadata["unit_id"], "valid": ts_data[col].notna()}
                )
                unit_valid_counts = temp_df.groupby("unit_id")["valid"].sum()
                median_valid_obs = unit_valid_counts.median()

                # Also track coverage percentage for reporting
                unit_coverage = temp_df.groupby("unit_id")["valid"].mean()
                median_coverage = unit_coverage.median()
                median_missing = 1 - median_coverage

                self.report.missing_data[col] = {
                    "n_missing": int(n_missing),
                    "frac_missing_global": float(frac_missing),
                    "frac_missing_per_unit_median": float(median_missing),
                    "median_valid_obs": int(median_valid_obs),
                    "accepted": median_valid_obs >= self.min_valid_obs,
                    "is_panel_data": True,
                }

                if median_valid_obs < self.min_valid_obs:
                    self.report.quality_flags.append(
                        f"{col}: {median_valid_obs:.0f} median valid obs (< {self.min_valid_obs} required) - DROPPED"
                    )
                    if verbose:
                        logger.warning(
                            f"  {col}: {median_valid_obs:.0f} median valid obs ({median_coverage:.1%} coverage) - DROPPING (need {self.min_valid_obs})"
                        )
                    cols_to_drop.append(col)
                elif verbose and n_missing > 0:
                    logger.info(
                        f"  {col}: {median_valid_obs:.0f} median valid obs ({median_coverage:.1%} coverage) ✓"
                    )
            else:
                # For simple time series, use global check
                self.report.missing_data[col] = {
                    "n_missing": int(n_missing),
                    "frac_missing": float(frac_missing),
                    "accepted": frac_missing <= self.max_missing_frac,
                    "is_panel_data": False,
                }

                if frac_missing > self.max_missing_frac:
                    self.report.quality_flags.append(
                        f"{col}: {frac_missing:.1%} missing (exceeds {self.max_missing_frac:.1%} threshold) - DROPPED"
                    )
                    if verbose:
                        logger.warning(
                            f"  {col}: {frac_missing:.1%} missing (exceeds threshold) - DROPPING"
                        )
                    cols_to_drop.append(col)
                elif verbose and n_missing > 0:
                    logger.info(f"  {col}: {frac_missing:.1%} missing")
        if cols_to_drop:
            if verbose:
                logger.warning(
                    f"\n  Dropping {len(cols_to_drop)} column(s) with excessive missing data: {cols_to_drop}"
                )
            ts_data = ts_data.drop(columns=cols_to_drop)
            # Update numeric_cols list
            numeric_cols = [c for c in numeric_cols if c not in cols_to_drop]

            # Check if we have enough variables left
            if len(numeric_cols) < 2:
                error_msg = (
                    f"After dropping columns with >{self.max_missing_frac:.0%} missing data, "
                    f"only {len(numeric_cols)} variable(s) remain. Need at least 2 for causal analysis."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if verbose:
                logger.info(f"  Remaining variables for analysis: {numeric_cols}")

        # Step 3: Interpolate missing values (only for remaining columns)
        if verbose:
            logger.info("\n[Stage 2/7] Interpolating missing values...")

        ts_data = self._interpolate_missing(ts_data, verbose=verbose)

        # Step 4: Detect and handle outliers
        if self.outlier_method != "none":
            if verbose:
                logger.info(
                    f"\n[Stage 3/7] Detecting outliers (method={self.outlier_method})..."
                )

            ts_data = self._detect_and_handle_outliers(ts_data, verbose=verbose)
        else:
            if verbose:
                logger.info("\n[Stage 3/7] Skipping outlier detection")

        # Step 5: Remove seasonality
        if self.remove_seasonality:
            if verbose:
                logger.info("\n[Stage 4/7] Removing seasonal components...")

            ts_data = self._remove_seasonality(ts_data, verbose=verbose)
        else:
            if verbose:
                logger.info("\n[Stage 4/7] Skipping seasonal decomposition")

        # Step 6: Apply HP filter
        if self.hp_filter_lambda is not None:
            if verbose:
                logger.info(
                    f"\n[Stage 5/7] Applying HP filter (lambda={self.hp_filter_lambda})..."
                )

            ts_data = self._apply_hp_filter(ts_data, verbose=verbose)
        else:
            if verbose:
                logger.info("\n[Stage 5/7] Skipping HP filter")

        # Step 7: Ensure stationarity
        # For panel data, skip stationarity testing (will be tested on aggregated data later)
        # Testing on mixed panel data is computationally expensive and conceptually incorrect
        if is_panel:
            if verbose:
                logger.info(
                    f"\n[Stage 6/7] Skipping stationarity testing (panel data - will test on aggregated data)"
                )
            # Mark in report that stationarity was skipped for panel data
            for col in ts_data.columns:
                self.report.stationarity_tests[col] = {
                    "skipped": True,
                    "reason": "panel_data",
                    "note": "Stationarity will be tested on aggregated data during causal discovery",
                }
        else:
            if verbose:
                logger.info(
                    f"\n[Stage 6/7] Testing stationarity (method={self.stationarity_test})..."
                )
            ts_data = self._ensure_stationarity(ts_data, verbose=verbose)

        # Step 8: Normalize
        if self.normalize:
            if verbose:
                logger.info("\n[Stage 7/7] Normalizing (z-score)...")

            ts_data = self._normalize(ts_data, verbose=verbose)
        else:
            if verbose:
                logger.info("\n[Stage 7/7] Skipping normalization")

        # Step 9: Recombine with metadata
        if metadata is not None:
            # Align indices (in case some rows were dropped)
            metadata = metadata.loc[ts_data.index]
            result = pd.concat([metadata, ts_data], axis=1)
        else:
            result = ts_data

        # Update final shape
        self.report.final_shape = result.shape

        # Check minimum length
        if len(result) < self.min_length:
            self.report.quality_flags.append(
                f"Final length {len(result)} < minimum {self.min_length}"
            )

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("PREPROCESSING COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Original shape: {self.report.original_shape}")
            logger.info(f"Final shape: {self.report.final_shape}")
            logger.info(f"Quality flags: {len(self.report.quality_flags)}")
            for flag in self.report.quality_flags:
                logger.warning(f"  ⚠️  {flag}")

        return result, self.report

    def _interpolate_missing(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Interpolate missing values using configured method"""

        for col in data.columns:
            if data[col].isna().sum() == 0:
                continue

            missing_frac = data[col].isna().sum() / len(data)

            if self.interpolation_method == "gp" and missing_frac <= 0.2:
                # Gaussian Process interpolation (robust for small gaps)
                try:
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

                    mask_valid = ~data[col].isna()
                    X_train = np.arange(len(data))[mask_valid].reshape(-1, 1)
                    y_train = data.loc[mask_valid, col].values

                    kernel = RBF() + WhiteKernel()
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        n_restarts_optimizer=3,
                        random_state=42,
                    )
                    gp.fit(X_train, y_train)

                    # Predict missing values
                    X_missing = np.arange(len(data))[~mask_valid].reshape(-1, 1)
                    if len(X_missing) > 0:
                        y_pred = gp.predict(X_missing)
                        data.loc[~mask_valid, col] = y_pred

                        self.report.transformations[col] = (
                            self.report.transformations.get(col, {})
                        )
                        self.report.transformations[col]["interpolation"] = "gp"

                        if verbose:
                            logger.info(
                                f"  {col}: GP interpolation ({len(X_missing)} points)"
                            )
                except Exception as e:
                    logger.warning(f"  {col}: GP failed ({e}), falling back to linear")
                    data[col] = data[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                    self.report.transformations[col] = self.report.transformations.get(
                        col, {}
                    )
                    self.report.transformations[col]["interpolation"] = (
                        "linear_fallback"
                    )

            elif self.interpolation_method == "spline":
                # Cubic spline interpolation
                data[col] = data[col].interpolate(
                    method="cubic", limit_direction="both"
                )
                self.report.transformations[col] = self.report.transformations.get(
                    col, {}
                )
                self.report.transformations[col]["interpolation"] = "spline"

                if verbose:
                    logger.info(f"  {col}: Spline interpolation")

            else:
                # Linear interpolation (default/fallback)
                data[col] = data[col].interpolate(
                    method="linear", limit_direction="both"
                )
                self.report.transformations[col] = self.report.transformations.get(
                    col, {}
                )
                self.report.transformations[col]["interpolation"] = "linear"

                if verbose:
                    logger.info(f"  {col}: Linear interpolation")

        return data

    def _detect_and_handle_outliers(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Detect and handle outliers using configured method"""

        for col in data.columns:
            if self.outlier_method == "iqr":
                # IQR method
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.outlier_threshold * IQR
                upper = Q3 + self.outlier_threshold * IQR

                outlier_mask = (data[col] < lower) | (data[col] > upper)

            elif self.outlier_method == "zscore":
                # Z-score method
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outlier_mask = z_scores > self.outlier_threshold

            else:
                outlier_mask = pd.Series(False, index=data.index)

            n_outliers = int(outlier_mask.sum())

            if n_outliers > 0:
                # Clip outliers to 1st and 99th percentiles
                lower = data[col].quantile(0.01)
                upper = data[col].quantile(0.99)
                # Ensure dtype compatibility by casting clipped values to original dtype
                original_dtype = data[col].dtype
                clipped_values = data.loc[outlier_mask, col].clip(lower, upper)
                data.loc[outlier_mask, col] = clipped_values.astype(original_dtype)

                self.report.outliers_detected[col] = n_outliers
                self.report.transformations[col] = self.report.transformations.get(
                    col, {}
                )
                self.report.transformations[col]["outliers_clipped"] = n_outliers

                if verbose:
                    logger.info(f"  {col}: {n_outliers} outliers clipped")

        return data

    def _remove_seasonality(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Remove seasonal component using STL decomposition"""

        for col in data.columns:
            try:
                # Auto-detect period if not provided
                if self.seasonal_period is None:
                    # Use autocorrelation to detect period
                    from statsmodels.tsa.stattools import acf

                    acf_vals = acf(data[col].dropna(), nlags=min(100, len(data) // 2))
                    # Find first peak after lag 2
                    peaks = []
                    for i in range(2, len(acf_vals) - 1):
                        if (
                            acf_vals[i] > acf_vals[i - 1]
                            and acf_vals[i] > acf_vals[i + 1]
                        ):
                            peaks.append((i, acf_vals[i]))

                    if peaks:
                        period = max(peaks, key=lambda x: x[1])[0]
                    else:
                        period = min(12, len(data) // 4)  # Default guess
                else:
                    period = self.seasonal_period

                # Require at least 2 full periods
                if len(data) >= 2 * period:
                    stl = STL(data[col], period=period, seasonal=13)
                    result = stl.fit()

                    # Keep trend + residual, remove seasonal
                    data[col] = result.trend + result.resid

                    self.report.transformations[col] = self.report.transformations.get(
                        col, {}
                    )
                    self.report.transformations[col]["seasonal_removed"] = True
                    self.report.transformations[col]["seasonal_period"] = period

                    if verbose:
                        logger.info(
                            f"  {col}: Seasonal component removed (period={period})"
                        )
                else:
                    if verbose:
                        logger.warning(f"  {col}: Too short for seasonal decomposition")

            except Exception as e:
                logger.warning(f"  {col}: Seasonal decomposition failed ({e})")

        return data

    def _apply_hp_filter(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Apply Hodrick-Prescott filter to remove trend"""

        from statsmodels.tsa.filters.hp_filter import hpfilter

        for col in data.columns:
            try:
                cycle, trend = hpfilter(data[col], lamb=self.hp_filter_lambda)

                # Keep cycle (detrended series)
                data[col] = cycle

                self.report.transformations[col] = self.report.transformations.get(
                    col, {}
                )
                self.report.transformations[col]["hp_filter_applied"] = True
                self.report.transformations[col]["hp_lambda"] = self.hp_filter_lambda

                if verbose:
                    logger.info(
                        f"  {col}: HP filter applied (lambda={self.hp_filter_lambda})"
                    )

            except Exception as e:
                logger.warning(f"  {col}: HP filter failed ({e})")

        return data

    def _ensure_stationarity(
        self, data: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Test stationarity and apply differencing if needed.

        WARNING: First-order differencing changes the causal semantics.
        Differenced data tests whether *changes* in X cause *changes* in Y,
        not whether levels of X cause levels of Y. Users should verify whether
        level or change causality is appropriate for their domain (Lütkepohl, 2005).
        """
        differenced_vars = []

        for col in data.columns:
            is_stationary = self._test_stationarity(data[col], verbose=verbose)

            if not is_stationary:
                # Apply first-order differencing
                data[col] = data[col].diff()

                self.report.transformations[col] = self.report.transformations.get(
                    col, {}
                )
                self.report.transformations[col]["differenced"] = 1

                differenced_vars.append(col)

                if verbose:
                    logger.info(f"  {col}: First-order differencing applied")

                # Re-test after differencing
                is_stationary_after = self._test_stationarity(
                    data[col].dropna(), verbose=False
                )
                self.report.stationarity_tests[col]["stationary_after_diff"] = (
                    is_stationary_after
                )

        # Drop NaN from differencing
        data = data.dropna()

        # Emit warning about causal semantics change
        if differenced_vars:
            warning_msg = (
                f"Differencing applied to {len(differenced_vars)} variable(s): "
                f"{differenced_vars}. Causal interpretation changes from "
                f"'levels cause levels' to 'changes cause changes'. "
                f"Verify this is appropriate for your domain."
            )
            logger.warning(f"  ⚠️  CAUSAL SEMANTICS: {warning_msg}")
            self.report.quality_flags.append(f"DIFFERENCING_SEMANTICS: {warning_msg}")

        return data

    def _test_stationarity(self, series: pd.Series, verbose: bool = False) -> bool:
        """
        Test stationarity using ADF and/or KPSS.

        Returns True if series is stationary according to configured test(s).
        """
        series_clean = series.dropna()

        if len(series_clean) < 20:
            # Too short for reliable testing
            return True

        results = {}
        is_stationary = False

        if self.stationarity_test in ["adf", "both"]:
            # ADF test (null hypothesis: non-stationary)
            try:
                adf_stat, adf_pval, _, _, _, _ = adfuller(series_clean, autolag="AIC")
                is_stationary_adf = adf_pval < self.stationarity_alpha
                results["adf_statistic"] = float(adf_stat)
                results["adf_pvalue"] = float(adf_pval)
                results["adf_stationary"] = is_stationary_adf

                if self.stationarity_test == "adf":
                    is_stationary = is_stationary_adf
            except Exception as e:
                logger.debug(f"ADF test failed: {e}")
                results["adf_error"] = str(e)

        if self.stationarity_test in ["kpss", "both"]:
            # KPSS test (null hypothesis: stationary)
            try:
                kpss_stat, kpss_pval, _, _ = kpss(
                    series_clean, regression="c", nlags="auto"
                )
                is_stationary_kpss = kpss_pval > self.stationarity_alpha
                results["kpss_statistic"] = float(kpss_stat)
                results["kpss_pvalue"] = float(kpss_pval)
                results["kpss_stationary"] = is_stationary_kpss

                if self.stationarity_test == "kpss":
                    is_stationary = is_stationary_kpss
            except Exception as e:
                logger.debug(f"KPSS test failed: {e}")
                results["kpss_error"] = str(e)

        if self.stationarity_test == "both":
            # Both tests must agree
            is_stationary = results.get("adf_stationary", False) and results.get(
                "kpss_stationary", False
            )
            results["both_agree"] = is_stationary

        self.report.stationarity_tests[series.name] = results

        return is_stationary

    def _normalize(self, data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Apply z-score normalization"""

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_normalized = pd.DataFrame(
            data_scaled,
            columns=data.columns,
            index=data.index,
        )

        for col in data.columns:
            self.report.transformations[col] = self.report.transformations.get(col, {})
            self.report.transformations[col]["normalized"] = True
            self.report.transformations[col]["original_mean"] = float(data[col].mean())
            self.report.transformations[col]["original_std"] = float(data[col].std())

        if verbose:
            logger.info("  All variables normalized (z-score)")

        return data_normalized


def quick_preprocess(
    data: pd.DataFrame,
    metadata_cols: Optional[List[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Convenience function for quick preprocessing with sensible defaults.

    Automatically handles variables with excessive missing data by dropping them
    (default threshold: 20%). This ensures memory-efficient processing even with
    large datasets containing sparse variables.

    Parameters:
        data: Input DataFrame
        metadata_cols: Columns to exclude from processing (e.g., 'unit_id', 'date')
        **kwargs: Additional arguments passed to TimeSeriesPreprocessor
                 (e.g., max_missing_frac=0.2 to adjust threshold)

    Returns:
        Tuple of (processed_df, PreprocessingReport)

    Example:
        >>> data, report = quick_preprocess(df, metadata_cols=['unit_id', 'date'])
        >>> print(f"Variables dropped: {len(report.quality_flags)}")
        >>> print(f"Remaining variables: {len(report.variables)}")
    """
    preprocessor = TimeSeriesPreprocessor(**kwargs)
    return preprocessor.preprocess(data, metadata_cols=metadata_cols)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Create synthetic test data
    np.random.seed(42)
    n = 100
    t = np.arange(n)

    data = pd.DataFrame(
        {
            "unit_id": ["A"] * n,
            "X": np.sin(t / 10) + np.random.normal(0, 0.1, n),
            "Y": np.cos(t / 10) + np.random.normal(0, 0.1, n),
            "Z": np.random.normal(0, 1, n) + 0.01 * t,  # Non-stationary
        }
    )

    # Add some missing values
    data.loc[10:15, "X"] = np.nan
    data.loc[30:32, "Y"] = np.nan

    # Add some outliers
    data.loc[50, "X"] = 10.0
    data.loc[75, "Y"] = -10.0

    print("\n" + "=" * 70)
    print("PREPROCESSING EXAMPLE")
    print("=" * 70)

    # Preprocess
    processed, report = quick_preprocess(
        data,
        metadata_cols=["unit_id"],
        interpolation_method="linear",
        stationarity_test="both",
        normalize=True,
        remove_seasonality=False,
        outlier_method="iqr",
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Original shape: {report.original_shape}")
    print(f"Final shape: {report.final_shape}")
    print("\nProcessed data (first 5 rows):")
    print(processed.head())

    # Save report
    report.save(Path("/tmp/preprocessing_report.json"))
    print("\n✅ Example completed!")
