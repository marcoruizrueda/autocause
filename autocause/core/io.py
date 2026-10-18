"""
I/O module for causal discovery framework.

Handles CSV/Parquet loading, time series harmonization, and data format detection.
Supports both wide and long/tidy formats with robust time indexing.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)


def load_timeseries_parquet(
    filepath: Union[str, Path],
    sample_rows: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load time series data from parquet file(s).

    Supports both single parquet files and directories containing multiple parquet files.
    Memory-efficient loading with optional sampling for large datasets.

    Parameters
    ----------
    filepath : str or Path
        Path to parquet file or directory containing parquet files
    sample_rows : int, optional
        Number of rows to randomly sample. If None, loads all data.
    random_state : int, default=42
        Random seed for reproducible sampling

    Returns
    -------
    pd.DataFrame
        Loaded dataframe

    Examples
    --------
    >>> # Load full dataset
    >>> df = load_timeseries_parquet("data/timeseries.parquet")

    >>> # Load with sampling for testing
    >>> df = load_timeseries_parquet("data/timeseries.parquet", sample_rows=100000)

    >>> # Load from directory of parquet files
    >>> df = load_timeseries_parquet("data/timeseries/", sample_rows=500000)
    """
    filepath = Path(filepath)

    if filepath.is_file():
        # Single parquet file
        logger.info(f"Loading parquet file: {filepath}")
        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    elif filepath.is_dir():
        # Directory of parquet files
        parquet_files = sorted(filepath.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {filepath}")

        logger.info(f"Loading {len(parquet_files)} parquet files from {filepath}")
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    else:
        raise ValueError(f"Path does not exist: {filepath}")

    # Optional sampling
    if sample_rows and sample_rows < len(df):
        logger.info(f"Sampling {sample_rows:,} rows from {len(df):,} total")
        df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)
        logger.info(f"Working with {len(df):,} sampled rows")

    return df


def load_timeseries_csv(
    filepath: str,
    time_column: Optional[str] = None,
    id_column: Optional[str] = None,
    variable_column: Optional[str] = None,
    value_column: Optional[str] = None,
    parse_dates: bool = True,
    infer_sampling_interval: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a CSV time series in either wide or long format.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    time_column : str, optional
        Name of the time column. If None, will be auto-detected.
    id_column : str, optional
        Name of the ID column (for long format). E.g., 'station_id', 'cube_id'.
    variable_column : str, optional
        Name of the variable column (for long format). E.g., 'variable', 'band'.
    value_column : str, optional
        Name of the value column (for long format). E.g., 'value', 'measurement'.
    parse_dates : bool, default=True
        Whether to parse dates automatically.
    infer_sampling_interval : bool, default=True
        Whether to infer the sampling interval (e.g., daily, 5-day).

    Returns
    -------
    tuple
        - df : pd.DataFrame
            Harmonized wide-format dataframe with DatetimeIndex.
        - metadata : dict
            Metadata including detected format, sampling interval, variables, etc.
    """

    # Load CSV
    df = pd.read_csv(filepath)
    logger.info(f"Loaded CSV from {filepath} with shape {df.shape}")

    # Detect format
    metadata = {
        "filepath": str(filepath),
        "original_shape": df.shape,
        "format": None,
        "time_column": time_column,
        "sampling_interval_days": None,
        "variables": None,
        "n_ids": None,
    }

    # Auto-detect time column if not provided
    if time_column is None:
        time_candidates = [
            col
            for col in df.columns
            if col.lower() in ["time", "date", "datetime", "timestamp"]
        ]
        if time_candidates:
            time_column = time_candidates[0]
            logger.info(f"Auto-detected time column: {time_column}")
        else:
            raise ValueError(
                "Could not detect time column. Please specify 'time_column'."
            )

    # Parse dates
    if parse_dates:
        df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
        if df[time_column].isna().sum() > 0:
            logger.warning(f"Could not parse {df[time_column].isna().sum()} dates.")
            df = df[df[time_column].notna()].copy()

    # Detect format (long vs wide)
    if id_column and variable_column and value_column:
        # Explicit long format
        metadata["format"] = "long"
        df = _pivot_long_to_wide(
            df, time_column, id_column, variable_column, value_column
        )
        metadata["n_ids"] = (
            df.index.get_level_values(0).nunique()
            if isinstance(df.index, pd.MultiIndex)
            else 1
        )
    elif id_column and value_column and variable_column is None:
        # Long format with implicit detection
        metadata["format"] = "long"
        df = _pivot_long_to_wide(df, time_column, id_column, None, value_column)
    else:
        # Wide format
        metadata["format"] = "wide"
        df.set_index(time_column, inplace=True)
        df.index.name = "time"

    # Sort by time
    df.sort_index(inplace=True)

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Time column must be datetime after parsing.")

    # Infer and set frequency to avoid warnings
    try:
        df.index.freq = pd.infer_freq(df.index)
    except (ValueError, TypeError):
        # If frequency cannot be inferred (e.g., irregular sampling), that's okay
        pass

    # Infer sampling interval
    if infer_sampling_interval:
        interval_days = _infer_sampling_interval(df.index)
        metadata["sampling_interval_days"] = interval_days
        logger.info(f"Inferred sampling interval: {interval_days} days")

    # Record variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata["variables"] = numeric_cols

    logger.info(f"Harmonized dataframe shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    return df, metadata


def _pivot_long_to_wide(
    df: pd.DataFrame,
    time_col: str,
    id_col: Optional[str],
    variable_col: Optional[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Pivot a long-format dataframe to wide format.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format data.
    time_col : str
        Time column name.
    id_col : str or None
        ID column (e.g., location, cube, station).
    variable_col : str or None
        Variable column (e.g., NDVI, RR, TG).
    value_col : str
        Value column.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe with DatetimeIndex.
    """

    if variable_col:
        # Pivot with both id and variable
        if id_col:
            df_wide = df.pivot_table(
                index=time_col,
                columns=[id_col, variable_col],
                values=value_col,
                aggfunc="mean",
            )
        else:
            df_wide = df.pivot_table(
                index=time_col, columns=variable_col, values=value_col, aggfunc="mean"
            )
    else:
        # Pivot with id only (one variable per id)
        if id_col:
            df_wide = df.pivot_table(
                index=time_col, columns=id_col, values=value_col, aggfunc="mean"
            )
        else:
            df_wide = df.set_index(time_col)[[value_col]]

    df_wide.index.name = "time"
    df_wide.index = pd.to_datetime(df_wide.index)

    # Infer and set frequency to avoid warnings
    try:
        df_wide.index.freq = pd.infer_freq(df_wide.index)
    except (ValueError, TypeError):
        pass

    return df_wide


def _infer_sampling_interval(time_index: pd.DatetimeIndex) -> float:
    """
    Infer the sampling interval in days from a datetime index.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index to analyze.

    Returns
    -------
    float
        Most common sampling interval in days.
    """

    if len(time_index) < 2:
        logger.warning("Cannot infer interval with < 2 time points. Assuming daily.")
        return 1.0

    # Calculate differences
    diffs = time_index.to_series().diff().dt.days.dropna()

    # Find most common interval
    mode_diff = diffs.mode()
    if len(mode_diff) > 0:
        interval = float(mode_diff[0])
    else:
        interval = diffs.mean()

    if interval <= 0:
        logger.warning(f"Invalid interval detected: {interval} days. Assuming daily.")
        interval = 1.0

    return interval


def harmonize_timeseries(
    df: pd.DataFrame,
    target_interval_days: Optional[float] = None,
    interpolation_method: str = "linear",
    max_gap_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Harmonize a time series to a consistent frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with DatetimeIndex.
    target_interval_days : float, optional
        Target interval in days. If None, inferred from data.
    interpolation_method : str, default="linear"
        Interpolation method for missing values.
    max_gap_days : int, optional
        Maximum gap (in days) to interpolate across. Beyond this, leave as NaN.

    Returns
    -------
    pd.DataFrame
        Harmonized dataframe.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Dataframe must have DatetimeIndex.")

    # Infer interval if not provided
    if target_interval_days is None:
        diffs = df.index.to_series().diff().dt.days.dropna()
        target_interval_days = float(diffs.mode()[0]) if len(diffs) > 0 else 1.0

    # Create target frequency string
    freq_str = f"{int(target_interval_days)}D"

    # Create complete date range
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_str)

    # Reindex
    df_harmonized = df.reindex(date_range)

    # Interpolate if needed
    if max_gap_days:
        max_gap_periods = int(max_gap_days / target_interval_days) + 1
    else:
        max_gap_periods = None

    df_harmonized = df_harmonized.interpolate(
        method=interpolation_method, limit=max_gap_periods, limit_direction="both"
    )

    logger.info(f"Harmonized to frequency: {freq_str}")
    logger.info(f"New shape: {df_harmonized.shape}")

    return df_harmonized


def apply_robust_na_policy(
    df: pd.DataFrame,
    max_missing_percent: float = 50,
    min_valid_rows: int = 30,
    outlier_handling: str = "winsorize",
    outlier_limits: Tuple[float, float] = (0.05, 0.95),
    outlier_z_threshold: float = 3.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a robust NA handling policy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_missing_percent : float, default=50
        Maximum allowed missing % per column (drop column if exceeded).
    min_valid_rows : int, default=30
        Minimum valid rows required after removing NaN.
    outlier_handling : str, default="winsorize"
        Method: 'winsorize', 'zscore', or 'none'.
    outlier_limits : tuple, default=(0.05, 0.95)
        Quantile limits for winsorization.
    outlier_z_threshold : float, default=3.5
        Z-score threshold for outlier removal.

    Returns
    -------
    tuple
        - df_clean : pd.DataFrame
            Cleaned dataframe.
        - qc_report : dict
            QC metrics.
    """

    qc_report = {
        "original_shape": df.shape,
        "columns_dropped": [],
        "missing_before": {},
        "missing_after": {},
        "outliers_handled": {},
    }

    df_clean = df.copy()

    # Check per-column missingness
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_pct = df_clean[col].isna().sum() / len(df_clean) * 100
        qc_report["missing_before"][col] = missing_pct

        if missing_pct > max_missing_percent:
            logger.warning(
                f"Dropping column '{col}': {missing_pct:.1f}% missing (> {max_missing_percent}%)"
            )
            df_clean.drop(columns=[col], inplace=True)
            qc_report["columns_dropped"].append(col)
        else:
            # Interpolate within column
            df_clean[col] = df_clean[col].interpolate(method="linear")

            # Handle outliers
            if outlier_handling == "winsorize":
                lower, upper = df_clean[col].quantile(outlier_limits)
                before_count = (df_clean[col] < lower) | (df_clean[col] > upper)
                df_clean[col] = df_clean[col].clip(lower, upper)
                qc_report["outliers_handled"][col] = int(before_count.sum())

            elif outlier_handling == "zscore":
                z_scores = np.abs(
                    (df_clean[col] - df_clean[col].mean()) / df_clean[col].std()
                )
                outlier_mask = z_scores > outlier_z_threshold
                df_clean.loc[outlier_mask, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method="linear")
                qc_report["outliers_handled"][col] = int(outlier_mask.sum())

    # Check rows
    df_clean.dropna(inplace=True)
    if len(df_clean) < min_valid_rows:
        logger.error(
            f"After cleaning, only {len(df_clean)} rows remain (< {min_valid_rows})"
        )

    for col in df_clean.select_dtypes(include=[np.number]).columns:
        qc_report["missing_after"][col] = df_clean[col].isna().sum()

    qc_report["final_shape"] = df_clean.shape

    logger.info(f"QC Policy applied. Final shape: {df_clean.shape}")

    return df_clean, qc_report
