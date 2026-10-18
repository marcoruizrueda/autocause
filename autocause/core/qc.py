"""
Quality control module for causal discovery framework.

Handles coverage statistics, missing data, outliers, and data flags.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any, Optional

logger = logging.getLogger(__name__)


def coverage_statistics(
    df: pd.DataFrame,
    groupby_column: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute coverage statistics per series or per group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    groupby_column : str, optional
        Column to group by (e.g., 'cube_id'). If None, computes per-column stats.

    Returns
    -------
    dict
        Coverage statistics.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}

    if groupby_column and groupby_column in df.columns:
        for group in df[groupby_column].unique():
            group_data = df[df[groupby_column] == group][numeric_cols]
            stats[group] = {
                col: {
                    "n_valid": group_data[col].notna().sum(),
                    "n_missing": group_data[col].isna().sum(),
                    "coverage_pct": (group_data[col].notna().sum() / len(group_data))
                    * 100,
                    "mean": group_data[col].mean(),
                    "std": group_data[col].std(),
                }
                for col in numeric_cols
            }
    else:
        for col in numeric_cols:
            stats[col] = {
                "n_valid": df[col].notna().sum(),
                "n_missing": df[col].isna().sum(),
                "coverage_pct": (df[col].notna().sum() / len(df)) * 100,
                "mean": df[col].mean(),
                "std": df[col].std(),
            }

    return stats


def flag_low_coverage_series(
    df: pd.DataFrame,
    min_coverage_pct: float = 10,
    groupby_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Flag series (or groups) with coverage below threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    min_coverage_pct : float, default=10
        Minimum coverage percentage required.
    groupby_column : str, optional
        Column to group by.

    Returns
    -------
    tuple
        - df_flagged : pd.DataFrame
            Dataframe with flag column added.
        - report : dict
            Flagging report.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_flagged = df.copy()
    df_flagged["__qc_flag__"] = "pass"

    report = {
        "n_flagged_series": 0,
        "flagged_series": [],
        "min_coverage_threshold_pct": min_coverage_pct,
    }

    if groupby_column and groupby_column in df.columns:
        for col in numeric_cols:
            group_coverage = df.groupby(groupby_column)[col].apply(
                lambda x: (x.notna().sum() / len(x)) * 100
            )
            low_coverage = group_coverage[group_coverage < min_coverage_pct]
            if len(low_coverage) > 0:
                report["n_flagged_series"] += len(low_coverage)
                for group, coverage in low_coverage.items():
                    flag_name = f"{groupby_column}={group},var={col}"
                    report["flagged_series"].append(
                        {
                            "series": flag_name,
                            "coverage_pct": coverage,
                        }
                    )
                    mask = df_flagged[groupby_column] == group
                    df_flagged.loc[mask, "__qc_flag__"] = "low_coverage"
    else:
        for col in numeric_cols:
            coverage = (df[col].notna().sum() / len(df)) * 100
            if coverage < min_coverage_pct:
                report["n_flagged_series"] += 1
                report["flagged_series"].append(
                    {
                        "series": col,
                        "coverage_pct": coverage,
                    }
                )
                df_flagged["__qc_flag__"] = "low_coverage"

    logger.info(
        f"Flagged {report['n_flagged_series']} series below {min_coverage_pct}% coverage"
    )

    return df_flagged, report


def detect_and_handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    handle_mode: str = "clip",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detect and handle outliers using statistical methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    method : str, default="iqr"
        Detection method: 'iqr', 'zscore', or 'both'.
    iqr_multiplier : float, default=1.5
        IQR multiplier for outlier bounds.
    z_threshold : float, default=3.0
        Z-score threshold.
    handle_mode : str, default="clip"
        Handling mode: 'clip', 'interpolate', or 'flag'.

    Returns
    -------
    tuple
        - df_clean : pd.DataFrame
        - report : dict
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_clean = df.copy()
    report = {
        "method": method,
        "n_outliers_per_col": {},
        "handling_mode": handle_mode,
    }

    for col in numeric_cols:
        outlier_mask = pd.Series(False, index=df.index)

        if method in ["iqr", "both"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            outlier_mask |= (df[col] < lower) | (df[col] > upper)

        if method in ["zscore", "both"]:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask |= z_scores > z_threshold

        n_outliers = int(outlier_mask.sum())
        report["n_outliers_per_col"][col] = n_outliers

        if n_outliers > 0:
            if handle_mode == "clip":
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df_clean.loc[outlier_mask, col] = df_clean.loc[outlier_mask, col].clip(
                    lower, upper
                )
            elif handle_mode == "interpolate":
                df_clean.loc[outlier_mask, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method="linear")
            elif handle_mode == "flag":
                if "__outlier_flag__" not in df_clean.columns:
                    df_clean["__outlier_flag__"] = False
                df_clean.loc[outlier_mask, "__outlier_flag__"] = True

    logger.info(
        f"Outlier detection completed. Total outliers: {sum(report['n_outliers_per_col'].values())}"
    )

    return df_clean, report


def detect_masking_flags(
    df: pd.DataFrame,
    mask_columns: list = None,
) -> Dict[str, Any]:
    """
    Detect and report on data masking flags (e.g., cloud masks, snow masks).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    mask_columns : list, optional
        Columns to check for mask flags. If None, auto-detect.

    Returns
    -------
    dict
        Masking report.
    """

    if mask_columns is None:
        # Auto-detect mask columns
        mask_columns = [
            col for col in df.columns if "mask" in col.lower() or "flag" in col.lower()
        ]

    report = {
        "mask_columns_found": mask_columns,
        "mask_coverage": {},
    }

    for col in mask_columns:
        if col in df.columns:
            # Assuming 0 = valid, 1 = masked
            valid_count = (
                (df[col] == 0).sum()
                if df[col].dtype in [int, float]
                else df[col].notna().sum()
            )
            total_count = len(df)
            coverage = (valid_count / total_count) * 100

            report["mask_coverage"][col] = {
                "valid_pixels": valid_count,
                "masked_pixels": total_count - valid_count,
                "coverage_pct": coverage,
            }

    logger.info(f"Detected {len(mask_columns)} mask columns")

    return report


def generate_qc_report(
    df: pd.DataFrame,
    groupby_column: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive QC report.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    groupby_column : str, optional
        Column to group by.
    output_file : str, optional
        File to save report.

    Returns
    -------
    dict
        Comprehensive QC report.
    """

    report = {
        "dataframe_shape": df.shape,
        "date_range": f"{df.index.min()} to {df.index.max()}"
        if isinstance(df.index, pd.DatetimeIndex)
        else "N/A",
        "coverage_stats": coverage_statistics(df, groupby_column),
        "masking_flags": detect_masking_flags(df),
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report["numeric_columns"] = list(numeric_cols)
    report["total_missing_values"] = int(df[numeric_cols].isna().sum().sum())
    report["total_valid_values"] = int(df[numeric_cols].notna().sum().sum())

    if output_file:
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            # Custom serializer for numpy types
            json.dump(report, f, indent=2, default=str)
        logger.info(f"QC report saved to {output_file}")

    return report
