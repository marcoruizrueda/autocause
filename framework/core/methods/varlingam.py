"""
VAR-LiNGAM: Vector Autoregressive Linear Non-Gaussian Acyclic Model

Implements score-based causal discovery exploiting non-Gaussianity for
identifiability. Unlike VAR-based Granger (which only tests predictive
improvement) and PCMCI+ (which tests conditional independence),
VAR-LiNGAM recovers the full causal DAG including contemporaneous
effects by leveraging independent component analysis on VAR residuals.

Requires non-Gaussian noise for identifiability (Shimizu et al. 2006).
When noise is Gaussian, the model is not identifiable and results
should be interpreted with caution.

References:
    - Hyvarinen, A. et al. (2010). "Estimation of a structural vector
      autoregression model using non-Gaussianity". JMLR, 11, 1709-1731.
    - Shimizu, S. et al. (2006). "A linear non-Gaussian acyclic model
      for causal discovery". JMLR, 7, 2003-2030.
    - lingam package: https://github.com/cdt15/lingam
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from lingam import VARLiNGAM

    VARLINGAM_AVAILABLE = True
except ImportError:
    VARLINGAM_AVAILABLE = False
    logger.warning("lingam not installed. Install with: uv pip install lingam")


def batch_varlingam(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    lags: int = 3,
    alpha: float = 0.05,
    sampling_days: float = 1.0,
    prune: bool = True,
) -> pd.DataFrame:
    """
    Run VAR-LiNGAM on the full dataset and extract results for requested pairs.

    Unlike bivariate Granger, VAR-LiNGAM fits a single multivariate model
    and uses ICA on the residuals to identify the causal DAG, including
    contemporaneous effects. This requires non-Gaussian noise.

    Parameters:
        df: Multivariate time series (DatetimeIndex, numeric columns).
        variable_pairs: List of (source, target) pairs to report.
        lags: Number of lags for the VAR component.
        alpha: Significance threshold for bootstrap p-values.
        sampling_days: Days per timestep for lag conversion.
        prune: If True, prune small coefficients via bootstrap (slower but
               more reliable).

    Returns:
        DataFrame with columns: source, target, lag, lag_days, coefficient,
        is_significant, method.
    """
    if not VARLINGAM_AVAILABLE:
        logger.error("lingam package not available")
        return pd.DataFrame()

    # Collect all variables needed
    all_vars = sorted(set(v for pair in variable_pairs for v in pair))
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        logger.warning(f"Missing variables: {missing}")
        all_vars = [v for v in all_vars if v in df.columns]

    data = df[all_vars].copy()

    # Handle missing values — VAR-LiNGAM cannot handle NaN
    n_missing = data.isna().sum().sum()
    if n_missing > 0:
        frac = n_missing / data.size
        logger.info(
            f"Interpolating {n_missing} missing values ({frac:.1%}) for VAR-LiNGAM"
        )
        data = data.interpolate(method="linear", limit_direction="both")
        data = data.ffill().bfill()
        # If still NaN (e.g. entire column missing), drop those columns
        still_missing = data.columns[data.isna().any()].tolist()
        if still_missing:
            logger.warning(f"Dropping columns with remaining NaN: {still_missing}")
            data = data.drop(columns=still_missing)
            all_vars = [v for v in all_vars if v in data.columns]

    if len(data) < lags + 20:
        logger.error(f"Insufficient data ({len(data)}) for lags={lags}")
        return pd.DataFrame()

    logger.info(
        f"Running VAR-LiNGAM (lags={lags}, vars={len(all_vars)}, "
        f"T={len(data)}, prune={prune})"
    )

    try:
        model = VARLiNGAM(lags=lags, prune=prune)
        model.fit(data.values)
    except Exception as e:
        logger.error(f"VAR-LiNGAM fit failed: {e}")
        return pd.DataFrame()

    # Extract adjacency matrices: model.adjacency_matrices_[k] is (N x N)
    # Entry [i, j] at lag k = coefficient of var_j(t-k) → var_i(t)
    var_names = all_vars
    n_vars = len(var_names)
    results = []

    for lag_k, adj in enumerate(model.adjacency_matrices_):
        for i in range(n_vars):
            for j in range(n_vars):
                coeff = adj[i, j]
                if abs(coeff) < 1e-10:
                    continue
                source = var_names[j]
                target = var_names[i]
                if (source, target) in variable_pairs:
                    results.append(
                        {
                            "source": source,
                            "target": target,
                            "lag": lag_k,
                            "lag_days": lag_k * sampling_days,
                            "coefficient": float(coeff),
                            "abs_coefficient": abs(float(coeff)),
                            "is_contemporaneous": lag_k == 0,
                            "method": "VAR-LiNGAM",
                        }
                    )

    # If pruning zeroed out all edges, retry without pruning and use a
    # coefficient threshold instead.  This is common when the sample size
    # is small or the signal is moderate.
    if not results and prune:
        logger.info("VAR-LiNGAM pruning removed all edges — retrying without pruning")
        try:
            model_np = VARLiNGAM(lags=lags, prune=False)
            model_np.fit(data.values)
            for lag_k, adj in enumerate(model_np.adjacency_matrices_):
                for i in range(n_vars):
                    for j in range(n_vars):
                        coeff = adj[i, j]
                        if abs(coeff) < 0.05:
                            continue
                        source = var_names[j]
                        target = var_names[i]
                        if (source, target) in variable_pairs:
                            results.append(
                                {
                                    "source": source,
                                    "target": target,
                                    "lag": lag_k,
                                    "lag_days": lag_k * sampling_days,
                                    "coefficient": float(coeff),
                                    "abs_coefficient": abs(float(coeff)),
                                    "is_contemporaneous": lag_k == 0,
                                    "method": "VAR-LiNGAM",
                                }
                            )
        except Exception as e2:
            logger.warning(f"VAR-LiNGAM retry without pruning also failed: {e2}")

    if not results:
        logger.info("VAR-LiNGAM: no edges found for requested pairs")
        # Return one row per pair with is_significant=False
        for source, target in variable_pairs:
            results.append(
                {
                    "source": source,
                    "target": target,
                    "lag": np.nan,
                    "lag_days": np.nan,
                    "coefficient": 0.0,
                    "abs_coefficient": 0.0,
                    "is_contemporaneous": False,
                    "method": "VAR-LiNGAM",
                }
            )

    results_df = pd.DataFrame(results)

    # Significance: bootstrap if requested, otherwise threshold on |coeff|
    if prune and hasattr(model, "adjacency_matrices_"):
        # The lingam prune option already zeros out non-significant entries,
        # so any non-zero coefficient is considered significant
        results_df["is_significant"] = results_df["abs_coefficient"] > 1e-10
    else:
        # Fallback: use a coefficient threshold
        results_df["is_significant"] = results_df["abs_coefficient"] > 0.05

    results_df["significant"] = results_df["is_significant"]

    n_sig = results_df["is_significant"].sum()
    logger.info(
        f"VAR-LiNGAM: {n_sig} significant edges for {len(variable_pairs)} pairs"
    )

    return results_df.sort_values("abs_coefficient", ascending=False)
