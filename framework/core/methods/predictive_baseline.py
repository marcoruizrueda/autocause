"""
Predictive attribution baseline (non-causal).

Uses Random Forest or XGBoost to predict each target variable from
lagged values of all other variables, then extracts feature importances
as a "pseudo-causal graph."  This is NOT causal discovery — it measures
predictive relevance, which conflates direct effects, indirect effects,
and confounding.  Including it in the benchmark answers the question:

    "Does causal discovery find something different from strong
     predictive attribution?"

If a causal method's graph is similar to the RF importance graph, the
causal method may just be picking up predictive associations rather
than genuine mechanisms.  If they differ, the causal method is adding
value beyond prediction.

References:
    Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32.
    Runge et al. (2019). Detecting and quantifying causal associations
    in large nonlinear time series datasets. Science Advances.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _build_lagged_features(
    df: pd.DataFrame, target: str, max_lag: int
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build a lagged feature matrix for predicting `target`.

    For each variable v (including target) and each lag 1..max_lag,
    creates a column ``v_lag_k``.  Returns (X, y) aligned and dropna'd.
    """
    cols = {}
    for var in df.columns:
        for lag in range(1, max_lag + 1):
            cols[f"{var}_lag_{lag}"] = df[var].shift(lag)
    X = pd.DataFrame(cols, index=df.index)
    y = df[target]
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def batch_predictive_baseline(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    lags: int = 3,
    method: str = "random_forest",
    n_estimators: int = 200,
    sampling_days: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute predictive feature importances for all requested pairs.

    For each unique target in `variable_pairs`, fits a Random Forest (or
    Gradient Boosting) regressor using lagged values of all variables as
    features.  The importance of source→target is the sum of importances
    across lags of the source variable.

    A permutation-based p-value is computed by shuffling the source's
    lagged features and measuring the drop in R².

    Parameters:
        df: Multivariate time series (DatetimeIndex, numeric columns).
        variable_pairs: List of (source, target) pairs to report.
        lags: Number of lags to include as features.
        method: "random_forest" or "gradient_boosting".
        n_estimators: Number of trees.
        sampling_days: Days per timestep.
        alpha: Significance threshold for permutation test.

    Returns:
        DataFrame with: source, target, importance, p_value, is_significant,
        r2_full, method.
    """
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn required for predictive baseline")
        return pd.DataFrame()

    # Group pairs by target (fit one model per target)
    targets = {}
    for src, tgt in variable_pairs:
        targets.setdefault(tgt, set()).add(src)

    all_vars = sorted(set(v for pair in variable_pairs for v in pair))
    data = df[[v for v in all_vars if v in df.columns]].copy()
    data = data.interpolate(method="linear", limit_direction="both").ffill().bfill()

    results = []

    for target, sources in targets.items():
        X, y = _build_lagged_features(data, target, lags)
        if len(X) < 50:
            logger.warning(f"Insufficient data for {target} ({len(X)} rows)")
            for src in sources:
                results.append(_empty_row(src, target, method))
            continue

        # Fit model
        if method == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, max_depth=4, random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )

        model.fit(X, y)
        r2_full = model.score(X, y)
        importances = dict(zip(X.columns, model.feature_importances_))

        # For each source, aggregate importance across lags
        for src in sources:
            src_cols = [f"{src}_lag_{k}" for k in range(1, lags + 1)]
            src_cols = [c for c in src_cols if c in importances]
            total_imp = sum(importances.get(c, 0) for c in src_cols)
            best_lag = 0
            best_imp = 0
            for k in range(1, lags + 1):
                col = f"{src}_lag_{k}"
                if col in importances and importances[col] > best_imp:
                    best_imp = importances[col]
                    best_lag = k

            # Permutation p-value: shuffle source lags, measure R² drop
            n_perm = 100
            r2_drops = []
            rng = np.random.default_rng(42)
            for _ in range(n_perm):
                X_perm = X.copy()
                for c in src_cols:
                    X_perm[c] = rng.permutation(X_perm[c].values)
                r2_perm = model.score(X_perm, y)
                r2_drops.append(r2_full - r2_perm)

            r2_drop = np.mean(r2_drops)
            # p-value: fraction of permutations where drop ≤ 0 (no importance)
            p_value = (np.array(r2_drops) <= 0).sum() / n_perm

            method_name = "RF-baseline" if method == "random_forest" else "GB-baseline"
            results.append(
                {
                    "source": src,
                    "target": target,
                    "importance": total_imp,
                    "best_lag": best_lag,
                    "best_lag_days": best_lag * sampling_days,
                    "r2_drop": r2_drop,
                    "p_value": p_value,
                    "is_significant": p_value < alpha,
                    "r2_full": r2_full,
                    "method": method_name,
                }
            )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("importance", ascending=False)
        n_sig = results_df["is_significant"].sum()
        logger.info(
            f"Predictive baseline ({method}): {n_sig} significant edges, "
            f"R²={results_df['r2_full'].mean():.3f}"
        )
    return results_df


def _empty_row(src, tgt, method):
    method_name = "RF-baseline" if method == "random_forest" else "GB-baseline"
    return {
        "source": src,
        "target": tgt,
        "importance": 0,
        "best_lag": np.nan,
        "best_lag_days": np.nan,
        "r2_drop": 0,
        "p_value": 1.0,
        "is_significant": False,
        "r2_full": 0,
        "method": method_name,
    }
