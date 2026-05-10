#!/usr/bin/env python3
"""
Causal Strength Quantification Module

This module provides normalized causal effect metrics for comparing
strength across different methods and variable pairs:

1. **Granger Causality**: Partial R² (variance explained)
2. **Transfer Entropy**: Normalized TE (bits normalized by entropy)
3. **PCMCI+**: Partial correlation coefficients and p-values

Additionally supports:
- Sensitivity analysis across lags
- Temporal window analysis (rolling causal strength)
- Effect size comparisons across methods
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalStrengthMetrics:
    """Normalized causal strength metrics for a single edge"""

    source: str
    target: str
    method: str

    # Raw effect sizes
    raw_effect: Optional[float] = None

    # Normalized metrics (0-1 scale where possible)
    normalized_effect: Optional[float] = None

    # Statistical significance
    p_value: Optional[float] = None
    q_value: Optional[float] = None  # FDR-corrected
    is_significant: bool = False

    # Confidence intervals
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    # Lag information
    lag_steps: Optional[int] = None
    lag_days: Optional[float] = None

    # Metadata
    n_observations: Optional[int] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class CausalStrengthQuantifier:
    """
    Quantify and normalize causal strength across different methods.

    Provides comparable effect sizes enabling:
    - Cross-method comparison
    - Sensitivity analysis
    - Rolling window analysis
    - Effect size ranking
    """

    def __init__(
        self,
        alpha: float = 0.05,
        confidence_level: float = 0.95,
    ):
        """
        Initialize quantifier.

        Parameters:
            alpha: Significance threshold
            confidence_level: Level for confidence intervals (e.g., 0.95 = 95%)
        """
        self.alpha = alpha
        self.confidence_level = confidence_level

    def quantify_granger(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        lag: int,
        **kwargs,
    ) -> CausalStrengthMetrics:
        """
        Quantify Granger causality strength using Partial R².

        Partial R² = (RSS_restricted - RSS_unrestricted) / RSS_restricted

        This represents the fraction of variance in target explained by
        source beyond what's explained by target's own history.
        """
        from framework.core.methods import granger

        result = granger.run_granger_causality(
            data,
            source,
            target,
            maxlag=lag,
            alpha=self.alpha,
            **kwargs,
        )

        # Extract effect size (beta coefficient standardized)
        raw_effect = result.get("granger_beta_std", np.nan)

        # Partial R² approximation from F-statistic
        # R² ≈ F / (F + df_denom)
        f_stat = result.get("f_statistic", np.nan)
        df_denom = result.get("df_denom", 1)

        if not np.isnan(f_stat) and f_stat > 0:
            partial_r2 = f_stat / (f_stat + df_denom)
        else:
            partial_r2 = np.nan

        return CausalStrengthMetrics(
            source=source,
            target=target,
            method="Granger",
            raw_effect=raw_effect,
            normalized_effect=partial_r2,  # Already 0-1 scale
            p_value=result.get("best_p_value"),
            is_significant=result.get("is_causal", False),
            lag_steps=result.get("best_lag"),
            lag_days=result.get("best_lag_days"),
            n_observations=result.get("n_observations"),
        )

    def quantify_transfer_entropy(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        delay: int,
        **kwargs,
    ) -> CausalStrengthMetrics:
        """
        Quantify Transfer Entropy strength.

        Normalized TE = TE / H(target)
        where H(target) is the entropy of the target variable.

        This gives a 0-1 scale representing the fraction of target's
        uncertainty explained by source.
        """
        from framework.core.methods import transfer_entropy

        result = transfer_entropy.run_transfer_entropy(
            data,
            source,
            target,
            delay=delay,
            **kwargs,
        )

        te_bits = result.get("te_bits", np.nan)

        # Compute target entropy for normalization
        target_data = data[target].dropna()

        # Discrete entropy (using histogram binning)
        n_bins = min(50, len(target_data) // 10)
        counts, _ = np.histogram(target_data, bins=n_bins)
        probs = counts[counts > 0] / counts[counts > 0].sum()
        h_target = -np.sum(probs * np.log2(probs))

        # Normalized TE
        if h_target > 0 and not np.isnan(te_bits):
            normalized_te = min(te_bits / h_target, 1.0)  # Cap at 1.0
        else:
            normalized_te = np.nan

        return CausalStrengthMetrics(
            source=source,
            target=target,
            method="TransferEntropy",
            raw_effect=te_bits,
            normalized_effect=normalized_te,
            p_value=result.get("p_value"),
            is_significant=result.get("is_significant", False),
            lag_steps=delay,
            lag_days=result.get("delay_days"),
            n_observations=result.get("n_observations"),
        )

    def quantify_pcmci(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        tau_max: int,
        **kwargs,
    ) -> CausalStrengthMetrics:
        """
        Quantify PCMCI+ strength using partial correlation.

        Partial correlation is already on [-1, 1] scale.
        We report absolute value for strength.
        """
        from framework.core.methods import tigramite_pcmci

        result = tigramite_pcmci.run_pcmci_pair(
            data,
            source,
            target,
            tau_max=tau_max,
            alpha=self.alpha,
            **kwargs,
        )

        # Partial correlation (already normalized)
        parcorr = result.get("best_parcorr", np.nan)

        return CausalStrengthMetrics(
            source=source,
            target=target,
            method="PCMCI+",
            raw_effect=parcorr,
            normalized_effect=abs(parcorr) if not np.isnan(parcorr) else np.nan,
            p_value=result.get("best_p_value"),
            is_significant=result.get("causal", False),
            lag_steps=result.get("best_lag"),
            lag_days=result.get("best_lag_days"),
            n_observations=result.get("n_observations"),
        )

    def sensitivity_analysis_lag(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        method: str = "granger",
        lag_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis across different lags.

        Tests how causal strength varies with lag parameter.

        Parameters:
            data: Time series data
            source: Source variable
            target: Target variable
            method: 'granger', 'te', or 'pcmci'
            lag_range: (min_lag, max_lag) tuple. Default: (1, 12)
            **kwargs: Additional method-specific arguments

        Returns:
            DataFrame with lag, strength, p_value columns
        """
        if lag_range is None:
            lag_range = (1, min(12, len(data) // 10))

        min_lag, max_lag = lag_range
        results = []

        logger.info(
            f"Lag sensitivity analysis: {source}→{target} "
            f"(method={method}, lags={min_lag}-{max_lag})"
        )

        for lag in range(min_lag, max_lag + 1):
            try:
                if method == "granger":
                    metrics = self.quantify_granger(
                        data, source, target, lag=lag, **kwargs
                    )
                elif method == "te":
                    metrics = self.quantify_transfer_entropy(
                        data, source, target, delay=lag, **kwargs
                    )
                elif method == "pcmci":
                    metrics = self.quantify_pcmci(
                        data, source, target, tau_max=lag, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                results.append(
                    {
                        "lag": lag,
                        "raw_effect": metrics.raw_effect,
                        "normalized_effect": metrics.normalized_effect,
                        "p_value": metrics.p_value,
                        "is_significant": metrics.is_significant,
                    }
                )

            except Exception as e:
                logger.debug(f"Lag {lag} failed: {e}")
                results.append(
                    {
                        "lag": lag,
                        "raw_effect": np.nan,
                        "normalized_effect": np.nan,
                        "p_value": np.nan,
                        "is_significant": False,
                    }
                )

        return pd.DataFrame(results)

    def rolling_window_analysis(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        method: str = "granger",
        window_size: int = 100,
        step_size: int = 20,
        lag: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Perform rolling window analysis to detect temporal changes in causality.

        Tests structural stability: is the causal relationship constant over time?

        Parameters:
            data: Time series data
            source: Source variable
            target: Target variable
            method: 'granger', 'te', or 'pcmci'
            window_size: Size of rolling window
            step_size: Step between windows
            lag: Fixed lag to use for all windows
            **kwargs: Additional method-specific arguments

        Returns:
            DataFrame with window_start, window_end, strength, p_value columns
        """
        n = len(data)

        if window_size > n:
            raise ValueError(f"Window size ({window_size}) exceeds data length ({n})")

        results = []

        logger.info(
            f"Rolling window analysis: {source}→{target} "
            f"(method={method}, window={window_size}, step={step_size})"
        )

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            window_data = data.iloc[start:end]

            try:
                if method == "granger":
                    metrics = self.quantify_granger(
                        window_data, source, target, lag=lag, **kwargs
                    )
                elif method == "te":
                    metrics = self.quantify_transfer_entropy(
                        window_data, source, target, delay=lag, **kwargs
                    )
                elif method == "pcmci":
                    metrics = self.quantify_pcmci(
                        window_data, source, target, tau_max=lag, **kwargs
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")

                results.append(
                    {
                        "window_start": start,
                        "window_end": end,
                        "window_center": (start + end) // 2,
                        "raw_effect": metrics.raw_effect,
                        "normalized_effect": metrics.normalized_effect,
                        "p_value": metrics.p_value,
                        "is_significant": metrics.is_significant,
                    }
                )

            except Exception as e:
                logger.debug(f"Window [{start}:{end}] failed: {e}")
                results.append(
                    {
                        "window_start": start,
                        "window_end": end,
                        "window_center": (start + end) // 2,
                        "raw_effect": np.nan,
                        "normalized_effect": np.nan,
                        "p_value": np.nan,
                        "is_significant": False,
                    }
                )

        return pd.DataFrame(results)

    def compare_methods(
        self,
        data: pd.DataFrame,
        source: str,
        target: str,
        lag: int = 5,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Compare causal strength across all methods for a single edge.

        Returns DataFrame with normalized effects for easy comparison.
        """
        results = []

        # Granger
        try:
            granger_metrics = self.quantify_granger(
                data, source, target, lag=lag, **kwargs
            )
            results.append(granger_metrics.to_dict())
        except Exception as e:
            logger.debug(f"Granger failed: {e}")

        # Transfer Entropy
        try:
            te_metrics = self.quantify_transfer_entropy(
                data, source, target, delay=lag, **kwargs
            )
            results.append(te_metrics.to_dict())
        except Exception as e:
            logger.debug(f"Transfer Entropy failed: {e}")

        # PCMCI+
        try:
            pcmci_metrics = self.quantify_pcmci(
                data, source, target, tau_max=lag, **kwargs
            )
            results.append(pcmci_metrics.to_dict())
        except Exception as e:
            logger.debug(f"PCMCI+ failed: {e}")

        return pd.DataFrame(results)

    def rank_edges_by_strength(
        self,
        edges_df: pd.DataFrame,
        metric: str = "normalized_effect",
    ) -> pd.DataFrame:
        """
        Rank edges by causal strength.

        Parameters:
            edges_df: DataFrame with causal edges (must have 'normalized_effect' column)
            metric: Column to rank by ('normalized_effect', 'raw_effect', or 'p_value')

        Returns:
            Sorted DataFrame with rank column added
        """
        if metric not in edges_df.columns:
            raise ValueError(f"Metric '{metric}' not found in edges DataFrame")

        # Sort by metric
        if metric == "p_value":
            # Lower p-value = stronger evidence
            sorted_df = edges_df.sort_values(metric, ascending=True)
        else:
            # Higher effect = stronger causality
            sorted_df = edges_df.sort_values(
                metric, ascending=False, na_position="last"
            )

        # Add rank
        sorted_df = sorted_df.copy()
        sorted_df["rank"] = range(1, len(sorted_df) + 1)

        return sorted_df


def batch_quantify(
    data: pd.DataFrame,
    edges: List[Tuple[str, str]],
    method: str = "granger",
    lag: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """
    Convenience function to quantify multiple edges at once.

    Parameters:
        data: Time series data
        edges: List of (source, target) tuples
        method: 'granger', 'te', or 'pcmci'
        lag: Fixed lag to use
        **kwargs: Additional arguments

    Returns:
        DataFrame with all quantified edges
    """
    quantifier = CausalStrengthQuantifier(**kwargs)
    results = []

    for source, target in edges:
        try:
            if method == "granger":
                metrics = quantifier.quantify_granger(data, source, target, lag=lag)
            elif method == "te":
                metrics = quantifier.quantify_transfer_entropy(
                    data, source, target, delay=lag
                )
            elif method == "pcmci":
                metrics = quantifier.quantify_pcmci(data, source, target, tau_max=lag)
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append(metrics.to_dict())

        except Exception as e:
            logger.warning(f"Failed to quantify {source}→{target}: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Create synthetic test data
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    # X causes Y with lag 5
    X = np.random.normal(0, 1, n)
    Y = np.zeros(n)
    Y[:5] = np.random.normal(0, 1, 5)
    for i in range(5, n):
        Y[i] = 0.5 * X[i - 5] + 0.3 * Y[i - 1] + np.random.normal(0, 0.5)

    data = pd.DataFrame({"X": X, "Y": Y})

    print("\n" + "=" * 70)
    print("CAUSAL STRENGTH QUANTIFICATION EXAMPLE")
    print("=" * 70)

    # 1. Compare methods
    quantifier = CausalStrengthQuantifier()

    print("\n1. Comparing methods for X→Y (lag=5):")
    comparison = quantifier.compare_methods(data, "X", "Y", lag=5)
    print(comparison[["method", "normalized_effect", "p_value", "is_significant"]])

    # 2. Lag sensitivity analysis
    print("\n2. Lag sensitivity analysis:")
    lag_sensitivity = quantifier.sensitivity_analysis_lag(
        data, "X", "Y", method="granger", lag_range=(1, 10)
    )
    print(lag_sensitivity[["lag", "normalized_effect", "p_value"]])

    # 3. Rolling window analysis
    print("\n3. Rolling window analysis (first 5 windows):")
    rolling = quantifier.rolling_window_analysis(
        data, "X", "Y", method="granger", window_size=100, step_size=20, lag=5
    )
    print(rolling[["window_center", "normalized_effect", "is_significant"]].head())

    print("\n✅ Example completed!")
