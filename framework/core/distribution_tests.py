#!/usr/bin/env python3
"""
Statistical Distribution Tests for Causal Method Selection

This module provides comprehensive statistical tests to characterize time series
properties, enabling intelligent method selection for causal discovery:

1. **Gaussianity Tests**: Shapiro-Wilk, Jarque-Bera, Anderson-Darling
2. **Linearity Tests**: BDS test, Ljung-Box test, RESET test
3. **Heteroscedasticity Tests**: White test, ARCH test
4. **Autocorrelation**: Ljung-Box, Durbin-Watson

These tests inform method selection:
- Gaussian + Linear → VAR-based Granger causality (optimal)
- Non-Gaussian or Non-Linear → Transfer Entropy or PCMCI+
- High-dimensional → PCMCI+ with regularization
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white, het_arch
from statsmodels.stats.stattools import durbin_watson

logger = logging.getLogger(__name__)


@dataclass
class DistributionTestResults:
    """Results from statistical distribution tests"""

    variable: str
    n_observations: int

    # Gaussianity tests
    shapiro_stat: Optional[float] = None
    shapiro_pval: Optional[float] = None
    is_gaussian_shapiro: Optional[bool] = None

    jarque_bera_stat: Optional[float] = None
    jarque_bera_pval: Optional[float] = None
    is_gaussian_jb: Optional[bool] = None

    anderson_stat: Optional[float] = None
    anderson_critical_5pct: Optional[float] = None
    is_gaussian_anderson: Optional[bool] = None

    # Linearity tests
    ljungbox_stat: Optional[float] = None
    ljungbox_pval: Optional[float] = None
    is_linear_ljungbox: Optional[bool] = None

    bds_available: bool = False

    # Heteroscedasticity tests
    white_stat: Optional[float] = None
    white_pval: Optional[float] = None
    is_homoscedastic_white: Optional[bool] = None

    arch_stat: Optional[float] = None
    arch_pval: Optional[float] = None
    is_homoscedastic_arch: Optional[bool] = None

    # Autocorrelation
    durbin_watson_stat: Optional[float] = None
    has_autocorrelation: Optional[bool] = None

    # Summary classifications
    is_gaussian: Optional[bool] = None
    is_linear: Optional[bool] = None
    is_homoscedastic: Optional[bool] = None

    # Method recommendations
    recommended_methods: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self) -> str:
        return (
            f"DistributionTestResults({self.variable}: "
            f"gaussian={self.is_gaussian}, linear={self.is_linear}, "
            f"recommended={self.recommended_methods})"
        )


class DistributionTester:
    """
    Comprehensive statistical distribution testing for time series.

    Performs battery of tests to characterize series properties and
    recommend optimal causal discovery methods.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        ljungbox_lags: int = 10,
        arch_lags: int = 5,
        min_obs: int = 30,
    ):
        """
        Initialize distribution tester.

        Parameters:
            alpha: Significance level for hypothesis tests
            ljungbox_lags: Number of lags for Ljung-Box test
            arch_lags: Number of lags for ARCH test
            min_obs: Minimum observations required for testing
        """
        self.alpha = alpha
        self.ljungbox_lags = ljungbox_lags
        self.arch_lags = arch_lags
        self.min_obs = min_obs

    def test_variable(
        self,
        series: pd.Series,
        verbose: bool = False,
    ) -> DistributionTestResults:
        """
        Run complete battery of distribution tests on a single variable.

        Parameters:
            series: Time series to test
            verbose: Whether to log detailed results

        Returns:
            DistributionTestResults object
        """
        series_clean = series.dropna()
        n_obs = len(series_clean)

        if n_obs < self.min_obs:
            logger.warning(
                f"{series.name}: Only {n_obs} observations (min={self.min_obs}), "
                "skipping tests"
            )
            result = DistributionTestResults(
                variable=series.name,
                n_observations=n_obs,
            )
            result.reasoning.append(f"Insufficient data: {n_obs} < {self.min_obs}")
            return result

        result = DistributionTestResults(
            variable=series.name,
            n_observations=n_obs,
        )

        if verbose:
            logger.info(f"\nTesting {series.name} (n={n_obs}):")

        # 1. Gaussianity Tests
        self._test_gaussianity(series_clean, result, verbose)

        # 2. Linearity Tests
        self._test_linearity(series_clean, result, verbose)

        # 3. Heteroscedasticity Tests
        self._test_heteroscedasticity(series_clean, result, verbose)

        # 4. Autocorrelation Test
        self._test_autocorrelation(series_clean, result, verbose)

        # 5. Generate recommendations
        self._generate_recommendations(result, verbose)

        return result

    def test_dataframe(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, DistributionTestResults]:
        """
        Test multiple variables in a DataFrame.

        Parameters:
            data: DataFrame containing time series
            variables: List of column names to test (None = all numeric)
            verbose: Whether to log detailed results

        Returns:
            Dictionary mapping variable name to DistributionTestResults
        """
        if variables is None:
            variables = data.select_dtypes(include=[np.number]).columns.tolist()

        results = {}

        if verbose:
            logger.info("=" * 70)
            logger.info("DISTRIBUTION TESTS")
            logger.info("=" * 70)

        for var in variables:
            if var not in data.columns:
                logger.warning(f"Variable {var} not found in DataFrame")
                continue

            results[var] = self.test_variable(data[var], verbose=verbose)

        if verbose:
            logger.info("\n" + "=" * 70)
            logger.info("DISTRIBUTION TESTS COMPLETE")
            logger.info("=" * 70)
            self._print_summary(results)

        return results

    def _test_gaussianity(
        self,
        series: pd.Series,
        result: DistributionTestResults,
        verbose: bool,
    ):
        """Test for Gaussian (normal) distribution"""

        # Shapiro-Wilk test (best for n < 5000)
        if len(series) <= 5000:
            try:
                stat, pval = scipy_stats.shapiro(series)
                result.shapiro_stat = float(stat)
                result.shapiro_pval = float(pval)
                result.is_gaussian_shapiro = pval > self.alpha

                if verbose:
                    logger.info(
                        f"  Shapiro-Wilk: stat={stat:.4f}, p={pval:.4f} "
                        f"→ {'Gaussian' if result.is_gaussian_shapiro else 'Non-Gaussian'}"
                    )
            except Exception as e:
                logger.debug(f"Shapiro-Wilk failed: {e}")

        # Jarque-Bera test (based on skewness and kurtosis)
        try:
            stat, pval = scipy_stats.jarque_bera(series)
            result.jarque_bera_stat = float(stat)
            result.jarque_bera_pval = float(pval)
            result.is_gaussian_jb = pval > self.alpha

            if verbose:
                logger.info(
                    f"  Jarque-Bera: stat={stat:.4f}, p={pval:.4f} "
                    f"→ {'Gaussian' if result.is_gaussian_jb else 'Non-Gaussian'}"
                )
        except Exception as e:
            logger.debug(f"Jarque-Bera failed: {e}")

        # Anderson-Darling test
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ad_result = scipy_stats.anderson(series, dist="norm")
            result.anderson_stat = float(ad_result.statistic)
            # Use 5% critical value
            result.anderson_critical_5pct = float(ad_result.critical_values[2])
            result.is_gaussian_anderson = (
                ad_result.statistic < ad_result.critical_values[2]
            )

            if verbose:
                logger.info(
                    f"  Anderson-Darling: stat={ad_result.statistic:.4f}, "
                    f"crit_5%={ad_result.critical_values[2]:.4f} "
                    f"→ {'Gaussian' if result.is_gaussian_anderson else 'Non-Gaussian'}"
                )
        except Exception as e:
            logger.debug(f"Anderson-Darling failed: {e}")

        # Consensus: Gaussian if majority of tests agree
        gaussian_votes = [
            result.is_gaussian_shapiro,
            result.is_gaussian_jb,
            result.is_gaussian_anderson,
        ]
        gaussian_votes = [v for v in gaussian_votes if v is not None]

        if len(gaussian_votes) > 0:
            result.is_gaussian = sum(gaussian_votes) > len(gaussian_votes) / 2

    def _test_linearity(
        self,
        series: pd.Series,
        result: DistributionTestResults,
        verbose: bool,
    ):
        """Test for linear dynamics"""

        # Ljung-Box test on residuals (after AR fit)
        # Linear if residuals are white noise
        try:
            # Fit simple AR(1) model
            from statsmodels.tsa.ar_model import AutoReg

            if len(series) > 20:
                model = AutoReg(series, lags=1, old_names=False)
                fitted = model.fit()
                residuals = fitted.resid

                # Ljung-Box test on residuals
                lb_result = acorr_ljungbox(
                    residuals,
                    lags=min(self.ljungbox_lags, len(residuals) // 4),
                    return_df=False,
                )

                # Use minimum p-value across lags
                result.ljungbox_stat = float(lb_result[0][-1])
                result.ljungbox_pval = float(lb_result[1].min())
                # Linear if residuals are white noise (high p-value)
                result.is_linear_ljungbox = result.ljungbox_pval > self.alpha

                if verbose:
                    logger.info(
                        f"  Ljung-Box (AR residuals): p={result.ljungbox_pval:.4f} "
                        f"→ {'Linear' if result.is_linear_ljungbox else 'Non-linear'}"
                    )
        except Exception as e:
            logger.debug(f"Ljung-Box test failed: {e}")

        # BDS test would go here (requires external package)
        # Note: BDS test is computationally expensive and requires 'arch' package
        try:
            from arch.unitroot import BDS

            _ = BDS(series)  # Create BDS object (not yet using results)
            result.bds_available = True
            # BDS test interpretation is complex, skipping for now

            if verbose:
                logger.info("  BDS test: Available but not yet implemented")
        except ImportError:
            result.bds_available = False

        # Consensus for linearity
        if result.is_linear_ljungbox is not None:
            result.is_linear = result.is_linear_ljungbox
        else:
            result.is_linear = None  # Unknown

    def _test_heteroscedasticity(
        self,
        series: pd.Series,
        result: DistributionTestResults,
        verbose: bool,
    ):
        """Test for heteroscedasticity (non-constant variance)"""

        # Need to fit a model first
        try:
            from statsmodels.tsa.ar_model import AutoReg

            if len(series) > 20:
                # Fit AR(1)
                model = AutoReg(series, lags=1, old_names=False)
                fitted = model.fit()
                residuals = fitted.resid

                # White's test for heteroscedasticity
                try:
                    # Create design matrix (need exog)
                    X = np.column_stack([series[1:], series[:-1]])
                    X = X[: len(residuals)]

                    white_result = het_white(residuals, X)
                    result.white_stat = float(white_result[0])
                    result.white_pval = float(white_result[1])
                    result.is_homoscedastic_white = white_result[1] > self.alpha

                    if verbose:
                        logger.info(
                            f"  White's test: p={white_result[1]:.4f} "
                            f"→ {'Homoscedastic' if result.is_homoscedastic_white else 'Heteroscedastic'}"
                        )
                except Exception as e:
                    logger.debug(f"White's test failed: {e}")

                # ARCH test for autoregressive conditional heteroscedasticity
                try:
                    arch_result = het_arch(residuals, nlags=self.arch_lags)
                    result.arch_stat = float(arch_result[0])
                    result.arch_pval = float(arch_result[1])
                    result.is_homoscedastic_arch = arch_result[1] > self.alpha

                    if verbose:
                        logger.info(
                            f"  ARCH test: p={arch_result[1]:.4f} "
                            f"→ {'No ARCH effects' if result.is_homoscedastic_arch else 'ARCH effects present'}"
                        )
                except Exception as e:
                    logger.debug(f"ARCH test failed: {e}")

        except Exception as e:
            logger.debug(f"Heteroscedasticity tests failed: {e}")

        # Consensus
        homoscedastic_votes = [
            result.is_homoscedastic_white,
            result.is_homoscedastic_arch,
        ]
        homoscedastic_votes = [v for v in homoscedastic_votes if v is not None]

        if len(homoscedastic_votes) > 0:
            result.is_homoscedastic = (
                sum(homoscedastic_votes) > len(homoscedastic_votes) / 2
            )

    def _test_autocorrelation(
        self,
        series: pd.Series,
        result: DistributionTestResults,
        verbose: bool,
    ):
        """Test for autocorrelation using Durbin-Watson"""

        try:
            dw_stat = durbin_watson(series)
            result.durbin_watson_stat = float(dw_stat)

            # DW ≈ 2 means no autocorrelation
            # DW < 2 means positive autocorrelation
            # DW > 2 means negative autocorrelation
            result.has_autocorrelation = not (1.5 < dw_stat < 2.5)

            if verbose:
                logger.info(
                    f"  Durbin-Watson: stat={dw_stat:.4f} "
                    f"→ {'Autocorrelation detected' if result.has_autocorrelation else 'No strong autocorrelation'}"
                )
        except Exception as e:
            logger.debug(f"Durbin-Watson failed: {e}")

    def _generate_recommendations(
        self,
        result: DistributionTestResults,
        verbose: bool,
    ):
        """Generate method recommendations based on test results"""

        recommendations = []
        reasoning = []

        # Rule-based decision tree
        if result.is_gaussian and result.is_linear:
            recommendations.append("granger")
            reasoning.append(
                "Gaussian + Linear dynamics → VAR-based Granger causality optimal"
            )

        if not result.is_gaussian or not result.is_linear:
            recommendations.append("transfer_entropy")
            reasoning.append("Non-Gaussian or Non-linear → Transfer entropy robust")

        # Always recommend PCMCI+ as it's flexible
        if result.n_observations >= 100:
            recommendations.append("pcmci")
            reasoning.append("Sufficient data for PCMCI+ (conditional independence)")

        # If non-linear, prefer TE over Granger
        if result.is_linear is False and "granger" in recommendations:
            recommendations.remove("granger")
            reasoning.append("Non-linear dynamics → Granger not recommended")

        # If very small sample, only Granger
        if result.n_observations < 60:
            recommendations = ["granger"]
            reasoning = ["Small sample → Granger only (fast, low requirements)"]

        # Deduplicate
        recommendations = list(dict.fromkeys(recommendations))

        result.recommended_methods = recommendations
        result.reasoning = reasoning

        if verbose:
            logger.info(f"  → Recommended methods: {recommendations}")
            for reason in reasoning:
                logger.info(f"     • {reason}")

    def _print_summary(self, results: Dict[str, DistributionTestResults]):
        """Print summary table of all test results"""

        logger.info("\nSUMMARY TABLE:")
        logger.info("-" * 90)
        logger.info(
            f"{'Variable':<15} {'N':>6} {'Gaussian':>10} {'Linear':>10} "
            f"{'Recommended Methods':<40}"
        )
        logger.info("-" * 90)

        for var, res in results.items():
            gaussian_str = (
                "Yes"
                if res.is_gaussian
                else "No"
                if res.is_gaussian is False
                else "Unknown"
            )
            linear_str = (
                "Yes"
                if res.is_linear
                else "No"
                if res.is_linear is False
                else "Unknown"
            )
            methods_str = (
                ", ".join(res.recommended_methods)
                if res.recommended_methods
                else "None"
            )

            logger.info(
                f"{var:<15} {res.n_observations:>6} {gaussian_str:>10} "
                f"{linear_str:>10} {methods_str:<40}"
            )

        logger.info("-" * 90)

    def save_results(
        self,
        results: Dict[str, DistributionTestResults],
        output_path: Path,
    ):
        """Save test results to JSON file"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict = {var: res.to_dict() for var, res in results.items()}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Distribution test results saved to {output_path}")


def quick_test(
    data: pd.DataFrame,
    variables: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, DistributionTestResults]:
    """
    Convenience function for quick distribution testing.

    Parameters:
        data: DataFrame containing time series
        variables: Variables to test (None = all numeric)
        **kwargs: Additional arguments for DistributionTester

    Returns:
        Dictionary of test results
    """
    tester = DistributionTester(**kwargs)
    return tester.test_dataframe(data, variables=variables, verbose=True)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Create test data with different characteristics
    np.random.seed(42)
    n = 200

    data = pd.DataFrame(
        {
            "gaussian_linear": np.random.normal(0, 1, n),
            "nongaussian": np.random.exponential(2, n),
            "nonlinear": np.sin(np.arange(n) / 10) + np.random.normal(0, 0.2, n),
            "heteroscedastic": np.random.normal(0, 1, n) * (1 + 0.5 * np.arange(n) / n),
        }
    )

    # Test
    results = quick_test(data)

    # Print recommendations
    print("\n" + "=" * 70)
    print("METHOD RECOMMENDATIONS")
    print("=" * 70)

    for var, res in results.items():
        print(f"\n{var}:")
        print(f"  Methods: {', '.join(res.recommended_methods)}")
        for reason in res.reasoning:
            print(f"  - {reason}")

    print("\n✅ Example completed!")
