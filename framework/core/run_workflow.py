#!/usr/bin/env python3
"""
Complete Causal Discovery Workflow

Provides a reusable framework for running causal discovery experiments:
1. Run causal discovery methods (Granger, Transfer Entropy, PCMCI+)
2. Compute consensus
3. Generate visualizations
4. Run diagnostics and validation
"""

import os

# Disable Numba JIT to avoid compatibility issues with newer pandas/numpy
os.environ["NUMBA_DISABLE_JIT"] = "1"

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _set_datetime_index_with_freq(
    df: pd.DataFrame, date_col: str, sampling_days: float = None
) -> pd.DataFrame:
    """
    Set datetime index and infer/set frequency to avoid statsmodels warnings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of date column
    sampling_days : float, optional
        Sampling interval in days (used as fallback if frequency cannot be inferred)

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and frequency set if possible
    """
    if date_col not in df.columns:
        return df

    df = df.set_index(date_col).sort_index()

    # Try to infer frequency from the index itself
    if isinstance(df.index, pd.DatetimeIndex):
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq:
                df.index.freq = inferred_freq
            elif sampling_days is not None:
                # Fallback: convert sampling_days to pandas frequency string
                # Handle common cases
                if abs(sampling_days - 0.5 / 24) < 1e-6:  # ~30 minutes
                    df.index.freq = "30min"
                elif abs(sampling_days - 1.0 / 24) < 1e-6:  # 1 hour
                    df.index.freq = "H"
                elif abs(sampling_days - 1.0) < 1e-6:  # 1 day
                    df.index.freq = "D"
                elif sampling_days < 1.0 / 24:  # Less than 1 hour
                    minutes = int(sampling_days * 24 * 60)
                    df.index.freq = f"{minutes}min"
                elif sampling_days < 1.0:  # Less than 1 day
                    hours = int(sampling_days * 24)
                    df.index.freq = f"{hours}H"
                else:  # Days or more
                    days = int(sampling_days)
                    df.index.freq = f"{days}D"
        except (ValueError, TypeError):
            # If frequency cannot be set, that's okay - just continue
            pass

    return df


def run_causal_discovery_workflow(
    data_df: pd.DataFrame,
    output_dir: Path,
    tau_max: Optional[int] = None,
    tau_max_method: str = "scientific",
    domain_max_days: int = 90,
    alpha: float = 0.05,
    sampling_days: int = 1,
    unit_id_col: Optional[str] = None,
    date_col: Optional[str] = "date",
    pairs: Optional[List[Tuple[str, str]]] = None,
    target_var: Optional[str] = None,
    clean_outputs: bool = True,
    experiment_name: Optional[str] = None,
    enable_preprocessing: bool = True,
    enable_distribution_tests: bool = True,
    enable_strength_analysis: bool = True,
    enable_temporal_validation: bool = True,
    enable_tracking: bool = True,
    method_config: Optional[Dict] = None,
    enable_consensus: bool = False,
    enable_causal_audit: bool = False,
    true_edges: Optional[set] = None,
    deseasonalize: bool = False,
) -> Dict:
    """
    Run complete causal discovery workflow on panel or single-unit data.

    This is the main framework function that orchestrates:
    0. (Optional) Experiment tracking initialization
    1. (Optional) Preprocessing pipeline
    2. (Optional) Distribution testing
    3. Experiment folder structure creation
    4. Output cleaning (if requested)
    5. Automatic pair generation (if not provided)
    6. Causal discovery (Granger, Transfer Entropy, PCMCI+) - configurable via method_config
    7. (Optional) Causal strength quantification
    8. (Optional) Consensus computation - only if enable_consensus=True
    9. (Optional) Temporal validation
    10. Visualization generation
    11. Results export

    Parameters:
        data_df (pd.DataFrame): Input data with variables and optional unit_id
        output_dir (Path): Directory to save all results (experiment folder)
        tau_max (Optional[int]): Maximum lag in timesteps. If None, auto-estimated from data.
        tau_max_method (str): Method for tau_max estimation if tau_max is None.
            Options: "scientific" (default, ACF+domain+Nyquist), "ensemble", "acf_zero"
        domain_max_days (int): Domain knowledge maximum lag in days (default: 90)
        alpha (float): Significance threshold
        sampling_days (int): Days per timestep for lag conversion
        unit_id_col (Optional[str]): Column name for panel units (None for single unit)
        date_col (Optional[str]): Column name for time index
        pairs (Optional[List[Tuple[str, str]]]): List of (cause, effect) pairs to test.
            If None, will auto-generate all pairs targeting target_var or all vs all.
        target_var (Optional[str]): If pairs is None, generate pairs with this as target.
            If None, generates all possible pairs from data variables.
        clean_outputs (bool): Whether to clean previous outputs before running
        experiment_name (Optional[str]): Name of experiment for logging
        enable_preprocessing (bool): Enable preprocessing pipeline (stationarity, outliers, etc.)
        enable_distribution_tests (bool): Enable distribution tests for method recommendation
        enable_strength_analysis (bool): Enable causal strength quantification
        enable_temporal_validation (bool): Enable temporal cross-validation
        enable_tracking (bool): Enable experiment tracking (reproducibility logging)
        method_config (Optional[Dict]): Configuration for which methods to run. Expected format:
            {
                "granger": {"enabled": True, ...},
                "transfer_entropy": {"enabled": True, ...},
                "pcmci": {"enabled": False, ...}
            }
            If None, loads from framework/config/defaults.json
        enable_consensus (bool): Enable consensus computation across methods (default: False).
            If False, only individual method results are reported.
        enable_causal_audit (bool): Run causal-audit assumption risk assessment after
            preprocessing (default: False). Requires causal-audit to be installed.
        true_edges (Optional[set]): Ground truth edges as a set of (source, target)
            tuples.  When provided, the workflow computes graph recovery metrics
            (F1, AUROC, AUPRC) for each method and saves them to
            ``graph_recovery_metrics.csv`` in the output directory.

    Returns:
        Dict: Results summary with paths to outputs
    """
    from framework.core.methods import (
        granger,
        consensus,
        _get_tigramite_pcmci,
        _get_transfer_entropy,
    )
    from framework.core.methods.visualize_results import visualize_all_results

    # Import new enhancement modules
    if enable_preprocessing:
        from framework.core.preprocessing import TimeSeriesPreprocessor
    if enable_distribution_tests:
        from framework.core.distribution_tests import DistributionTester
    if enable_strength_analysis:
        from framework.core.causal_strength import CausalStrengthQuantifier
    if enable_temporal_validation:
        from framework.core.validate import temporal_cross_validation
    if enable_tracking:
        from framework.core.experiment_tracker import ExperimentTracker

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler to capture all log output to experiment.log
    log_file = output_dir / "experiment.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    exp_name = experiment_name or output_dir.name

    # Load method configuration
    if method_config is None:
        # Load from defaults.json
        import json

        defaults_path = Path(__file__).parent.parent / "config" / "defaults.json"
        with open(defaults_path) as f:
            defaults = json.load(f)
            method_config = defaults.get("methods", {})

    # Determine which methods are enabled
    enable_granger = method_config.get("granger", {}).get("enabled", True)
    enable_transfer_entropy = method_config.get("transfer_entropy", {}).get(
        "enabled", True
    )
    enable_pcmci = method_config.get("pcmci", {}).get("enabled", True)
    enable_correlation = method_config.get("correlation", {}).get("enabled", False)
    enable_varlingam = method_config.get("varlingam", {}).get("enabled", False)
    enable_lpcmci = method_config.get("lpcmci", {}).get("enabled", False)
    enable_predictive_baseline = method_config.get("predictive_baseline", {}).get(
        "enabled", False
    )
    enable_ci_sensitivity = method_config.get("ci_sensitivity", {}).get(
        "enabled", False
    )

    # Lazy load optional methods
    transfer_entropy = None
    tigramite_pcmci = None
    varlingam_mod = None
    if enable_transfer_entropy:
        try:
            transfer_entropy = _get_transfer_entropy()
        except ImportError as e:
            logger.warning(f"Transfer Entropy disabled: {e}")
            enable_transfer_entropy = False

    if enable_pcmci:
        try:
            tigramite_pcmci = _get_tigramite_pcmci()
        except ImportError as e:
            logger.warning(f"PCMCI+ disabled: {e}")
            enable_pcmci = False

    if enable_varlingam:
        try:
            from framework.core.methods import _get_varlingam

            varlingam_mod = _get_varlingam()
        except ImportError as e:
            logger.warning(f"VAR-LiNGAM disabled: {e}")
            enable_varlingam = False

    lpcmci_mod = None
    if enable_lpcmci:
        try:
            from framework.core.methods import _get_lpcmci

            lpcmci_mod = _get_lpcmci()
        except ImportError as e:
            logger.warning(f"LPCMCI disabled: {e}")
            enable_lpcmci = False

    predictive_baseline_mod = None
    if enable_predictive_baseline:
        try:
            from framework.core.methods import _get_predictive_baseline

            predictive_baseline_mod = _get_predictive_baseline()
        except ImportError as e:
            logger.warning(f"Predictive baseline disabled: {e}")
            enable_predictive_baseline = False

    logger.info("=" * 70)
    logger.info(f"CAUSAL DISCOVERY WORKFLOW: {exp_name}")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Methods enabled: Granger={enable_granger}, TE={enable_transfer_entropy}, "
        f"PCMCI+={enable_pcmci}, LPCMCI={enable_lpcmci}, "
        f"VAR-LiNGAM={enable_varlingam}, Correlation={enable_correlation}"
    )

    # STAGE 0A: TAU_MAX ESTIMATION (if tau_max is None)
    tau_max_result = None
    if tau_max is None:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 0A: TAU_MAX ESTIMATION")
        logger.info("=" * 70)

        from framework.core.tau_max_estimation import estimate_tau_max_scientific

        # Use sample data for estimation (first unit if panel data)
        if unit_id_col and unit_id_col in data_df.columns:
            sample_unit = data_df[unit_id_col].iloc[0]
            sample_data = data_df[data_df[unit_id_col] == sample_unit].copy()
            logger.info(f"Estimating tau_max from sample unit: {sample_unit}")
        else:
            sample_data = data_df.copy()

        # Estimate tau_max across ALL numeric variable pairs (not just the first two).
        # Different variables may have different memory lengths; use the maximum
        # ACF zero-crossing across all variables to avoid underestimating the
        # system's temporal memory (Issue 6 fix).
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            tau_max_estimates = []
            # Sample up to 5 variable pairs for efficiency
            pairs_to_test = min(5, len(numeric_cols))
            for i in range(pairs_to_test):
                for j in range(i + 1, min(i + 2, len(numeric_cols))):
                    series_x = sample_data[numeric_cols[i]].dropna()
                    series_y = sample_data[numeric_cols[j]].dropna()
                    if len(series_x) > 50 and len(series_y) > 50:
                        try:
                            result_ij = estimate_tau_max_scientific(
                                series_x=series_x,
                                series_y=series_y,
                                sampling_interval_days=sampling_days,
                                domain_max_days=domain_max_days,
                            )
                            tau_max_estimates.append(result_ij)
                        except Exception:
                            pass

            if tau_max_estimates:
                # Use the maximum tau_max across all tested pairs (conservative:
                # ensures the longest memory in the system is captured)
                all_tau = [r["tau_max"] for r in tau_max_estimates]
                max_idx = int(np.argmax(all_tau))
                tau_max_result = tau_max_estimates[max_idx]
                tau_max = tau_max_result["tau_max"]

                logger.info(
                    f"✅ tau_max Estimation Results (max across {len(tau_max_estimates)} variable pairs):"
                )
                logger.info(f"   Method: {tau_max_result['method']}")
                logger.info(
                    f"   ACF zero-crossing: {tau_max_result['acf_estimate']} timesteps ({tau_max_result['acf_estimate_days']:.0f} days)"
                )
                logger.info(
                    f"   Domain constraint: {tau_max_result['domain_constraint']} timesteps ({tau_max_result['domain_constraint_days']:.0f} days)"
                )
                logger.info(
                    f"   Nyquist constraint: {tau_max_result['nyquist_constraint']} timesteps ({tau_max_result['nyquist_constraint_days']:.0f} days)"
                )
                logger.info(
                    f"   Selected (minimum of constraints): {tau_max} timesteps ({tau_max_result['tau_max_days']:.0f} days)"
                )
                logger.info(
                    f"   Binding constraint: {tau_max_result['binding_constraint']}"
                )
                logger.info(f"   All estimates: {all_tau} → max = {tau_max}")
            else:
                logger.warning(
                    "tau_max estimation failed for all variable pairs. Using default: 12"
                )
                tau_max = 12
                tau_max_result = None
        else:
            logger.warning(
                "Insufficient variables for tau_max estimation. Using default: 12"
            )
            tau_max = 12
            tau_max_result = None
    else:
        logger.info(
            f"Using provided tau_max: {tau_max} timesteps ({tau_max * sampling_days} days)"
        )
        tau_max_result = None

    logger.info(
        f"Parameters: tau_max={tau_max}, alpha={alpha}, sampling_days={sampling_days}"
    )
    logger.info(
        f"Enhancements: preprocessing={enable_preprocessing}, dist_tests={enable_distribution_tests}, "
        f"strength={enable_strength_analysis}, validation={enable_temporal_validation}, tracking={enable_tracking}"
    )

    # Initialize experiment tracker
    tracker = None
    if enable_tracking:
        tracker = ExperimentTracker(
            experiment_name=exp_name,
            output_dir=output_dir,
            description=f"Causal discovery with tau_max={tau_max}, alpha={alpha}",
        )
        tracker.log_parameters(
            {
                "tau_max": tau_max,
                "tau_max_method": tau_max_method if tau_max_result else "provided",
                "tau_max_estimated": tau_max_result is not None,
                "tau_max_binding_constraint": tau_max_result.get("binding_constraint")
                if tau_max_result
                else None,
                "alpha": alpha,
                "sampling_days": sampling_days,
                "domain_max_days": domain_max_days,
                "enable_preprocessing": enable_preprocessing,
                "enable_distribution_tests": enable_distribution_tests,
                "enable_strength_analysis": enable_strength_analysis,
                "enable_temporal_validation": enable_temporal_validation,
            }
        )

        # Log tau_max estimation results if available
        if tau_max_result:
            import json

            tau_max_log_path = output_dir / "tau_max_estimation.json"
            with open(tau_max_log_path, "w") as f:
                json.dump(tau_max_result, f, indent=2, default=str)
            tracker.log_file_path(tau_max_log_path, "tau_max_estimation")
            logger.info(f"✅ tau_max estimation log saved: {tau_max_log_path}")

        logger.info("✅ Experiment tracker initialized")

    # Preprocessing stage
    if enable_preprocessing:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 0: PREPROCESSING")
        logger.info("=" * 70)

        metadata_cols = [col for col in [unit_id_col, date_col] if col is not None]

        preprocessor = TimeSeriesPreprocessor(
            interpolation_method="linear",  # Faster than GP for large datasets
            stationarity_test="adf",  # Use ADF test
            normalize=True,
            outlier_method="iqr",
            remove_seasonality=False,  # Keep domain semantics
        )

        data_df, prep_report = preprocessor.preprocess(
            data_df, metadata_cols=metadata_cols, verbose=True
        )

        # Save preprocessing report
        prep_report_path = output_dir / "preprocessing_report.json"
        prep_report.save(prep_report_path)
        logger.info(f"✅ Preprocessing report saved: {prep_report_path}")

        if prep_report.quality_flags:
            logger.warning("⚠️  Quality issues detected:")
            for flag in prep_report.quality_flags:
                logger.warning(f"   - {flag}")

        if tracker:
            tracker.log_data_hash(data_df, data_name="preprocessed_data")
            tracker.log_file_path(prep_report_path, "preprocessing_report")
    else:
        if tracker:
            tracker.log_data_hash(data_df, data_name="raw_data")

    # Deseasonalization (optional — removes annual/periodic cycles)
    if deseasonalize:
        logger.info("\n" + "=" * 70)
        logger.info("DESEASONALIZATION (anomaly series)")
        logger.info("=" * 70)
        try:
            metadata_cols = [col for col in [unit_id_col, date_col] if col is not None]
            numeric_cols = data_df.select_dtypes(include=["number"]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in metadata_cols]

            for col in numeric_cols:
                series = data_df[col].dropna()
                if len(series) < 10:
                    continue
                # STL-like: subtract rolling mean (window = min(365, T//3))
                window = min(365, max(7, len(series) // 3))
                if window % 2 == 0:
                    window += 1
                seasonal = series.rolling(
                    window=window, center=True, min_periods=1
                ).mean()
                data_df[col] = data_df[col] - seasonal

            data_df = data_df.dropna(how="all")
            logger.info(
                f"  Deseasonalized {len(numeric_cols)} variables (window={window})"
            )
        except Exception as e:
            logger.warning(f"Deseasonalization failed: {e}")

    # Causal-audit stage: assumption risk assessment (optional)
    causal_audit_result = None
    if enable_causal_audit:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 0B: CAUSAL-AUDIT (ASSUMPTION RISK ASSESSMENT)")
        logger.info("=" * 70)

        try:
            from causal_audit import RiskAwareGatekeeper

            audit_output_dir = output_dir / "causal_audit"
            audit_output_dir.mkdir(parents=True, exist_ok=True)

            gatekeeper = RiskAwareGatekeeper(random_seed=42)

            # For panel data, audit a sample unit
            if unit_id_col and unit_id_col in data_df.columns:
                sample_unit = data_df[unit_id_col].iloc[0]
                audit_data = data_df[data_df[unit_id_col] == sample_unit].copy()
                audit_data = audit_data.drop(
                    columns=[
                        c
                        for c in [unit_id_col, date_col]
                        if c and c in audit_data.columns
                    ],
                    errors="ignore",
                )
                if date_col and date_col in data_df.columns:
                    audit_data.index = pd.to_datetime(
                        data_df[data_df[unit_id_col] == sample_unit][date_col]
                    )
                logger.info(f"Auditing sample unit: {sample_unit}")
            else:
                audit_data = data_df.select_dtypes(include=["number"]).copy()
                if not isinstance(audit_data.index, pd.DatetimeIndex):
                    audit_data = _set_datetime_index_with_freq(
                        data_df.copy(), date_col, sampling_days
                    )
                    audit_data = audit_data.select_dtypes(include=["number"])

            causal_audit_result = gatekeeper.analyze(
                data=audit_data,
                output_dir=str(audit_output_dir),
            )

            policy = causal_audit_result["policy"]
            risks = causal_audit_result["risk_profile"].risks

            logger.info(f"  Decision: {policy.decision.upper()}")
            if policy.recommended_method:
                logger.info(
                    f"  Recommended: {policy.recommended_method} "
                    f"(confidence={policy.confidence:.0%})"
                )
            for name, r in risks.items():
                logger.info(
                    f"    {name}: {r['mean']:.3f} "
                    f"[{r['lower_95']:.3f}, {r['upper_95']:.3f}]"
                )

            if policy.decision == "abstain":
                logger.warning(
                    "⚠️  causal-audit recommends ABSTENTION. "
                    "Discovery will proceed but results may be unreliable."
                )

            if tracker:
                tracker.log_file_path(
                    audit_output_dir / "risk_profile.json", "causal_audit_risk_profile"
                )
                tracker.log_file_path(
                    audit_output_dir / "recommendation_policy.json",
                    "causal_audit_policy",
                )

        except ImportError:
            logger.warning(
                "causal-audit not installed. Skipping assumption risk assessment. "
                "Install with: uv pip install -e /path/to/causal-audit"
            )
        except Exception as e:
            logger.warning(f"causal-audit failed: {e}. Continuing without it.")

    # Distribution testing stage
    distribution_results = None
    if enable_distribution_tests:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: DISTRIBUTION TESTING")
        logger.info("=" * 70)

        tester = DistributionTester(alpha=alpha)

        # For panel data, test a sample unit
        if unit_id_col and unit_id_col in data_df.columns:
            sample_unit = data_df[unit_id_col].iloc[0]
            sample_data = data_df[data_df[unit_id_col] == sample_unit].copy()
            sample_data = _set_datetime_index_with_freq(
                sample_data, date_col, sampling_days
            )
            logger.info(f"Testing distributions on sample unit: {sample_unit}")
        else:
            sample_data = data_df.copy()
            sample_data = _set_datetime_index_with_freq(
                sample_data, date_col, sampling_days
            )

        # Test only numeric variables (exclude metadata)
        metadata_cols = [col for col in [unit_id_col, date_col] if col is not None]
        numeric_cols = sample_data.select_dtypes(include=["number"]).columns.tolist()
        test_cols = [c for c in numeric_cols if c not in metadata_cols]

        distribution_results = {}
        for col in test_cols:
            if col in sample_data.columns:
                distribution_results[col] = tester.test_variable(
                    sample_data[col].dropna()
                )
                logger.info(
                    f"  {col}: Gaussian={distribution_results[col].is_gaussian}, "
                    f"Linear={distribution_results[col].is_linear}"
                )
                logger.info(
                    f"    → Recommended methods: {distribution_results[col].recommended_methods}"
                )

        # Save distribution test results
        dist_test_path = output_dir / "distribution_tests.json"
        tester.save_results(distribution_results, dist_test_path)
        logger.info(f"✅ Distribution tests saved: {dist_test_path}")

        if tracker:
            tracker.log_file_path(dist_test_path, "distribution_tests")

    # Clean previous outputs if requested
    if clean_outputs:
        logger.info("\nCleaning previous outputs...")
        for sub in ["figures", "checkpoints", "diagnostics"]:
            subdir = output_dir / sub
            if subdir.exists():

                def handle_remove_error(func, path, exc_info):
                    """Handle errors when removing files (e.g., OS X metadata files)."""
                    pass  # Ignore errors for files we can't remove

                shutil.rmtree(subdir, onerror=handle_remove_error)
        for f in output_dir.glob("results_*.csv"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        for f in output_dir.glob("consensus*.csv"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        for f in output_dir.glob("*.graphml"):
            try:
                f.unlink()
            except FileNotFoundError:
                pass

    # Create experiment folder structure
    logger.info("\nCreating experiment folder structure...")
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Determine which methods are enabled
    enabled_methods = []
    if method_config:
        if method_config.get("granger", {}).get("enabled", False):
            enabled_methods.append("granger")
        if method_config.get("transfer_entropy", {}).get("enabled", False):
            enabled_methods.append("transfer_entropy")
        if method_config.get("pcmci", {}).get("enabled", False):
            enabled_methods.append("pcmci")
        if method_config.get("correlation", {}).get("enabled", False):
            enabled_methods.append("correlation")
        if method_config.get("varlingam", {}).get("enabled", False):
            enabled_methods.append("varlingam")
        if method_config.get("lpcmci", {}).get("enabled", False):
            enabled_methods.append("lpcmci")
        if method_config.get("predictive_baseline", {}).get("enabled", False):
            enabled_methods.append("predictive_baseline")
    else:
        # Fallback to individual flags if method_config not provided
        if enable_granger:
            enabled_methods.append("granger")
        if enable_transfer_entropy:
            enabled_methods.append("transfer_entropy")
        if enable_pcmci:
            enabled_methods.append("pcmci")
        if enable_correlation:
            enabled_methods.append("correlation")
        if enable_varlingam:
            enabled_methods.append("varlingam")
        if enable_lpcmci:
            enabled_methods.append("lpcmci")

    logger.info(f"Creating directories for enabled methods: {enabled_methods}")

    # Create numbered-stage directory structure for publication-ready organization
    # Only create directories for enabled methods - directly under output_dir, not in nested "results" folder
    (output_dir / "method").mkdir(parents=True, exist_ok=True)
    for method in enabled_methods:
        (output_dir / "method" / method / "1-raw").mkdir(parents=True, exist_ok=True)
    # Only create consensus directories if consensus is enabled
    if enable_consensus:
        (output_dir / "consensus").mkdir(parents=True, exist_ok=True)

    # Auto-generate pairs if not provided
    if pairs is None:
        logger.info("\nAuto-generating variable pairs...")
        # Get variable columns (exclude unit_id, date, and non-numeric columns)
        exclude_cols = [unit_id_col, date_col] if unit_id_col else [date_col]

        # Only include numeric columns for causal analysis
        numeric_cols = data_df.select_dtypes(include=["number"]).columns.tolist()
        var_cols = [c for c in numeric_cols if c not in exclude_cols]

        excluded_non_numeric = [
            c
            for c in data_df.columns
            if c not in numeric_cols and c not in exclude_cols
        ]
        if excluded_non_numeric:
            logger.info(f"Excluding non-numeric columns: {excluded_non_numeric}")

        if target_var:
            # Generate pairs: all variables -> target
            pairs = [(cause, target_var) for cause in var_cols if cause != target_var]
            logger.info(
                f"Generated {len(pairs)} pairs with target variable '{target_var}'"
            )
        else:
            # Generate all possible pairs
            pairs = [
                (cause, effect)
                for cause in var_cols
                for effect in var_cols
                if cause != effect
            ]
            logger.info(f"Generated {len(pairs)} pairs from {len(var_cols)} variables")

        logger.info(f"Variable pairs: {pairs}")
    else:
        logger.info(f"Using provided {len(pairs)} variable pairs")

    # Step 1: Run causal discovery methods
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: CAUSAL DISCOVERY")
    logger.info("=" * 70)

    results = {}

    # Determine if panel data or single unit
    if unit_id_col and unit_id_col in data_df.columns:
        # Panel data: process each unit
        unique_units = data_df[unit_id_col].unique()
        logger.info(f"Processing {len(unique_units)} units...")

        # Initialize CSV file paths for streaming output
        granger_csv = output_dir / "results_granger.csv"
        te_csv = output_dir / "results_transfer_entropy.csv"
        pcmci_csv = output_dir / "results_pcmci.csv"
        correlation_csv = output_dir / "results_correlation.csv"

        # Track if headers have been written
        granger_header_written = False
        te_header_written = False
        pcmci_header_written = False
        correlation_header_written = False

        try:
            for idx, unit_id in enumerate(unique_units, 1):
                if idx % 50 == 0:
                    logger.info(f"Progress: {idx}/{len(unique_units)} units")

                unit_df = data_df[data_df[unit_id_col] == unit_id].copy()
                unit_df = _set_datetime_index_with_freq(
                    unit_df, date_col, sampling_days
                )

                # Extract variable columns
                var_cols = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
                unit_df = unit_df[var_cols]

                # Skip if insufficient data
                if len(unit_df.dropna()) < 60:
                    continue

                # Granger
                if enable_granger:
                    try:
                        granger_df = granger.batch_granger_causality(
                            unit_df,
                            pairs,
                            maxlag=tau_max,
                            alpha=alpha,
                            sampling_days=sampling_days,
                        )
                        if granger_df is not None and len(granger_df) > 0:
                            granger_df[unit_id_col] = unit_id
                            # Stream to CSV with header only on first write
                            mode = "w" if not granger_header_written else "a"
                            granger_df.to_csv(
                                granger_csv,
                                index=False,
                                header=(not granger_header_written),
                                mode=mode,
                            )
                            granger_header_written = True
                            results["granger"] = granger_df  # Keep schema for metadata
                    except Exception as e:
                        logger.debug(f"Granger failed for {unit_id}: {e}")

                # Transfer Entropy - Test at representative lag subset
                if enable_transfer_entropy:
                    try:
                        candidate_lags = [1, 2, 3, 5, 7, 10, 14, 21, 30]
                        te_delays = [d for d in candidate_lags if d <= tau_max]
                        if tau_max not in te_delays:
                            te_delays.append(tau_max)

                        te_df = transfer_entropy.batch_transfer_entropy(
                            unit_df,
                            pairs,
                            delays=te_delays,
                            alpha=alpha,
                            apply_global_fdr=False,
                            reduce_to_best_per_pair=True,
                        )
                        if te_df is not None and len(te_df) > 0:
                            te_df[unit_id_col] = unit_id
                            # Stream to CSV with header only on first write
                            mode = "w" if not te_header_written else "a"
                            te_df.to_csv(
                                te_csv,
                                index=False,
                                header=(not te_header_written),
                                mode=mode,
                            )
                            te_header_written = True
                            results["transfer_entropy"] = (
                                te_df  # Keep schema for metadata
                            )
                    except Exception as e:
                        logger.debug(f"TE failed for {unit_id}: {e}")

                # PCMCI+ - OPTIMIZED: Aggregate across units instead of running per-unit
                # This prevents memory explosion (4,356 units × memory per run)
                # Store unit data for aggregation after all units are processed
                if enable_pcmci:
                    if idx == 1:
                        # Initialize list to store unit dataframes
                        unit_data_list = []
                    # Store unit data for later aggregation
                    # Preserve time column by resetting index before storing
                    unit_df_for_agg = unit_df.copy()
                    time_col_name = date_col if date_col else "time"

                    if isinstance(unit_df_for_agg.index, pd.DatetimeIndex):
                        # Reset index to convert DatetimeIndex to column
                        unit_df_for_agg = unit_df_for_agg.reset_index()
                        # Find and rename the datetime column to standard name
                        for col in unit_df_for_agg.columns:
                            if (
                                col not in var_cols
                                and pd.api.types.is_datetime64_any_dtype(
                                    unit_df_for_agg[col]
                                )
                            ):
                                if col != time_col_name:
                                    unit_df_for_agg = unit_df_for_agg.rename(
                                        columns={col: time_col_name}
                                    )
                                break
                    elif date_col and date_col not in unit_df_for_agg.columns:
                        # If date_col is not in columns, it might be in index
                        unit_df_for_agg = unit_df_for_agg.reset_index()
                        # Try to find and rename datetime column
                        for col in unit_df_for_agg.columns:
                            if (
                                col not in var_cols
                                and pd.api.types.is_datetime64_any_dtype(
                                    unit_df_for_agg[col]
                                )
                            ):
                                if col != time_col_name:
                                    unit_df_for_agg = unit_df_for_agg.rename(
                                        columns={col: time_col_name}
                                    )
                                break

                    # Ensure time column exists with correct name
                    if time_col_name not in unit_df_for_agg.columns:
                        # Try to find any datetime column
                        for col in unit_df_for_agg.columns:
                            if (
                                col not in var_cols
                                and pd.api.types.is_datetime64_any_dtype(
                                    unit_df_for_agg[col]
                                )
                            ):
                                unit_df_for_agg = unit_df_for_agg.rename(
                                    columns={col: time_col_name}
                                )
                                break

                    unit_data_list.append(unit_df_for_agg)

                    # Run PCMCI+ once on aggregated data after processing all units
                    if idx == len(unique_units):
                        try:
                            logger.info(
                                "Aggregating panel data for PCMCI+ (median across units)..."
                            )
                            logger.info(f"Aggregating {len(unit_data_list)} units...")

                            # Combine all unit data
                            all_units_data = pd.concat(
                                unit_data_list, ignore_index=True
                            )

                            # Aggregate by time (median to be robust to outliers)
                            var_cols = list(
                                set([p[0] for p in pairs] + [p[1] for p in pairs])
                            )

                            # Determine time column for aggregation
                            time_col_for_agg = None
                            if date_col and date_col in all_units_data.columns:
                                time_col_for_agg = date_col
                            elif "time" in all_units_data.columns:
                                time_col_for_agg = "time"
                            else:
                                # Find datetime column
                                for col in all_units_data.columns:
                                    if (
                                        col not in var_cols
                                        and pd.api.types.is_datetime64_any_dtype(
                                            all_units_data[col]
                                        )
                                    ):
                                        time_col_for_agg = col
                                        break

                            if time_col_for_agg:
                                # Aggregate by time column
                                df_agg = (
                                    all_units_data.groupby(time_col_for_agg)[var_cols]
                                    .median()
                                    .reset_index()
                                )
                                df_agg = df_agg.sort_values(
                                    time_col_for_agg
                                ).reset_index(drop=True)
                                df_agg = df_agg.set_index(time_col_for_agg)
                                logger.info(
                                    f"Aggregated to {len(df_agg)} time points (from {len(all_units_data)} unit-time observations)"
                                )
                            else:
                                # Fallback: aggregate by index position (shouldn't happen with proper time column)
                                logger.warning(
                                    "No time column found for aggregation, using index position fallback"
                                )
                                all_units_data["_time_idx"] = all_units_data.groupby(
                                    unit_id_col
                                ).cumcount()
                                df_agg = (
                                    all_units_data.groupby("_time_idx")[var_cols]
                                    .median()
                                    .reset_index(drop=True)
                                )

                            # Interpolate missing values
                            df_agg = df_agg.interpolate(
                                method="linear", limit_direction="both"
                            )
                            df_agg = df_agg.ffill().bfill()

                            logger.info(
                                f"Running PCMCI+ on aggregated data (shape: {df_agg.shape})..."
                            )
                            pcmci_df = tigramite_pcmci.batch_pcmci(
                                df_agg,
                                pairs,
                                tau_max=tau_max,
                                alpha=alpha,
                                sampling_days=sampling_days,
                            )
                            if pcmci_df is not None and len(pcmci_df) > 0:
                                # Mark as aggregated (no unit_id column)
                                pcmci_df.to_csv(
                                    pcmci_csv,
                                    index=False,
                                    mode="w",  # Overwrite any previous writes
                                )
                                pcmci_header_written = True
                                results["pcmci"] = pcmci_df
                                logger.info(
                                    f"✓ PCMCI+ completed on aggregated data: {len(pcmci_df)} results"
                                )
                        except Exception as e:
                            logger.error(f"PCMCI+ failed on aggregated data: {e}")
                            import traceback

                            logger.debug(traceback.format_exc())

                # Correlation (pairwise streaming or sparse)
                # Correlation - OPTIMIZED: Aggregate across units instead of running per-unit
                # This prevents memory explosion (4,356 units × memory per run)
                # Store unit data for aggregation after all units are processed
                if enable_correlation:
                    if idx == 1:
                        # Initialize list to store unit dataframes
                        correlation_unit_data_list = []
                    # Store unit data for later aggregation
                    # Preserve time column by resetting index before storing
                    unit_df_for_corr = unit_df.copy()
                    time_col_name = date_col if date_col else "time"

                    if isinstance(unit_df_for_corr.index, pd.DatetimeIndex):
                        # Reset index to convert DatetimeIndex to column
                        unit_df_for_corr = unit_df_for_corr.reset_index()
                        if time_col_name not in unit_df_for_corr.columns:
                            # If reset created 'index' column, rename it
                            if "index" in unit_df_for_corr.columns:
                                unit_df_for_corr = unit_df_for_corr.rename(
                                    columns={"index": time_col_name}
                                )
                    elif time_col_name not in unit_df_for_corr.columns:
                        # Create synthetic time index if missing
                        unit_df_for_corr[time_col_name] = pd.date_range(
                            start="2017-01-01",
                            periods=len(unit_df_for_corr),
                            freq=f"{int(sampling_days)}D",
                        )

                    # Add unit_id for tracking (optional, for debugging)
                    unit_df_for_corr[unit_id_col] = unit_id
                    correlation_unit_data_list.append(unit_df_for_corr.copy())

                    # Run correlation on aggregated data after processing all units
                    if idx == len(unique_units):
                        logger.info(
                            "Aggregating panel data for Correlation (median across units)..."
                        )
                        logger.info(
                            f"Aggregating {len(correlation_unit_data_list)} units..."
                        )

                        all_units_corr_data = pd.concat(
                            correlation_unit_data_list, ignore_index=True
                        )

                        var_cols = list(
                            set([p[0] for p in pairs] + [p[1] for p in pairs])
                        )

                        # Group by time and aggregate (median across units)
                        if time_col_name in all_units_corr_data.columns:
                            df_agg_corr = (
                                all_units_corr_data.groupby(time_col_name)[var_cols]
                                .median()
                                .reset_index()
                            )
                            df_agg_corr = df_agg_corr.sort_values(
                                time_col_name
                            ).reset_index(drop=True)
                            df_agg_corr = df_agg_corr.set_index(time_col_name)
                        else:
                            # Fallback: group by integer index
                            df_agg_corr = (
                                all_units_corr_data.groupby(all_units_corr_data.index)[
                                    var_cols
                                ]
                                .median()
                                .reset_index(drop=True)
                            )

                        # Handle missing values
                        df_agg_corr = df_agg_corr.interpolate(
                            method="linear", limit_direction="both"
                        )
                        df_agg_corr = df_agg_corr.ffill().bfill()

                        logger.info(
                            f"Running Correlation on aggregated data (shape: {df_agg_corr.shape})..."
                        )

                        try:
                            from framework.core.methods import (
                                correlation as corr_module,
                            )

                            # Get correlation config options
                            corr_config = method_config.get("correlation", {})
                            use_sparse = corr_config.get("sparse_thresholding", False)
                            p_threshold = corr_config.get("p_threshold", alpha)
                            use_streaming = corr_config.get("use_streaming", True)

                            if use_sparse:
                                # Sparse correlation analysis
                                corr_df = corr_module.sparse_pairwise_correlations(
                                    df_agg_corr,
                                    alpha=alpha,
                                    p_threshold=p_threshold,
                                    use_streaming=use_streaming,
                                )
                            else:
                                # Pairwise streaming correlation (default)
                                correlation_results = []
                                for x_var, y_var in pairs:
                                    x = df_agg_corr[x_var].values
                                    y = df_agg_corr[y_var].values
                                    result = corr_module.pairwise_streaming_correlation(
                                        x, y
                                    )
                                    correlation_results.append(
                                        {
                                            "cause": x_var,
                                            "effect": y_var,
                                            "lag": 0,  # Correlation is lag-0
                                            "correlation": result["correlation"],
                                            "p_value": result["p_value"],
                                            "n_obs": result["n_obs"],
                                            "significant": result["p_value"] < alpha,
                                        }
                                    )
                                corr_df = pd.DataFrame(correlation_results)

                            if corr_df is not None and len(corr_df) > 0:
                                # Save to CSV
                                corr_df.to_csv(correlation_csv, index=False)
                                logger.info(
                                    f"✅ Correlation results saved: {correlation_csv}"
                                )
                                results["correlation"] = corr_df
                            else:
                                logger.warning("No correlation results generated")
                                results["correlation"] = pd.DataFrame()
                        except Exception as e:
                            logger.error(
                                f"Correlation analysis failed: {e}", exc_info=True
                            )
                            results["correlation"] = pd.DataFrame()

                # Memory cleanup after each unit
                import gc

                del unit_df
                gc.collect()

        except KeyboardInterrupt:
            logger.info("Experiment interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Error during unit processing: {e}")
            raise

        # Reload results from disk as DataFrames for subsequent processing
        if granger_csv.exists() and granger_csv.stat().st_size > 0:
            results["granger"] = pd.read_csv(granger_csv)
        else:
            results["granger"] = pd.DataFrame()

        if te_csv.exists() and te_csv.stat().st_size > 0:
            results["transfer_entropy"] = pd.read_csv(te_csv)
        else:
            results["transfer_entropy"] = pd.DataFrame()

        if pcmci_csv.exists() and pcmci_csv.stat().st_size > 0:
            results["pcmci"] = pd.read_csv(pcmci_csv)
        else:
            results["pcmci"] = pd.DataFrame()

        if correlation_csv.exists() and correlation_csv.stat().st_size > 0:
            results["correlation"] = pd.read_csv(correlation_csv)
        else:
            results["correlation"] = pd.DataFrame()

    else:
        # Single unit data
        logger.info("Processing single unit data...")
        data_df = _set_datetime_index_with_freq(data_df, date_col, sampling_days)

        var_cols = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        data_df = data_df[var_cols]

        # Helper: save method result to disk immediately after computation
        def _save_method_result(method_name: str, result_df):
            """Save a single method's result to disk immediately."""
            if result_df is not None and len(result_df) > 0:
                method_dir = output_dir / "method" / method_name / "1-raw"
                method_dir.mkdir(parents=True, exist_ok=True)
                out_file = method_dir / f"results_{method_name}.csv"
                result_df.to_csv(out_file, index=False)
                logger.info(
                    f"  ✓ {method_name}: {len(result_df)} edges saved → {out_file}"
                )
            else:
                logger.info(f"  ✓ {method_name}: 0 edges (no significant results)")

        # Granger
        if enable_granger:
            logger.info("  Running Granger causality...")
            results["granger"] = granger.batch_granger_causality(
                data_df, pairs, maxlag=tau_max, alpha=alpha, sampling_days=sampling_days
            )
            _save_method_result("granger", results["granger"])
        else:
            results["granger"] = pd.DataFrame()

        # Transfer Entropy - Test at representative lag subset
        # Using logarithmically-spaced lags to cover key timescales efficiently:
        # short (1-3d), medium (5-10d), long (14-30d) — standard in TE analysis
        if enable_transfer_entropy:
            logger.info("  Running Transfer Entropy...")
            # Build representative lag grid up to tau_max
            candidate_lags = [1, 2, 3, 5, 7, 10, 14, 21, 30]
            te_delays = [d for d in candidate_lags if d <= tau_max]
            # Ensure tau_max itself is included
            if tau_max not in te_delays:
                te_delays.append(tau_max)

            results["transfer_entropy"] = transfer_entropy.batch_transfer_entropy(
                data_df,
                pairs,
                delays=te_delays,
                alpha=alpha,
                apply_global_fdr=False,
                reduce_to_best_per_pair=True,
            )
            _save_method_result("transfer_entropy", results["transfer_entropy"])
        else:
            results["transfer_entropy"] = pd.DataFrame()

        # PCMCI+
        if enable_pcmci:
            logger.info("  Running PCMCI+...")
            pcmci_test_method = method_config.get("pcmci", {}).get(
                "test_method", "auto"
            )
            results["pcmci"] = tigramite_pcmci.batch_pcmci(
                data_df,
                pairs,
                tau_max=tau_max,
                test_method=pcmci_test_method,
                alpha=alpha,
                sampling_days=sampling_days,
            )
            _save_method_result("pcmci", results["pcmci"])
        else:
            results["pcmci"] = pd.DataFrame()

        # VAR-LiNGAM
        if enable_varlingam:
            logger.info("  Running VAR-LiNGAM...")
            results["varlingam"] = varlingam_mod.batch_varlingam(
                data_df,
                pairs,
                lags=tau_max,
                alpha=alpha,
                sampling_days=sampling_days,
            )
            _save_method_result("varlingam", results["varlingam"])
        else:
            results["varlingam"] = pd.DataFrame()

        # LPCMCI (latent confounders)
        if enable_lpcmci:
            logger.info("  Running LPCMCI...")
            results["lpcmci"] = lpcmci_mod.batch_lpcmci(
                data_df,
                pairs,
                tau_max=tau_max,
                alpha=alpha,
                sampling_days=sampling_days,
            )
            _save_method_result("lpcmci", results["lpcmci"])
        else:
            results["lpcmci"] = pd.DataFrame()

        # Predictive baseline (RF feature importance — non-causal)
        if enable_predictive_baseline:
            logger.info("  Running Predictive Baseline (RF)...")
            results["predictive_baseline"] = (
                predictive_baseline_mod.batch_predictive_baseline(
                    data_df,
                    pairs,
                    lags=tau_max,
                    alpha=alpha,
                    sampling_days=sampling_days,
                )
            )
            _save_method_result("predictive_baseline", results["predictive_baseline"])
        else:
            results["predictive_baseline"] = pd.DataFrame()

        # Correlation (pairwise streaming or sparse)
        if enable_correlation:
            logger.info("  Running Correlation analysis...")
            from framework.core.methods import correlation as corr_module

            # Get correlation config options
            corr_config = method_config.get("correlation", {})
            use_sparse = corr_config.get("sparse_thresholding", False)
            p_threshold = corr_config.get("p_threshold", alpha)
            use_streaming = corr_config.get("use_streaming", True)

            if use_sparse:
                # Sparse correlation analysis
                results["correlation"] = corr_module.sparse_pairwise_correlations(
                    data_df,
                    alpha=alpha,
                    p_threshold=p_threshold,
                    use_streaming=use_streaming,
                )
                _save_method_result("correlation", results["correlation"])
            else:
                # Pairwise streaming correlation (default)
                correlation_results = []
                for x_var, y_var in pairs:
                    x = data_df[x_var].values
                    y = data_df[y_var].values
                    result = corr_module.pairwise_streaming_correlation(x, y)
                    correlation_results.append(
                        {
                            "cause": x_var,
                            "effect": y_var,
                            "lag": 0,  # Correlation is lag-0
                            "correlation": result["correlation"],
                            "p_value": result["p_value"],
                            "n_obs": result["n_obs"],
                            "significant": result["p_value"] < alpha,
                        }
                    )
                results["correlation"] = pd.DataFrame(correlation_results)
            _save_method_result("correlation", results["correlation"])
        else:
            results["correlation"] = pd.DataFrame()

    # Save individual method results (for multi-unit, already streamed to disk; for single-unit, save now)
    # Use numbered stages: method/<method>/1-raw/
    methods_results_dir = output_dir / "method"
    methods_results_dir.mkdir(parents=True, exist_ok=True)

    if unit_id_col and unit_id_col in data_df.columns:
        # Multi-unit path - results already streamed to disk, just log
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: CAUSAL DISCOVERY - Streaming complete")
        logger.info("=" * 70)
        for method, df in results.items():
            if df is not None and len(df) > 0:
                method_dir = methods_results_dir / method / "1-raw"
                method_dir.mkdir(parents=True, exist_ok=True)
                output_file = method_dir / f"results_{method}.csv"
                logger.info(f"  {method}: Results streamed to disk → {output_file}")
            else:
                logger.warning(f"  {method}: No results")
    else:
        # Single-unit path - save results now
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: CAUSAL DISCOVERY - Complete")
        logger.info("=" * 70)
        for method, df in results.items():
            if df is not None and len(df) > 0:
                method_dir = methods_results_dir / method / "1-raw"
                method_dir.mkdir(parents=True, exist_ok=True)
                output_file = method_dir / f"results_{method}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"  {method}: {len(df)} edges → {output_file}")
            else:
                logger.warning(
                    f"  {method}: No results"
                )  # Step 2: Compute consensus (optional)

    # CI-test sensitivity analysis (optional)
    if enable_ci_sensitivity and not (unit_id_col and unit_id_col in data_df.columns):
        logger.info("\n" + "=" * 70)
        logger.info("CI-TEST SENSITIVITY ANALYSIS")
        logger.info("=" * 70)
        try:
            from framework.core.ci_sensitivity import (
                run_ci_sensitivity,
                save_ci_sensitivity,
            )

            ci_report = run_ci_sensitivity(
                data_df,
                pairs,
                tau_max=tau_max,
                alpha=alpha,
                sampling_days=sampling_days,
            )
            save_ci_sensitivity(ci_report, output_dir / "ci_sensitivity")
            logger.info(
                f"  Robust edges (all 3 tests): {len(ci_report['robust_edges'])}"
            )
            logger.info(f"  Any test: {len(ci_report['any_edges'])}")
        except Exception as e:
            logger.warning(f"CI sensitivity analysis failed: {e}")

    # Graph recovery evaluation (when ground truth is provided)
    graph_metrics = None
    if true_edges is not None:
        logger.info("\n" + "=" * 70)
        logger.info("GRAPH RECOVERY EVALUATION (vs. ground truth)")
        logger.info("=" * 70)
        try:
            from framework.core.graph_metrics import (
                evaluate_graph_recovery,
                save_evaluation,
            )

            # Collect variable names
            eval_var_names = sorted(set(v for pair in (pairs or []) for v in pair))
            if not eval_var_names:
                eval_var_names = list(data_df.select_dtypes(include=["number"]).columns)

            graph_metrics = evaluate_graph_recovery(
                results_dict=results,
                true_edges=true_edges,
                var_names=eval_var_names,
            )

            metrics_df = save_evaluation(
                graph_metrics, output_dir / "graph_recovery_metrics.csv"
            )

            for method_name, m in graph_metrics.items():
                f1_v = m.get("f1", 0.0)
                if f1_v is None or (isinstance(f1_v, float) and np.isnan(f1_v)):
                    continue
                prec_v = m.get("precision", 0.0) or 0.0
                rec_v = m.get("recall", 0.0) or 0.0
                auroc_v = m.get("auroc", float("nan"))
                auroc_str = (
                    f"AUROC={auroc_v:.3f}"
                    if auroc_v is not None and not np.isnan(auroc_v)
                    else "AUROC=n/a"
                )
                logger.info(
                    f"  {method_name:<20} F1={f1_v:.3f}  Prec={prec_v:.3f}  "
                    f"Rec={rec_v:.3f}  {auroc_str}"
                )

            if tracker:
                tracker.log_file_path(
                    output_dir / "graph_recovery_metrics.csv", "graph_recovery"
                )
        except Exception as e:
            logger.warning(f"Graph recovery evaluation failed: {e}")

    # Cache method results for plot regeneration
    logger.info("\n" + "=" * 70)
    logger.info("CACHING RESULTS FOR PLOT REGENERATION")
    logger.info("=" * 70)
    try:
        from framework.core.plot_cache import PlotCacheManager

        cache_mgr = PlotCacheManager(output_dir)
        cache_mgr.save_method_results(results, force=False)
        logger.info("✅ Method results cached successfully")
    except Exception as e:
        logger.warning(f"⚠ Failed to cache results: {e}")

    consensus_results = None
    if enable_consensus:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: CONSENSUS")
        logger.info("=" * 70)

        # Save consensus to numbered stage: consensus/2-core/
        consensus_core_dir = output_dir / "consensus" / "2-core"
        consensus_core_dir.mkdir(parents=True, exist_ok=True)

        consensus_results = consensus.merge_method_results(
            granger_df=results.get("granger"),
            te_df=results.get("transfer_entropy"),
            pcmci_df=results.get("pcmci"),
            min_votes=2,
            lag_tolerance_steps=1,
            sampling_days=sampling_days,
            alpha=alpha,
            output_dir=str(consensus_core_dir),
        )

        # Cache consensus results for regeneration
        try:
            from framework.core.plot_cache import PlotCacheManager

            cache_mgr = PlotCacheManager(output_dir)
            consensus_csv_path_check = consensus_core_dir / "consensus.csv"
            if consensus_csv_path_check.exists():
                consensus_df_cache = pd.read_csv(consensus_csv_path_check)
                cache_mgr.save_consensus_results(
                    consensus_df_cache, consensus_results or {}
                )
        except Exception as e:
            logger.warning(f"⚠ Failed to cache consensus: {e}")
    else:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: CONSENSUS - SKIPPED (enable_consensus=False)")
        logger.info("=" * 70)
        logger.info(
            "Individual method results are available in results_<method>.csv files"
        )

    # Step 3: Generate visualizations
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: VISUALIZATIONS")
    logger.info("=" * 70)

    # Filter results to only include methods with actual data
    filtered_results = {}
    for method_name, df in results.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            filtered_results[method_name] = df
        elif isinstance(df, list) and len(df) > 0:
            filtered_results[method_name] = pd.DataFrame(df)

    if filtered_results:
        visualize_all_results(filtered_results, output_dir)
    else:
        logger.warning("No results available for visualization")

    # Clean up empty method directories in figures folder
    figures_dir = output_dir / "figures" / "method"
    if figures_dir.exists():
        for method_dir in figures_dir.iterdir():
            if method_dir.is_dir():
                # Check if directory is empty (only contains ._ files or is truly empty)
                contents = [
                    f for f in method_dir.rglob("*") if not f.name.startswith("._")
                ]
                if not contents:
                    try:
                        method_dir.rmdir()
                        logger.debug(f"Removed empty method directory: {method_dir}")
                    except OSError:
                        pass  # Directory not empty or can't be removed

    # Memory optimization: Free method results after visualizations (already saved to disk)
    import gc

    logger.debug("Freeing method results memory after visualizations...")
    results.clear()
    gc.collect()

    # Step 3.3: Falsification Testing (NEW)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.3: FALSIFICATION TESTING")
    logger.info("=" * 70)

    consensus_csv_path = output_dir / "consensus" / "2-core" / "consensus.csv"
    if enable_consensus and consensus_csv_path.exists() and enable_strength_analysis:
        try:
            from framework.core.falsification import run_falsification_battery

            consensus_df_falsification = pd.read_csv(consensus_csv_path)
            logger.info(
                f"Running falsification tests on {len(consensus_df_falsification)} consensus edges..."
            )

            # Prepare data without metadata columns
            test_data = data_df.drop(
                columns=[
                    col
                    for col in [unit_id_col, date_col]
                    if col and col in data_df.columns
                ],
                errors="ignore",
            )

            falsification_results = []
            for idx, edge in consensus_df_falsification.iterrows():
                source = edge["source"]
                target = edge["target"]
                lag = int(edge["lag_steps"])

                logger.info(f"  Testing {source} → {target} (lag={lag})...")

                # Run falsification battery
                try:
                    # Test function using Granger causality p-value
                    def test_func(x, y, lag):
                        from framework.core.methods import granger

                        # Create DataFrame and ensure no NaN values
                        df_test = pd.DataFrame({source: x, target: y}).dropna()
                        if len(df_test) < lag + 20:  # Need enough data points
                            return 1.0

                        test_results = granger.run_granger_causality(
                            df_test,
                            cause_var=source,
                            effect_var=target,
                            maxlag=lag,
                            alpha=0.999,  # High alpha to not filter (1.0 causes error)
                            verbose=False,
                            skip_stationarity=True,  # Skip stationarity for speed on surrogates
                        )
                        if test_results is not None and "results" in test_results:
                            # Extract p-value for the specific lag
                            results_df = test_results["results"]
                            if len(results_df) > 0:
                                # Get the result for the specific lag (or best lag)
                                lag_result = results_df[results_df["lag"] == lag]
                                if len(lag_result) > 0:
                                    return float(lag_result.iloc[0]["p_value"])
                                # Fallback to best lag if specific lag not found
                                return float(results_df.iloc[0]["p_value"])
                        return 1.0  # Non-significant if test fails

                    results_falsif = run_falsification_battery(
                        source=test_data[source].values,
                        target=test_data[target].values,
                        lag=lag,
                        test_func=test_func,
                        n_surrogates=200,  # Increased for more stable p-value estimation
                        alpha=alpha,
                    )

                    # Summarize results
                    from framework.core.falsification import (
                        summarize_falsification_results,
                    )

                    summary = summarize_falsification_results(results_falsif)

                    falsification_results.append(
                        {
                            "source": source,
                            "target": target,
                            "lag_steps": lag,
                            "falsification_passed": summary["tests_passed"],
                            "falsification_total": summary["tests_total"],
                            "block_perm_passed": results_falsif["block_permutation"][
                                "passed"
                            ],
                            "iaaft_passed": results_falsif["iaaft"]["passed"],
                        }
                    )
                    logger.info(
                        f"    ✓ Passed {summary['tests_passed']}/{summary['tests_total']} tests"
                    )
                except Exception as e:
                    logger.warning(f"    ✗ Falsification test failed: {e}")
                    falsification_results.append(
                        {
                            "source": source,
                            "target": target,
                            "lag_steps": lag,
                            "falsification_passed": 0,
                            "falsification_total": 2,
                            "block_perm_passed": False,
                            "iaaft_passed": False,
                        }
                    )

            # Save falsification results
            falsification_df = pd.DataFrame(falsification_results)
            falsification_path = output_dir / "falsification_results.csv"
            falsification_df.to_csv(falsification_path, index=False)
            logger.info(f"✅ Falsification results saved: {falsification_path}")

            # Merge with consensus
            consensus_df_falsification = consensus_df_falsification.merge(
                falsification_df[
                    ["source", "target", "lag_steps", "falsification_passed"]
                ],
                on=["source", "target", "lag_steps"],
                how="left",
            )
            consensus_df_falsification["falsification_passed"] = (
                consensus_df_falsification["falsification_passed"].fillna(0).astype(int)
            )

            if tracker:
                tracker.log_file_path(falsification_path, "falsification_results")
                tracker.log_metric(
                    "falsification_tests_run", len(falsification_results)
                )
        except Exception as e:
            logger.warning(f"⚠️  Falsification testing failed: {e}")
            consensus_df_falsification = pd.read_csv(consensus_csv_path)
            consensus_df_falsification["falsification_passed"] = 0
    else:
        if consensus_csv_path.exists():
            consensus_df_falsification = pd.read_csv(consensus_csv_path)
            consensus_df_falsification["falsification_passed"] = 0
        logger.warning("Skipping falsification testing")

    # Step 3.4: ICP Stability Testing (NEW)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.4: ICP STABILITY TESTING")
    logger.info("=" * 70)

    if (
        enable_consensus
        and consensus_csv_path.exists()
        and unit_id_col
        and unit_id_col in data_df.columns
    ):
        # Panel data: use spatial/unit environments
        try:
            from framework.core.icp_stability import test_consensus_stability

            logger.info(
                f"Testing coefficient stability across {unit_id_col} environments..."
            )

            stability_results = test_consensus_stability(
                data=data_df,
                consensus_df=consensus_df_falsification,
                environment_col=unit_id_col,
            )

            stability_path = output_dir / "icp_stability_results.csv"
            stability_results.to_csv(stability_path, index=False)
            logger.info(f"✅ ICP stability results saved: {stability_path}")
            logger.info(
                f"   Stable edges: {stability_results['icp_stable'].sum()}/{len(stability_results)}"
            )

            # Merge with consensus
            consensus_df_falsification = consensus_df_falsification.merge(
                stability_results[["source", "target", "lag_steps", "icp_stable"]],
                on=["source", "target", "lag_steps"],
                how="left",
            )
            consensus_df_falsification["icp_stable"] = consensus_df_falsification[
                "icp_stable"
            ].fillna(False)

            if tracker:
                tracker.log_file_path(stability_path, "icp_stability_results")
                tracker.log_metric(
                    "icp_stable_edges", int(stability_results["icp_stable"].sum())
                )
        except Exception as e:
            logger.warning(f"⚠️  ICP stability testing failed: {e}")
            consensus_df_falsification["icp_stable"] = False
    elif enable_consensus and consensus_csv_path.exists():
        # Single-unit data: use temporal block splitting as fallback
        try:
            from framework.core.icp_stability import test_consensus_stability_temporal

            logger.info(
                "No panel structure available. Testing coefficient stability "
                "across temporal blocks (Peters et al. 2016 invariance principle)..."
            )

            # Use the numeric data (drop metadata columns)
            test_data = data_df.drop(
                columns=[
                    col
                    for col in [unit_id_col, date_col]
                    if col and col in data_df.columns
                ],
                errors="ignore",
            )

            stability_results = test_consensus_stability_temporal(
                data=test_data,
                consensus_df=consensus_df_falsification,
                n_blocks=3,
                min_obs_per_block=max(30, len(test_data) // 10),
            )

            stability_path = output_dir / "icp_stability_results.csv"
            stability_results.to_csv(stability_path, index=False)
            n_stable = stability_results["icp_stable"].sum()
            n_total = len(stability_results)
            logger.info(f"✅ ICP stability results saved: {stability_path}")
            logger.info(f"   Stable edges (temporal blocks): {n_stable}/{n_total}")

            # Merge with consensus
            consensus_df_falsification = consensus_df_falsification.merge(
                stability_results[["source", "target", "lag_steps", "icp_stable"]],
                on=["source", "target", "lag_steps"],
                how="left",
            )
            consensus_df_falsification["icp_stable"] = consensus_df_falsification[
                "icp_stable"
            ].fillna(False)

            if tracker:
                tracker.log_file_path(stability_path, "icp_stability_results")
                tracker.log_metric("icp_stable_edges", int(n_stable))
        except Exception as e:
            logger.warning(f"⚠️  ICP temporal stability testing failed: {e}")
            consensus_df_falsification["icp_stable"] = False
    else:
        logger.info("Skipping ICP stability (consensus not enabled or no edges)")
        if "consensus_df_falsification" in locals():
            consensus_df_falsification["icp_stable"] = False

    # Step 3.5: Tiered Classification (NEW)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.5: TIERED CLASSIFICATION")
    logger.info("=" * 70)

    if "consensus_df_falsification" in locals() and len(consensus_df_falsification) > 0:
        try:
            from framework.core.tiered_consensus import (
                add_tier_classification,
                generate_tier_report,
                summarize_tiers,
            )

            # Add tier classification
            consensus_with_tiers = add_tier_classification(consensus_df_falsification)

            # Generate tier report to consensus/5-tiers/
            tier_report = generate_tier_report(consensus_with_tiers)
            tier_report_path = output_dir / "consensus" / "5-tiers" / "tier_report.txt"
            tier_report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tier_report_path, "w", encoding="utf-8") as f:
                f.write(tier_report)
            logger.info(f"✅ Tier report saved: {tier_report_path}")

            # Log tier summary
            tier_summary = summarize_tiers(consensus_with_tiers)
            logger.info("\nTier Summary:")
            for _, row in tier_summary.iterrows():
                logger.info(
                    f"  Tier-{int(row['tier'])} ({row['name']}): {int(row['n_edges'])} edges ({row['percentage']:.1f}%)"
                )

            # Save consensus with tiers to consensus/5-tiers/
            consensus_tiers_path = (
                output_dir / "consensus" / "5-tiers" / "consensus_with_tiers.csv"
            )
            consensus_tiers_path.parent.mkdir(parents=True, exist_ok=True)
            consensus_with_tiers.to_csv(consensus_tiers_path, index=False)
            logger.info(f"✅ Consensus with tiers saved: {consensus_tiers_path}")

            if tracker:
                tracker.log_file_path(tier_report_path, "tier_report")
                tracker.log_file_path(consensus_tiers_path, "consensus_with_tiers")
                for _, row in tier_summary.iterrows():
                    tracker.log_metric(
                        f"tier_{int(row['tier'])}_edges", int(row["n_edges"])
                    )
        except Exception as e:
            logger.warning(f"⚠️  Tiered classification failed: {e}")

    # Step 3.6: Robust Correlation Analysis (CONDITIONAL - METHOD CONFIG DRIVEN)
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.6: ROBUST CORRELATION ANALYSIS")
    logger.info("=" * 70)

    # Check if correlation analysis is enabled in config
    correlation_enabled = method_config and method_config.get("correlation", {}).get(
        "enabled", False
    )

    if not correlation_enabled:
        logger.info("ℹ️  Correlation analysis disabled in configuration")
    else:
        try:
            from framework.plots.correlation_plots import create_all_correlation_plots

            # Prepare data without metadata
            corr_data = data_df.drop(
                columns=[
                    col
                    for col in [unit_id_col, date_col]
                    if col and col in data_df.columns
                ],
                errors="ignore",
            )

            correlation_dir = output_dir / "figures" / "correlation"
            logger.info(
                f"Creating correlation plots for {len(corr_data.columns)} variables..."
            )

            plot_paths = create_all_correlation_plots(
                data=corr_data,
                output_dir=correlation_dir,
                max_lag=tau_max,
            )

            logger.info(
                f"✅ Created {len(plot_paths)} correlation plots in {correlation_dir}"
            )

            if tracker:
                for plot_name, plot_path in plot_paths.items():
                    tracker.log_file_path(plot_path, f"correlation_{plot_name}")
        except Exception as e:
            logger.warning(f"⚠️  Correlation plot generation failed: {e}")

    # Memory cleanup before interactive visualizations
    import gc

    logger.debug("Freeing memory before interactive visualizations...")
    if "corr_data" in locals():
        del corr_data
    gc.collect()

    # Step 3.7: Generate interactive visualizations and plain-text summary
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3.7: INTERACTIVE VISUALIZATIONS & PLAIN-TEXT SUMMARY")
    logger.info("=" * 70)

    # Generate plain-text summary (always available)
    try:
        from framework.reporting import generate_full_summary_report

        consensus_csv_path = output_dir / "consensus" / "2-core" / "consensus.csv"
        if consensus_csv_path.exists():
            consensus_df_summary = pd.read_csv(consensus_csv_path)

            # Collect results from each method (1-raw stage)
            results_dict_summary = {}
            methods_results_dir = output_dir / "method"
            for method_name in ["granger", "transfer_entropy", "pcmci"]:
                method_file = (
                    methods_results_dir
                    / method_name
                    / "1-raw"
                    / f"results_{method_name}.csv"
                )
                if method_file.exists():
                    results_dict_summary[method_name.replace("_", " ").title()] = (
                        pd.read_csv(method_file)
                    )
                method_file = method_dir / f"results_{method_name}.csv"
                if method_file.exists():
                    results_dict_summary[method_name.replace("_", " ").title()] = (
                        pd.read_csv(method_file)
                    )

            summary_path = output_dir / "causal_summary.txt"
            generate_full_summary_report(
                consensus_df=consensus_df_summary,
                results_dict=results_dict_summary,
                output_path=summary_path,
                experiment_name=experiment_name or "Causal Discovery",
                alpha=alpha,
                top_n_per_method=5,
            )
            logger.info(f"✅ Plain-text summary saved: {summary_path}")

            if tracker:
                tracker.log_file_path(summary_path, "causal_summary_report")
    except Exception as e:
        logger.warning(f"⚠️  Plain-text summary generation failed: {e}")

    # Generate interactive visualizations (requires plotly)
    try:
        from framework.plots import create_interactive_dashboard

        consensus_csv_path = output_dir / "consensus" / "2-core" / "consensus.csv"
        if consensus_csv_path.exists():
            consensus_df_viz = pd.read_csv(consensus_csv_path)

            # Collect results from each method (1-raw stage)
            results_dict_viz = {}
            methods_results_dir = output_dir / "method"
            for method_name in ["granger", "transfer_entropy", "pcmci"]:
                method_file = (
                    methods_results_dir
                    / method_name
                    / "1-raw"
                    / f"results_{method_name}.csv"
                )
                if method_file.exists():
                    results_dict_viz[method_name.replace("_", " ").title()] = (
                        pd.read_csv(method_file)
                    )

            interactive_dir = output_dir / "figures" / "interactive"
            dashboard_files = create_interactive_dashboard(
                consensus_df=consensus_df_viz,
                results_dict=results_dict_viz,
                output_dir=interactive_dir,
                experiment_name=experiment_name or "Causal Discovery",
            )

            if dashboard_files:
                logger.info(
                    f"✅ Interactive dashboard created: {len(dashboard_files)} visualizations"
                )
                for name, path in dashboard_files.items():
                    logger.info(f"   - {name}: {path.name}")

                    if tracker:
                        tracker.log_file_path(path, f"interactive_{name}")
    except ImportError:
        logger.warning(
            "⚠️  Plotly not installed - skipping interactive visualizations (pip install plotly)"
        )
    except Exception as e:
        logger.warning(f"⚠️  Interactive visualization generation failed: {e}")

    # NOTE: Literature comparison is experiment-specific and moved to experiments/earthnet/common/
    # Experiments can implement their own literature comparison if needed
    # logger.info("\n" + "=" * 70)
    # logger.info("LITERATURE COMPARISON (EXPERIMENT-SPECIFIC)")
    # logger.info("=" * 70)

    # Step 4: Lag-Stratified Analysis
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: LAG-STRATIFIED ANALYSIS")
    logger.info("=" * 70)

    from framework.core.lag_analysis import generate_lag_report

    # Prepare results dictionary for lag analysis
    # Ensure we have DataFrames (not empty lists) and filter out empty ones
    lag_results = {}
    for method_name in ["granger", "transfer_entropy", "pcmci", "correlation"]:
        if method_name in results:
            df = results[method_name]
            # Handle both DataFrame and list/None cases
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                lag_results[method_name] = df
                logger.debug(f"Added {method_name} to lag analysis: {len(df)} rows")
            elif isinstance(df, list) and len(df) > 0:
                # Convert list of dicts to DataFrame if needed
                lag_results[method_name] = pd.DataFrame(df)
                logger.debug(
                    f"Added {method_name} to lag analysis (from list): {len(df)} items"
                )
        else:
            logger.debug(f"Method {method_name} not found in results dictionary")

    # Also check if results were saved to CSV files and reload if needed
    if not lag_results:
        logger.info("Results dictionary empty, checking CSV files...")
        pcmci_csv = output_dir / "results_pcmci.csv"
        if pcmci_csv.exists() and pcmci_csv.stat().st_size > 0:
            try:
                df = pd.read_csv(pcmci_csv)
                if len(df) > 0:
                    lag_results["pcmci"] = df
                    logger.info(f"Loaded PCMCI results from CSV: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load PCMCI CSV: {e}")

    if lag_results:
        logger.info(
            f"Running lag-stratified analysis on {len(lag_results)} method(s): {list(lag_results.keys())}"
        )
        generate_lag_report(
            lag_results,
            output_dir=output_dir,
            short_lag_max=3,  # ≤3 timesteps (~5-15 days)
            long_lag_min=10,  # ≥10 timesteps (~50-60 days)
        )
    else:
        logger.warning("No results available for lag-stratified analysis")

    # Step 5: Causal Strength Quantification (NEW)
    if (
        enable_strength_analysis
        and consensus_results
        and consensus_results.get("n_consensus_edges", 0) > 0
    ):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: CAUSAL STRENGTH QUANTIFICATION")
        logger.info("=" * 70)

        quantifier = CausalStrengthQuantifier(alpha=alpha)

        # Load consensus edges
        consensus_df = pd.read_csv(output_dir / "consensus.csv")

        strength_results = []
        for idx, edge in consensus_df.iterrows():
            source, target = edge["source"], edge["target"]
            lag = int(edge.get("lag_steps", 5))

            # Compare all methods for this edge
            try:
                comparison = quantifier.compare_methods(
                    data_df if unit_id_col is None else None,
                    source,
                    target,
                    lag=lag,
                )
                strength_results.append(comparison)

                logger.info(
                    f"  {source}→{target} (lag={lag}): "
                    f"Granger={comparison[comparison['method'] == 'Granger']['normalized_effect'].values[0]:.3f}, "
                    f"TE={comparison[comparison['method'] == 'TransferEntropy']['normalized_effect'].values[0]:.3f}, "
                    f"PCMCI+={comparison[comparison['method'] == 'PCMCI+']['normalized_effect'].values[0]:.3f}"
                )
            except Exception as e:
                logger.debug(f"Strength analysis failed for {source}→{target}: {e}")

        if strength_results:
            strength_df = pd.concat(strength_results, ignore_index=True)
            strength_path = output_dir / "causal_strength.csv"
            strength_df.to_csv(strength_path, index=False)
            logger.info(f"✅ Causal strength analysis saved: {strength_path}")

            if tracker:
                tracker.log_results(strength_df, "causal_strength")
                tracker.log_file_path(strength_path, "causal_strength_analysis")

    # Step 6: Temporal Validation (NEW)
    if enable_temporal_validation and len(pairs) > 0:
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: TEMPORAL VALIDATION")
        logger.info("=" * 70)

        # For panel data, we need to handle differently
        if unit_id_col and unit_id_col in data_df.columns:
            logger.info(
                "⚠️  Temporal validation on panel data requires sampling - using first unit"
            )
            sample_unit = data_df[unit_id_col].iloc[0]
            validation_data = data_df[data_df[unit_id_col] == sample_unit].copy()
            validation_data = _set_datetime_index_with_freq(
                validation_data, date_col, sampling_days
            )
        else:
            validation_data = data_df.copy()
            validation_data = _set_datetime_index_with_freq(
                validation_data, date_col, sampling_days
            )

        # Extract variable columns
        var_cols = list(set([p[0] for p in pairs] + [p[1] for p in pairs]))
        validation_data = validation_data[var_cols]

        if len(validation_data.dropna()) >= 100:  # Need sufficient data for CV
            try:
                # Run temporal cross-validation on Granger method
                cv_results = temporal_cross_validation(
                    validation_data,
                    pairs[:10],  # Validate top 10 pairs to save time
                    granger.batch_granger_causality,
                    n_splits=3,
                    alpha=alpha,
                    maxlag=tau_max,
                )

                logger.info(
                    f"  Stable edges: {len(cv_results['stable_edges'])} / {len(pairs[:10])}"
                )
                logger.info(f"  Stability score: {cv_results['stability_score']:.1%}")

                # Save validation results
                validation_path = output_dir / "temporal_validation.json"
                import json

                with open(validation_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "stable_edges": cv_results["stable_edges"],
                            "stability_score": cv_results["stability_score"],
                            "n_folds": cv_results["n_folds"],
                            "edge_votes": {
                                f"{e[0]}->{e[1]}": v
                                for e, v in cv_results["edge_votes"].items()
                            },
                        },
                        f,
                        indent=2,
                    )

                logger.info(f"✅ Temporal validation saved: {validation_path}")

                if tracker:
                    tracker.log_metric("stability_score", cv_results["stability_score"])
                    tracker.log_file_path(validation_path, "temporal_validation")

            except Exception as e:
                logger.warning(f"Temporal validation failed: {e}")
        else:
            logger.warning(
                "Insufficient data for temporal validation (need ≥100 observations)"
            )

    # Step 7: Summary
    logger.info("\n" + "=" * 70)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Results directory: {output_dir}")
    logger.info(f"  - results_granger.csv: {len(results.get('granger', []))} edges")
    logger.info(
        f"  - results_transfer_entropy.csv: {len(results.get('transfer_entropy', []))} edges"
    )
    logger.info(f"  - results_pcmci.csv: {len(results.get('pcmci', []))} edges")
    if consensus_results is not None:
        logger.info(
            f"  - consensus.csv: {consensus_results.get('n_consensus_edges', 0)} edges"
        )
    logger.info("  - figures/: Visualizations generated")

    # Finalize experiment tracker
    if tracker:
        tracker.log_results(results.get("granger"), "granger_edges")
        tracker.log_results(results.get("transfer_entropy"), "te_edges")
        tracker.log_results(results.get("pcmci"), "pcmci_edges")

        # Only log consensus if consensus was enabled and file exists
        consensus_csv_path = output_dir / "consensus" / "2-core" / "consensus.csv"
        if consensus_csv_path.exists():
            consensus_df = pd.read_csv(consensus_csv_path)
            tracker.log_results(consensus_df, "consensus_edges")
            tracker.log_metric(
                "n_consensus_edges", consensus_results.get("n_consensus_edges", 0)
            )

        tracker.log_metric("n_granger_edges", len(results.get("granger", [])))
        tracker.log_metric("n_te_edges", len(results.get("transfer_entropy", [])))
        tracker.log_metric("n_pcmci_edges", len(results.get("pcmci", [])))

        log_path = tracker.save()
        logger.info(f"✅ Experiment log saved: {log_path}")

    # Remove file handler to avoid accumulation on repeated calls
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    logger.info(f"Full experiment log: {log_file}")

    return {
        "success": True,
        "output_dir": str(output_dir),
        "results": results,
        "consensus": consensus_results,
        "n_edges": {k: len(v) if v is not None else 0 for k, v in results.items()},
        "tracker_log": str(log_path) if tracker else None,
    }


def run_complete_workflow(
    results_dir: Path,
    edges_csv: Path,
    sample: bool = True,
) -> dict:
    """
    Run complete causal analysis workflow.

    Parameters:
        results_dir (Path): Directory for results
        edges_csv (Path): Path to edges CSV (can be generated or provided)
        sample (bool): Whether to use sample data for testing

    Returns:
        dict: Workflow results
    """
    logger.info("=" * 70)
    logger.info("PHASE 8-9 INTEGRATION: Complete Causal Analysis Workflow")
    logger.info("=" * 70)

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    workflow_results = {}

    try:
        # Step 1: Verify edges CSV exists
        logger.info("\n[Step 1/5] Verifying edge data...")
        if edges_csv.exists():
            logger.info(f"  ✅ Found edges file: {edges_csv}")
            edges_df = pd.read_csv(edges_csv)
            logger.info(f"  ✅ Loaded {len(edges_df)} edges")
        else:
            logger.warning(f"  Edges file not found: {edges_csv}")
            logger.info("  Skipping edge-based analysis")
            return {"success": False, "error": "No edges file"}

        # Step 2: Run diagnostics
        logger.info("\n[Step 2/5] Running comprehensive diagnostics...")
        from framework.core.diagnostics import Diagnostics

        diagnostics = Diagnostics(edges_df, cadence_days=5)
        diag_results = diagnostics.run_all()

        diag_output_dir = results_dir / "diagnostics"
        diagnostics.save_results(diag_output_dir)
        logger.info(f"  ✅ Diagnostics saved to {diag_output_dir}")
        workflow_results["diagnostics"] = diag_results

        # Step 3: Run sanity tests (DISABLED - module not implemented)
        # logger.info("\n[Step 3/5] Running sanity tests...")
        # from framework.core.sanity_tests import SanityTests
        #
        # sanity_tests = SanityTests(edges_df, cadence_days=5)
        # sanity_results = sanity_tests.run_all()
        #
        # sanity_output_dir = results_dir / "sanity_tests"
        # sanity_output_dir.mkdir(parents=True, exist_ok=True)
        # sanity_tests.save_results(sanity_output_dir / "sanity_test_results.csv")
        # logger.info(f"  ✅ Sanity tests saved to {sanity_output_dir}")
        # workflow_results["sanity_tests"] = sanity_results

        # Step 4: Generate summary report

        report_lines = [
            "# Phase 8-9 Complete Workflow Report\n",
            f"Generated: {pd.Timestamp.now().isoformat()}\n\n",
            "## Edge Analysis Summary\n",
            f"- Total edges: {len(edges_df)}\n",
            f"- Methods used: {', '.join(edges_df['method'].unique())}\n\n",
            "## Sanity Tests Results\n",
            "- Sanity tests disabled (module not implemented)\n",
        ]

        report_lines.extend(
            [
                "\n## Diagnostics Summary\n",
                "### P-value Distributions\n",
            ]
        )

        if "pvalue_distributions" in diag_results:
            pval_diag = diag_results["pvalue_distributions"]
            for method, stats_dict in pval_diag.get("by_method", {}).items():
                report_lines.append(
                    f"- {method}: mean={stats_dict['mean']:.4f}, median={stats_dict['median']:.4f}\n"
                )

        report_lines.extend(
            [
                "\n### Lag Distributions (days)\n",
            ]
        )

        if "lag_distributions" in diag_results:
            lag_diag = diag_results["lag_distributions"]
            for method, stats_dict in lag_diag.get("by_method", {}).items():
                report_lines.append(
                    f"- {method}: mean={stats_dict['mean_days']:.1f}, median={stats_dict['median_days']:.1f}\n"
                )

        report_lines.extend(
            [
                "\n### Method Agreement\n",
            ]
        )

        if "method_agreement" in diag_results:
            method_diag = diag_results["method_agreement"]
            total_pairs = method_diag.get("total_unique_pairs", 0)
            report_lines.append(f"- Unique variable pairs: {total_pairs}\n")

        report_lines.extend(
            [
                "\n### Wilson Confidence Intervals (95%)\n",
            ]
        )

        if "detection_ci" in diag_results:
            ci_diag = diag_results["detection_ci"]
            for method, ci_dict in ci_diag.get("by_method", {}).items():
                report_lines.append(
                    f"- {method}: {ci_dict['detection_rate']:.1%} "
                    f"[{ci_dict['ci_lower']:.1%}, {ci_dict['ci_upper']:.1%}]\n"
                )

        report_lines.extend(
            [
                "\n## Status\n",
                "- Sanity tests: ⚠️ DISABLED (module not implemented)\n",
                "- Diagnostics: ✅ COMPLETE\n",
                "- Overall: Ready for Phase 10 (Final Validation)\n",
            ]
        )

        report_text = "".join(report_lines)
        report_path = results_dir / "workflow_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"  ✅ Report saved to {report_path}")
        workflow_results["report"] = str(report_path)

        # Step 5: Final status
        logger.info("\n[Step 5/5] Finalizing workflow...")
        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW SUMMARY")
        logger.info("=" * 70)
        logger.info("⚠️  Sanity tests: DISABLED (module not implemented)")
        logger.info("✅ Diagnostics: Complete")
        logger.info(f"✅ Report: {report_path}")
        logger.info("\n✅ Workflow completed successfully!")
        logger.info("=" * 70)

        workflow_results["success"] = True
        return workflow_results

    except Exception as e:
        logger.error(f"❌ Workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 8-9 complete workflow")
    parser.add_argument("--edges", type=Path, help="Path to edges CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default="results/workflow_output",
        help="Output directory",
    )

    args = parser.parse_args()

    # Use test data if no edges provided
    edges_file = args.edges or Path("tests/test_data/test_edges.csv")

    if not edges_file.exists():
        logger.error(f"Edges file not found: {edges_file}")
        sys.exit(1)

    results = run_complete_workflow(args.output, edges_file)

    sys.exit(0 if results["success"] else 1)
