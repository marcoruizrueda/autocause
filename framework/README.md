# Causal Discovery Framework

A **reusable, generic, adaptive Python framework** for robust causal discovery analysis on CSV time-series datasets. Automatically selects optimal methods per variable pair, supports ensemble analysis, and produces publication-ready visualizations.

## ✨ Recent Enhancements

### 🎯 Dataset-Agnostic Preprocessing (New!)

The framework now **automatically adapts** to different data structures and temporal sampling patterns:

- **Panel-Aware Quality Checks**: Detects panel data (multiple units) vs. single time series
- **Smart Missing Data Handling**: 
  - For **dense/continuous sensors** (e.g., climate stations): checks percentage thresholds
  - For **sparse/irregular sensors** (e.g., satellite observations): checks absolute observation counts
- **Satellite Data Support**: Correctly handles optical remote sensing data with ~5-day revisits
- **Automatic Detection**: No manual configuration needed - inspects data structure and adapts logic

**Example**: Sentinel-2 NDVI observations every ~5 days appear as "89% missing" in daily grids, but preprocessing now recognizes this as **expected temporal sampling** rather than poor quality.

### 🔬 Enhanced Analysis Capabilities

- **Preprocessing Pipeline**: Automatic outlier detection, interpolation, stationarity testing
- **Distribution Testing**: Normality, stationarity, and independence tests with visual diagnostics
- **Causal Strength Metrics**: Effect size quantification (correlation, mutual information, Granger F-statistics)
- **Temporal Validation**: Cross-validation, out-of-sample testing, lag stability analysis
- **Experiment Tracking**: Reproducible runs with parameter logging and hash-based data versioning

### 🤖 Adaptive Method Selection

The framework **automatically selects the best causal discovery method(s)** for each variable pair based on data characteristics:

- **Linearity**: Pearson correlation test
- **Stationarity**: ADF test with automatic differencing
- **Sample Size**: Adequacy assessment per pair
- **Missingness**: Adaptive imputation strategies

See [**Adaptive Framework Guide**](ADAPTIVE_FRAMEWORK_GUIDE.md) for complete documentation.

### Quick Example

```python
from framework.core import method_selector

# Automatically select methods for all pairs
selections = method_selector.batch_select_methods(
    df, variable_pairs, tau_max=12, alpha=0.05
)

# Review selections
print(selections[['source', 'target', 'primary_method', 'reasoning']])
```

Or use the orchestrator:
```bash
python experiments/<your_experiment>/run_adaptive_experiment.py
```

## Features

- **Dataset-Agnostic Preprocessing**: Automatically handles panel data, time series, dense/sparse sampling
- **Adaptive Method Selection**: Automatically chooses Granger, Transfer Entropy, or PCMCI+ per pair
- **Multiple Execution Modes**: Optimized (single method) or Ensemble (consensus)
- **Multiple Input Formats**: Wide or long/tidy CSV with automatic detection
- **Automatic tau_max Estimation**: Data-driven maximum lag selection using ACF + domain constraints + Nyquist criterion (Runge et al. 2019)
- **Robust Data QC**: 
  - Panel-aware quality checks (per-unit vs. global statistics)
  - Coverage filtering with observation count thresholds
  - Outlier handling (IQR, Z-score, or custom)
  - Missing-data policies (interpolation, forward-fill, etc.)
- **Statistical Safeguards**:
  - Stationarity testing (ADF/KPSS) with automatic differencing
  - Normality and independence testing with visual diagnostics
  - Linearity assessment to guide method selection
  - Max-lag logic with ACF/PACF heuristics
  - Multiple-testing correction (FDR/Bonferroni)
- **Three Causal Methods**:
  - **Granger Causality**: Linear, lag-based (fast, precise for linear systems)
  - **Transfer Entropy**: Information-theoretic (robust, model-free, nonlinear)
  - **PCMCI+/LPCMCI**: Constraint-based (handles latent confounders)
- **Enhanced Analytics**:
  - **Causal Strength Metrics**: Effect size quantification (correlation, MI, F-stats)
  - **Temporal Validation**: Cross-validation, out-of-sample testing, lag stability
  - **Distribution Tests**: Comprehensive statistical testing with visualization
  - **Experiment Tracking**: Reproducible runs with parameter/data versioning
- **Consensus Detection**: Flags when ≥2 methods agree on direction + lag window
- **Paper-Ready Outputs**:
  - P-value distributions, lag histograms (in days), causal graphs with arrows
  - Europe maps with detection rates by UTM zone
  - Comparison panels across experiments
  - **SVG-only exports** (high resolution, no PNG/PDF)

## Installation

```bash
# Clone or navigate to framework root
cd framework

# Install dependencies (add to your environment)
pip install pandas numpy scipy scikit-learn statsmodels

# Optional (for nonlinear methods)
pip install tigramite  # For PCMCI+/LPCMCI
pip install pyinform   # For advanced TE estimation

# Optional (for interactive visualizations) 🆕
pip install plotly     # For interactive network graphs and dashboards
```

## Enhanced Features 🆕

### 1. Interactive Causal Visualization

Create **browser-based interactive visualizations** with Plotly:

```python
from framework.plots import (
    create_interactive_causal_network,
    create_interactive_lag_explorer,
    create_interactive_dashboard
)

# Interactive consensus network with hover details
create_interactive_causal_network(
    consensus_df,
    output_path="figures/interactive_network.html",
    color_by="vote_count",  # or "best_p_value", "n_significant"
    size_by="n_significant",
    min_votes=2
)

# Interactive lag distribution explorer
create_interactive_lag_explorer(
    granger_results,
    method_name="Granger",
    output_path="figures/lag_explorer.html"
)

# Complete dashboard with all methods
create_interactive_dashboard(
    consensus_df,
    results_dict={"Granger": df1, "PCMCI+": df2},
    output_dir="figures/interactive/",
    experiment_name="My Experiment"
)
```

**Features**:
- 🔍 Hover over nodes/edges for detailed information
- 🎨 Color-coded by agreement strength or p-value
- 📏 Edge width proportional to effect strength
- 🔎 Zoom, pan, and filter interactively
- 💾 Export to HTML for sharing and presentation
- 🌐 No additional software required (browser-based)

### 2. Plain-Text Causal Summary Generator

Generate **standardized, publication-ready causal statements**:

```python
from framework.reporting import (
    generate_causal_statement,
    summarize_consensus_edges,
    generate_full_summary_report
)

# Single causal statement
statement = generate_causal_statement(
    source="RR",
    target="NDVI", 
    lag_days=35,
    p_value=0.0001,
    strength=0.65,
    strength_metric="correlation",
    n_units=125,
    n_significant=91
)
print(statement)
# Output: "RR → NDVI (lag=35 days, p<0.001, strength=0.65 [strong], 
#          significant in 91/125 units [73%])"

# Summarize all consensus edges
summaries = summarize_consensus_edges(
    consensus_df,
    alpha=0.05,
    include_method_details=True
)
for summary in summaries:
    print(summary)

# Generate comprehensive plain-text report
report = generate_full_summary_report(
    consensus_df,
    results_dict={"Granger": df1, "PCMCI+": df2},
    output_path="causal_summary.txt",
    experiment_name="My Experiment",
    alpha=0.05,
    top_n_per_method=5
)
```

**Features**:
- 📝 Standardized format: `X → Y (lag=N, p<threshold, strength=S)`
- 📊 Effect size classification (weak/moderate/strong/very_strong)
- 🎯 Confidence levels from multi-method agreement
- 📈 Panel data statistics (% of units with effect)
- 📖 Compatible with academic writing and documentation
- 🔬 Auto-formatted p-values (p<0.001, p<0.01, etc.)

**Try the demo**: `python framework/demo_interactive_summary.py`

## Core Modules

The framework consists of several independent, reusable modules in `framework/core/`:

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `preprocessing.py` | Data preparation & QC | `TimeSeriesPreprocessor`, automatic interpolation, outlier detection |
| `tau_max_estimation.py` | Maximum lag selection | Scientific method (ACF+domain+Nyquist), 7 estimation methods |
| `distribution_tests.py` | Statistical assumption testing | Normality (Shapiro-Wilk, KS), stationarity (ADF, KPSS), independence |
| `causal_strength.py` | Effect size quantification | Correlation, mutual information, Granger F-statistics |
| `validate.py` | Temporal validation | Cross-validation, out-of-sample testing, lag stability |
| `experiment_tracker.py` | Reproducibility | Parameter logging, data versioning (hash-based) |
| `lag_analysis.py` | Lag-stratified analysis | Short-lag vs. long-lag breakdowns, method agreement |
| `run_workflow.py` | Main orchestrator | Complete pipeline with optional enhancement flags |
| `io.py` | Data loading | CSV parsing, wide/long format detection |
| `qc.py` | Traditional QC | Legacy quality control functions |
| `stats.py` | Statistical utilities | Stationarity tests, linearity assessment, lag suggestions |
| `decision.py` | Consensus logic | Multi-method agreement detection |
| `diagnostics.py` | Result inspection | P-value diagnostics, FDR analysis |
| `multiple_testing.py` | Correction methods | Bonferroni, Holm, BH, BY |
| `sensitivity.py` | Robustness checks | Parameter sensitivity analysis |

### Usage Pattern

```python
# Complete workflow (recommended)
from framework.core import run_workflow
results = run_workflow.run_causal_workflow(data, metadata, pairs, ...)

# Or use modules individually
from framework.core import preprocessing, distribution_tests, causal_strength

preprocessor = preprocessing.TimeSeriesPreprocessor(...)
df_clean, _, report = preprocessor.process(df, metadata)

dist_tests = distribution_tests.test_all_distributions(df_clean, ...)
strength = causal_strength.compute_causal_strength(results, df_clean, ...)
```

## Automatic tau_max Estimation 🆕

The framework now includes **scientifically defensible, automatic maximum lag (tau_max) estimation** following best practices from Earth system sciences (Runge et al. 2019, Box & Jenkins 1976).

### Scientific Method (Recommended)

**Formula**: `tau_max = min(ACF_zero_crossing, domain_max/sampling, N/3)`

Combines three independent constraints:
- **ACF Zero-Crossing**: Statistical memory length from autocorrelation (Box-Jenkins 1976)
- **Domain Constraint**: Physical plausibility limit (Runge et al. 2019)
- **Nyquist Constraint**: Sample adequacy (N/3 rule for statistical power)

**Conservative**: Takes minimum → avoids overfitting  
**Transparent**: All constraints logged for reproducibility  
**Defensible**: Follows standards from *Nature Communications* and time-series analysis literature

### Usage

#### Automatic (Recommended)
```python
from framework.core.run_workflow import run_causal_discovery_workflow

results = run_causal_discovery_workflow(
    data_df=df,
    tau_max=None,  # Auto-estimate using scientific method
    tau_max_method="scientific",
    domain_max_days=90,
    sampling_days=5,
    ...
)

# Check estimation results
with open("results/tau_max_estimation.json") as f:
    tau_info = json.load(f)
    print(f"Estimated tau_max: {tau_info['tau_max']} timesteps")
    print(f"Binding constraint: {tau_info['binding_constraint']}")
```

#### Manual Estimation
```python
from framework.core.tau_max_estimation import estimate_tau_max_scientific

result = estimate_tau_max_scientific(
    series_x=rainfall,
    series_y=ndvi,
    sampling_interval_days=5,
    domain_max_days=90
)

tau_max = result["tau_max"]
print(f"Recommended: {tau_max} timesteps ({result['tau_max_days']:.0f} days)")
print(f"ACF constraint: {result['acf_constraint']}")
print(f"Domain constraint: {result['domain_constraint']}")
print(f"Nyquist constraint: {result['nyquist_constraint']}")
print(f"Binding: {result['binding_constraint']}")
```

### All Available Methods

The `tau_max_estimation.py` module provides 7 methods:

| Method | Best For | Reference |
|--------|----------|----------|
| **`estimate_tau_max_scientific()`** | **General use (recommended)** | Runge et al. 2019, Box & Jenkins 1976 |
| `estimate_tau_max_acf_zero_crossing()` | Pure statistical approach | Box & Jenkins 1976 |
| `estimate_tau_max_pacf_cutoff()` | Linear AR processes | Time series standards |
| `estimate_tau_max_mi_decay()` | Nonlinear systems | Information theory |
| `estimate_tau_max_aic_bic()` | Model-based selection | AIC/BIC criteria |
| `estimate_tau_max_nyquist_domain()` | Conservative hybrid | Sample adequacy |
| `estimate_tau_max_first_mi_minimum()` | Takens embedding | Dynamical systems |
| `estimate_tau_max_ensemble()` | Multi-method consensus | Ensemble voting |

### Configuration

**File**: `framework/config/defaults.json`

```json
{
  "tau_max": 6,
  "tau_max_method": "scientific",
  "tau_max_auto_estimate": false,
  "temporal": {
    "max_lag_days": 90,
    "tau_max_confidence_level": 0.95,
    "tau_max_safety_factor": 3.0
  }
}
```

Set `tau_max_auto_estimate: true` or pass `tau_max=None` to enable automatic estimation.

### Why tau_max Matters

- **Too small**: Misses long-term causal effects (Type II error)
- **Too large**: Overfitting, spurious correlations, reduced statistical power
- **Data-driven**: Adapts to actual temporal dependencies in your data
- **Publication-ready**: Transparent, reproducible, scientifically justified

**Example**: Climate→vegetation studies typically need 60-90 days (12-18 timesteps at 5-day sampling) to capture soil moisture memory and phenological responses. Fixed tau_max=6 (30 days) may be too conservative.

### References

- **Runge, J., et al. (2019)**. Inferring causation from time series in Earth system sciences. *Nature Communications*, 10(1), 2553.
- **Box, G. E., & Jenkins, G. M. (1976)**. Time series analysis: forecasting and control.
- **Peters, J., Janzing, D., & Schölkopf, B. (2017)**. Elements of causal inference.

For complete theoretical justification, see `docs/tau_max_scientific_justification.md`.

## Quick Start

### 1. Load Data

```python
from framework.core import io

df, metadata = io.load_timeseries_csv(
    "my_data.csv",
    time_column="date",
    parse_dates=True,
    infer_sampling_interval=True
)

print(metadata)
```

**Supported formats**:
- **Wide**: Columns per variable, rows per timestamp
- **Long**: Columns for `time`, `id`, `variable`, `value`

### 2. Apply Preprocessing & Quality Control

```python
from framework.core import preprocessing, qc

# Create preprocessing pipeline with automatic data structure detection
preprocessor = preprocessing.TimeSeriesPreprocessor(
    interpolation_method="linear",
    max_missing_frac=0.2,        # For time series
    min_valid_obs=60,             # For panel data (satellite observations)
    normalize=True,
    remove_seasonality=False,
    outlier_method="iqr"
)

# Process data - automatically detects panel vs. time series
df_processed, metadata_processed, report = preprocessor.process(
    df, 
    metadata=metadata,  # Include for panel data
    verbose=True
)

# Review preprocessing report
print(report.to_dict())
# Shows: missing data handling, outliers clipped, stationarity tests, etc.

# Optional: Run distribution tests
from framework.core import distribution_tests

dist_report = distribution_tests.test_all_distributions(
    df_processed,
    variables=["NDVI", "RR", "TG", "PP"],
    alpha=0.05,
    output_dir="results/diagnostics"
)

# Traditional QC (if not using preprocessing pipeline)
df_flagged, report = qc.flag_low_coverage_series(
    df,
    min_coverage_pct=10,
    groupby_column="cube_id"  # Optional for panel data
)

df_clean, qc_results = qc.detect_and_handle_outliers(
    df_flagged,
    method="iqr",
    handle_mode="clip"
)
```

### 3. Prepare Data & Test Statistics

```python
from framework.core import stats

# Test stationarity
for col in df_clean.select_dtypes(include=[float]).columns:
    result = stats.test_stationarity(df_clean[col], method="adf")
    print(f"{col}: {result['interpretation']}")

# Assess linearity
linearity = stats.assess_linearity(
    df_clean["RR"],
    df_clean["NDVI"],
    pearson_r2_threshold=0.8
)
print(f"RR->NDVI linear: {linearity['is_linear']}")

# Suggest max lag
lag_info = stats.suggest_max_lag(
    df_clean["NDVI"],
    sampling_interval_days=5,
    default_max_days=90
)
print(f"Max lag: {lag_info['max_lag_timesteps']} timesteps ({lag_info['max_lag_days']} days)")
```

### 4. Run Enhanced Causal Analysis

```python
from framework.core import run_workflow

# Run complete workflow with all enhancements enabled
results = run_workflow.run_causal_workflow(
    data=df_processed,
    metadata=metadata_processed,
    pairs=[("RR", "NDVI"), ("TG", "NDVI"), ("PP", "NDVI")],
    tau_max=12,
    alpha=0.05,
    output_dir="results/exp1",
    # Enable enhancements
    enable_preprocessing=True,        # Already done above, but can re-run
    enable_distribution_tests=True,   # Test assumptions
    enable_causal_strength=True,      # Quantify effect sizes
    enable_temporal_validation=True,  # Cross-validation
    enable_experiment_tracking=True,  # Log parameters & data hash
    verbose=True
)

# Results include:
# - Consensus edges with agreement counts
# - Causal strength metrics (correlation, MI, F-stats)
# - Validation scores (out-of-sample accuracy)
# - Distribution test results
# - Preprocessing report
# - Experiment metadata

# Or run via CLI
```bash
python -m framework.cli run \
    --input data.csv \
    --methods granger te pcmci consensus \
    --maxlag 12 \
    --alpha 0.05 \
    --output results/ \
    --enable-preprocessing \
    --enable-strength \
    --enable-validation
```

## Configuration

### Method Selection (New! 🆕)

You can now **selectively enable/disable causal discovery methods** per experiment. This allows you to:
- Skip computationally expensive methods (e.g., PCMCI+ for large datasets)
- Focus on specific method types (e.g., only Granger for linear systems)
- Run method-specific experiments for comparison studies

#### Framework Defaults

**File**: `framework/config/defaults.json`

```json
{
  "methods": {
    "granger": {
      "enabled": true
    },
    "transfer_entropy": {
      "enabled": true
    },
    "pcmci": {
      "enabled": true
    }
  }
}
```

#### Per-Experiment Configuration

Create a `config.json` in your experiment folder:

```json
{
  "experiment": {
    "name": "my_experiment",
    "dataset": "data.parquet"
  },
  "methods": {
    "granger": {
      "enabled": true
    },
    "transfer_entropy": {
      "enabled": false
    },
    "pcmci": {
      "enabled": false
    }
  },
  "parameters": {
    "tau_max": 12,
    "alpha": 0.05
  }
}
```

#### Programmatic Usage

```python
from framework.core import run_causal_discovery_workflow

# Option 1: Pass method_config directly
method_config = {
    "granger": {"enabled": True},
    "transfer_entropy": {"enabled": False},
    "pcmci": {"enabled": True}
}

results = run_causal_discovery_workflow(
    data_df=df,
    output_dir="results/",
    method_config=method_config,
    ...
)

# Option 2: Let framework load from experiment config.json
# (see experiment structure below)
```

#### Experiment Structure Pattern

```
autocause_resources/
├── data/
│   └── my_dataset/
│       └── my_data.parquet           # Dataset files
└── experiments/
    └── my_dataset/
        └── my_experiment/
            ├── config.json           # Experiment configuration
            ├── run_experiment.py     # Experiment runner script
            └── results/              # Generated by framework
                ├── results_granger.csv
                ├── consensus.csv
                ├── figures/
                └── experiment_metadata.json
```

**Example `run_experiment.py`**:
```python
import json
from pathlib import Path
from framework.core import run_causal_discovery_workflow
import pandas as pd

# Load config
config_file = Path(__file__).parent / "config.json"
with open(config_file) as f:
    config = json.load(f)

# Dataset path can be absolute or relative to experiment directory
dataset_path = Path(config["experiment"]["dataset"])
if not dataset_path.is_absolute():
    dataset_path = config_file.parent / dataset_path

# Load data
df = pd.read_parquet(dataset_path)

# Run workflow with method config
results = run_causal_discovery_workflow(
    data_df=df,
    output_dir=Path(__file__).parent / "results",
    method_config=config.get("methods"),  # Pass to framework
    tau_max=config["parameters"]["tau_max"],
    alpha=config["parameters"]["alpha"],
    ...
)
```

Run your experiment:
```bash
cd autocause_resources/experiments/my_dataset/my_experiment
python run_experiment.py
```

See `/Volumes/X10 Pro/autocause_resources/experiments/earthnet/earthnet_timeseries/` for a complete working example.

### Global Configuration

The `config/defaults.json` file controls all framework behavior:

### Key Sections

| Section | Purpose | Key Options |
|---------|---------|-------------|
| `general` | Global settings | `sampling_interval_days`, `verbose`, `random_seed` |
| `temporal` | Lag configuration | `max_lag_days`, `max_lag_timesteps`, `lag_selection_method` |
| `tau_max_*` | Automatic lag estimation | `tau_max_method` (scientific/acf/ensemble), `tau_max_auto_estimate`, confidence levels |
| `preprocessing` | Data preprocessing | `interpolation_method`, `max_missing_frac`, `min_valid_obs` (for panel data), `normalize`, `remove_seasonality` |
| `stationarity` | Non-stationarity handling | `test`, `treat_failure` (difference/detrend), `seasonal_detrending` |
| `linearity` | Linearity assessment | Thresholds for R², MI/R² ratio, BDS test |
| `missing_data` | NA handling | `strategy`, `outlier_handling`, `max_missing_percent` |
| `multiple_testing` | p-value correction | `correction_method` (auto/bonferroni/fdr_bh), thresholds |
| `methods` | Method settings | Enable/disable each method, set CI tests, FDR options |
| `causal_strength` | Effect size metrics | `compute_correlation`, `compute_mutual_info`, `compute_granger_f` |
| `validation` | Temporal validation | `cross_validation_folds`, `test_size`, `lag_stability_tests` |
| `plotting` | Visualization | Theme, formats (svg/pdf/png), colors, fonts |
| `mapping` | Geographic plots | CRS, Europe extent, colormaps |
| `reporting` | Report generation | Literature sources, baseline lag ranges |

### Example Override

```json
{
  "temporal": {
    "max_lag_days": 120,
    "lag_selection_method": "bic"
  },
  "tau_max": null,
  "tau_max_method": "scientific",
  "tau_max_auto_estimate": true,
  "preprocessing": {
    "interpolation_method": "gp",
    "min_valid_obs": 60,
    "normalize": true,
    "outlier_method": "zscore"
  },
  "methods": {
    "granger": {"enabled": true},
    "pcmci": {"tau_max": 20, "ci_test": "GPDC"},
    "transfer_entropy": {"enabled": false}
  },
  "causal_strength": {
    "compute_correlation": true,
    "compute_mutual_info": true,
    "compute_granger_f": true
  },
  "validation": {
    "cross_validation_folds": 5,
    "test_size": 0.2
  }
}
```

## Output Structure

After running causal discovery, results are organized as:

```
results/
├── preprocessing_report.json    # Data quality, transformations applied
├── tau_max_estimation.json      # Automatic tau_max estimation results (ACF/domain/Nyquist constraints)
├── distribution_tests.json      # Normality, stationarity, independence tests
├── granger.csv                  # Granger results (pair, lag, p-value, corrected_p, significant)
├── pcmci.csv                    # PCMCI+ results (link, lag, MI, p-value, contemp_edge)
├── te.csv                       # Transfer Entropy results (pair, TE_value, p-value)
├── consensus.csv                # Consensus results (pair, agreement_count, methods)
├── causal_strength.csv          # Effect sizes (correlation, MI, F-statistics)
├── validation_results.json      # Cross-validation scores, lag stability
├── experiment_log.json          # Parameters, data hash, timestamps
├── figures/
│   ├── preprocessing/
│   │   ├── missing_data_*.svg
│   │   └── stationarity_tests_*.svg
│   ├── diagnostics/
│   │   ├── normality_tests_*.svg
│   │   └── qq_plots_*.svg
│   ├── pvalues_dist.svg        # P-value distribution plots
│   ├── lags_histogram.svg      # Lag distribution (in days)
│   ├── causal_graphs_*.svg     # Tigramite causal network graphs per region
│   ├── strength_heatmap.svg    # Effect size visualization
│   ├── validation_curves.svg   # Cross-validation performance
│   ├── detection_map.svg       # Europe map with detection rates by UTM zone
│   └── method_comparison.svg
├── report/
│   ├── causal_summary.md       # Main findings
│   ├── statistics_table.csv    # Per-experiment detection %, lag stats
│   ├── literature_alignment.md
│   └── references.bib
```

## Methods & Statistical Care

### Preprocessing Pipeline

- **Panel Data Detection**: Automatically identifies panel structure via `unit_id` column
- **Adaptive Quality Checks**:
  - **Panel data**: Checks median valid observations per unit (e.g., ≥60 obs)
  - **Time series**: Checks global missing percentage (e.g., ≤20%)
- **Interpolation**: Linear, spline, or Gaussian Process for missing values
- **Outlier Detection**: IQR, Z-score, or custom methods with clipping/interpolation
- **Stationarity Testing**: ADF/KPSS with automatic differencing if needed
- **Normalization**: Z-score standardization option
- **Seasonal Decomposition**: STL decomposition with configurable periods

**Key Innovation**: Correctly handles sparse satellite observations (e.g., Sentinel-2 every ~5 days) by checking absolute observation counts rather than percentage coverage, preventing false rejection of high-quality but temporally sparse data.

### Granger Causality

- **Implementation**: Vector AR with lag selection (AIC/BIC)
- **Stationarity**: Automatic differencing if needed
- **Linearity**: Applied only to pairs detected as linear
- **p-value Correction**: BH (Benjamini-Hochberg) FDR across cubes/regions
- **Output**: p-values per lag, detection rates, lag distributions, F-statistics

### PCMCI+ / LPCMCI

- **Reference**: Runge (2020, PMLR v124); Gerhardus & Runge (2020, NeurIPS)
- **Features**: Handles autocorrelated time series, latent confounders, contemporaneous edges
- **CI Tests**: ParCorr (fast, linear) or GPDC (nonlinear, slow)
- **Lag Selection**: tau_max controls maximum lag to explore (default 18 timesteps)
- **Output**: Causal graphs (tigramite), lagged + contemporaneous edges, q-values

### Transfer Entropy

- **Estimator**: Kraskov (continuous, k-NN) with fallback to adaptive discretization
- **Permutation Surrogates**: Block bootstrap for finite-sample p-values
- **Nonlinearity**: Preferred when linear tests reject or R² is very high
- **Output**: TE values, p-values, significance flags

### Lag Window Selection

The framework provides **automatic tau_max estimation** (see [Automatic tau_max Estimation](#automatic-tau_max-estimation-🆕) section above) using the scientific hybrid method:

- **Automatic** (recommended): `tau_max = min(ACF_zero_crossing, domain_max/sampling, N/3)`
- **Manual fallback**: min(⌊90 days / sampling_interval⌋, 18 timesteps)
- **E.g., 5-day composites**: 18 timesteps = 90 days ≈ **12.9 weeks**
- **Literature baseline**: 1-12 weeks expected per Erasmi et al. 2009, Chen et al. 2020

For climate→vegetation studies, the scientific method typically suggests 12-18 timesteps (60-90 days) based on soil moisture memory and vegetation response times.

### Multiple Testing Control

- **Auto-selection**: FDR if >100 tests, else FWER
- **Methods**: Bonferroni (conservative), Holm (step-down), BH (FDR), BY (arbitrary dependence)
- **Decision rule**: Computed for each comparison per method

### Causal Strength Metrics

- **Correlation**: Pearson/Spearman for linear relationships
- **Mutual Information**: KSG estimator for nonlinear dependencies
- **Granger F-statistics**: Effect size from VAR models
- **Normalized Effect Sizes**: Cohen's d, η² (eta-squared) where applicable

### Temporal Validation

- **Cross-Validation**: K-fold temporal splits preserving time order
- **Out-of-Sample Testing**: Hold-out test sets for validation
- **Lag Stability**: Robustness checks across different lag specifications
- **Bootstrap Confidence Intervals**: Uncertainty quantification for effect sizes

## Baseline Literature & Theory

See `reporting/summarize.py` for automated literature alignment. Key references:

1. **Papagiannopoulou et al. 2017 (ERL)**:  
   ~61% vegetated land is water-limited. Antecedent precipitation effects up to 3 months in dry regions.

2. **Papagiannopoulou et al. 2017 (GMD)**:  
   Nonlinear methods reveal additional signal. Explained variance in water-limited regions can exceed 40%.

3. **Lag expectations**: 1-12 weeks typical; 1-2 months in semi-arid systems (literature consensus).

4. **Mediterranean regime** (Vogel et al. 2021):  
   Soil-moisture limited (summer), energy-limited (winter). Hottest vulnerability May-July.

## Citation

Please cite this framework as:

```bibtex
@software{CausalDiscoveryFramework2024,
  title={Causal Discovery Framework for Time-Series Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## References (Detailed)

- **PCMCI+**: Runge, J. (2020). "Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets." *PMLR* 124, 1-16.
- **LPCMCI**: Gerhardus, A., & Runge, J. (2020). "High-recall causal discovery for autocorrelated time series with latent confounders." *NeurIPS*, 6387-6397.
- **Transfer Entropy**: Kraskov, A., et al. (2004). "Estimating mutual information." *Phys. Rev. E*, 69(6), 066138.
- **EarthNet**: Requena-Mesa, C., et al. (2021). "EarthNet2021: A large-scale dataset and challenge for Earth surface forecasting." *CVPRW*, 184-199.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "Series is non-stationary" | Data has trend/seasonality | Enable `seasonal_detrending` or increase `max_lag_timesteps` |
| Low detection rates | High missing data | Lower `min_coverage_pct` or check `missing_data.strategy` |
| Method runs slowly | Too many pairs / large tau_max | Reduce `tau_max`, filter variables, enable GPU (if available) |
| Outliers skew results | Extreme values present | Change `outlier_handling` to "interpolate", adjust thresholds |
| Satellite data rejected | Sparse temporal sampling | Framework auto-detects panel data; ensure `min_valid_obs` is set appropriately (default 60) |
| NDVI dropped despite good quality | Wrong quality check mode | Verify panel data has `unit_id` column in metadata - triggers observation count check instead of percentage |
| Preprocessing too slow | Large panel datasets | Vectorized operations are used; ensure pandas/numpy are optimized |

## License

[Specify your license]

## Support

For issues or questions, please open an issue on the project repository.
