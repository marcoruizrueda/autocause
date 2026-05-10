# AutoCause

Multi-method causal discovery framework for time series.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: GPLv3+](https://img.shields.io/badge/License-GPLv3%2B-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Tests: 14/14](https://img.shields.io/badge/tests-14%2F14-brightgreen.svg)]()

AutoCause orchestrates multiple causal discovery methods on time-series data, with automatic tau_max estimation, preprocessing, adaptive CI-test selection, FDR correction, graph recovery evaluation (F1 + AUROC), consensus voting, falsification testing, and tiered edge classification.

| Method | Paradigm | CI test / engine | Library |
|--------|----------|------------------|---------|
| VAR-based Granger | Regression (conditional) | F-test on multivariate VAR | `statsmodels` |
| Transfer entropy | Information-theoretic | CMIknn surrogates | `tigramite` |
| PCMCI+ | Constraint-based | Auto: ParCorr / RobustParCorr / CMIknn | `tigramite` |
| LPCMCI | Constraint-based (latent confounders) | Auto | `tigramite` |
| VAR-LiNGAM | Score-based (ICA) | Bootstrap pruning | `lingam` |
| RF-baseline | Predictive (non-causal) | Permutation importance | `scikit-learn` |

The first five are causal discovery methods representing the dominant paradigms. The RF-baseline is a predictive attribution method included to answer: "does causal discovery find something different from feature importance?"

## Installation

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

## Input data

A `pandas.DataFrame` with a `DatetimeIndex`, one column per variable (`float64`), `NaN` for missing values.

```python
#                      TA       TS     FCH4
# 2020-05-01       12.30    8.71    0.042
# 2020-05-02       13.10    8.95    0.038
# 2020-05-03        NaN     9.02    0.045
```

## Usage

```python
import pandas as pd
from framework.core.run_workflow import run_causal_discovery_workflow

df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

results = run_causal_discovery_workflow(
    data_df=df,
    output_dir="results/",
    target_var="FCH4",
    alpha=0.05,
    sampling_days=1,
    method_config={
        "granger":              {"enabled": True},
        "transfer_entropy":     {"enabled": True},
        "pcmci":                {"enabled": True},
        "varlingam":            {"enabled": True},
        "lpcmci":               {"enabled": False},
        "predictive_baseline":  {"enabled": True},
    },
)
```

### Key options

```python
run_causal_discovery_workflow(
    # ...
    deseasonalize=True,              # subtract rolling mean before discovery
    true_edges={("X","Y")},          # ground truth → auto-computes F1, AUROC, AUPRC
    enable_consensus=True,           # multi-method voting
    enable_causal_audit=True,        # run causal-audit assumption diagnostics first
    method_config={
        "pcmci": {"enabled": True, "test_method": "parcorr"},  # force a specific CI test
        "ci_sensitivity": {"enabled": True},  # compare ParCorr vs RobustParCorr vs CMIknn
    },
)
```

### Adaptive CI-test selection

PCMCI+ defaults to `test_method="auto"`, which runs a Ramsey RESET test on the data and selects:

- **ParCorr** - linear, Gaussian data (fast, high power)
- **RobustParCorr** - linear, non-Gaussian data (robust to heavy tails)
- **CMIknn** - nonlinear data (nonparametric, slower)

Override with `method_config={"pcmci": {"test_method": "cmiknn"}}` for explicit control.

### PCMCI vs. PCMCI+

The `"pcmci"` method key in AutoCause runs **PCMCI+** by default (discovers both lagged and contemporaneous causal links). To control this:

```python
method_config={
    # PCMCI+ (default): discovers lagged AND contemporaneous edges
    "pcmci": {"enabled": True, "test_method": "auto"},

    # PCMCI (lagged-only): set allow_contemporaneous=False
    # Useful when you know there are no instantaneous effects
    "pcmci": {"enabled": True, "test_method": "parcorr", "allow_contemporaneous": False},

    # LPCMCI: handles latent confounders (outputs a PAG, not a DAG)
    "lpcmci": {"enabled": True},
}
```

| Config | Algorithm | Discovers | Use when |
|--------|-----------|-----------|----------|
| `"pcmci": {"enabled": True}` | PCMCI+ | Lagged + contemporaneous | Default (most general) |
| `"pcmci": {"enabled": True, "allow_contemporaneous": False}` | PCMCI | Lagged only | No instantaneous effects expected |
| `"lpcmci": {"enabled": True}` | LPCMCI | Lagged + latent confounders | Hidden common causes suspected |

### Graph recovery evaluation

When `true_edges` is provided, the workflow saves `graph_recovery_metrics.csv` with both binary (F1, precision, recall, SHD) and ranking (AUROC, AUPRC) metrics per method - following the evaluation protocols of TimeGraph (Ferdous et al. 2025) and CausalTime (Cheng et al. 2023).

## Integration with causal-audit

```python
from causal_audit import RiskAwareGatekeeper
from framework.core.run_workflow import run_causal_discovery_workflow

gk = RiskAwareGatekeeper(random_seed=42)
audit = gk.analyze(data=df, output_dir="audit_results/")

if audit["policy"].decision == "recommend":
    run_causal_discovery_workflow(
        data_df=df, output_dir="results/", target_var="FCH4",
        enable_causal_audit=True,
    )
```

## Pipeline

**A. Data and assumptions**
1. tau_max estimation - ACF zero-crossing, domain constraint, Nyquist bound
2. Preprocessing - outlier removal, interpolation, normalization, stationarity check
3. Deseasonalization (optional) - rolling-mean subtraction to remove annual cycles
4. [causal-audit](https://github.com/marcoruizrueda/causal-audit) (optional) - assumption diagnostics, risk scores, method recommendation
5. Distribution tests - Gaussianity, linearity → CI test recommendation
6. Correlation analysis - Pearson, Spearman, distance correlation (symmetric baseline)

**B. Discovery**
7. VAR-based Granger causality - conditional F-tests (all controls for N ≤ 10)
8. PCMCI+ - momentary conditional independence with adaptive CI test
9. LPCMCI - PCMCI+ extended for latent confounders (outputs PAG)
10. VAR-LiNGAM - ICA on VAR residuals (non-Gaussian identifiability)
11. Transfer entropy - k-NN conditional mutual information with surrogate p-values
12. RF-baseline - Random Forest feature importance (non-causal comparison)

**C. Validation and synthesis**
13. Graph recovery evaluation - F1, AUROC, AUPRC against ground truth (when available)
14. CI-test sensitivity - compare edges across ParCorr / RobustParCorr / CMIknn
15. Falsification - block permutation + IAAFT surrogate tests
16. ICP stability - coefficient stability across environments
17. Consensus - multi-method voting with lag tolerance
18. Tiered classification - edges ranked by evidence strength

## Tests

```bash
pytest tests/ -v
```

## Adding a new method

Every method follows the same pattern: `batch_<method>(df, variable_pairs, ...) → DataFrame`.

**1. Create** `framework/core/methods/mymethod.py` with a `batch_mymethod()` function returning a DataFrame with `source`, `target`, `is_significant` columns.

**2. Register** in `framework/core/methods/__init__.py`:

```python
def _get_mymethod():
    from . import mymethod
    return mymethod
```

**3. Wire** into `framework/core/run_workflow.py`:

```python
enable_mymethod = method_config.get("mymethod", {}).get("enabled", False)
if enable_mymethod:
    results["mymethod"] = mymethod_mod.batch_mymethod(data_df, pairs, ...)
```

**4. Register** column conventions in `framework/core/graph_metrics.py`:

```python
METHOD_COLUMNS["mymethod"] = {
    "src": "source", "tgt": "target",
    "sig": "is_significant", "score": "p_value", "score_type": "pvalue",
}
```

The consensus, visualization, and graph evaluation modules automatically pick up any new method.

## Experiments

AutoCause includes four benchmark experiments in `experiments/`. Each has a `run.py` script and produces a standard output structure. Full results are in [`experiments/RESULTS.md`](experiments/RESULTS.md).

### Running the experiments

```bash
# Synthetic DGP Atlas (our dataset, ~10 min)
python experiments/atlas_validation/run.py

# TimeGraph benchmark (KDD 2025, ~15 min)
python experiments/timegraph_validation/run.py

# CausalTime benchmark (NeurIPS 2024, ~30 min per domain)
python experiments/causaltime_validation/run.py

# FLUXNET-CH4 real-world wetland methane (~5 min)
python experiments/fluxnet_ch4/run.py
```

Each experiment creates a folder per dataset/site with the standard AutoCause output:

```
experiments/<name>/
├── results.csv              # summary across all datasets/sites
├── RESULTS.md               # detailed analysis (root-level)
└── families/ or sites/      # per-dataset outputs
    └── <dataset>/
        ├── method/
        │   ├── granger/1-raw/results_granger.csv
        │   ├── pcmci/1-raw/results_pcmci.csv
        │   ├── varlingam/1-raw/results_varlingam.csv
        │   └── predictive_baseline/1-raw/results_predictive_baseline.csv
        ├── figures/
        │   ├── per_method/      # causal graphs, p-value distributions
        │   ├── comparison/      # cross-method panels
        │   └── diagnostics/     # FDR, DAG, lag analysis
        └── graph_recovery_metrics.csv   # F1, AUROC, AUPRC (when ground truth available)
```

### Reading the results

**1. Summary CSV** - `experiments/<name>/results.csv` has one row per (dataset, method) with F1, precision, recall. Load it to compare methods:

```python
import pandas as pd
r = pd.read_csv("experiments/timegraph_validation/results.csv")
for m in r["method"].unique():
    s = r[r["method"] == m]
    print(f"{m:<14} F1={s['f1'].mean():.2f}")
```

**2. Graph recovery metrics** - `graph_recovery_metrics.csv` in each dataset folder has both binary (F1) and ranking (AUROC, AUPRC) metrics per method. AUROC is more informative than F1 because it evaluates the ranking quality of confidence scores, not just the binary threshold.

```python
m = pd.read_csv("experiments/atlas_validation/families/F1_clean_var/graph_recovery_metrics.csv")
print(m[["method", "f1", "auroc", "auprc"]])
```

**3. Per-method CSVs** - each method's raw output is in `method/<name>/1-raw/results_<name>.csv`. These contain per-edge details (source, target, lag, p-value, significance):

```python
g = pd.read_csv("experiments/fluxnet_ch4/sites/FI-Lom_raw/method/granger/1-raw/results_granger.csv")
drivers = g[g["significant"] & (g["effect"] == "FCH4")]
print(drivers[["cause", "best_lag", "best_p_value"]])
```

**4. Causal vs. predictive** - compare the causal methods against the RF-baseline to see what causal discovery adds beyond prediction:

```python
# RF finds everything significant (no directional selectivity)
rf = pd.read_csv(".../predictive_baseline/1-raw/results_predictive_baseline.csv")
rf_drivers = set(rf[rf["is_significant"] & (rf["target"] == "FCH4")]["source"])

# Granger is more selective (conditional testing filters indirect paths)
gr = pd.read_csv(".../granger/1-raw/results_granger.csv")
gr_drivers = set(gr[gr["significant"] & (gr["effect"] == "FCH4")]["cause"])

print(f"RF finds {len(rf_drivers)} drivers, Granger finds {len(gr_drivers)}")
print(f"Granger-only (not in RF): {gr_drivers - rf_drivers}")
print(f"RF-only (filtered by Granger): {rf_drivers - gr_drivers}")
```

**5. Deseasonalization comparison** - the FLUXNET-CH4 experiment runs each site twice (raw and deseasonalized). Drivers that disappear after deseasonalization may be artifacts of shared annual cycles:

```python
r = pd.read_csv("experiments/fluxnet_ch4/results.csv")
for var in ["GPP_DT", "WTD", "VPD", "TA"]:
    raw = r[(r["variant"] == "raw") & r["drivers"].str.contains(var, na=False)]
    des = r[(r["variant"] == "deseasonalized") & r["drivers"].str.contains(var, na=False)]
    print(f"{var}: raw={len(raw)}, deseas={len(des)}")
```

### Writing a new experiment

Use the existing experiments as templates. The minimal pattern:

```python
from framework.core.run_workflow import run_causal_discovery_workflow

df = load_your_data()  # DataFrame with DatetimeIndex

run_causal_discovery_workflow(
    data_df=df,
    output_dir="experiments/my_experiment/output/",
    target_var="Y",
    alpha=0.05,
    sampling_days=1,
    true_edges={("X", "Y"), ("Z", "Y")},  # optional ground truth
    method_config={
        "granger": {"enabled": True},
        "pcmci": {"enabled": True},
        "varlingam": {"enabled": True},
        "predictive_baseline": {"enabled": True},
    },
)
```

The workflow handles everything: method execution, visualization, graph evaluation, and output organization.

## Citation

```bibtex
@software{autocause2025,
  title  = {AutoCause: Multi-Method Causal Discovery Framework},
  author = {Ruiz, Marco},
  year   = {2025},
  url    = {https://github.com/marcoruizrueda/autocause}
}
```

## License

Copyright (C) 2025 Marco Ruiz.

AutoCause is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

AutoCause is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See LICENSE for full text.

GNU General Public License v3.0+

AUTOCAUSE is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version. AUTOCAUSE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
