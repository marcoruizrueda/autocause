# AutoCause

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Tests](https://img.shields.io/badge/tests-14%2F14-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/paper-arxiv-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**AutoCause** automates expert decisions in environmental time-series causal discovery:
method selection, lag choice, CI-test selection, sample-size adequacy, FDR correction, and evidence grading.

AutoCause orchestrates multiple causal discovery methods on time-series data, with automatic tau_max estimation, preprocessing, adaptive CI-test selection, sample size adequacy diagnostics, FDR correction, graph recovery evaluation (F1, SHD, AUROC), consensus voting, falsification testing, and tiered edge classification.

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
    causal_audit_only=True,          # run ONLY causal-audit, skip all discovery methods
    method_config={
        "pcmci": {"enabled": True, "test_method": "parcorr"},  # force a specific CI test
        "ci_sensitivity": {"enabled": True},  # compare ParCorr vs RobustParCorr vs CMIknn
    },
)
```

When `causal_audit_only=True`, the workflow runs the full pre-discovery diagnostics (stationarity, nonlinearity, confounding, seasonality) and returns immediately with the risk profile and method recommendation. No causal discovery methods are executed. This is useful for fast data screening before committing to a full run.

```python
# Fast pre-screening: ~30 seconds instead of hours
result = run_causal_discovery_workflow(
    data_df=df, output_dir="audit_only/", causal_audit_only=True,
    enable_preprocessing=False,
)
policy = result["causal_audit"]["policy"]
print(f"Recommended: {policy.recommended_method} ({policy.confidence:.0%})")
# → Recommended: PCMCI+ (83%)
```

### Adaptive CI-test selection

PCMCI+ defaults to `test_method="auto"`, which runs a Ramsey RESET test on the data and selects:

- **ParCorr** - linear, Gaussian data (fast, high power)
- **RobustParCorr** - linear, non-Gaussian data (robust to heavy tails)
- **CMIknn** - nonlinear data (nonparametric, slower)

When nonlinear data is detected but the sample size is too small for CMIknn (T_eff < 200), the selector automatically falls back to RobustParCorr with a logged warning. This sample-size-aware fallback prevents unreliable results on short series.

Override with `method_config={"pcmci": {"test_method": "cmiknn"}}` for explicit control.

### Sample size adequacy

Before running any method, AutoCause checks whether the effective sample size (T minus tau_max minus missing-value rows) is sufficient for each method. The check runs automatically and saves `sample_size_adequacy.json` in the output directory.

```python
from framework.core.sample_size_adequacy import assess_sample_size

report = assess_sample_size(df, tau_max=10)
print(report.t_effective)          # e.g., 185
print(report.recommended_methods)  # e.g., ['parcorr', 'robust_parcorr', 'granger']
print(report.warnings)             # e.g., ['Methods with insufficient data: cmiknn, gpdc']
```

Method-specific minimums (example for N=4, tau_max=5):

| Method | T_min | Rationale |
|--------|:-----:|-----------|
| ParCorr | 50 | 3 obs per OLS parameter |
| RobustParCorr | 66 | Rank transformation overhead |
| Granger (VAR) | 73 | F-test degrees of freedom |
| VARLiNGAM | 120 | ICA stability |
| CMIknn | 200 | k-NN MI convergence |
| GPDC | 500 | GP hyperparameter optimization |

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

When `true_edges` is provided, the workflow saves `graph_recovery_metrics.csv` with both binary (F1, precision, recall, SHD) and ranking (AUROC, AUPRC) metrics per method, following the evaluation protocols of TimeGraph (Ferdous et al. 2025) and CausalTime (Cheng et al. 2023).

SHD (Structural Hamming Distance) counts the minimum number of edge additions, deletions, and direction reversals needed to transform the discovered graph into the true graph. A reversed edge (A→B discovered as B→A) counts as 1 edit, not 2. This matches the standard definition used in TimeGraph and the broader causal discovery literature.

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
2. Sample size adequacy - method-specific T_min checks, CI-test fallback recommendations
3. Preprocessing - outlier removal, interpolation, normalization, stationarity check
4. Deseasonalization (optional) - rolling-mean subtraction to remove annual cycles
5. [causal-audit](https://github.com/marcoruizrueda/causal-audit) (optional) - assumption diagnostics, risk scores, method recommendation
6. Distribution tests - Gaussianity, linearity → CI test recommendation
7. Correlation analysis - Pearson, Spearman, distance correlation (symmetric baseline)

**B. Discovery**
8. VAR-based Granger causality - conditional F-tests (all controls for N ≤ 10)
9. PCMCI+ - momentary conditional independence with adaptive CI test
10. LPCMCI - PCMCI+ extended for latent confounders (outputs PAG)
11. VAR-LiNGAM - ICA on VAR residuals (non-Gaussian identifiability)
12. Transfer entropy - k-NN conditional mutual information with surrogate p-values
13. RF-baseline - Random Forest feature importance (non-causal comparison)

**C. Validation and synthesis**
14. Graph recovery evaluation - F1, SHD, AUROC, AUPRC against ground truth (when available)
15. Ensemble scoring - confidence-weighted aggregation across methods (regime-adaptive weights)
16. Power analysis - minimum detectable effect size (MDES) per method
17. CI-test sensitivity - compare edges across ParCorr / RobustParCorr / CMIknn
18. Falsification - block permutation + IAAFT surrogate tests
19. ICP stability - coefficient stability across environments
20. Consensus - multi-method voting with lag tolerance
21. Tiered classification - edges ranked by evidence strength

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

## Reproducing the paper experiments

Experiment scripts and data for the benchmarks reported in the paper are maintained in the
[`causal-audit`](https://github.com/marcoruizrueda/causal-audit) companion repository:

| Benchmark | Datasets | Run script |
|-----------|----------|------------|
| DGP-Atlas | 97 synthetic DGPs, 10 families | `experiments/atlas_validation/run.py` |
| TimeGraph | 18 categories (linear, nonlinear, trend, missing) | `experiments/timegraph_validation/run.py` |
| CausalRivers | 30 five-station subgraphs, Bavaria | `experiments/causalrivers_validation/run.py` |

See the paper for full evaluation protocol, metric definitions, and numerical results.

## Citation

If you use AutoCause in your research, please cite both the paper and the software:

```bibtex
@article{autocause2026,
  title   = {AutoCause: A Python framework that automates expert decisions
             in environmental time-series causal discovery},
  author  = {Ruiz, Marco and Arana-Catania, Miguel and Ardila, David R.
             and Ventura, Rodrigo},
  year    = {2026},
  journal = {Environmental Modelling \& Software},
  doi     = {10.1016/j.envsoft.2026.xxxxx}
}

@software{autocause2026code,
  title  = {AutoCause: Multi-Method Causal Discovery Framework},
  author = {Ruiz, Marco},
  year   = {2026},
  url    = {https://github.com/marcoruizrueda/autocause},
  doi    = {10.5281/zenodo.xxxxx}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 or later
(AGPL-3.0-or-later).

This project depends on Tigramite, which is licensed under the GNU General Public
License v3.0 or later (GPL-3.0-or-later). Tigramite remains under its original
license.
