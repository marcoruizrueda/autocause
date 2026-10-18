<div align="center">

# AutoCause

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Tests](https://img.shields.io/badge/tests-70%20collected-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/paper-arxiv-red.svg)](https://arxiv.org/abs/2608.00198)

<p align="center">
  <a href="https://arxiv.org/abs/2608.00198">📄 Paper (arXiv:2608.00198)</a>
</p>

<img src="https://raw.githubusercontent.com/marcoruizrueda/autocause/main/docs/figures/graphical_abstract.png" alt="AutoCause graphical abstract" width="90%">

**Automated expert decisions for environmental time-series causal discovery**: method selection, lag choice, CI-test selection, sample-size adequacy, FDR correction, and evidence grading.

</div>

> Ruiz, M.¹, Arana-Catania, M.², Ardila, D. R.³ & Ventura, R.¹ (2026).
> *Environmental Modelling & Software.*
>
> ¹ ISR-Lisbon, Instituto Superior Técnico, Lisbon &nbsp;
> ² Digital Scholarship at Oxford, University of Oxford &nbsp;
> ³ Jet Propulsion Lab., Caltech

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Usage](#usage)
- [Framework &amp; pipeline](#framework--pipeline)
- [Citation](#citation)
- [License](#license)

## Install

### Users

Install the released package with `uv` (recommended) or `pip`:

```bash
uv pip install autocause
# or
pip install autocause
```

The optional causal-audit pre-discovery module is installed with the
`causal-audit` extra, or on its own:

```bash
uv pip install "autocause[causal-audit]"
# or install the companion package directly
pip install causal-audit
```

### Contributors

Clone the repository and install an editable copy into a fresh environment:

```bash
git clone https://github.com/marcoruizrueda/autocause.git
cd autocause
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

For development with the pre-discovery audit, clone `causal-audit` beside
`autocause` and install both editable packages:

```bash
git clone https://github.com/marcoruizrueda/causal-audit.git ../causal-audit
uv pip install -e ../causal-audit -e '.[causal-audit]'
```

The `uv` source configuration and lockfile use that sibling checkout. The
supported interpreter is Python 3.13, including the companion audit package.
`numba` is a required dependency for the CMIknn transfer-entropy estimator;
LiNGAM brings its own dependencies, including `psy`, through the resolver.

## Quickstart

```python
import pandas as pd
from autocause.core.run_workflow import run_causal_discovery_workflow

# DataFrame with a DatetimeIndex and one column per variable
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

result = run_causal_discovery_workflow(
    data_df=df,
    output_dir="results/",
    enable_causal_audit=True,
    apply_audit_recommendation=True,
    enable_consensus=True,
)
print(result["success"])
```

This enables the companion audit and consensus alongside the default workflow
stages. The audit fills unset configuration fields; review saved warnings and
evidence before interpreting links. See [Usage](#usage) for configured runs.

Run the [quickstart notebook](https://github.com/marcoruizrueda/autocause/blob/main/examples/quickstart.ipynb) for the complete workflow
and its figures. Use the independent [walkthrough](https://github.com/marcoruizrueda/autocause/blob/main/examples/walkthrough.ipynb) for
configuration, statistical interpretation, and auditing. See
[Quickstart validation](https://github.com/marcoruizrueda/autocause/blob/main/docs/QUICKSTART_VALIDATION.md) for reproducible checks.

## Usage

```python
import pandas as pd
from autocause.core.run_workflow import run_causal_discovery_workflow

# Input: DataFrame with DatetimeIndex, one column per variable, NaN for missing
df = pd.read_csv("data.csv", index_col=0, parse_dates=True)

results = run_causal_discovery_workflow(
    data_df=df,
    output_dir="results/",
    target_var="FCH4",
    alpha=0.05,
    sampling_days=1,
    deseasonalize=True,  # subtract rolling mean before discovery
    true_edges={
        ("X", "Y")
    },  # optional: if ground truth available, computes F1, AUROC, AUPRC
    enable_consensus=True,  # multi-method voting
    enable_causal_audit=True,  # run causal-audit assumption diagnostics first
    method_config={
        "granger": {"enabled": True},  # conditional VAR-based Granger (statsmodels)
        "transfer_entropy": {
            "enabled": True
        },  # CMIknn nearest-neighbour estimator (tigramite)
        "pcmci": {
            "enabled": True,
            "test_method": "parcorr",
        },  # PCMCI+ constraint-based (tigramite)
        "varlingam": {"enabled": True},  # ICA-based VAR-LiNGAM (lingam)
        "lpcmci": {"enabled": False},  # LPCMCI for latent confounders (tigramite)
        "correlation": {
            "enabled": True
        },  # lagged Pearson/Spearman (non-causal baseline)
        "predictive_baseline": {
            "enabled": True
        },  # Random Forest permutation importance (scikit-learn)
        "ci_sensitivity": {
            "enabled": True
        },  # compare ParCorr vs RobustParCorr vs CMIknn
    },
)
```

For detailed options (audit-only mode, adaptive CI-test selection, sample size adequacy, PCMCI vs PCMCI+, graph recovery metrics), see [`docs/USAGE.md`](https://github.com/marcoruizrueda/autocause/blob/main/docs/USAGE.md).

### Methods

| Method | Paradigm | CI test / engine | Library |
|--------|----------|------------------|---------|
| VAR-based Granger | Regression (conditional) | *F*-test on multivariate VAR | `statsmodels` |
| Transfer entropy | Information-theoretic | CMIknn surrogates | `tigramite` |
| PCMCI+ | Constraint-based | Auto: ParCorr / RobustParCorr / CMIknn | `tigramite` |
| LPCMCI | Constraint-based (latent confounders) | Auto | `tigramite` |
| VAR-LiNGAM | Score-based (ICA) | Bootstrap pruning | `lingam` |
| RF-baseline | Predictive (non-causal) | Permutation importance | `scikit-learn` |

The first five are causal discovery methods from the dominant paradigms. The RF-baseline is a predictive reference included to answer: "does causal discovery find something different from feature importance?"

## Framework &amp; pipeline

The end-to-end architecture figure and the full 21-step workflow (data and
assumptions, discovery, validation and synthesis) are documented in
[FRAMEWORK.md](https://github.com/marcoruizrueda/autocause/blob/main/autocause/FRAMEWORK.md).

## Reproducing the paper experiments

All experiment scripts, configurations, and reproduction instructions are in
[`experiments/README.md`](https://github.com/marcoruizrueda/autocause/blob/main/experiments/README.md).

## Citation

```
@article{ruiz2026autocause,
  title={AutoCause: A Python framework that automates expert decisions in environmental time-series causal discovery},
  author={Ruiz, Marco and Arana-Catania, Miguel and Ardila, David R and Ventura, Rodrigo},
  journal={arXiv preprint arXiv:2608.00198},
  year={2026}
}
# APA:
Ruiz, M., Arana-Catania, M., Ardila, D. R., & Ventura, R. (2026). AutoCause: A Python framework that automates expert decisions in environmental time-series causal discovery. arXiv preprint arXiv:2608.00198.
```
## License

AGPL-3.0-or-later. Depends on Tigramite (GPL-3.0-or-later), which remains under its original license.
