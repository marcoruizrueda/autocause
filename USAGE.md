# Usage Guide

Detailed reference for AutoCause configuration and features. For a quick start, see the main [README](README.md).

## Audit-only mode

When `causal_audit_only=True`, the workflow runs the full pre-discovery diagnostics (stationarity, nonlinearity, confounding, seasonality) and returns immediately with the risk profile and method recommendation. No causal discovery methods are executed. Useful for fast data screening before committing to a full run.

```python
result = run_causal_discovery_workflow(
    data_df=df, output_dir="audit_only/", causal_audit_only=True,
    enable_preprocessing=False,
)
policy = result["causal_audit"]["policy"]
print(f"Recommended: {policy.recommended_method} ({policy.confidence:.0%})")
# → Recommended: PCMCI+ (83%)
```

## Adaptive CI-test selection

PCMCI+ defaults to `test_method="auto"`, which runs a Ramsey RESET test on the data and selects:

- **ParCorr**: linear, Gaussian data (fast, high power)
- **RobustParCorr**: linear, non-Gaussian data (robust to heavy tails)
- **CMIknn**: nonlinear data (nonparametric, slower)

When nonlinear data is detected but the sample size is too small for CMIknn (*T*<sub>eff</sub> < 200), the selector falls back to RobustParCorr with a logged warning.

Override with `method_config={"pcmci": {"test_method": "cmiknn"}}` for explicit control.

## Sample size adequacy

Before running any method, AutoCause checks whether the effective sample size (*T* − τ<sub>max</sub> − missing rows) is sufficient for each method. The check runs automatically and saves `sample_size_adequacy.json` in the output directory.

```python
from framework.core.sample_size_adequacy import assess_sample_size

report = assess_sample_size(df, tau_max=10)
print(report.t_effective)          # e.g., 185
print(report.recommended_methods)  # e.g., ['parcorr', 'robust_parcorr', 'granger']
print(report.warnings)             # e.g., ['Methods with insufficient data: cmiknn, gpdc']
```

Method-specific minimums (example for *N* = 4, τ<sub>max</sub> = 5):

| Method | *T*<sub>min</sub> | Why |
|--------|:-----:|-----------|
| ParCorr | 50 | 3 obs per OLS parameter |
| RobustParCorr | 66 | Rank transformation overhead |
| Granger (VAR) | 73 | *F*-test degrees of freedom |
| VARLiNGAM | 120 | ICA stability |
| CMIknn | 200 | *k*-NN MI convergence |
| GPDC | 500 | GP hyperparameter optimization |

## PCMCI vs. PCMCI+

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

## Graph recovery evaluation

When `true_edges` is provided, the workflow saves `graph_recovery_metrics.csv` with both binary (F1, precision, recall, SHD) and ranking (AUROC, AUPRC) metrics per method, following the evaluation protocols of TimeGraph (Ferdous et al. 2025) and CausalTime (Cheng et al. 2023).

SHD (Structural Hamming Distance) counts the minimum number of edge additions, deletions, and direction reversals needed to transform the discovered graph into the true graph. A reversed edge (A → B discovered as B → A) counts as 1 edit, not 2. This matches the standard definition used in TimeGraph and the broader causal discovery literature.

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

