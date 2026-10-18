# Quickstart corrections and verification

The independent examples are [quickstart.ipynb](../examples/quickstart.ipynb)
and [walkthrough.ipynb](../examples/walkthrough.ipynb).
After changing source code, restart each notebook's kernel and run every cell.

## Choose a notebook

The **quickstart** includes installation and sample-data setup, then executes one
`run_causal_discovery_workflow` call with the audit and consensus explicitly
enabled. It uses the packaged discovery-method list and lets the audit fill
unset configuration fields. Its evidence table and two package-generated
figures are saved under `examples/quickstart_output/easy_run/`.

The **walkthrough** has its own setup and makes an explicitly
configured run in `examples/walkthrough_output/`. It demonstrates the raw-data
audit and standalone preprocessing, explains each configuration choice, adds
two noncausal baselines, and inspects recovery metrics and provenance. Both
full workflow calls receive the raw input; the standalone transformed data
is for teaching, not input to another preprocessing pass.

The short path enables framework stages; it does not verify causal assumptions.
An audit abstention currently does not automatically stop discovery. The
notebook explains this behavior and points readers to warnings and per-method
outputs rather than treating `success=True` as evidence that every check passed.

## Corrections

- Transfer entropy uses `(b + 1) / (B + 1)` for Monte Carlo tail probabilities,
  counting ties as extreme. Invalid statistics fail instead of silently
  producing significance. This matches the finite permutation convention
  documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
- Observed and surrogate TE statistics use the same estimator. Estimator
  failures are surfaced; no discrete fallback is mixed with continuous null
  statistics. Discrete TE now computes conditional mutual information given
  target history. CMIknn histories are aligned and use the correct conditioning
  labels. The explicit lagged-MI proxy is reported in bits.
- Falsification ranks the original callback p-value against the surrogate
  p-values. Smaller callback values mean stronger evidence. Both the original
  and empirical p-values must be below alpha. This replaces the binomial test
  on the fraction of significant surrogate results. Output CSVs retain the
  empirical probabilities, simulation counts, resolution and failure status.
- Tier-2 requires a supporting check in addition to method agreement. Tier-1
  is named **Strong Internal Support**. Missing stability flags cannot pass.
- The companion causal-audit policy no longer infers non-Gaussian innovations
  from low diagnostic risks. Its linear default is PCMCI+ with ParCorr.
  [VARLiNGAM requires non-Gaussian errors](https://lingam.readthedocs.io/en/latest/tutorial/var.html).
- The workflow retains method results for its return value, lag analysis and
  provenance. Previously, clearing the dictionary produced empty returned
  results and misleading zero-result completion logs. The legacy `n_edges`
  return field counts result rows, including non-significant candidates.
- IAAFT converts ordinal ranks to integer indices, including its final amplitude
  adjustment. Recent [SciPy rankdata documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html)
  specifies floating-point output; indexing with those ranks caused every
  IAAFT check to fail in a freshly installed environment. A regression test
  forces floating-point ordinal ranks and checks preservation of the input's
  amplitude distribution, including tied values.
- The inline PCMCI matrix uses mathematical labels and a reported-value
  threshold. The notebook's values include Tigramite's BH adjustment, so
  labeling them unadjusted was incorrect. Stars indicate significance rather
  than causal orientation; lags and evidence grades remain in the tables.

Dependency and plot-cache fixes are included in the source checkout;
`pyproject.toml` declares numba for CMIknn and the uv source for the companion
audit repository.

## Reproduce the checks

From the AutoCause checkout, with both packages installed:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q ../causal-audit/tests -o cache_dir=/tmp/causal-audit-pytest-cache
.venv/bin/python experiments/quickstart_surrogate_validation.py
# Restart Kernel and Run All in both examples/quickstart.ipynb
# and examples/walkthrough.ipynb, then:
.venv/bin/python experiments/verify_quickstart_outputs.py
```

Regression tests first reproduced the incorrect probabilities, estimator
mixing, history alignment, tier decisions, recommendation and empty return.
The output verifier independently reconstructs skeleton metrics from raw
method CSVs, checks tier rules and surrogate probabilities, compares displayed
tables to CSVs, and compares embedded figure bytes to their saved files.
Cells are located by their source content rather than positional cell numbers,
so the notebooks can be reorganized without invalidating the verifier.

The simulation script uses 100 seeds per setting, 300 retained observations,
200 burn-in observations and 99 surrogates per check. It tests two source AR
coefficients (0.3 and 0.8), with either no cross-variable effect or a lag-1
effect of 0.6. Its F statistic is independently checked against statsmodels.
Per-seed outputs and grouped counts are written to
`examples/quickstart_output/validation/surrogate_controls*.csv`.

The checked run produced:

| Source AR coefficient | Cross-variable effect | Block passes / 100 | IAAFT passes / 100 |
|---|---|---|---|
| 0.3 | 0 | 6 | 2 |
| 0.8 | 0 | 5 | 5 |
| 0.3 | 0.6 | 100 | 100 |
| 0.8 | 0.6 | 100 | 100 |

These are observed counts in specified controls, with sampling uncertainty;
they do not establish universal calibration or power.

## Interpretation of the walkthrough

For its fixed seed, the consensus contains all four true cross-variable
lag-1 links as Tier-1, plus one spurious Tier-3 candidate (`X2 → X3`, lag 2).
The notebook explicitly reports direction-and-lag recovery separately from
the per-method skeleton table. Its five method rows recover all four true
skeleton links, but each includes false positives. The verifier checks these
values against the generated CSVs rather than treating the displayed output
as authoritative by itself.

The following limitations remain essential to interpreting these results:

- Pairwise TE and bivariate Granger checks can capture mediated or confounded
  associations. Source shuffling is not a conditional randomization sampler
  given all the other variables.
- Block permutation and IAAFT provide approximate time-series null models.
  The finite simulation correction does not establish their validity for a
  particular dataset. Selected-link checks do not supply post-selection or
  global multiplicity control.
- TE's default BH adjustment is across lags within each ordered pair, not
  across every pair in the graph.
- The legacy `icp_stable` field describes a coefficient-variation heuristic,
  not the formal ICP procedure. Method agreement, surrogate checks and
  stability on the same data are not independent validation.
- Risk scores, recommendation confidence and evidence tiers are not
  probabilities that a recovered graph is causally correct.

Historical experiment outputs and paper tables have not been regenerated.
Their previous falsification and tier results must not be described as outputs
of these corrected rules without rerunning the relevant experiments.
