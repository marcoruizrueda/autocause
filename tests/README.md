# Tests

```bash
uv run pytest tests/ -v
```

Requires `pytest` and `pytest-timeout` to be installed in the environment.

| File | What it covers |
|------|----------------|
| `test_causal_discovery.py` | PCMCI+, VAR-LiNGAM, Granger, Transfer Entropy — link detection, lag orientation, graph recovery |
| `test_fixes.py` | Lagged correlation, paradigm diversity, multi-variable tau_max, differencing warnings |
| `test_icp_temporal.py` | ICP stability across time environments |
| `test_power_and_ensemble.py` | Power analysis (MDES), ensemble scoring, regime detection |
| `test_sample_size_adequacy.py` | Effective sample size, method-specific T_min, CI-test suggestions |
| `test_shd_metric.py` | Structural Hamming Distance — perfect recovery, missing edges, reversals |

Test data is synthetic (random or parametrized fixtures in `conftest.py`). No external datasets required.
