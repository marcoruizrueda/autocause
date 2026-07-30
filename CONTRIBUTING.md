# Contributing to AutoCause

We welcome contributions that improve AutoCause as a tool for environmental
time-series causal discovery.

## Scope

AutoCause wraps established causal discovery methods (Tigramite, LiNGAM).
The contribution is the decision logic, pre-discovery auditing, and
reproducible evaluation. Contributions that add new causal discovery
algorithms are out of scope unless they follow the existing plugin pattern.

## Before contributing

1. Open an issue to discuss the change before writing code.
2. Check whether the change affects the paper's numerical results.
3. Ensure all tests pass: `pytest tests/ -v`

## Development setup

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -e ".[dev]"
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

## Code style

- Follow existing patterns: lazy imports for optional dependencies,
  column-name conventions in `graph_metrics.py`, batch_* function signature.
- No comments in source code unless the logic is non-obvious.
- Every new method needs a `test_` in `tests/`.
