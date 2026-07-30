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

## Code style

- Follow existing patterns: lazy imports for optional dependencies,
  column-name conventions in `graph_metrics.py`, batch_* function signature.
- No comments in source code unless the logic is non-obvious.
- Every new method needs a `test_` in `tests/`.
