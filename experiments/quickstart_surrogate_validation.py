"""Reproducible null/positive controls for quickstart surrogate checks.

Run from the repository root:
    .venv/bin/python experiments/quickstart_surrogate_validation.py

These small simulations are regression evidence for specified stationary VARs,
not a calibration guarantee for arbitrary data or discovery-selected edges.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import f

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autocause.core.falsification import run_falsification_battery


def granger_f_pvalue(source, target, lag):
    """Nested OLS F test of source lags given target lags and an intercept."""
    y = target[lag:]
    restricted = np.column_stack([np.ones(len(y))] +
                                 [target[lag-k:len(target)-k] for k in range(1, lag+1)])
    full = np.column_stack([restricted] +
                           [source[lag-k:len(source)-k] for k in range(1, lag+1)])
    residual_r = y - restricted @ np.linalg.lstsq(restricted, y, rcond=None)[0]
    residual_f = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    rss_r, rss_f = residual_r @ residual_r, residual_f @ residual_f
    df_resid = len(y) - full.shape[1]
    statistic = max(0., (rss_r - rss_f) / lag / (rss_f / df_resid))
    return float(f.sf(statistic, lag, df_resid))


def simulate(seed, persistence, effect, n=300):
    rng = np.random.default_rng(seed)
    # Burn-in keeps initial transients out of the controls.
    values = np.zeros((n+200, 2))
    for t in range(1, len(values)):
        values[t, 0] = persistence * values[t-1, 0] + rng.normal()
        values[t, 1] = .3 * values[t-1, 1] + effect * values[t-1, 0] + rng.normal()
    return values[200:, 0], values[200:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--surrogates', type=int, default=99)
    parser.add_argument('--output', type=Path,
                        default=Path('examples/quickstart_output/validation/surrogate_controls.csv'))
    args = parser.parse_args()
    rows = []
    for persistence in [.3, .8]:
        for effect in [0., .6]:
            for seed in range(args.seeds):
                source, target = simulate(seed, persistence, effect)
                battery = run_falsification_battery(
                    source, target, 1, granger_f_pvalue,
                    n_surrogates=args.surrogates, seed=1000+seed)
                for method, result in battery.items():
                    rows.append(dict(seed=seed, persistence=persistence, effect=effect,
                                     method=method, **{k: result[k] for k in
                                     ['original_pvalue', 'empirical_pvalue', 'passed', 'status', 'n_surrogates']}))
            print(f'Completed persistence={persistence}, effect={effect}', flush=True)
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    summary = df.groupby(['persistence', 'effect', 'method']).agg(
        n=('passed', 'size'), passes=('passed', 'sum'), pass_rate=('passed', 'mean'))
    summary.to_csv(args.output.with_name('surrogate_controls_summary.csv'))
    print(summary.to_string())
    assert df.status.eq('ok').all(), 'At least one control failed to execute'
    assert df.empirical_pvalue.between(1/(args.surrogates+1), 1).all()


if __name__ == '__main__':
    main()
