"""Statistical contracts for surrogate tests and evidence grading."""

import numpy as np
import pandas as pd
import pytest

from autocause.core import falsification
from autocause.core.methods import transfer_entropy as te
from autocause.core.tiered_consensus import classify_consensus_tier


def data():
    return pd.DataFrame(np.random.default_rng(0).normal(size=(40, 2)), columns=['X', 'Y'])


def test_iaaft_accepts_float_ordinal_ranks_and_preserves_amplitudes(monkeypatch):
    # SciPy 1.18 returns floating-point ranks even for method='ordinal'.
    rankdata = falsification.stats.rankdata
    monkeypatch.setattr(
        falsification.stats, 'rankdata',
        lambda *args, **kwargs: np.asarray(rankdata(*args, **kwargs), dtype=float),
    )
    x = np.random.default_rng(0).normal(size=100)
    x[0] = x[1]  # Ties must still receive distinct ordinal positions.
    surrogate = falsification.iaaft(x, seed=42)
    assert surrogate.shape == x.shape
    np.testing.assert_array_equal(np.sort(surrogate), np.sort(x))


def test_te_monte_carlo_pvalue_is_never_zero(monkeypatch):
    values = iter([1.0] + [0.0] * 9)
    monkeypatch.setattr(te, 'transfer_entropy_mi_proxy', lambda *a, **k: next(values))
    result = te.run_transfer_entropy(data(), 'X', 'Y', method='proxy', n_surrogates=9)
    assert result['p_value'] == pytest.approx(0.1)
    assert not result['is_significant']


def test_explicit_mi_proxy_is_reported_in_bits(monkeypatch):
    monkeypatch.setattr(te, 'transfer_entropy_mi_proxy', lambda *a, **k: 1.)
    result = te.run_transfer_entropy(data(), 'X', 'Y', method='proxy', n_surrogates=9)
    assert result['te_bits'] == pytest.approx(np.log2(5))
    assert result['p_value'] == 1.


@pytest.mark.parametrize('invalid', [np.nan, np.inf])
def test_te_rejects_invalid_surrogate_statistics(monkeypatch, invalid):
    values = iter([1.0, invalid] + [0.0] * 8)
    monkeypatch.setattr(te, 'transfer_entropy_mi_proxy', lambda *a, **k: next(values))
    with pytest.raises(ValueError, match='finite'):
        te.run_transfer_entropy(data(), 'X', 'Y', method='proxy', n_surrogates=9)


def test_discrete_te_conditions_on_target_history():
    x = np.tile([0.0, 1.0], 100)
    # The target's past already determines its future; source adds nothing.
    assert te.transfer_entropy_discrete(x, x, delay=1, history_len=1, bins=2) == pytest.approx(0)


def test_cmiknn_history_alignment_and_labels(monkeypatch):
    import tigramite.independence_tests.cmiknn as module
    captured = []
    class FakeCMI:
        def __init__(self, **kwargs): pass
        def get_dependence_measure(self, array, xyz):
            captured.append((array.copy(), xyz.copy()))
            return 0.1
    monkeypatch.setattr(module, 'CMIknn', FakeCMI)
    df = data()
    te.run_transfer_entropy(df, 'X', 'Y', delay=1, history_len=3, n_surrogates=2)
    array, xyz = captured[0]
    np.testing.assert_array_equal(xyz, [0, 1, 2, 2, 2])
    expected = np.vstack([df.Y.values[3:], df.X.values[2:-1], df.Y.values[2:-1], df.Y.values[1:-2], df.Y.values[:-3]])
    np.testing.assert_allclose(array, np.vstack([te.standardize_array(row) for row in expected]))


def test_cmiknn_failure_does_not_mix_estimators(monkeypatch):
    import tigramite.independence_tests.cmiknn as module
    class FakeCMI:
        def __init__(self, **kwargs): pass
        def get_dependence_measure(self, array, xyz):
            raise RuntimeError('estimator failure')
    monkeypatch.setattr(module, 'CMIknn', FakeCMI)
    with pytest.raises(RuntimeError, match='estimator failure'):
        te.run_transfer_entropy(data(), 'X', 'Y', n_surrogates=2)


@pytest.mark.parametrize('original,null,expected', [(1e-8, 0.001, 0.01), (0.01, 1e-8, 1.0)])
def test_falsification_compares_original_with_surrogate_distribution(monkeypatch, original, null, expected):
    values = iter([original] + [null] * 99)
    monkeypatch.setattr(falsification, 'block_permutation', lambda x, **k: x)
    result = falsification.test_edge_falsification(np.arange(50), np.arange(50), 1,
                                                  lambda *a: next(values), n_surrogates=99)
    assert result['empirical_pvalue'] == pytest.approx(expected)
    assert bool(result['passed']) == (expected < 0.05)


def test_failed_surrogate_does_not_produce_a_pass(monkeypatch):
    values = iter([0.001, np.nan] + [0.9] * 98)
    monkeypatch.setattr(falsification, 'block_permutation', lambda x, **k: x)
    result = falsification.test_edge_falsification(np.arange(50), np.arange(50), 1,
                                                  lambda *a: next(values), n_surrogates=99)
    assert result['status'] == 'failed'
    assert not result['passed']
    assert np.isnan(result['empirical_pvalue'])


@pytest.mark.parametrize('votes,passes,stable,oos,tier', [
    (3, 0, True, False, 3), (2, 1, False, False, 2),
    (2, 0, False, True, 2), (3, 2, True, False, 1),
    (1, 2, True, True, 3), (3, 2, np.nan, False, 2),
])
def test_tiers_require_the_documented_evidence(votes, passes, stable, oos, tier):
    assert classify_consensus_tier(votes, passes, icp_stable=stable, oos_significant=oos) == tier


@pytest.mark.parametrize('alternative', ['greater', 'less'])
def test_monte_carlo_ties_are_not_significant(alternative):
    from autocause.core.surrogate_inference import monte_carlo_pvalue
    assert monte_carlo_pvalue(0, np.zeros(99), alternative) == 1
    assert monte_carlo_pvalue(1, np.ones(99), alternative) == 1


def test_falsification_seed_zero_is_reproducible_without_changing_global_rng():
    from experiments.quickstart_surrogate_validation import granger_f_pvalue, simulate
    x, y = simulate(0, .3, .6, n=100)
    np.random.seed(123)
    before = np.random.get_state()
    a = falsification.run_falsification_battery(x, y, 1, granger_f_pvalue,
                                               n_surrogates=9, seed=0, store_surrogates=True)
    after = np.random.get_state()
    b = falsification.run_falsification_battery(x, y, 1, granger_f_pvalue,
                                               n_surrogates=9, seed=0, store_surrogates=True)
    assert a == b
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    # Nine simulations cannot reach the .05 threshold, however strong the link.
    assert all(not result['passed'] for result in a.values())


def test_control_statistic_matches_statsmodels():
    from experiments.quickstart_surrogate_validation import granger_f_pvalue, simulate
    from statsmodels.tsa.stattools import grangercausalitytests
    x, y = simulate(4, .8, .6)
    reference = grangercausalitytests(np.column_stack([y, x]), maxlag=[1])[1][0]['ssr_ftest'][1]
    assert granger_f_pvalue(x, y, 1) == pytest.approx(reference, rel=1e-9, abs=0)


@pytest.mark.parametrize('method', ['block_permutation', 'iaaft'])
@pytest.mark.parametrize('effect', [0., .6])
def test_surrogates_known_null_and_causal_controls(method, effect):
    from experiments.quickstart_surrogate_validation import granger_f_pvalue, simulate
    x, y = simulate(0, .8, effect)
    result = falsification.test_edge_falsification(x, y, 1, granger_f_pvalue,
                                                  n_surrogates=39, seed=42,
                                                  falsification_type=method)
    assert result['status'] == 'ok'
    assert result['passed'] == (effect > 0)
