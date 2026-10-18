"""Cross-check the completed quickstart and walkthrough against their exports.

Run after Restart Kernel and Run All:
    .venv/bin/python experiments/verify_quickstart_outputs.py
"""
import base64
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'examples/walkthrough_output'
    notebooks = [nbformat.read(root / f'examples/{name}.ipynb', 4)
                 for name in ['quickstart', 'walkthrough']]
    cells = []
    for nb in notebooks:
        notebook_cells = [c for c in nb.cells if c.cell_type == 'code']
        assert [c.execution_count for c in notebook_cells] == list(range(1, len(notebook_cells)+1))
        assert not any(o.output_type == 'error' for c in notebook_cells for o in c.outputs)
        cells.extend(notebook_cells)

    truth = {('X0', 'X1'), ('X1', 'X2'), ('X0', 'X3'), ('X3', 'X4')}
    truth_skeleton = {frozenset(edge) for edge in truth}
    metrics = pd.read_csv(output / 'graph_recovery_metrics.csv').set_index('method')
    raw = {}
    for path in (output / 'method').glob('*/1-raw/results_*.csv'):
        method = path.parents[1].name
        df = pd.read_csv(path)
        raw[method] = df
        source, target = ('cause', 'effect') if method == 'granger' else ('source', 'target')
        flag = 'is_significant' if 'is_significant' in df else 'significant'
        assert df[flag].notna().all()
        selected = df.loc[df[flag]]
        skeleton = {frozenset(edge) for edge in zip(selected[source], selected[target])}
        tp, fp, fn = len(skeleton & truth_skeleton), len(skeleton - truth_skeleton), len(truth_skeleton - skeleton)
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        np.testing.assert_allclose(metrics.loc[method, ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']].astype(float),
                                   [tp, fp, fn, precision, recall, f1])
    te = raw['transfer_entropy']
    assert len(te) == 20 and np.isfinite(te.p_value).all()
    assert te.p_value.between(te.p_value_resolution-1e-14, 1).all()
    np.testing.assert_allclose(te.p_value_resolution, 1/(te.n_surrogates+1))
    assert (te.q_value >= te.p_value-1e-14).all()

    tiers = pd.read_csv(output / 'consensus/5-tiers/consensus_with_tiers.csv')
    falsification = pd.read_csv(output / 'falsification_results.csv')
    assert len(tiers) == len(falsification)
    for prefix in ['block_perm', 'iaaft']:
        assert falsification[f'{prefix}_status'].eq('ok').all()
        empirical = falsification[f'{prefix}_empirical_pvalue']
        resolution = falsification[f'{prefix}_p_value_resolution']
        assert empirical.between(resolution-1e-14, 1).all()
        expected = (empirical < .05) & (falsification[f'{prefix}_original_pvalue'] < .05)
        np.testing.assert_array_equal(falsification[f'{prefix}_passed'], expected)
    np.testing.assert_array_equal(tiers.falsification_passed,
                                  tiers.block_perm_passed.astype(int)+tiers.iaaft_passed.astype(int))
    for row in tiers.itertuples():
        expected = 3
        if row.vote_count >= 2:
            if row.falsification_passed >= 2 and row.icp_stable:
                expected = 1
            elif row.falsification_passed >= 1 or row.oos_significant:
                expected = 2
        assert row.tier == expected
        assert row.vote_count == len(row.agreeing_methods.split(','))
    recovered = {(r.source, r.target, int(r.lag_steps)) for r in tiers.itertuples()}
    truth_lagged = {(s, t, 1) for s, t in truth}
    graph_counts = (len(recovered & truth_lagged), len(recovered-truth_lagged), len(truth_lagged-recovered))

    def cell_with(fragment):
        matches = [c for c in cells if fragment in c.source]
        assert len(matches) == 1, f'Expected one code cell containing {fragment!r}'
        return matches[0]

    def text_output(fragment):
        return ''.join(o.get('text', '') for o in cell_with(fragment).outputs)

    assert 'Recommended: PCMCI+' in text_output('audit_dir =')
    assert 'success=True' in text_output('run_log =')
    assert 'success=True' in text_output('easy_result =')
    assert f'TP={graph_counts[0]}, FP={graph_counts[1]}, FN={graph_counts[2]}' in text_output('import networkx as nx')
    columns = ['method', 'precision', 'recall', 'f1']
    table = metrics.reset_index()
    table = table[table.method.isin(raw)][columns].to_string(index=False, float_format=lambda x: f'{x:.3f}')
    assert table in text_output('enabled_methods =')
    tier_columns = ['source', 'target', 'lag_steps', 'vote_count', 'paradigm_diversity',
                    'falsification_passed', 'block_perm_empirical_pvalue',
                    'iaaft_empirical_pvalue', 'icp_stable', 'tier']
    display_columns = {
        'source': 'source', 'target': 'target', 'lag_steps': 'lag',
        'vote_count': 'votes', 'falsification_passed': 'checks',
        'block_perm_empirical_pvalue': 'block p', 'iaaft_empirical_pvalue': 'IAAFT p',
        'icp_stable': 'stable', 'tier': 'tier',
    }
    displayed_tiers = tiers.sort_values('tier')[list(display_columns)].rename(columns=display_columns)
    assert displayed_tiers.to_string(index=False, float_format=lambda x: f'{x:.3g}') in text_output('tiers_csv =')

    # The short path uses the same causal methods; adding noncausal baselines
    # in the detailed path must not change consensus votes or evidence grades.
    easy = root / 'examples/quickstart_output/easy_run'
    easy_tiers = pd.read_csv(easy / 'consensus/5-tiers/consensus_with_tiers.csv')
    sort_keys = ['source', 'target', 'lag_steps']
    pd.testing.assert_frame_equal(
        tiers[tier_columns].sort_values(sort_keys).reset_index(drop=True),
        easy_tiers[tier_columns].sort_values(sort_keys).reset_index(drop=True),
    )
    easy_falsification = pd.read_csv(easy / 'falsification_results.csv')
    for prefix in ['block_perm', 'iaaft']:
        assert easy_falsification[f'{prefix}_status'].eq('ok').all()
        empirical = easy_falsification[f'{prefix}_empirical_pvalue']
        resolution = easy_falsification[f'{prefix}_p_value_resolution']
        assert empirical.between(resolution-1e-14, 1).all()
        expected = (empirical < .05) & (easy_falsification[f'{prefix}_original_pvalue'] < .05)
        np.testing.assert_array_equal(easy_falsification[f'{prefix}_passed'], expected)

    # Locate cells by their content so teaching-section reordering cannot silently
    # invalidate output checks or require maintaining positional cell numbers.
    figures = {
        '"fig_raw_series.png"': output / 'fig_raw_series.png',
        'summary_png =': output / 'causal_audit_standalone/figures/prediscovery_summary.png',
        '"fig_preprocess_x0.png"': output / 'fig_preprocess_x0.png',
        '"fig_consensus_graph.png"': output / 'fig_consensus_graph.png',
        '"fig_method_precision.png"': output / 'fig_method_precision.png',
        'easy_policy =': easy / 'causal_audit/figures/prediscovery_summary.png',
        'easy_method_figure =': easy / 'figures/per_method/pcmci_ci_matrix.png',
    }
    for fragment, filename in figures.items():
        embedded = [o.data['image/png'] for o in cell_with(fragment).outputs if 'image/png' in o.get('data', {})]
        assert len(embedded) == 1
        assert base64.b64decode(embedded[0]) == filename.read_bytes()
    for path in output.glob('*.json'):
        json.loads(path.read_text())
    print(f'Checked {len(notebooks)} notebooks, {len(cells)} executed cells, {len(raw)} method tables, '
          f'{len(figures)} embedded figures, surrogate probabilities and tier rules.')
    print(f'Directed + exact-lag counts: TP={graph_counts[0]}, FP={graph_counts[1]}, FN={graph_counts[2]}')


if __name__ == '__main__':
    main()
