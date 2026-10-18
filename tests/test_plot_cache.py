"""Cache regressions exercised with the actual consensus return schema."""

import json

import pandas as pd

from autocause.core.methods.consensus import merge_method_results
from autocause.core.plot_cache import PlotCacheManager


def test_consensus_cache_round_trip(tmp_path):
    granger = pd.DataFrame([
        {"source": "X", "target": "Y", "lag": 1, "p_value": 0.001}
    ])
    result = merge_method_results(granger_df=granger)
    cache = PlotCacheManager(tmp_path)
    edges = result["consensus_edges"]
    cache.save_consensus_results(edges, result)
    pd.testing.assert_frame_equal(cache.load_consensus_results(), edges)
    metadata = json.loads(cache.consensus_metadata_path.read_text())
    assert metadata["n_edges"] == len(edges)
    assert metadata["info"]["n_consensus_edges"] == len(edges)
    assert metadata["info"]["voting_matrix"] == result["voting_matrix"]
    # The frame is already stored losslessly in the pickle, not JSON metadata.
    assert "consensus_edges" not in metadata["info"]
    assert result["consensus_edges"] is edges


def test_workflow_refreshes_method_cache(monkeypatch, tmp_path):
    import numpy as np
    from autocause.core.methods import visualize_results
    from autocause.core.run_workflow import run_causal_discovery_workflow

    cache = PlotCacheManager(tmp_path)
    cache.save_method_results({"granger": pd.DataFrame({"stale": [True]})})
    monkeypatch.setattr(visualize_results, "visualize_all_results", lambda *args: None)
    rng = np.random.default_rng(0)
    x = rng.normal(size=150)
    y = np.r_[0, x[:-1]] + rng.normal(scale=0.1, size=150)
    run_causal_discovery_workflow(
        data_df=pd.DataFrame({"X": x, "Y": y}), output_dir=tmp_path,
        pairs=[("X", "Y")], tau_max=1,
        enable_preprocessing=False, enable_distribution_tests=False,
        enable_strength_analysis=False, enable_temporal_validation=False,
        enable_tracking=False, enable_consensus=False,
        method_config={"granger": {"enabled": True},
                       "pcmci": {"enabled": False},
                       "transfer_entropy": {"enabled": False}},
    )
    current = cache.load_method_results()["granger"]
    assert "stale" not in current.columns
    assert current.iloc[0]["best_p_value"] < 0.001
