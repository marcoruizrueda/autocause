"""
Framework Results Visualization

Generates standard figures for causal discovery results.

Output structure (flat, researcher-friendly):
    figures/
    ├── per_method/           # one graph + p-values + lags per method
    ├── comparison/           # cross-method panels
    └── diagnostics/          # FDR, DAG, lag analysis
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from framework.plots import (
    plot_pvalue_distribution,
    plot_pvalue_comparison,
    plot_lag_histogram,
    plot_lag_distribution,
    plot_causal_graph,
    plot_method_comparison,
    plot_europe_map,
    plot_transfer_entropy_flow,
    plot_conditional_independence_matrix,
    plot_lag_distribution_analysis,
    plot_fdr_diagnostics,
    plot_dag_learning_diagnostics,
)

logger = logging.getLogger(__name__)


def visualize_all_results(results_dict, output_dir):
    """
    Generate standard causal discovery figures.

    Output:
        figures/
        ├── per_method/
        │   ├── granger_graph.png
        │   ├── granger_pvalues.svg
        │   ├── te_information_flow.png
        │   └── pcmci_ci_matrix.png
        ├── comparison/
        │   ├── method_comparison.svg
        │   ├── pvalue_comparison.svg
        │   └── lag_distribution.svg
        └── diagnostics/
            ├── lag_analysis.png
            ├── granger_fdr.png
            └── granger_dag.png
    """
    from framework.core.plot_cache import PlotCacheManager

    figures_dir = Path(output_dir) / "figures"
    per_method = figures_dir / "per_method"
    comparison = figures_dir / "comparison"
    diagnostics = figures_dir / "diagnostics"
    for d in [per_method, comparison, diagnostics]:
        d.mkdir(parents=True, exist_ok=True)

    cache_mgr = PlotCacheManager(output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FIGURES")
    logger.info("=" * 70)

    for method, df in results_dict.items():
        if df is None or len(df) == 0:
            continue
        logger.info(f"  {method}...")
        df_norm = _normalize_columns(df)
        cache_mgr.cache_plot_intermediate_data("normalized", df_norm, method=method)

        # P-value distribution
        pval_col = next(
            (c for c in ["p_value", "best_p_value", "q_value"] if c in df.columns), None
        )
        if pval_col:
            plot_df = df.copy()
            plot_df["p_value"] = plot_df[pval_col]
            try:
                plot_pvalue_distribution(
                    plot_df, method=method, output_path=per_method / f"{method}_pvalues"
                )
            except Exception:
                pass

        # Lag histogram
        lag_col = next(
            (c for c in ["delay", "lag", "best_lag"] if c in df.columns), None
        )
        if lag_col:
            try:
                plot_lag_histogram(
                    df,
                    method=method,
                    lag_column=lag_col,
                    output_path=per_method / f"{method}_lags",
                )
            except Exception:
                pass

        # Causal graph
        try:
            plot_causal_graph(
                df_norm, method=method, output_path=per_method / f"{method}_graph"
            )
        except Exception:
            pass

        # FDR diagnostics
        if "p_value" in df_norm.columns:
            p_vals = df_norm["p_value"].dropna().values
            p_vals = p_vals[(p_vals >= 0) & (p_vals <= 1)]
            if len(p_vals) > 0:
                try:
                    from scipy.stats import false_discovery_control

                    q_vals = false_discovery_control(p_vals, method="bh")
                    plot_fdr_diagnostics(
                        p_vals,
                        q_vals,
                        alpha=0.05,
                        output_path=diagnostics / f"{method}_fdr.png",
                    )
                except Exception:
                    pass

        # DAG diagnostics
        if len(df_norm) > 0:
            try:
                plot_dag_learning_diagnostics(
                    df_norm, output_path=diagnostics / f"{method}_dag.png"
                )
            except Exception:
                pass

    # TE information flow
    if "transfer_entropy" in results_dict:
        te_df = _normalize_columns(results_dict["transfer_entropy"])
        if len(te_df) > 0:
            try:
                plot_transfer_entropy_flow(
                    te_df, output_path=per_method / "te_information_flow.png"
                )
            except Exception:
                pass

    # PCMCI conditional independence matrix
    if "pcmci" in results_dict:
        pcmci_df = _normalize_columns(results_dict["pcmci"])
        if len(pcmci_df) > 0:
            try:
                plot_conditional_independence_matrix(
                    pcmci_df, output_path=per_method / "pcmci_ci_matrix.png"
                )
            except Exception:
                pass

    # Cross-method comparison
    try:
        plot_lag_distribution(results_dict, output_path=comparison / "lag_distribution")
    except Exception:
        pass
    try:
        plot_pvalue_comparison(
            results_dict, output_path=comparison / "pvalue_comparison"
        )
    except Exception:
        pass
    try:
        plot_method_comparison(
            results_dict, output_path=comparison / "method_comparison"
        )
    except Exception:
        pass

    # Comprehensive lag analysis
    all_norm = []
    for method, df in results_dict.items():
        df_n = _normalize_columns(df)
        if len(df_n) > 0 and "lag_steps" in df_n.columns:
            df_n["method"] = method
            all_norm.append(df_n)
    if all_norm:
        combined = pd.concat(all_norm, ignore_index=True)
        cache_mgr.cache_plot_intermediate_data("lag_distribution_combined", combined)
        try:
            plot_lag_distribution_analysis(
                combined, output_path=diagnostics / "lag_analysis.png"
            )
        except Exception:
            pass

    logger.info(f"  Figures saved to {figures_dir}")
    return figures_dir


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for cross-method compatibility."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
    rename_map = {}

    # Source/target normalization
    if "cause" in df.columns and "source" not in df.columns:
        rename_map["cause"] = "source"
    if "effect" in df.columns and "target" not in df.columns:
        rename_map["effect"] = "target"

    # Lag normalization
    if "best_lag" in df.columns and "lag_steps" not in df.columns:
        rename_map["best_lag"] = "lag_steps"
    elif "delay" in df.columns and "lag_steps" not in df.columns:
        rename_map["delay"] = "lag_steps"
    elif "lag" in df.columns and "lag_steps" not in df.columns:
        rename_map["lag"] = "lag_steps"

    # P-value normalization
    if "best_p_value" in df.columns and "p_value" not in df.columns:
        rename_map["best_p_value"] = "p_value"

    # Significance normalization
    if "is_significant" in df.columns and "significant" not in df.columns:
        rename_map["is_significant"] = "significant"
    if "is_causal" in df.columns and "significant" not in df.columns:
        rename_map["is_causal"] = "significant"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df
