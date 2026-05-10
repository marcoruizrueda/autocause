"""
Graph recovery metrics for causal discovery evaluation.

Provides binary (F1, precision, recall, SHD) and ranking (AUROC, AUPRC)
metrics following the evaluation protocols of:
  - TimeGraph (Ferdous et al. 2025, KDD): TPR, FDR, SHD
  - CausalTime (Cheng et al. 2023, NeurIPS): AUROC, AUPRC

Usage inside the workflow (when ground truth is available):

    from framework.core.graph_metrics import evaluate_graph_recovery
    metrics = evaluate_graph_recovery(
        results_dict=results,
        true_edges={("X", "Y"), ("Z", "Y")},
        var_names=["X", "Y", "Z"],
    )
    # metrics["granger"]["f1"], metrics["granger"]["auroc"], ...

Usage standalone:

    from framework.core.graph_metrics import binary_metrics, auroc_metrics
    m = binary_metrics(discovered_edges, true_edges)
    a = auroc_metrics(score_matrix, true_adj, var_names)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column conventions per method (source, target, significance, score)
METHOD_COLUMNS = {
    "granger": {
        "src": "cause",
        "tgt": "effect",
        "sig": "significant",
        "score": "best_p_value",
        "score_type": "pvalue",
    },
    "transfer_entropy": {
        "src": "source",
        "tgt": "target",
        "sig": "significant",
        "score": "p_value",
        "score_type": "pvalue",
    },
    "pcmci": {
        "src": "source",
        "tgt": "target",
        "sig": "is_significant",
        "score": "best_p_value",
        "score_type": "pvalue",
    },
    "varlingam": {
        "src": "source",
        "tgt": "target",
        "sig": "is_significant",
        "score": "abs_coefficient",
        "score_type": "coefficient",
    },
    "lpcmci": {
        "src": "source",
        "tgt": "target",
        "sig": "is_significant",
        "score": "best_p_value",
        "score_type": "pvalue",
    },
    "predictive_baseline": {
        "src": "source",
        "tgt": "target",
        "sig": "is_significant",
        "score": "importance",
        "score_type": "coefficient",
    },
}


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


def binary_metrics(
    discovered: Set[Tuple[str, str]],
    true_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute precision, recall, F1 from directed edge sets."""
    tp = len(true_edges & discovered)
    fp = len(discovered - true_edges)
    fn = len(true_edges - discovered)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    shd = fp + fn  # structural Hamming distance (no reversal penalty)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "shd": shd,
    }


def binary_metrics_undirected(
    discovered: Set[Tuple[str, str]],
    true_edges: Set[Tuple[str, str]],
) -> Dict[str, float]:
    """Compute metrics on undirected (frozenset) pairs."""
    disc_u = {frozenset(e) for e in discovered}
    true_u = {frozenset(e) for e in true_edges}
    tp = len(true_u & disc_u)
    fp = len(disc_u - true_u)
    fn = len(true_u - disc_u)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "shd": fp + fn,
    }


# ---------------------------------------------------------------------------
# AUROC / AUPRC metrics
# ---------------------------------------------------------------------------


def auroc_metrics(
    results_df: pd.DataFrame,
    true_edges: Set[Tuple[str, str]],
    var_names: List[str],
    src_col: str = "source",
    tgt_col: str = "target",
    score_col: str = "p_value",
    score_type: str = "pvalue",
) -> Dict[str, float]:
    """Compute AUROC and AUPRC from continuous method scores.

    Builds an N*(N-1) vector of scores (one per ordered pair excluding
    self-loops) and compares against the binary ground truth.

    Parameters:
        results_df: Method output with source, target, and a score column.
        true_edges: Set of (source, target) directed ground truth edges.
        var_names: Ordered list of variable names.
        src_col, tgt_col: Column names for source/target.
        score_col: Column with continuous confidence (p-value or coefficient).
        score_type: "pvalue" (lower = more confident) or "coefficient"
                    (higher = more confident).

    Returns:
        {"auroc": float, "auprc": float}
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except ImportError:
        logger.warning(
            "scikit-learn required for AUROC/AUPRC. pip install scikit-learn"
        )
        return {"auroc": float("nan"), "auprc": float("nan")}

    n = len(var_names)
    if n < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}

    var_idx = {v: i for i, v in enumerate(var_names)}

    # Ordered pair list (excluding diagonal)
    pair_list = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_pairs = len(pair_list)

    # Binary ground truth
    y_true = np.zeros(n_pairs, dtype=int)
    for k, (i, j) in enumerate(pair_list):
        if (var_names[j], var_names[i]) in true_edges:
            y_true[k] = 1

    # Build score vector
    y_score = np.zeros(n_pairs)

    if results_df is not None and len(results_df) > 0:
        for _, row in results_df.iterrows():
            s, t = row.get(src_col), row.get(tgt_col)
            if s not in var_idx or t not in var_idx:
                continue
            si, ti = var_idx[s], var_idx[t]
            if si == ti:
                continue
            try:
                k = pair_list.index((ti, si))
            except ValueError:
                continue

            raw = row.get(score_col)
            if pd.isna(raw):
                continue
            raw = float(raw)

            if score_type == "pvalue":
                score = 1.0 - raw  # lower p → higher confidence
            else:
                score = abs(raw)

            y_score[k] = max(y_score[k], score)

    # Edge case: all true or all false
    if y_true.sum() == 0 or y_true.sum() == n_pairs:
        return {"auroc": float("nan"), "auprc": float("nan")}

    try:
        auroc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, y_score))
    except ValueError:
        auprc = float(y_true.mean())

    return {"auroc": auroc, "auprc": auprc}


# ---------------------------------------------------------------------------
# High-level: evaluate all methods at once
# ---------------------------------------------------------------------------


def evaluate_graph_recovery(
    results_dict: Dict[str, pd.DataFrame],
    true_edges: Set[Tuple[str, str]],
    var_names: List[str],
    undirected: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all methods in results_dict against ground truth.

    Parameters:
        results_dict: {"granger": df, "pcmci": df, ...}
        true_edges: Set of (source, target) directed edges.
        var_names: Variable names for AUROC matrix construction.
        undirected: If True, evaluate on undirected pairs (for benchmarks
                    where direction is ambiguous, e.g. symmetric graphs).

    Returns:
        {"granger": {"f1": ..., "auroc": ..., ...}, ...}
    """
    out = {}
    for method, df in results_dict.items():
        if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
            out[method] = {
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": len(true_edges),
                "shd": len(true_edges),
                "auroc": float("nan"),
                "auprc": float("nan"),
            }
            continue

        cols = METHOD_COLUMNS.get(method, {})
        src = cols.get("src", "source")
        tgt = cols.get("tgt", "target")
        sig = cols.get("sig", "significant")
        score = cols.get("score", "p_value")
        stype = cols.get("score_type", "pvalue")

        # Binary metrics
        if sig in df.columns and src in df.columns and tgt in df.columns:
            sig_rows = df[df[sig]]
            discovered = set(zip(sig_rows[src], sig_rows[tgt]))
        else:
            discovered = set()

        if undirected:
            bm = binary_metrics_undirected(discovered, true_edges)
        else:
            bm = binary_metrics(discovered, true_edges)

        # AUROC/AUPRC
        am = auroc_metrics(
            df,
            true_edges,
            var_names,
            src_col=src,
            tgt_col=tgt,
            score_col=score,
            score_type=stype,
        )

        out[method] = {**bm, **am}

    return out


def save_evaluation(
    metrics: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
) -> pd.DataFrame:
    """Save evaluation metrics to CSV and return as DataFrame."""
    rows = []
    for method, m in metrics.items():
        rows.append({"method": method, **m})
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Graph recovery metrics saved: {output_path}")
    return df
