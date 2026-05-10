"""
CI-test sensitivity analysis for PCMCI+/LPCMCI.

Runs the same causal discovery algorithm with multiple conditional
independence tests and compares the resulting graphs.  This answers:

    "Are the discovered edges robust to the choice of CI test, or
     are they artifacts of a specific test's assumptions?"

Edges that appear across all three tests (ParCorr, RobustParCorr,
CMIknn) are robust.  Edges that appear only with one test may be
driven by that test's specific assumptions (linearity, Gaussianity).

Usage:
    from framework.core.ci_sensitivity import run_ci_sensitivity
    report = run_ci_sensitivity(df, pairs, tau_max=5)
    # report["robust_edges"]  — edges found by all tests
    # report["parcorr_only"]  — edges found only by ParCorr
"""

import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CI_TESTS = ["parcorr", "robust_parcorr", "cmiknn"]
CI_TEST_LABELS = {
    "parcorr": "ParCorr (linear, Gaussian)",
    "robust_parcorr": "RobustParCorr (linear, robust marginals)",
    "cmiknn": "CMIknn (nonparametric, nonlinear)",
}


def run_ci_sensitivity(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    tau_max: int = 5,
    alpha: float = 0.05,
    sampling_days: float = 1.0,
    tests: List[str] = None,
) -> Dict:
    """
    Run PCMCI+ with each CI test and compare results.

    Parameters:
        df: Multivariate time series.
        variable_pairs: Pairs to test.
        tau_max: Maximum lag.
        alpha: Significance threshold.
        sampling_days: Days per timestep.
        tests: List of CI tests to compare (default: all three).

    Returns:
        Dict with keys:
            "per_test": {test_name: DataFrame of results}
            "edge_votes": DataFrame with one row per (source, target) and
                columns for each test (True/False)
            "robust_edges": set of edges found by ALL tests
            "any_edges": set of edges found by ANY test
            "summary": human-readable summary string
    """
    from framework.core.methods.tigramite_pcmci import batch_pcmci

    if tests is None:
        tests = list(CI_TESTS)

    per_test = {}
    edge_sets = {}

    for test_name in tests:
        logger.info(f"CI sensitivity: running PCMCI+ with {test_name}...")
        try:
            result_df = batch_pcmci(
                df,
                variable_pairs,
                tau_max=tau_max,
                test_method=test_name,
                alpha=alpha,
                sampling_days=sampling_days,
            )
            per_test[test_name] = result_df

            sig = (
                result_df[result_df["is_significant"]]
                if "is_significant" in result_df.columns
                else pd.DataFrame()
            )
            edges = set()
            for _, row in sig.iterrows():
                edges.add((row["source"], row["target"]))
            edge_sets[test_name] = edges
        except Exception as e:
            logger.warning(f"CI sensitivity: {test_name} failed: {e}")
            per_test[test_name] = pd.DataFrame()
            edge_sets[test_name] = set()

    # Build vote matrix
    all_edges = set()
    for edges in edge_sets.values():
        all_edges |= edges

    vote_rows = []
    for src, tgt in sorted(all_edges):
        row = {"source": src, "target": tgt}
        for test_name in tests:
            row[test_name] = (src, tgt) in edge_sets[test_name]
        row["n_votes"] = sum(row[t] for t in tests)
        vote_rows.append(row)

    vote_df = pd.DataFrame(vote_rows)
    if not vote_df.empty:
        vote_df = vote_df.sort_values("n_votes", ascending=False)

    # Classify edges
    n_tests = len(tests)
    robust = (
        {
            (r["source"], r["target"])
            for _, r in vote_df.iterrows()
            if r["n_votes"] == n_tests
        }
        if not vote_df.empty
        else set()
    )

    # Summary
    lines = [f"CI-test sensitivity analysis ({n_tests} tests, α={alpha})"]
    lines.append(f"  Total unique edges found: {len(all_edges)}")
    lines.append(f"  Robust (all {n_tests} tests): {len(robust)}")
    for test_name in tests:
        n = len(edge_sets[test_name])
        label = CI_TEST_LABELS.get(test_name, test_name)
        lines.append(f"  {label}: {n} edges")

    if not vote_df.empty:
        lines.append("")
        lines.append("Edge robustness:")
        for _, r in vote_df.iterrows():
            votes = r["n_votes"]
            tag = "ROBUST" if votes == n_tests else f"{votes}/{n_tests}"
            lines.append(f"  {r['source']} → {r['target']}: [{tag}]")

    summary = "\n".join(lines)

    return {
        "per_test": per_test,
        "edge_votes": vote_df,
        "robust_edges": robust,
        "any_edges": all_edges,
        "summary": summary,
    }


def save_ci_sensitivity(report: Dict, output_dir) -> None:
    """Save CI sensitivity report to files."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Vote matrix
    if not report["edge_votes"].empty:
        report["edge_votes"].to_csv(out / "ci_sensitivity_votes.csv", index=False)

    # Per-test results
    for test_name, df in report["per_test"].items():
        if not df.empty:
            df.to_csv(out / f"ci_sensitivity_{test_name}.csv", index=False)

    # Summary
    with open(out / "ci_sensitivity_summary.txt", "w") as f:
        f.write(report["summary"])

    logger.info(f"CI sensitivity report saved to {out}")
