"""Tests for SHD (Structural Hamming Distance) metric."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.core.graph_metrics import binary_metrics, binary_metrics_undirected


def test_perfect_recovery():
    """SHD=0 when discovered matches true exactly."""
    true = {("X", "Y"), ("Z", "Y"), ("X", "Z")}
    discovered = {("X", "Y"), ("Z", "Y"), ("X", "Z")}
    m = binary_metrics(discovered, true)
    assert m["shd"] == 0
    assert m["f1"] == 1.0
    assert m["n_reversals"] == 0
    print(f"  PASS: Perfect recovery → SHD={m['shd']}, F1={m['f1']}")


def test_one_missing_edge():
    """SHD=1 for one missing edge (deletion)."""
    true = {("X", "Y"), ("Z", "Y")}
    discovered = {("X", "Y")}
    m = binary_metrics(discovered, true)
    assert m["shd"] == 1  # 1 deletion
    assert m["fn"] == 1
    assert m["fp"] == 0
    assert m["n_reversals"] == 0
    print(f"  PASS: One missing → SHD={m['shd']}, fn={m['fn']}")


def test_one_extra_edge():
    """SHD=1 for one spurious edge (addition)."""
    true = {("X", "Y")}
    discovered = {("X", "Y"), ("Z", "Y")}
    m = binary_metrics(discovered, true)
    assert m["shd"] == 1  # 1 addition
    assert m["fp"] == 1
    assert m["fn"] == 0
    assert m["n_reversals"] == 0
    print(f"  PASS: One extra → SHD={m['shd']}, fp={m['fp']}")


def test_one_reversal():
    """SHD=1 for a reversed edge (not 2)."""
    true = {("X", "Y"), ("Z", "Y")}
    discovered = {("Y", "X"), ("Z", "Y")}  # X→Y reversed to Y→X
    m = binary_metrics(discovered, true)
    # Y→X is FP (not in true), X→Y is FN (not in discovered)
    # But they form a reversal pair, so SHD = 1 (not 2)
    assert m["shd"] == 1
    assert m["n_reversals"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    print(f"  PASS: One reversal → SHD={m['shd']}, reversals={m['n_reversals']}")


def test_two_reversals():
    """SHD=2 for two reversed edges."""
    true = {("X", "Y"), ("Z", "W")}
    discovered = {("Y", "X"), ("W", "Z")}  # both reversed
    m = binary_metrics(discovered, true)
    assert m["shd"] == 2
    assert m["n_reversals"] == 2
    print(f"  PASS: Two reversals → SHD={m['shd']}, reversals={m['n_reversals']}")


def test_mixed_errors():
    """SHD counts additions + deletions + reversals correctly."""
    true = {("A", "B"), ("C", "D"), ("E", "F")}
    discovered = {("B", "A"), ("C", "D"), ("G", "H")}
    # A→B reversed to B→A: 1 reversal
    # E→F missing: 1 deletion
    # G→H spurious: 1 addition
    # SHD = 1 (reversal) + 1 (deletion) + 1 (addition) = 3
    m = binary_metrics(discovered, true)
    assert m["shd"] == 3, f"Expected 3, got {m['shd']}"
    assert m["n_reversals"] == 1
    assert m["tp"] == 1  # C→D correct
    print(
        f"  PASS: Mixed errors → SHD={m['shd']}, rev={m['n_reversals']}, tp={m['tp']}"
    )


def test_empty_discovered():
    """SHD = number of true edges when nothing is discovered."""
    true = {("X", "Y"), ("Z", "Y"), ("X", "Z")}
    discovered = set()
    m = binary_metrics(discovered, true)
    assert m["shd"] == 3  # 3 deletions
    assert m["fn"] == 3
    assert m["f1"] == 0.0
    print(f"  PASS: Empty discovered → SHD={m['shd']}")


def test_empty_true():
    """SHD = number of discovered edges when true graph is empty."""
    true = set()
    discovered = {("X", "Y"), ("Z", "Y")}
    m = binary_metrics(discovered, true)
    assert m["shd"] == 2  # 2 additions
    assert m["fp"] == 2
    print(f"  PASS: Empty true → SHD={m['shd']}")


def test_both_empty():
    """SHD=0 when both are empty."""
    m = binary_metrics(set(), set())
    assert m["shd"] == 0
    assert m["f1"] == 0.0
    print(f"  PASS: Both empty → SHD={m['shd']}")


def test_timegraph_a1_perfect():
    """Simulate TimeGraph A1 perfect recovery (paper: TPR=1.0, FDR=0.0, SHD=0)."""
    # TimeGraph 4-var graph: X1→X4, X4→X3, X3→X2, X2→X1
    true = {("X1", "X4"), ("X4", "X3"), ("X3", "X2"), ("X2", "X1")}
    discovered = {("X1", "X4"), ("X4", "X3"), ("X3", "X2"), ("X2", "X1")}
    m = binary_metrics(discovered, true)
    assert m["shd"] == 0
    assert m["recall"] == 1.0  # TPR
    assert m["precision"] == 1.0  # 1 - FDR
    print(
        f"  PASS: TimeGraph A1 perfect → SHD={m['shd']}, TPR={m['recall']}, FDR={1 - m['precision']}"
    )


def test_timegraph_a1_paper_lpcmci():
    """Simulate TimeGraph A1 LPCMCI result (paper: TPR=0.78, FDR=0.13, SHD=3)."""
    # 4 true edges, TPR=0.78 → ~3 TP, 1 FN
    # FDR=0.13 → ~0.5 FP (round to 0 or 1)
    # Paper reports SHD=3 for LPCMCI on A1 Gaussian
    # With 4 true edges: 3 TP + 1 FN + some FP/reversals → SHD=3
    # Possible: 3 correct, 1 missed, 2 extra → SHD = 2 + 1 = 3
    true = {("X1", "X4"), ("X4", "X3"), ("X3", "X2"), ("X2", "X1")}
    # 3 correct, miss X2→X1, add 2 spurious (to get FDR ≈ 0.13 is hard with integers)
    # Actually with TPR=7/9≈0.78 the paper uses 9 true edges (not 4)
    # For our 4-edge graph: TPR=3/4=0.75, FDR=0 → SHD=1
    discovered = {("X1", "X4"), ("X4", "X3"), ("X3", "X2")}  # miss X2→X1
    m = binary_metrics(discovered, true)
    assert m["shd"] == 1  # 1 deletion
    assert m["recall"] == 0.75
    print(f"  PASS: Simulated LPCMCI → SHD={m['shd']}, TPR={m['recall']:.2f}")


def test_shd_vs_naive_with_reversals():
    """SHD with reversals is strictly less than naive fp+fn."""
    true = {("A", "B"), ("C", "D")}
    discovered = {("B", "A"), ("D", "C")}  # both reversed
    m = binary_metrics(discovered, true)
    naive = m["fp"] + m["fn"]  # would be 4 without reversal detection
    assert m["shd"] == 2  # 2 reversals
    assert m["shd"] < naive  # 2 < 4
    print(f"  PASS: SHD={m['shd']} < naive(fp+fn)={naive}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SHD metric")
    print("=" * 60)

    tests = [
        test_perfect_recovery,
        test_one_missing_edge,
        test_one_extra_edge,
        test_one_reversal,
        test_two_reversals,
        test_mixed_errors,
        test_empty_discovered,
        test_empty_true,
        test_both_empty,
        test_timegraph_a1_perfect,
        test_timegraph_a1_paper_lpcmci,
        test_shd_vs_naive_with_reversals,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed}/{passed + failed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
