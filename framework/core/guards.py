"""
Guards and diagnostics enforcing framework invariants.
"""

from typing import Optional
import hashlib
import pandas as pd


def assert_lag_days_consistency(
    df: pd.DataFrame, steps_col: str, days_col: str, sampling_days: int
) -> None:
    if steps_col in df.columns and days_col in df.columns:
        incons = df[
            (df[steps_col].notna())
            & (df[days_col].notna())
            & (df[days_col] != df[steps_col] * sampling_days)
        ]
        if len(incons) > 0:
            raise AssertionError(
                f"lag-day mismatch in {len(incons)} rows: expected {days_col} == {steps_col} * {sampling_days}"
            )


def assert_q_ge_p(
    df: pd.DataFrame, p_col: str = "p_value", q_col: str = "q_value"
) -> None:
    if p_col in df.columns and q_col in df.columns:
        bad = df[(df[p_col].notna()) & (df[q_col].notna()) & (df[q_col] < df[p_col])]
        if len(bad) > 0:
            raise AssertionError("Found q-values smaller than p-values (illegal).")


def hash_edge(exp: str, method: str, source: str, target: str) -> str:
    key = f"{exp}|{method}|{source}|{target}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def check_cross_experiment_collisions(
    df: pd.DataFrame, exp_col: str = "experiment_name"
) -> None:
    required = [exp_col, "method", "source", "target"]
    if not all(c in df.columns for c in required):
        return
    hashes = df.apply(
        lambda r: hash_edge(
            str(r[exp_col]), str(r["method"]), str(r["source"]), str(r["target"])
        ),
        axis=1,
    )
    if hashes.duplicated().any():
        raise AssertionError(
            "Cross-experiment hash collision detected (potential overwrite)."
        )


def alert_if_te_granger_p_correlated(
    df: pd.DataFrame, threshold: float = 0.8
) -> Optional[float]:
    # Requires per-edge rows for both methods with same (source,target)
    if df.empty or "method" not in df.columns:
        return None
    pvt = df.pivot_table(
        index=["source", "target"], columns="method", values="p_value", aggfunc="min"
    )
    if {"Granger", "TransferEntropy"}.issubset(set(pvt.columns)):
        sub = pvt.dropna(subset=["Granger", "TransferEntropy"])  # type: ignore[index]
        if len(sub) >= 3:
            corr = sub["Granger"].corr(sub["TransferEntropy"])  # type: ignore[index]
            if corr is not None and corr > threshold:
                raise AssertionError(
                    f"TE and Granger p-values highly correlated (r={corr:.2f}). Check pipeline independence."
                )
            return float(corr) if corr is not None else None
    return None
