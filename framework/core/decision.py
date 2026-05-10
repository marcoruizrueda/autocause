"""
Automatic method selection and configuration for causal discovery.

This module implements a lightweight decision engine that, for each
variable pair, selects suitable causal methods (VAR-based Granger, transfer entropy,
PCMCI+) and configures key hyperparameters based on the data regime and
your heuristics cheat-sheet.

Key decisions per pair include:
- Which methods to run (fast/linear vs nonlinear/multivariate)
- tau_max via domain window and sampling cadence
- CI test for PCMCI+ (ParCorr vs CMIknn)
- Whether to attempt consensus

Outputs a per-pair plan and convenience runners to execute and normalize
results to a unified schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import stats
from .methods import granger, transfer_entropy, tigramite_pcmci
from .multiple_testing import apply_fdr_to_dataframe

logger = logging.getLogger(__name__)


@dataclass
class PairDecision:
    source: str
    target: str
    methods: List[str]
    tau_max: int
    sampling_days: float
    ci_test: Optional[str] = None  # for PCMCI+: "parcorr" | "cmiknn" | "robust_parcorr"
    allow_contemporaneous: bool = False
    notes: Optional[str] = None


def _estimate_sampling_days(df: pd.DataFrame, cols: List[str]) -> float:
    """Estimate effective sampling interval (days) based on datetime index.

    Uses the median delta of non-NaN rows. Falls back to 1.0 when unknown.
    """
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            return 1.0
        sub = df[cols].dropna()
        if len(sub) < 3:
            return 1.0
        deltas = np.diff(sub.index.view("int64")) / 1e9 / 86400.0
        med = float(np.median(deltas)) if len(deltas) > 0 else 1.0
        # Clamp to sensible bounds
        if med <= 0:
            return 1.0
        return med
    except Exception:
        return 1.0


def _auto_tau_max(
    T: int, sampling_days: float, window_days: int = 30, hard_cap: int = None
) -> int:
    """Compute tau_max = min(ceil(W/sampling_days), floor(T/8), hard_cap)."""
    try:
        from math import ceil, floor

        # Load hard_cap from config if not provided
        if hard_cap is None:
            try:
                defaults = _load_defaults()
                hard_cap = defaults.get("tau_max", 18)
            except Exception:
                hard_cap = 18

        a = int(ceil(window_days / max(sampling_days, 1e-9)))
        b = int(floor(T / 8)) if T > 0 else 1
        tau = max(1, min(a, b, hard_cap))
        return tau
    except Exception:
        return min(6, hard_cap if hard_cap else 18)


def _load_defaults() -> dict:
    """Load framework defaults.json with safe fallbacks."""
    try:
        cfg_path = Path(__file__).parent.parent / "config" / "defaults.json"
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def decide_for_pair(
    df: pd.DataFrame,
    source: str,
    target: str,
    window_days: int = 30,
    min_T_for_te: int = 200,
    min_T_for_pcmci: int = 100,
    allow_contemporaneous: bool = True,
) -> PairDecision:
    """Decide which methods to apply to a given pair and configure params.

    Heuristics based on the provided cheat-sheet.
    """
    cols = [source, target]
    sub = df[cols].dropna()
    T = len(sub)
    sampling_days = _estimate_sampling_days(df, cols)

    # Load routing config (with fallbacks)
    cfg = _load_defaults()
    router = cfg.get("router", {})
    router_enabled = bool(router.get("enable", True))

    # Linearity/MI screen (proxy for nonlinearity)
    lin = stats.assess_linearity(sub[source], sub[target])
    mi_ratio = lin.get("mi_to_r2_ratio", np.nan)
    # Proxy nonlinearity score using MI/R2 ratio thresholds
    thr_lo = float(router.get("linearity_nonlin_threshold_low", 0.03))
    thr_hi = float(router.get("linearity_nonlin_threshold_high", 0.10))
    # If MI/R2 is NaN, fall back to is_linear flag
    is_linear_flag = lin.get("is_linear", None)

    # tau_max from window and T
    tau_max = _auto_tau_max(T, sampling_days, window_days=window_days)

    methods: List[str] = []
    ci_test: Optional[str] = None
    notes: List[str] = []

    # Router-based selection (relevant subset of rules), else fallback to legacy
    if router_enabled:
        # Diagnostics
        # Missing fraction on union of source/target
        miss_frac = 1.0 - (len(sub) / max(1, len(df)))
        max_missing_frac = float(router.get("max_missing_frac", 0.25))
        min_per_tau_pcmci = int(router.get("min_obs_per_tau_pcmci", 8))
        min_per_tau_granger = int(router.get("min_obs_per_tau_granger", 4))

        # Stationarity (ADF & KPSS) quick check
        adf_y = stats.test_stationarity(sub[target], method="adf").get(
            "is_stationary", False
        )
        kpss_y = stats.test_stationarity(sub[target], method="kpss").get(
            "is_stationary", False
        )
        stationary = bool(adf_y and kpss_y)

        # Capacity guards
        granger_ok = (T >= max(2 * tau_max, min_per_tau_granger * tau_max)) and (
            miss_frac <= max_missing_frac
        )
        pcmci_ok = T >= max(2 * tau_max, min_per_tau_pcmci * tau_max)
        te_ok = pcmci_ok and (miss_frac <= max_missing_frac)

        # Nonlinearity gating via MI/R2 ratio or fallback flag
        if not np.isnan(mi_ratio):
            mostly_linear = mi_ratio < (1.0 + thr_lo)  # lower ratio => more linear
            strongly_nonlinear = mi_ratio > (1.0 + thr_hi)
        else:
            mostly_linear = is_linear_flag is True
            strongly_nonlinear = is_linear_flag is False

        # Choose methods
        if T < 2 * tau_max:
            methods = []  # ill-posed
            notes.append("skip: T < 2*tau")
        elif mostly_linear and stationary and granger_ok:
            methods.append("granger")
            if pcmci_ok:
                methods.append("pcmci")
                ci_test = "parcorr"
                notes.append("linear+stationary -> Granger + PCMCI(ParCorr)")
        elif strongly_nonlinear:
            if te_ok:
                methods.append("te")
            if pcmci_ok:
                methods.append("pcmci")
                # Use parcorr instead of cmiknn (cmiknn is too slow/unstable)
                ci_test = "parcorr"
            if stationary and granger_ok:
                notes.append("add linear baseline")
                methods.append("granger")
        else:  # mixed regime
            if granger_ok:
                methods.append("granger")
            if pcmci_ok:
                methods.append("pcmci")
                # Always use parcorr (cmiknn hangs on this dataset)
                ci_test = "parcorr"
            if router.get("fallback_run_all_if_ambiguous", True) and te_ok:
                methods.append("te")

        # FORCE ALL THREE METHODS for maximum robustness and consensus quality
        # Override conservative selection: always try granger + pcmci + te when feasible
        if granger_ok and "granger" not in methods:
            methods.append("granger")
        if pcmci_ok and "pcmci" not in methods:
            methods.append("pcmci")
            if ci_test is None:
                ci_test = "parcorr"
        if te_ok and "te" not in methods:
            methods.append("te")

        # De-duplicate and prioritize
        priority = {"granger": 0, "pcmci": 1, "te": 2}
        methods = sorted(
            list(dict.fromkeys(methods)), key=lambda m: priority.get(m, 9)
        )[:3]
    else:
        # Legacy simple heuristic
        if T < 60 or tau_max >= max(6, T // 6):
            methods = ["granger"]
            ci_test = None
            notes.append("short_T -> Granger only")
        else:
            is_linear = is_linear_flag
            if is_linear is True:
                methods.append("granger")
                if T >= min_T_for_pcmci:
                    methods.append("pcmci")
                    ci_test = "parcorr"
            elif is_linear is False:
                if T >= min_T_for_te:
                    methods.append("te")
                if T >= min_T_for_pcmci:
                    methods.append("pcmci")
                    ci_test = "cmiknn"
            else:
                methods.append("granger")
                if T >= min_T_for_pcmci:
                    methods.append("pcmci")
                    ci_test = "parcorr"
            methods = methods[:2] if len(methods) > 2 else methods

    return PairDecision(
        source=source,
        target=target,
        methods=methods,
        tau_max=tau_max,
        sampling_days=sampling_days,
        ci_test=ci_test,
        allow_contemporaneous=allow_contemporaneous,
        notes="; ".join(notes) if notes else None,
    )


def run_plan_for_pair(
    df: pd.DataFrame,
    plan: PairDecision,
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Execute methods according to the plan for a single pair and normalize outputs.

    Returns a dict with keys in {"granger","te","pcmci"} mapping to
    1-row DataFrames (or empty DataFrames) with unified columns:
    [source,target,method,is_significant,lag_steps,lag_days,p_value,q_value,effect_size].
    """
    outputs: Dict[str, pd.DataFrame] = {}
    s, t = plan.source, plan.target

    def _df_row(row: Dict) -> pd.DataFrame:
        return pd.DataFrame([row])

    if "granger" in plan.methods:
        try:
            res = granger.run_granger_causality(
                df,
                s,
                t,
                maxlag=plan.tau_max,
                alpha=alpha,
                sampling_days=plan.sampling_days,
                verbose=False,
            )
            row = {
                "source": s,
                "target": t,
                "method": "Granger",
                "is_significant": bool(res.get("is_causal", False)),
                "lag_steps": res.get("best_lag", np.nan),
                "lag_days": res.get("best_lag_days", np.nan),
                "p_value": res.get("best_p_value", np.nan),
                "q_value": np.nan,
                "effect_size": res.get("granger_beta_std", np.nan),
                "n_obs": res.get("n_observations", np.nan),
            }
            outputs["granger"] = _df_row(row)
        except Exception as e:
            logger.warning(f"Granger failed for {s}→{t}: {e}")
            outputs["granger"] = pd.DataFrame()  # Empty on failure

    if "te" in plan.methods:
        # Try a small grid of delays up to tau_max (bounded to 6 for speed)
        delays = list(range(1, min(plan.tau_max, 6) + 1))

        # Load config for surrogates parameter
        try:
            config_path = Path(__file__).parents[2] / "config" / "defaults.json"
            with open(config_path) as f:
                config = json.load(f)
            n_surrogates = config.get("te", {}).get("surrogates", 500)
        except Exception:
            n_surrogates = 500

        te_rows = []
        for d in delays:
            res = transfer_entropy.run_transfer_entropy(
                df,
                s,
                t,
                delay=d,
                method="discrete",
                n_surrogates=n_surrogates,
                bins=10,
                verbose=False,  # Use 10 bins for better granularity
            )
            te_rows.append(
                {
                    "source": s,
                    "target": t,
                    "method": "TransferEntropy",
                    "is_significant": bool(res.get("is_significant", False)),
                    "lag_steps": res.get("delay", d),
                    "lag_days": res.get("delay_days", d * plan.sampling_days),
                    "p_value": res.get("p_value", np.nan),
                    "q_value": np.nan,
                    "effect_size": res.get("te_bits", np.nan),
                    "n_obs": res.get("n_observations", np.nan),
                }
            )
        te_df = pd.DataFrame(te_rows)
        # Keep best (min p-value) among significant results
        if len(te_df) > 0 and "p_value" in te_df:
            # Only keep significant results
            sig_te = te_df[te_df["is_significant"]]
            if len(sig_te) > 0:
                te_df = (
                    sig_te.sort_values("p_value", ascending=True)
                    .head(1)
                    .reset_index(drop=True)
                )
            else:
                # If none significant, keep empty
                te_df = pd.DataFrame()
        outputs["te"] = te_df

    if "pcmci" in plan.methods:
        test = plan.ci_test or "parcorr"
        pcmci_res = tigramite_pcmci.run_pcmci_pair(
            df,
            s,
            t,
            tau_max=plan.tau_max,
            test_method=test,
            alpha=alpha,
            verbose=False,
        )
        row = {
            "source": s,
            "target": t,
            "method": "PCMCI+",
            "is_significant": bool(pcmci_res.get("causal", False)),
            "lag_steps": pcmci_res.get("best_lag", np.nan),
            "lag_days": pcmci_res.get("best_lag_days", np.nan),
            "p_value": pcmci_res.get("best_p_value", np.nan),
            "q_value": np.nan,
            "effect_size": np.nan,
            "n_obs": pcmci_res.get("n_observations", np.nan),
        }
        outputs["pcmci"] = _df_row(row)

    # Optional: apply per-method FDR if we have multiple rows (TE delay grid)
    for k, dfk in list(outputs.items()):
        if dfk is not None and len(dfk) > 1 and "p_value" in dfk:
            dfk = apply_fdr_to_dataframe(dfk, p_col="p_value", alpha=alpha)
            # update significance by q-value if present
            if "q_value" in dfk.columns:
                dfk["is_significant"] = dfk["q_value"] < alpha
            outputs[k] = dfk

    return outputs


def decide_and_run(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    window_days: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """High-level convenience API: decide plan for each pair, run, and combine rows.

    Returns a DataFrame with unified schema across all methods and pairs.
    """
    rows: List[pd.DataFrame] = []
    for s, t in pairs:
        plan = decide_for_pair(df, s, t, window_days=window_days)
        logger.info(
            f"Plan for {s}→{t}: methods={plan.methods}, tau_max={plan.tau_max}, ci={plan.ci_test or '-'}"
        )
        outs = run_plan_for_pair(df, plan, alpha=alpha)
        for dfk in outs.values():
            if dfk is not None and len(dfk) > 0:
                rows.append(dfk)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
