"""
PCMCI+ and LPCMCI Methods for Causal Discovery

Implements constraint-based causal discovery using momentary conditional
independence tests. Supports linear (ParCorr), robust (RobustParCorr),
and nonlinear (GPDC, CMIknn) test statistics.
Handles lagged causal edges and contemporaneous links.

References:
    - Runge, J. (2020). "Discovering causal relations from multivariate data using
      the PC algorithm". Nature Communications, 11(1), 1-8.
    - Runge, J. (2020). "Causal inference for time series". arXiv:1905.13407.
    - Tigramite: https://github.com/jakobrunge/tigramite
    - Gerhardus, A., & Runge, J. (2020). "High-recall causal discovery for autocorrelated time series
      with latent confounders". NeurIPS.

This implementation wraps tigramite's PCMCI and PCMCI+ algorithms with automatic
significance testing and result formatting compatible with the framework.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Try to import tigramite; fall back gracefully if unavailable
try:
    import importlib.util

    tigramite_spec = importlib.util.find_spec("tigramite")
    TIGRAMITE_AVAILABLE = tigramite_spec is not None

    if TIGRAMITE_AVAILABLE:
        # Fix numpy 2.x incompatibility in tigramite ≤5.2.10.1:
        # np.corrcoef() never accepted ddof, but tigramite passes it.
        # Monkey-patch _get_acf to remove the invalid kwarg.
        try:
            import tigramite.independence_tests.independence_tests_base as _itb
            import inspect as _inspect

            _src = _inspect.getsource(_itb.CondIndTest._get_acf)
            if "ddof=0" in _src:
                _orig_get_acf = _itb.CondIndTest._get_acf

                def _patched_get_acf(self, series, max_lag=None):
                    """Patched _get_acf: removes ddof kwarg for numpy ≥2.0."""
                    import numpy as _np

                    if max_lag is None:
                        max_lag = int(max(5, 0.1 * len(series)))
                    autocorr = _np.ones(max_lag + 1)
                    for lag in range(1, max_lag + 1):
                        y1_vals = series[lag:]
                        y2_vals = series[: len(series) - lag]
                        autocorr[lag] = _np.corrcoef(y1_vals, y2_vals)[0, 1]
                    return autocorr

                _itb.CondIndTest._get_acf = _patched_get_acf
                logger.debug("Applied numpy 2.x corrcoef patch to tigramite")
        except Exception:
            pass  # If patching fails, the original code may still work on older numpy

        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.independence_tests.robust_parcorr import RobustParCorr

        try:
            from tigramite.independence_tests.gpdc import GPDC

            GPDC_AVAILABLE = True
        except (ImportError, Exception):
            GPDC_AVAILABLE = False
            logger.debug(
                "GPDC not available (dcor/Numba incompatibility); using ParCorr/CMIknn instead"
            )
        try:
            from tigramite.independence_tests.cmiknn import CMIknn

            CMIKNN_AVAILABLE = True
        except ImportError:
            CMIKNN_AVAILABLE = False
            logger.warning("CMIknn not available. Install with: pip install tigramite")
except (ImportError, OSError):
    TIGRAMITE_AVAILABLE = False
    logger.warning("tigramite not available. Install with: pip install tigramite")


def _validate_tigramite_installed():
    """Raise informative error if tigramite is not available."""
    if not TIGRAMITE_AVAILABLE:
        raise ImportError(
            "tigramite is required for PCMCI+ analysis.\n"
            "Install with: pip install tigramite\n"
            "For GPU support: pip install tigramite[gpu]"
        )


def select_ci_test(df: pd.DataFrame, method: str = "auto", verbose: bool = False):
    """Select the conditional independence test based on data properties.

    When method="auto", runs nonlinearity detection using:
    1. Ramsey RESET test on all variable pairs (contemporaneous + lagged)
    2. Distance correlation vs. linear correlation ratio (dcor/|r|)

    Picks CMIknn for nonlinear data or ParCorr for linear data. This
    implements the key insight from the TimeGraph benchmark (Ferdous et
    al. 2025): ParCorr fails completely on nonlinear data (TPR=0), while
    CMIknn can capture nonlinear dependencies at the cost of higher
    computational cost and lower power on linear data.

    Parameters:
        df: Multivariate time series (used for linearity check when auto)
        method: "auto", "parcorr", "robust_parcorr", "cmiknn", "gpdc"
        verbose: Print selection rationale

    Returns:
        (test_object, test_name) tuple
    """
    _validate_tigramite_installed()

    if method != "auto":
        return _make_ci_test(method, verbose), method

    # Nonlinearity detection using two complementary approaches:
    #
    # 1. Ramsey RESET test on both contemporaneous AND lagged pairs.
    #    Testing lagged pairs is essential because in time series the
    #    nonlinearity is often in the causal mechanism (e.g., X1(t-2) →
    #    X4(t) via polynomial), not in contemporaneous associations.
    #
    # 2. Distance correlation vs. linear correlation ratio (dcor/|r|).
    #    If dcor >> |r| for any pair, there is nonlinear dependence that
    #    linear methods cannot capture.
    #
    # Decision: declare nonlinear if EITHER test triggers on ANY pair.
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        if verbose:
            logger.info("Auto CI test: <2 variables, defaulting to ParCorr")
        return _make_ci_test("parcorr", verbose), "parcorr"

    try:
        from scipy import stats as sp_stats
        from scipy.spatial.distance import pdist, squareform
        from numpy.linalg import lstsq
        from itertools import combinations

        n_vars = len(numeric_cols)
        all_pairs = list(combinations(range(n_vars), 2))
        if len(all_pairs) > 20:
            rng = np.random.default_rng(42)
            pair_indices = rng.choice(len(all_pairs), size=20, replace=False)
            pairs_to_test = [all_pairs[i] for i in pair_indices]
        else:
            pairs_to_test = all_pairs

        min_p_reset = 1.0
        most_nonlinear_pair = (numeric_cols[0], numeric_cols[1])
        nonlinear_reason = ""

        def _reset_test(x, y):
            """Run RESET test, return p-value."""
            n = len(x)
            slope, intercept, _, _, _ = sp_stats.linregress(x, y)
            y_hat = slope * x + intercept
            rss_lin = np.sum((y - y_hat) ** 2)
            X_aug = np.column_stack([np.ones(n), x, y_hat**2, y_hat**3])
            _, rss_aug_arr, _, _ = lstsq(X_aug, y, rcond=None)
            rss_aug = rss_aug_arr[0] if len(rss_aug_arr) > 0 else rss_lin
            q, k = 2, 4
            if rss_aug > 0 and n > k:
                f_stat = ((rss_lin - rss_aug) / q) / (rss_aug / (n - k))
                return 1 - sp_stats.f.cdf(f_stat, q, n - k)
            return 1.0

        # --- RESET on contemporaneous pairs ---
        for i_idx, j_idx in pairs_to_test:
            col_i, col_j = numeric_cols[i_idx], numeric_cols[j_idx]
            x = df[col_i].dropna().values
            y = df[col_j].dropna().values
            n = min(len(x), len(y), 500)
            if n < 30:
                continue
            p_val = _reset_test(x[:n], y[:n])
            if p_val < min_p_reset:
                min_p_reset = p_val
                most_nonlinear_pair = (col_i, col_j)
                nonlinear_reason = f"RESET contemp p={p_val:.2e}"

        # --- RESET on lagged pairs (lag=1,2,3) ---
        lags_to_test = [1, 2, 3]
        for lag in lags_to_test:
            for i_idx in range(n_vars):
                for j_idx in range(n_vars):
                    if i_idx == j_idx:
                        continue
                    col_i, col_j = numeric_cols[i_idx], numeric_cols[j_idx]
                    x_full = df[col_i].dropna().values
                    y_full = df[col_j].dropna().values
                    min_len = min(len(x_full), len(y_full))
                    if min_len <= lag + 30:
                        continue
                    x = x_full[: min_len - lag]
                    y = y_full[lag:min_len]
                    n = min(len(x), 500)
                    p_val = _reset_test(x[:n], y[:n])
                    if p_val < min_p_reset:
                        min_p_reset = p_val
                        most_nonlinear_pair = (col_i, col_j)
                        nonlinear_reason = f"RESET lag={lag} p={p_val:.2e}"

        # --- Distance correlation ratio (dcor / |r|) ---
        max_dcor_ratio = 0.0
        dcor_nonlinear = False
        for i_idx, j_idx in pairs_to_test[:10]:
            col_i, col_j = numeric_cols[i_idx], numeric_cols[j_idx]
            x = df[col_i].dropna().values[:300]
            y = df[col_j].dropna().values[:300]
            n = min(len(x), len(y))
            if n < 30:
                continue
            x, y = x[:n], y[:n]
            r = abs(np.corrcoef(x, y)[0, 1])
            # Distance correlation
            a = squareform(pdist(x.reshape(-1, 1)))
            b = squareform(pdist(y.reshape(-1, 1)))
            A = a - a.mean(axis=0) - a.mean(axis=1, keepdims=True) + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1, keepdims=True) + b.mean()
            dcov2 = (A * B).mean()
            dvar_x = (A * A).mean()
            dvar_y = (B * B).mean()
            if dvar_x > 0 and dvar_y > 0:
                dcor = np.sqrt(max(0, dcov2) / np.sqrt(dvar_x * dvar_y))
            else:
                dcor = 0.0
            if dcor > 0.05 and r < 0.05:
                ratio = dcor / (r + 1e-10)
                if ratio > max_dcor_ratio:
                    max_dcor_ratio = ratio
                if ratio > 3.0:
                    dcor_nonlinear = True
                    if nonlinear_reason == "" or min_p_reset > 0.05:
                        most_nonlinear_pair = (col_i, col_j)
                        nonlinear_reason = f"dcor/|r| ratio={ratio:.1f}"

        # Decision: nonlinear if RESET p < 0.01 (strict, accounts for multiple
        # testing across ~40 pairs) OR dcor ratio > 3 (independent signal)
        is_nonlinear = (min_p_reset < 0.01) or dcor_nonlinear

        if is_nonlinear and CMIKNN_AVAILABLE:
            # Sample size guard: check if T is sufficient for CMIknn
            from framework.core.sample_size_adequacy import (
                suggest_ci_test_for_sample_size,
            )

            n_vars = len(numeric_cols)
            t_eff = int(df[numeric_cols].notna().all(axis=1).sum())
            # Use a conservative tau_max estimate (will be refined later)
            tau_max_est = min(12, max(1, t_eff // 10))
            suggested = suggest_ci_test_for_sample_size(
                t_effective=t_eff,
                n_vars=n_vars,
                tau_max=tau_max_est,
                is_nonlinear=True,
            )
            if suggested != "cmiknn":
                if verbose:
                    logger.info(
                        f"Auto CI test: nonlinear detected "
                        f"({nonlinear_reason}) but T_eff={t_eff} insufficient for CMIknn. "
                        f"Using {suggested} instead."
                    )
                return _make_ci_test(suggested, verbose), suggested

            try:
                col_i, col_j = most_nonlinear_pair
                _x = df[col_i].dropna().values[:20]
                _y = df[col_j].dropna().values[:20]
                _test_cmi = CMIknn(verbosity=0)
                _arr = np.column_stack([_x, _y])
                _test_cmi.get_dependence_measure(_arr.T, np.array([0, 1]))
                if verbose:
                    logger.info(
                        f"Auto CI test: nonlinear data detected "
                        f"({nonlinear_reason}, pair={most_nonlinear_pair}), using CMIknn"
                    )
                return _make_ci_test("cmiknn", verbose), "cmiknn"
            except Exception as cmi_err:
                if verbose:
                    logger.info(
                        f"Auto CI test: nonlinear but CMIknn failed ({cmi_err}), "
                        f"falling back to ParCorr"
                    )
                return _make_ci_test("parcorr", verbose), "parcorr"
        else:
            x_check = df[numeric_cols[0]].dropna().values
            n_check = min(len(x_check), 200)
            try:
                _, p_shapiro = sp_stats.shapiro(x_check[:n_check])
                is_nongaussian = p_shapiro < 0.01
            except Exception:
                is_nongaussian = False

            if is_nongaussian:
                if verbose:
                    logger.info(
                        f"Auto CI test: linear but non-Gaussian (Shapiro p={p_shapiro:.4f}), "
                        f"using RobustParCorr"
                    )
                return _make_ci_test("robust_parcorr", verbose), "robust_parcorr"
            else:
                if verbose:
                    logger.info(
                        f"Auto CI test: linear Gaussian "
                        f"(RESET min_p={min_p_reset:.4f}, dcor_ratio={max_dcor_ratio:.1f}), "
                        f"using ParCorr"
                    )
                return _make_ci_test("parcorr", verbose), "parcorr"
    except Exception as e:
        if verbose:
            logger.info(
                f"Auto CI test: linearity check failed ({e}), defaulting to ParCorr"
            )
        return _make_ci_test("parcorr", verbose), "parcorr"


def _make_ci_test(method: str, verbose: bool = False, knn=None):
    """Instantiate a tigramite CI test object.

    The optional ``knn`` argument applies only to CMIknn and accepts either a
    fraction of T (``0 < knn < 1``, the Tigramite default) or an absolute
    integer count of nearest neighbours (``knn >= 1``). When unset, the
    Tigramite default (``knn=0.2``) is preserved.
    """
    verbosity = 1 if verbose else 0
    if method == "parcorr":
        return ParCorr(verbosity=verbosity)
    elif method == "robust_parcorr":
        return RobustParCorr(verbosity=verbosity)
    elif method == "cmiknn":
        if not CMIKNN_AVAILABLE:
            logger.warning("CMIknn not available, falling back to ParCorr")
            return ParCorr(verbosity=verbosity)
        if knn is None:
            return CMIknn(verbosity=verbosity)
        return CMIknn(verbosity=verbosity, knn=knn)
    elif method == "gpdc":
        if not GPDC_AVAILABLE:
            logger.warning("GPDC not available, falling back to ParCorr")
            return ParCorr(verbosity=verbosity)
        return GPDC(verbosity=verbosity)
    else:
        logger.warning(f"Unknown test method '{method}', using ParCorr")
        return ParCorr(verbosity=verbosity)


def prepare_data_for_tigramite(df: pd.DataFrame, standardize: bool = True):
    """
    Prepare DataFrame for tigramite (returns tigramite.DataFrame object).

    Tigramite requires:
    1. NaN values in the data array to indicate missing values
    2. A boolean mask where True = missing value
    3. The mask must match the NaN positions exactly

    Parameters:
        df (pd.DataFrame): Multivariate time series
        standardize (bool): Standardize to zero mean, unit variance

    Returns:
        tigramite.data_processing.DataFrame: Data object suitable for PCMCI
    """
    from tigramite.data_processing import DataFrame as TimeseriesDataFrame

    data = df.values.astype(float)

    # Create mask for original NaN positions
    mask = np.isnan(data)

    logger.debug(
        f"Data shape: {data.shape}, NaN count: {mask.sum()}, NaN %: {100 * mask.sum() / data.size:.1f}%"
    )

    if standardize:
        # Standardize column-wise, preserving NaN positions
        # Use nanmean/nanstd to compute statistics only from observed values
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero for constant columns

        # Apply standardization - this will preserve NaNs automatically
        # (NaN - mean) / std = NaN
        data = (data - means) / stds

    # Verify mask still matches NaN positions after standardization
    assert np.array_equal(mask, np.isnan(data)), "Mask mismatch after standardization"

    logger.debug(f"After standardization - NaN count: {np.isnan(data).sum()}")

    # Create tigramite DataFrame with variable names, mask, and datatime
    # Note: tigramite expects mask=True for missing values
    # datatime is required for proper time series handling and memory efficiency
    var_names = list(df.columns)

    # Create datatime parameter - use index if it's datetime, otherwise use integer range
    if isinstance(df.index, pd.DatetimeIndex):
        datatime = {0: df.index.values}
    else:
        # Use integer time axis (0, 1, 2, ...)
        datatime = {0: np.arange(len(data))}

    tigramite_df = TimeseriesDataFrame(
        data, datatime=datatime, var_names=var_names, mask=mask
    )

    # Safely log mask information: tigramite implementations may expose mask
    # as a numpy array or as a dict-like structure depending on version.
    mask_info = None
    try:
        mask_obj = tigramite_df.mask
        if mask_obj is None:
            mask_info = None
        elif hasattr(mask_obj, "shape"):
            mask_info = getattr(mask_obj, "shape")
        elif isinstance(mask_obj, dict):
            # summarize dict masks
            mask_info = {
                k: (v.sum() if hasattr(v, "sum") else None) for k, v in mask_obj.items()
            }
        else:
            mask_info = str(type(mask_obj))
    except Exception:
        mask_info = "<unavailable>"

    logger.debug(
        f"Tigramite DataFrame created: T={tigramite_df.T}, N={tigramite_df.N}, mask_info={mask_info}"
    )

    return tigramite_df


def run_pcmci_algorithm(
    df: pd.DataFrame,
    test_method: str = "parcorr",
    tau_max: int = 12,
    pc_alpha: float = 0.05,
    contemp_alpha: float = None,
    fdr_method: str = "fdr_bh",
    missing_threshold: float = 0.5,
    verbose: bool = True,
    knn=None,
) -> Dict:
    """
    Run PCMCI+ algorithm on multivariate time series.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        test_method (str): "parcorr" (linear) or "gpdc" (nonlinear) or "robust_parcorr"
        tau_max (int): Maximum time lag to explore
        pc_alpha (float): Significance threshold for lagged causality
        contemp_alpha (float): Significance threshold for contemporaneous links
        verbose (bool): Print detailed output

    Returns:
        Dict: Results dict with graph, significances, and edge list
    """
    _validate_tigramite_installed()

    logger.info(f"\n{'=' * 70}")
    logger.info(
        f"PCMCI+ Algorithm: method={test_method}, tau_max={tau_max}, α={pc_alpha}, contemp_alpha={contemp_alpha}, fdr_method={fdr_method}"
    )
    logger.info(f"{'=' * 70}")

    try:
        # Prepare data
        data = prepare_data_for_tigramite(df, standardize=True)

        # Select independence test — use auto-selection when "parcorr" is requested
        # to automatically switch to CMIknn for nonlinear data
        if test_method.lower() == "auto":
            test, actual_method = select_ci_test(df, method="auto", verbose=verbose)
        elif test_method.lower() == "parcorr":
            test = ParCorr(verbosity=1 if verbose else 0)
            actual_method = "parcorr"
        elif test_method.lower() == "robust_parcorr":
            test = RobustParCorr(verbosity=1 if verbose else 0)
            actual_method = "robust_parcorr"
        elif test_method.lower() == "gpdc":
            if not GPDC_AVAILABLE:
                logger.warning("GPDC not available, falling back to ParCorr")
                test = ParCorr(verbosity=1 if verbose else 0)
                actual_method = "parcorr"
            else:
                test = GPDC(verbosity=1 if verbose else 0)
                actual_method = "gpdc"
        elif test_method.lower() == "cmiknn":
            if not CMIKNN_AVAILABLE:
                logger.warning("CMIknn not available, falling back to ParCorr")
                test = ParCorr(verbosity=1 if verbose else 0)
                actual_method = "parcorr"
            else:
                if knn is None:
                    test = CMIknn(verbosity=1 if verbose else 0)
                else:
                    test = CMIknn(verbosity=1 if verbose else 0, knn=knn)
                actual_method = "cmiknn"
        else:
            logger.warning(f"Unknown test method '{test_method}', using ParCorr")
            test = ParCorr(verbosity=1 if verbose else 0)
            actual_method = "parcorr"

        # Run PCMCI+ (not PCMCI) to handle contemporaneous effects
        pcmci = PCMCI(dataframe=data, cond_ind_test=test, verbosity=2 if verbose else 0)

        # Check for high-missingness variables and log warnings
        data_array = df.values.astype(float)
        mask_check = np.isnan(data_array)
        for col_idx, col_name in enumerate(df.columns):
            col_missing_frac = mask_check[:, col_idx].mean()
            if col_missing_frac > missing_threshold:
                logger.warning(
                    f"Variable '{col_name}' has {col_missing_frac:.1%} missing values "
                    f"(threshold: {missing_threshold:.0%})"
                )

        # Use run_pcmciplus instead of run_pcmci to properly handle contemporaneous links
        # This is the correct method according to Tigramite best practices
        allow_contemporaneous = contemp_alpha is not None and contemp_alpha > 0

        if allow_contemporaneous:
            # Run PCMCI+ with contemporaneous effects and internal FDR correction
            results = pcmci.run_pcmciplus(
                tau_max=tau_max,
                pc_alpha=pc_alpha
                if pc_alpha is not None
                else None,  # None = auto-optimize
                contemp_collider_rule="majority",
                conflict_resolution=True,
                fdr_method=fdr_method,
            )

            # Get graph from p_matrix with threshold (separate from condition selection)
            alpha_level = contemp_alpha if contemp_alpha is not None else pc_alpha
            results["graph"] = pcmci.get_graph_from_pmatrix(
                p_matrix=results["p_matrix"],
                alpha_level=alpha_level,
                tau_min=0,
                tau_max=tau_max,
                link_assumptions=None,
            )
        else:
            # For lagged-only, we can use run_pcmci (tau_min=1)
            results = pcmci.run_pcmci(
                tau_min=1,
                tau_max=tau_max,
                pc_alpha=pc_alpha,
                fdr_method=fdr_method,
            )
        # Debug: ensure returned object has expected structure
        logger.debug(f"PCMCI+ returned type: {type(results)}")
        logger.debug(
            f"PCMCI+ returned keys/length: {results.keys() if isinstance(results, dict) else len(results)}"
        )

        # Tigramite's API changed across versions: older versions return a
        # tuple (graph, significances, ...) while newer versions return a
        # dict with keys like 'graph' and 'p_matrix'. Support both.
        if isinstance(results, dict):
            logger.debug(f"Results dict keys: {list(results.keys())}")
            graph = results.get("graph")
            # p-values are commonly in 'p_matrix' or 'pvalues'
            # Use explicit None checks to avoid "truth value of array" errors
            significances = results.get("p_matrix")
            if significances is None:
                significances = results.get("pvalues")
            if significances is None:
                significances = results.get("p_vals")
            logger.debug(
                f"Extracted graph type: {type(graph)}, shape: {getattr(graph, 'shape', 'N/A')}"
            )
            logger.debug(
                f"Extracted significances type: {type(significances)}, shape: {getattr(significances, 'shape', 'N/A')}"
            )
        else:
            try:
                graph = results[0]
                significances = results[1]
                logger.debug(
                    f"Extracted graph (tuple) type: {type(graph)}, shape: {getattr(graph, 'shape', 'N/A')}"
                )
                logger.debug(
                    f"Extracted significances (tuple) type: {type(significances)}, shape: {getattr(significances, 'shape', 'N/A')}"
                )
            except Exception as e:
                logger.error(
                    f"Unexpected return from PCMCI.run_pcmci: {e} (type={type(results)})"
                )
                raise

        logger.debug(
            f"About to return from run_pcmci_algorithm with graph={'None' if graph is None else 'valid'}"
        )
        return {
            "graph": graph,
            "significances": significances,
            "pcmci_obj": pcmci,
            "test_method": test_method,
            "tau_max": tau_max,
            "pc_alpha": pc_alpha,
            "contemp_alpha": contemp_alpha,
            "fdr_method": fdr_method,
            "n_variables": data.N,
            "n_samples": data.T,
        }
    except Exception as e:
        logger.error(f"PCMCI+ failed: {e}")
        return {
            "graph": None,
            "significances": None,
            "error": str(e),
            "test_method": test_method,
            "contemp_alpha": contemp_alpha,
        }


def extract_causal_edges(
    graph: np.ndarray,
    significances: np.ndarray,
    variable_names: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Extract causal edges from a PCMCI+ graph in standard (source, target) format.

    Graph convention (Tigramite): ``graph[i, j, tau]`` encodes the link from
    variable ``i`` at time ``t-tau`` to variable ``j`` at time ``t``. For
    ``tau > 0`` the only valid marker is ``'-->'`` (lagged links are oriented
    by time order). For ``tau == 0`` the marker is ``'-->'`` (i -> j),
    ``'<--'`` (j -> i), ``'o-o'`` (unoriented, within the contemporaneous
    Markov equivalence class) or ``'x-x'`` (conflicting orientation). The
    p-value matrix shares this indexing, so ``significances[i, j, tau]`` is the
    p-value of the ``i -> j`` link at lag ``tau``.

    The ``directed`` column flags whether the orientation is identified
    (lagged links and oriented contemporaneous links) or left undirected
    (``o-o``/``x-x`` contemporaneous links), so that downstream evaluation and
    tiering can score orientation only where it is identifiable.

    Parameters:
        graph (np.ndarray): Adjacency matrix (N x N x tau_max+1).
        significances (np.ndarray): P-values for each link, same indexing.
        variable_names (List[str]): Variable names indexed as in the graph.
        alpha (float): Significance threshold.

    Returns:
        pd.DataFrame: Edges with source, target, lag, p-value, direction.
    """
    edges = []

    if graph is None or significances is None:
        return pd.DataFrame(edges)

    n_vars = graph.shape[0]

    # Contemporaneous links are symmetric in the graph array: both (i, j, 0)
    # and (j, i, 0) carry a marker. Deduplicate on the unordered pair so that
    # each contemporaneous link is emitted once.
    seen_contemp = set()

    def _p_value(i, j, lag):
        p = significances[i, j, lag]
        if np.ndim(p) > 0:
            return float(p.item()) if p.size == 1 else float(p[0])
        return float(p)

    # i indexes the source (variable at t-tau); j indexes the target (at t).
    for i in range(n_vars):
        for j in range(n_vars):
            for lag in range(graph.shape[2]):
                edge_val = graph[i, j, lag]
                if isinstance(edge_val, (str, np.str_)):
                    marker = edge_val.strip()
                    edge_exists = marker not in ("", "0")
                else:
                    marker = None
                    edge_exists = (
                        np.any(edge_val != 0)
                        if np.ndim(edge_val) > 0
                        else edge_val != 0
                    )

                if not edge_exists:
                    continue

                if lag == 0:
                    pair_key = (min(i, j), max(i, j))
                    if pair_key in seen_contemp:
                        continue
                    seen_contemp.add(pair_key)
                    if marker == "<--":
                        src_idx, tgt_idx, directed = j, i, True
                    elif marker == "-->":
                        src_idx, tgt_idx, directed = i, j, True
                    elif marker in ("o-o", "x-x"):
                        # Unoriented or conflicting: orientation is not
                        # identified from the data. Keep a canonical ordering
                        # and flag the link as undirected.
                        src_idx, tgt_idx, directed = i, j, False
                    else:
                        src_idx, tgt_idx, directed = i, j, True
                    p_value = _p_value(i, j, lag)
                else:
                    # Lagged link: oriented from i (past) to j (present).
                    src_idx, tgt_idx, directed = i, j, True
                    p_value = _p_value(i, j, lag)

                src_name = variable_names[src_idx]
                tgt_name = variable_names[tgt_idx]
                direction = (
                    f"{src_name} \u2192 {tgt_name}"
                    if directed
                    else f"{src_name} \u2014 {tgt_name}"
                )

                is_significant = p_value < alpha

                edges.append(
                    {
                        "source": src_name,
                        "target": tgt_name,
                        "lag": lag,
                        "lag_steps": lag,
                        "p_value": p_value,
                        "is_significant": is_significant,
                        "edge_type": "lagged" if lag > 0 else "contemp",
                        "link_category": (
                            "autoregressive"
                            if src_name == tgt_name
                            else ("contemporaneous" if lag == 0 else "lagged_directed")
                        ),
                        "direction": direction,
                        "directed": directed,
                    }
                )

    return pd.DataFrame(edges).sort_values("p_value") if edges else pd.DataFrame(edges)


def run_pcmci_pair(
    df: pd.DataFrame,
    source_var: str,
    target_var: str,
    tau_max: int = 12,
    test_method: str = "parcorr",
    alpha: float = 0.05,
    controls: List[str] = None,
    allow_contemporaneous: bool = False,
    verbose: bool = True,
    sampling_days: float = 1.0,
) -> Dict:
    """
    Run PCMCI+ for specific source-target pair (subset selection).

    Parameters:
        df (pd.DataFrame): Multivariate time series (must include conditioning set)
        source_var (str): Source variable name
        target_var (str): Target variable name
        tau_max (int): Maximum lag
        test_method (str): "parcorr", "gpdc", or "robust_parcorr"
        alpha (float): Significance threshold
        verbose (bool): Print results

    Returns:
        Dict: Results for specific pair
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"PCMCI+ Pair Analysis: {source_var} → {target_var}")
    logger.info(f"{'=' * 70}")

    # Build multivariate subset including optional controls
    cols = [source_var, target_var]
    if controls:
        for c in controls:
            if c not in cols:
                cols.append(c)
    data = df[cols].copy()

    # Tigramite's ParCorr (and some tests) do not handle NaNs even when a mask
    # is provided. To make PCMCI+ robust to moderate missingness we impute
    # missing values with a simple, local method that preserves temporal spacing:
    # linear interpolation followed by forward/backfill for edge cases.
    n_missing_before = data.isna().sum().sum()
    if n_missing_before > 0:
        logger.info(
            f"PCMCI+ pre-processing: {n_missing_before} missing values detected — applying linear interpolation"
        )
        # Use index-based interpolation; this preserves timestep spacing
        try:
            data = data.interpolate(method="linear", axis=0, limit_direction="both")
        except Exception:
            # Fallback to generic interpolation
            data = data.interpolate()
        # Fill any remaining NaNs at the start/end
        data = data.ffill().bfill()
        n_missing_after = data.isna().sum().sum()
        if n_missing_after > 0:
            logger.error(
                f"Imputation left {n_missing_after} NaNs — PCMCI+ cannot run on this pair"
            )
            return {
                "source": source_var,
                "target": target_var,
                "causal": False,
                "error": "Imputation left NaNs",
                "n_observations": len(data),
                "method": "PCMCI+",
            }
        else:
            logger.info("✓ Successfully imputed all missing values for PCMCI+ analysis")

    if len(data) < tau_max + 5:
        logger.error(f"Insufficient data ({len(data)} samples) for tau_max={tau_max}")
        return {
            "source": source_var,
            "target": target_var,
            "causal": False,
            "n_observations": len(data),
            "method": "PCMCI+",
        }

    try:
        results = run_pcmci_algorithm(
            data,
            test_method=test_method,
            tau_max=tau_max,
            pc_alpha=alpha,
            contemp_alpha=(alpha if allow_contemporaneous else None),
            verbose=verbose,
        )

        if results["graph"] is None:
            logger.warning("PCMCI+ returned empty results for pair")
            return {
                "source": source_var,
                "target": target_var,
                "causal": False,
                "error": results.get("error", "Unknown error"),
                "n_observations": len(data),
                "method": "PCMCI+",
            }

        # Extract edges involving our pair
        edges = extract_causal_edges(
            results["graph"],
            results["significances"],
            list(data.columns),
            alpha=alpha,
        )

        # Check for causality from source to target
        source_to_target = edges[
            (edges["source"] == source_var)
            & (edges["target"] == target_var)
            & (edges["is_significant"])
        ]

        causal_found = len(source_to_target) > 0
        best_lag = source_to_target["lag"].min() if causal_found else np.nan
        best_pvalue = source_to_target["p_value"].min() if causal_found else np.nan

        result = {
            "source": source_var,
            "target": target_var,
            "causal": causal_found,
            "best_lag": best_lag,
            "best_lag_days": best_lag * sampling_days
            if not np.isnan(best_lag)
            else np.nan,  # Convert to days
            "best_p_value": best_pvalue,
            "n_edges_found": len(edges),
            "n_significant": edges["is_significant"].sum(),
            "n_observations": len(data),
            "test_method": test_method,
            "tau_max": tau_max,
            "tau_max_days": tau_max * sampling_days,  # Convert to days
            "method": "PCMCI+",
            "sampling_days": sampling_days,
            "cadence_note": f"Lags are in timesteps; multiply by {sampling_days} to get days",
            "controls": controls or [],
        }

        if verbose:
            lag_days = best_lag * sampling_days if not np.isnan(best_lag) else "N/A"
            logger.info(
                f"Causal: {causal_found}, Lag: {best_lag} steps ({lag_days} days), p-value: {best_pvalue:.6f}"
            )

        return result

    except Exception as e:
        logger.error(f"PCMCI+ pair analysis failed: {e}")
        return {
            "source": source_var,
            "target": target_var,
            "causal": False,
            "error": str(e),
            "n_observations": len(data),
            "method": "PCMCI+",
        }


def batch_pcmci(
    df: pd.DataFrame,
    variable_pairs: List[Tuple[str, str]],
    tau_max: int = 12,
    test_method: str = "auto",
    alpha: float = 0.05,
    controls: List[str] = None,
    apply_fdr: bool = False,
    fdr_method: str = "fdr_bh",
    sampling_days: float = 1.0,
    knn=None,
) -> pd.DataFrame:
    """
    Run PCMCI+ on multiple source-target pairs.

    Parameters:
        df (pd.DataFrame): Multivariate time series
        variable_pairs (List[Tuple[str, str]]): List of (source, target) pairs
        tau_max (int): Maximum lag
        test_method (str): "parcorr" (default), "auto" (selects ParCorr or CMIknn
            based on linearity — requires tigramite ≥5.3 with numpy 2.x fix),
            "cmiknn", "gpdc", or "robust_parcorr"
        alpha (float): Significance threshold
        apply_fdr (bool): Apply post-hoc FDR correction across pairs (for non-PCMCI methods)
        fdr_method (str): FDR method passed to PCMCI+ internally ('fdr_bh', 'fdr_by', or None)
        sampling_days (float): Days per timestep for lag conversion (default: 1.0)

    Returns:
        pd.DataFrame: Results for all pairs
    """
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Batch PCMCI+ for {len(variable_pairs)} pairs (method={test_method})")
    logger.info("OPTIMIZED: Running PCMCI+ once on full dataset, then extracting pairs")
    logger.info(f"{'=' * 70}")

    # Collect all unique variables from pairs and controls
    all_vars = set()
    for source, target in variable_pairs:
        all_vars.add(source)
        all_vars.add(target)
    if controls:
        all_vars.update(controls)

    # Select only the variables we need
    var_cols = [v for v in all_vars if v in df.columns]
    if len(var_cols) != len(all_vars):
        missing = set(all_vars) - set(var_cols)
        logger.warning(f"Missing variables in data: {missing}")

    data_subset = df[var_cols].copy()

    # NaN handling: pass data with NaNs directly to tigramite which uses
    # its mask parameter to handle missing values without breaking temporal
    # continuity. Log warnings for high-missingness variables.
    for col in var_cols:
        col_missing_frac = data_subset[col].isna().mean()
        if col_missing_frac > 0.5:
            logger.warning(
                f"Variable '{col}' has {col_missing_frac:.1%} missing values "
                f"(exceeds 50% threshold)"
            )

    # Check sufficient data (non-NaN rows for at least some variables)
    min_valid = data_subset.notna().all(axis=1).sum()
    if min_valid < tau_max + 5:
        # Fall back to checking if any variable has enough non-NaN values
        max_valid_per_var = data_subset.notna().sum().max()
        if max_valid_per_var < tau_max + 5:
            logger.error(
                f"Insufficient valid data (max {max_valid_per_var} non-NaN observations) for tau_max={tau_max}"
            )
            # Return empty results for all pairs
            results = []
            for source, target in variable_pairs:
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "causal": False,
                        "error": "Insufficient data",
                        "n_observations": len(data_subset),
                        "method": "PCMCI+",
                    }
                )
            return pd.DataFrame(results)

    # Run PCMCI+ ONCE on the full dataset
    logger.info("Running PCMCI+ on full dataset...")
    try:
        full_results = run_pcmci_algorithm(
            data_subset,
            test_method=test_method,
            tau_max=tau_max,
            pc_alpha=alpha,
            contemp_alpha=alpha,  # Allow contemporaneous by default
            fdr_method=fdr_method,
            verbose=True,  # Show PCMCI+ progress (variable-by-variable)
            knn=knn,
        )

        if full_results["graph"] is None or full_results["significances"] is None:
            logger.error("PCMCI+ returned empty results")
            # Return empty results for all pairs
            results = []
            for source, target in variable_pairs:
                results.append(
                    {
                        "source": source,
                        "target": target,
                        "causal": False,
                        "error": full_results.get("error", "Empty results"),
                        "n_observations": len(data_subset),
                        "method": "PCMCI+",
                    }
                )
            return pd.DataFrame(results)

        # Extract all edges from full graph
        all_edges = extract_causal_edges(
            full_results["graph"],
            full_results["significances"],
            var_cols,
            alpha=alpha,
        )

        # Extract results for each requested pair
        results = []
        for source, target in variable_pairs:
            # Find edges from source to target
            if len(all_edges) > 0 and "source" in all_edges.columns:
                source_to_target = all_edges[
                    (all_edges["source"] == source)
                    & (all_edges["target"] == target)
                    & (all_edges["is_significant"])
                ]
            else:
                source_to_target = pd.DataFrame()

            causal_found = len(source_to_target) > 0
            best_lag = source_to_target["lag"].min() if causal_found else np.nan
            best_pvalue = source_to_target["p_value"].min() if causal_found else np.nan
            # Orientation status: a contemporaneous link returned as 'o-o' by
            # PCMCI+ is not identified (Markov equivalence class); lagged links
            # are oriented by time order. Propagate this so downstream
            # evaluation can score orientation only where it is identifiable.
            if causal_found and "directed" in source_to_target.columns:
                is_directed = bool(source_to_target["directed"].any())
            else:
                is_directed = causal_found

            results.append(
                {
                    "source": source,
                    "target": target,
                    "causal": causal_found,
                    "best_lag": best_lag,
                    "best_lag_days": best_lag * sampling_days
                    if not np.isnan(best_lag)
                    else np.nan,
                    "is_contemporaneous": best_lag == 0 if causal_found else False,
                    "is_directed": is_directed,
                    "best_p_value": best_pvalue,
                    "n_edges_found": len(
                        all_edges[
                            (all_edges["source"] == source)
                            & (all_edges["target"] == target)
                        ]
                    )
                    if len(all_edges) > 0 and "source" in all_edges.columns
                    else 0,
                    "n_significant": len(source_to_target),
                    "n_observations": len(data_subset),
                    "test_method": test_method,
                    "tau_max": tau_max,
                    "tau_max_days": tau_max * sampling_days,
                    "method": "PCMCI+",
                }
            )

    except Exception as e:
        logger.error(f"PCMCI+ batch analysis failed: {e}")
        # Return error results for all pairs
        results = []
        for source, target in variable_pairs:
            results.append(
                {
                    "source": source,
                    "target": target,
                    "causal": False,
                    "error": str(e),
                    "n_observations": len(data_subset),
                    "method": "PCMCI+",
                }
            )

    results_df = pd.DataFrame(results)

    # Sort by p-value (most significant first)
    if "best_p_value" in results_df.columns:
        results_df = results_df.sort_values("best_p_value")

    # Add is_significant column for compatibility with other methods
    if "causal" in results_df.columns:
        results_df["is_significant"] = results_df["causal"]

    # Optionally apply FDR correction across pairs
    if apply_fdr and "best_p_value" in results_df.columns:
        try:
            from ..multiple_testing import apply_fdr_to_dataframe

            logger.info("Applying FDR correction across all pairs...")
            tmp = results_df.rename(columns={"best_p_value": "p_value"})
            tmp = apply_fdr_to_dataframe(tmp, p_col="p_value", alpha=alpha)
            results_df["q_value"] = tmp["q_value"]
            results_df["fdr_significant"] = tmp["significant"]

            # Update causal status based on FDR-corrected significance
            results_df["causal"] = results_df["fdr_significant"]
            results_df["is_significant"] = results_df["fdr_significant"]

            logger.info(
                f"FDR correction: {results_df['causal'].sum()} relationships remain significant after correction"
            )
        except Exception as e:
            logger.warning(f"FDR application failed: {e}")

    logger.info(
        f"\nCompleted. Found {results_df['causal'].sum()} causal relationships (α={alpha})"
    )

    return results_df
