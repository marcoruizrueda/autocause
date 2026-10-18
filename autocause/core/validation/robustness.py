"""Robustness analysis via multi-dimensional sensitivity testing.

Generates variant configurations by toggling analysis dimensions one at a time,
runs causal discovery under each variant, and classifies links by their detection
stability across variants.
"""

from __future__ import annotations

import copy
import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS: dict[str, list] = {
    "detrend": [True, False],
    "season": ["full_year", "growing_season"],
    "ci_test": ["parcorr", "robust_parcorr"],
    "tau_max": [7, 14],
}


def generate_variants(
    base_config: dict,
    dimensions: dict[str, list] | None = None,
) -> list[dict]:
    """Generate variant configurations using a one-at-a-time design.

    For each dimension, creates a variant by setting that dimension to each
    non-baseline value. Also includes the baseline itself and one combined
    variant toggling all dimensions simultaneously.

    Parameters
    ----------
    base_config : dict
        Baseline configuration values.
    dimensions : dict[str, list] | None
        Dimensions to vary. Each key maps to a list of possible values.
        Defaults to DEFAULT_DIMENSIONS.

    Returns
    -------
    list[dict]
        Each dict is a complete variant configuration with a 'label' key.
        Guaranteed to contain at least 6 distinct configurations when
        dimensions has >= 2 keys each with >= 2 levels.
    """
    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS

    variants: list[dict] = []
    seen_labels: set[str] = set()

    # Baseline variant
    baseline = copy.deepcopy(base_config)
    baseline["label"] = "baseline"
    variants.append(baseline)
    seen_labels.add("baseline")

    # One-at-a-time variants: toggle each dimension to each alternative value
    for dim_name, levels in dimensions.items():
        baseline_value = base_config.get(dim_name, levels[0])
        for level in levels:
            if level == baseline_value:
                continue
            variant = copy.deepcopy(base_config)
            variant[dim_name] = level
            label = f"{dim_name}={level}"
            if label not in seen_labels:
                variant["label"] = label
                variants.append(variant)
                seen_labels.add(label)

    # Combined variant: set all dimensions to their last non-baseline value
    combined = copy.deepcopy(base_config)
    combined["label"] = "combined"
    for dim_name, levels in dimensions.items():
        baseline_value = base_config.get(dim_name, levels[0])
        alternatives = [lv for lv in levels if lv != baseline_value]
        if alternatives:
            combined[dim_name] = alternatives[-1]
    if "combined" not in seen_labels:
        variants.append(combined)
        seen_labels.add("combined")

    # If we still have fewer than 6, add pairwise combinations
    if len(variants) < 6:
        dim_names = list(dimensions.keys())
        for idx_a in range(len(dim_names)):
            for idx_b in range(idx_a + 1, len(dim_names)):
                if len(variants) >= 6:
                    break
                dim_a = dim_names[idx_a]
                dim_b = dim_names[idx_b]
                variant = copy.deepcopy(base_config)
                baseline_a = base_config.get(dim_a, dimensions[dim_a][0])
                baseline_b = base_config.get(dim_b, dimensions[dim_b][0])
                alt_a = [lv for lv in dimensions[dim_a] if lv != baseline_a]
                alt_b = [lv for lv in dimensions[dim_b] if lv != baseline_b]
                if alt_a:
                    variant[dim_a] = alt_a[0]
                if alt_b:
                    variant[dim_b] = alt_b[0]
                label = f"{dim_a}+{dim_b}"
                if label not in seen_labels:
                    variant["label"] = label
                    variants.append(variant)
                    seen_labels.add(label)

    return variants


def compute_stability(detection_arrays: list[np.ndarray]) -> np.ndarray:
    """Compute per-link stability as fraction of arrays detecting each link.

    Parameters
    ----------
    detection_arrays : list[np.ndarray]
        Each array is boolean, shape (N, N, tau_max+1).

    Returns
    -------
    np.ndarray
        Stability fractions, shape (N, N, tau_max+1), values in [0, 1].
    """
    if not detection_arrays:
        raise ValueError("detection_arrays must not be empty.")
    stacked = np.stack(detection_arrays, axis=0).astype(float)
    return stacked.mean(axis=0)


def classify_link(stability: float) -> str:
    """Classify a link by its stability fraction.

    Parameters
    ----------
    stability : float
        Detection fraction in [0, 1].

    Returns
    -------
    str
        "robust" if >= 0.8, "moderate" if in [0.5, 0.8), "fragile" if < 0.5.
    """
    if stability >= 0.8:
        return "robust"
    elif stability >= 0.5:
        return "moderate"
    else:
        return "fragile"


def run_robustness(
    data: np.ndarray,
    var_names: list[str],
    base_config: dict,
    dimensions: dict[str, list] | None = None,
    preprocess_fn: Callable | None = None,
    causal_discovery_fn: Callable | None = None,
) -> dict:
    """Full robustness pipeline.

    Generates variant configurations, runs causal discovery under each,
    and computes per-link stability and classification.

    Parameters
    ----------
    data : np.ndarray
        Shape (T, N) multivariate time series.
    var_names : list[str]
        Variable names.
    base_config : dict
        Baseline configuration for causal discovery.
    dimensions : dict[str, list] | None
        Dimensions to vary. Defaults to DEFAULT_DIMENSIONS.
    preprocess_fn : Callable | None
        Optional function (data, variant_config) -> preprocessed_data.
    causal_discovery_fn : Callable | None
        Function with signature (data, var_names, **kwargs) returning a result
        with a .graph attribute or a tuple (graph, val_matrix).

    Returns
    -------
    dict
        Keys: variants, stability, classification, n_successful, n_failed.
    """
    if causal_discovery_fn is None:
        raise NotImplementedError(
            "No default causal discovery function available. "
            "Pass causal_discovery_fn explicitly."
        )

    variants = generate_variants(base_config, dimensions)
    detection_arrays: list[np.ndarray] = []
    n_failed = 0

    for variant in variants:
        try:
            # Preprocess if function provided
            if preprocess_fn is not None:
                processed_data = preprocess_fn(data, variant)
            else:
                processed_data = data

            # Remove label key before passing to discovery function
            kwargs = {k: v for k, v in variant.items() if k != "label"}
            result = causal_discovery_fn(processed_data, var_names, **kwargs)

            if isinstance(result, tuple):
                graph = result[0]
            else:
                graph = result.graph

            # Convert to boolean detection array
            detection = np.array(
                [
                    [
                        [
                            graph[i, j, tau] != ""
                            and str(graph[i, j, tau]).strip() != ""
                            for tau in range(graph.shape[2])
                        ]
                        for j in range(graph.shape[1])
                    ]
                    for i in range(graph.shape[0])
                ],
                dtype=bool,
            )
            detection_arrays.append(detection)

        except Exception as e:
            logger.warning(f"Variant '{variant.get('label', '?')}' failed: {e}")
            n_failed += 1
            continue

    n_successful = len(detection_arrays)

    if n_successful == 0:
        logger.error("All robustness variants failed.")
        return {
            "variants": variants,
            "stability": np.zeros((1, 1, 1)),
            "classification": pd.DataFrame(
                columns=["parent", "target", "lag", "stability", "class"]
            ),
            "n_successful": 0,
            "n_failed": n_failed,
        }

    stability = compute_stability(detection_arrays)

    # Build classification DataFrame for links with nonzero stability
    rows: list[dict] = []
    N = stability.shape[0]
    tau_dim = stability.shape[2]
    for i in range(N):
        for j in range(N):
            for tau in range(tau_dim):
                s = stability[i, j, tau]
                if s > 0:
                    rows.append(
                        {
                            "parent": var_names[i] if i < len(var_names) else str(i),
                            "target": var_names[j] if j < len(var_names) else str(j),
                            "lag": tau,
                            "stability": s,
                            "class": classify_link(s),
                        }
                    )

    classification = pd.DataFrame(
        rows, columns=["parent", "target", "lag", "stability", "class"]
    )

    return {
        "variants": variants,
        "stability": stability,
        "classification": classification,
        "n_successful": n_successful,
        "n_failed": n_failed,
    }
