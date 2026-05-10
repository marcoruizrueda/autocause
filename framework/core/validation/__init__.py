"""Validation and post-analysis modules for causal discovery results.

Provides bootstrap confidence estimation, robustness analysis, surrogate-based
false positive rate computation, split-half stability testing, cross-site
consensus graphs, effect size extraction, and link classification.
"""

from __future__ import annotations

from framework.core.validation.link_classifier import (
    ClassifiedLinks,
    classify_links,
)
from framework.core.validation.bootstrap import (
    BootstrapResult,
    aggregate_bootstrap,
    run_bootstrap,
)
from framework.core.validation.surrogates import (
    generate_shuffle_surrogates,
    generate_phase_surrogates,
    compute_fpr,
)
from framework.core.validation.stability import (
    compute_jaccard,
    run_split_half_stability,
)
from framework.core.validation.robustness import (
    DEFAULT_DIMENSIONS,
    generate_variants,
    compute_stability,
    classify_link,
    run_robustness,
)
from framework.core.validation.cross_site import (
    compute_pairwise_jaccard,
    compute_consensus_graph,
)
from framework.core.validation.effects import (
    extract_effects,
    rank_drivers,
)

__all__ = [
    # Link classifier
    "ClassifiedLinks",
    "classify_links",
    # Bootstrap
    "BootstrapResult",
    "aggregate_bootstrap",
    "run_bootstrap",
    # Surrogates
    "generate_shuffle_surrogates",
    "generate_phase_surrogates",
    "compute_fpr",
    # Stability
    "compute_jaccard",
    "run_split_half_stability",
    # Robustness
    "DEFAULT_DIMENSIONS",
    "generate_variants",
    "compute_stability",
    "classify_link",
    "run_robustness",
    # Cross-site
    "compute_pairwise_jaccard",
    "compute_consensus_graph",
    # Effects
    "extract_effects",
    "rank_drivers",
]
