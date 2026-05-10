"""
Framework Reporting Module

Generates comprehensive analysis reports with:
- Per-experiment summary statistics
- Causal discovery statistics (detection rate, lag distribution)
- Statistical tests (chi-square, binomial)
- Literature alignment (Papagiannopoulou et al. 2017 baseline)
- Publication-ready LaTeX tables and BibTeX references
- Plain-text causal summaries in standardized format
"""

from .summarize import (
    generate_summary_report,
    compare_with_baseline,
    compute_detection_statistics,
    generate_latex_table,
)
from .causal_summary import (
    generate_causal_statement,
    summarize_consensus_edges,
    summarize_method_results,
    generate_full_summary_report,
    classify_strength,
    format_p_value,
)

__all__ = [
    "generate_summary_report",
    "compare_with_baseline",
    "compute_detection_statistics",
    "generate_latex_table",
    # Plain-text causal summaries
    "generate_causal_statement",
    "summarize_consensus_edges",
    "summarize_method_results",
    "generate_full_summary_report",
    "classify_strength",
    "format_p_value",
]
