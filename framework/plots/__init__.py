"""
Framework Plotting Module

Provides publication-quality visualizations for causal discovery results:
- pvalues: P-value distributions across methods
- lags: Lag histograms (in days/weeks)
- graphs: Causal network graphs
- maps: Geographic distribution maps (Europe)
- comparison: Multi-method comparison panels
- correlation: Correlation analysis visualizations
- causal_analysis: Advanced mathematical analyses (spectral, TE flow, FDR, etc.)
"""

from .pvalues import plot_pvalue_distribution, plot_pvalue_comparison
from .lags import plot_lag_histogram, plot_lag_distribution
from .graphs import plot_causal_graph, plot_network_structure
from .maps import plot_europe_map, plot_geographic_causality
from .comparison import plot_method_comparison, plot_agreement_matrix
from .correlation import (
    plot_correlation_matrix,
    plot_correlation_comparison,
    plot_correlation_network,
    plot_all_correlation_visualizations,
)
from .causal_analysis import (
    plot_granger_spectrum,
    plot_transfer_entropy_flow,
    plot_conditional_independence_matrix,
    plot_lag_distribution_analysis,
    plot_bootstrap_uncertainty,
    plot_fdr_diagnostics,
    plot_dag_learning_diagnostics,
)
from .interactive import (
    create_interactive_causal_network,
    create_interactive_lag_explorer,
    create_interactive_dashboard,
)

__all__ = [
    "plot_pvalue_distribution",
    "plot_pvalue_comparison",
    "plot_lag_histogram",
    "plot_lag_distribution",
    "plot_causal_graph",
    "plot_network_structure",
    "plot_europe_map",
    "plot_geographic_causality",
    "plot_method_comparison",
    "plot_agreement_matrix",
    "plot_correlation_matrix",
    "plot_correlation_comparison",
    "plot_correlation_network",
    "plot_all_correlation_visualizations",
    # Advanced causal analysis plots
    "plot_granger_spectrum",
    "plot_transfer_entropy_flow",
    "plot_conditional_independence_matrix",
    "plot_lag_distribution_analysis",
    "plot_bootstrap_uncertainty",
    "plot_fdr_diagnostics",
    "plot_dag_learning_diagnostics",
    # Interactive visualizations
    "create_interactive_causal_network",
    "create_interactive_lag_explorer",
    "create_interactive_dashboard",
]
