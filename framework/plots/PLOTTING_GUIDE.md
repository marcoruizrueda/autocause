# Advanced Causal Discovery Plotting Guide

## Overview

The framework now includes publication-quality visualizations and mathematical analyses typical for causal inference research. All plotting code is in `framework/plots/` (dataset-agnostic) and can be used with any experiment.

## New Plotting Capabilities

### 1. Enhanced Causal Network Graphs (`graphs.py`)

**Improvements:**
- 🎨 **Node coloring by role**: Orange (sources), Blue (sinks), Green (intermediates)
- 📍 **Multiple layout algorithms**: Spring, circular, hierarchical, Kamada-Kawai
- 🔄 **Curved edges** for bidirectional relationships
- 📝 **Background boxes** for node/edge labels (better readability)
- 🎨 **Subtle background** and enhanced styling
- 💾 **Dual export**: PNG (high-res) + SVG (editable)

**Usage:**
```python
from framework.plots import plot_causal_graph

plot_causal_graph(
    results_df=df,
    method="Granger",
    output_path="causal_graph.svg",
    layout="spring",  # or 'circular', 'hierarchical', 'kamada_kawai'
    figsize=(16, 14),
    node_size=3500,
    edge_alpha=0.85,
    show_legend=True
)
```

### 2. Granger Causality Spectral Analysis (`causal_analysis.py`)

**What it shows:**
- Time series plots (cause and effect)
- Cross-correlation function (temporal dependencies)
- Power spectral density (frequency content)
- Coherence (frequency-domain correlation)

**Why it matters:** Reveals at which frequencies X Granger-causes Y, providing insight into timescale-specific relationships (e.g., seasonal vs daily effects).

**Usage:**
```python
from framework.plots import plot_granger_spectrum

plot_granger_spectrum(
    X=precipitation_timeseries,
    Y=ndvi_timeseries,
    max_lag=12,
    output_path="granger_spectrum.png",
    figsize=(16, 10)
)
```

### 3. Transfer Entropy Information Flow (`causal_analysis.py`)

**What it shows:**
- Information flow network with edge widths ∝ TE value
- Node sizes ∝ in-degree (information reception)
- Color gradient indicating TE strength
- Quantitative TE values on strong connections

**Why it matters:** Transfer entropy quantifies directed information transfer in bits, making it ideal for visualizing information flow networks.

**Usage:**
```python
from framework.plots import plot_transfer_entropy_flow

plot_transfer_entropy_flow(
    results_df=te_results,
    output_path="te_flow.png",
    figsize=(14, 12)
)
```

### 4. Conditional Independence Matrix (`causal_analysis.py`)

**What it shows:**
- P-value matrix (log scale) with significance markers (*, **)
- Binary adjacency matrix (significant edges)
- Which variable pairs are conditionally independent

**Why it matters:** Fundamental for structure learning - shows which relationships persist after conditioning on other variables.

**Usage:**
```python
from framework.plots import plot_conditional_independence_matrix

plot_conditional_independence_matrix(
    results_df=pcmci_results,
    method="PCMCI+",
    output_path="ci_matrix.png",
    figsize=(12, 10)
)
```

### 5. Comprehensive Lag Distribution Analysis (`causal_analysis.py`)

**What it shows:**
- (A) Overall lag histogram with mean/median
- (B) Lag vs p-value scatter (colored by lag)
- (C) Per-method lag boxplot
- (D) Temporal clustering (lag differences histogram)
- (E) Cumulative distribution of lags

**Why it matters:** Reveals temporal patterns, identifies clustering, shows method-specific biases in lag detection.

**Usage:**
```python
from framework.plots import plot_lag_distribution_analysis

plot_lag_distribution_analysis(
    results_df=combined_results,  # All methods combined
    output_path="lag_analysis.png",
    figsize=(16, 12)
)
```

### 6. Bootstrap Uncertainty Quantification (`causal_analysis.py`)

**What it shows:**
- Bootstrap distribution histogram
- Point estimate with confidence intervals
- Forest plots (horizontal error bars)

**Why it matters:** Quantifies uncertainty in lag estimates via resampling, critical for assessing reliability.

**Usage:**
```python
from framework.plots import plot_bootstrap_uncertainty

plot_bootstrap_uncertainty(
    bootstrap_results={
        'RR->NDVI': {
            'point_estimate': 10,
            'ci_lower': 1,
            'ci_upper': 10.7,
            'bootstrap_samples': [1, 10, 9, ...]
        }
    },
    output_path="bootstrap_ci.png",
    figsize=(14, 8)
)
```

### 7. FDR Diagnostic Plots (`causal_analysis.py`)

**What it shows:**
- (A) P-value histogram (should be uniform under null)
- (B) Q-Q plot (observed vs theoretical quantiles)
- (C) Benjamini-Hochberg procedure threshold line
- (D) Q-value vs P-value relationship

**Why it matters:** Validates multiple testing correction, detects violations of assumptions, shows impact of FDR control.

**Usage:**
```python
from framework.plots import plot_fdr_diagnostics

plot_fdr_diagnostics(
    p_values=results_df['p_value'].values,
    q_values=results_df['q_value'].values,
    alpha=0.05,
    output_path="fdr_diagnostics.png",
    figsize=(14, 10)
)
```

### 8. DAG Structure Learning Diagnostics (`causal_analysis.py`)

**What it shows:**
- (A) Degree distribution (in-degree and out-degree)
- (B) If ground truth available: TP/FP/FN counts, precision/recall/F1
- (B) If no ground truth: DAG statistics (nodes, edges, is_DAG)

**Why it matters:** Evaluates structure learning performance, identifies highly connected nodes (hubs).

**Usage:**
```python
from framework.plots import plot_dag_learning_diagnostics

plot_dag_learning_diagnostics(
    results_df=results,
    true_dag={'('RR', 'NDVI')': True, ...},  # Optional ground truth
    output_path="dag_diagnostics.png",
    figsize=(16, 6)
)
```

## Complete Example: Generate All Advanced Plots

Use the provided script to generate all advanced visualizations:

```bash
# Generate all plot types for an experiment
cd experiments/earthnet
python generate_advanced_plots.py --exp exp2_mediterranean --all

# Or generate specific plot types
python generate_advanced_plots.py --exp exp1_all_data --graphs --lag-analysis --fdr

# Available flags:
#   --all          Generate all plot types
#   --graphs       Enhanced causal network graphs (3 layouts)
#   --spectral     Granger spectral analysis
#   --te-flow      Transfer entropy flow diagrams
#   --ci-matrix    Conditional independence matrix
#   --lag-analysis Comprehensive lag distribution analysis
#   --bootstrap    Bootstrap confidence intervals
#   --fdr          FDR diagnostic plots
#   --dag          DAG structure learning diagnostics
```

## Output Structure

Plots are organized in subdirectories:

```
experiments/earthnet/exp2_mediterranean/figures/advanced_plots/
├── causal_graphs/
│   ├── granger_spring_graph.svg
│   ├── granger_circular_graph.svg
│   ├── pcmci_kamada_kawai_graph.png
│   └── ...
├── spectral_analysis/
│   ├── spectrum_RR_NDVI_unit1.png
│   └── ...
├── te_flow/
│   └── te_information_flow.png
├── ci_analysis/
│   └── conditional_independence_matrix.png
├── lag_analysis/
│   └── comprehensive_lag_analysis.png
├── bootstrap/
│   └── bootstrap_confidence_intervals.png
├── fdr_diagnostics/
│   ├── fdr_diagnostics_granger.png
│   └── ...
└── dag_diagnostics/
    ├── dag_diagnostics_granger.png
    └── ...
```

## Mathematical Background

### Granger Spectral Analysis
- **Cross-correlation**: Measures linear predictability at different lags
- **Power Spectral Density**: Decomposes variance across frequencies
- **Coherence**: $$\text{Coherence}(f) = \frac{|P_{XY}(f)|^2}{P_{XX}(f) \cdot P_{YY}(f)}$$

### Transfer Entropy
- **Definition**: $$TE_{X \to Y} = \sum p(y_t, y_{t-1}, x_{t-1}) \log \frac{p(y_t | y_{t-1}, x_{t-1})}{p(y_t | y_{t-1})}$$
- **Interpretation**: Information in bits transferred from X to Y
- **Properties**: Non-negative, asymmetric, model-free

### FDR Control (Benjamini-Hochberg)
- **Procedure**: Sort p-values $p_{(1)} \leq \cdots \leq p_{(m)}$
- **Threshold**: Reject all $p_{(i)} \leq \frac{i}{m} \alpha$
- **Guarantee**: $\mathbb{E}[\text{FDR}] \leq \alpha$ under independence

### Bootstrap Confidence Intervals
- **Block Bootstrap**: Preserves temporal structure via blocks of size $\sqrt{n}$
- **Percentile CI**: $[\hat{\theta}_{\alpha/2}, \hat{\theta}_{1-\alpha/2}]$ from bootstrap distribution
- **Bias-corrected**: Adjusts for bias in point estimate

## Integration with Existing Workflow

All new plotting functions integrate seamlessly with existing code:

```python
# In your analysis script
from framework.plots import (
    plot_causal_graph,
    plot_granger_spectrum,
    plot_lag_distribution_analysis,
    plot_fdr_diagnostics
)

# Load results
results_df = pd.read_csv('results_granger.csv')

# Generate plots
plot_causal_graph(results_df, method='Granger', output_path='graph.svg')
plot_lag_distribution_analysis(results_df, output_path='lags.png')
plot_fdr_diagnostics(results_df['p_value'], results_df['q_value'], output_path='fdr.png')
```

## Best Practices

1. **Always generate FDR diagnostics** - Validate multiple testing correction
2. **Use spectral analysis** for time series with strong temporal patterns
3. **Generate bootstrap CIs** for final reported lags (uncertainty quantification)
4. **Compare layouts** for causal graphs - different layouts reveal different structures
5. **Export both PNG and SVG** - PNG for papers, SVG for editing

## Publication Quality

All plots are designed for publication with:
- **High DPI** (300 dpi for PNG exports)
- **Vector graphics** (SVG for scalability)
- **Professional styling** (clear labels, legends, colors)
- **Mathematical notation** (LaTeX-style where appropriate)
- **Comprehensive legends** (self-contained figures)

## Dependencies

New plots require:
- `matplotlib` (already installed)
- `networkx` (for graph visualizations)
- `scipy` (for spectral analysis)
- `numpy` and `pandas` (already installed)

Install missing dependencies:
```bash
pip install networkx scipy
```
