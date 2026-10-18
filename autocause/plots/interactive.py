"""
Interactive Causal Visualization

Provides interactive network visualizations of causal relationships using Plotly.
Enables exploration of causal graphs with hover information, filtering, and zooming.

Key Features:
- Interactive node-link diagrams with hover details
- Color-coded by method agreement or strength
- Edge width proportional to causal strength
- Filterable by p-value, lag, or method
- Exportable to HTML for sharing
"""

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import networkx as nx
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Lazy import of plotly (optional dependency)
PLOTLY_AVAILABLE = False
go = None
make_subplots = None

try:
    import plotly.graph_objects as go_module
    from plotly.subplots import make_subplots as make_subplots_func

    go = go_module
    make_subplots = make_subplots_func
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning(
        "Plotly not available. Install with: pip install plotly\n"
        "Interactive visualizations will be disabled."
    )


def create_interactive_causal_network(
    consensus_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    title: str = "Interactive Causal Network",
    color_by: str = "vote_count",
    size_by: str = "n_significant",
    min_votes: int = 1,
    include_lags: bool = True,
    height: int = 800,
    width: int = 1200,
):
    """
    Create an interactive network visualization of causal relationships.

    Parameters:
        consensus_df (pd.DataFrame): Consensus edges with columns:
            - source, target, vote_count, n_significant, best_p_value, lag_days
        output_path (Path): Path to save HTML file (optional)
        title (str): Plot title
        color_by (str): Variable to color edges by ('vote_count', 'best_p_value', 'n_significant')
        size_by (str): Variable to size edges by ('n_significant', 'vote_count')
        min_votes (int): Minimum vote count to display edge
        include_lags (bool): Include lag information in hover text
        height (int): Plot height in pixels
        width (int): Plot width in pixels

    Returns:
        plotly Figure object or None if plotly not available
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create interactive visualization.")
        return None

    if consensus_df is None or len(consensus_df) == 0:
        logger.warning("No consensus edges to visualize")
        return None

    # Filter by minimum votes
    df_filtered = consensus_df[consensus_df["vote_count"] >= min_votes].copy()

    if len(df_filtered) == 0:
        logger.warning(f"No edges with vote_count >= {min_votes}")
        return None

    logger.info(
        f"Creating interactive network with {len(df_filtered)} edges (min_votes={min_votes})"
    )

    # Build NetworkX graph
    G = nx.DiGraph()

    for _, row in df_filtered.iterrows():
        source = row["source"]
        target = row["target"]
        G.add_edge(
            source,
            target,
            vote_count=row["vote_count"],
            n_significant=row.get("n_significant", 0),
            best_p_value=row.get("best_p_value", 1.0),
            lag_days=row.get("lag_days", 0),
            agreeing_methods=row.get("agreeing_methods", ""),
        )

    # Compute layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Extract edge data
    edge_traces = []
    for edge in G.edges(data=True):
        source, target, data = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        # Color mapping - convert values to colors
        if color_by == "vote_count":
            vote_count = data["vote_count"]
            # Map vote counts to colors: 1=red, 2=orange, 3=green
            color_map = {1: "#e74c3c", 2: "#f39c12", 3: "#27ae60"}
            edge_color = color_map.get(vote_count, "#95a5a6")
            color_label = "Votes"
        elif color_by == "best_p_value":
            p_val = data["best_p_value"]
            # Map p-values to colors: lower p-value = darker red
            if p_val < 0.001:
                edge_color = "#8b0000"  # Dark red
            elif p_val < 0.01:
                edge_color = "#c0392b"  # Red
            elif p_val < 0.05:
                edge_color = "#e74c3c"  # Light red
            else:
                edge_color = "#95a5a6"  # Gray
            color_label = "-log10(p)"
        else:  # n_significant
            n_sig = data["n_significant"]
            # Map to blue intensity based on number of significant units
            max_sig = df_filtered["n_significant"].max()
            intensity = int(255 * (1 - n_sig / max(max_sig, 1)))
            edge_color = f"rgb({intensity},{intensity},255)"
            color_label = "Significant Units"

        # Size mapping (for edge line width)
        if size_by == "n_significant":
            line_width = 1 + 10 * (
                data["n_significant"] / df_filtered["n_significant"].max()
            )
        else:  # vote_count
            line_width = 2 + 6 * (data["vote_count"] / df_filtered["vote_count"].max())

        # Hover text
        hover_text = (
            f"{source} → {target}<br>"
            f"Votes: {data['vote_count']}<br>"
            f"P-value: {data['best_p_value']:.2e}<br>"
            f"Significant: {data['n_significant']}<br>"
        )
        if include_lags:
            hover_text += f"Lag: {data['lag_days']} days<br>"
        hover_text += f"Methods: {data['agreeing_methods']}"

        # Create edge trace (arrow)
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(
                width=line_width,
                color=edge_color,
            ),
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_hover = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Count in-degree and out-degree
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        node_hover.append(f"{node}<br>In: {in_deg}, Out: {out_deg}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont=dict(size=14, color="black"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=30,
            color="lightblue",
            line=dict(width=2, color="darkblue"),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=20)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=40, r=40, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Save to HTML if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Interactive plot saved to: {output_path}")

    return fig


def create_interactive_lag_explorer(
    results_df: pd.DataFrame,
    method_name: str,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 1200,
):
    """
    Create an interactive lag distribution explorer.

    Allows filtering by source-target pairs and visualizing lag distributions
    with p-value overlays.

    Parameters:
        results_df (pd.DataFrame): Results with columns: source, target, lag, p_value
        method_name (str): Name of causal method
        output_path (Path): Path to save HTML file (optional)
        title (str): Plot title (auto-generated if None)
        height (int): Plot height in pixels
        width (int): Plot width in pixels

    Returns:
        plotly Figure object or None
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create interactive visualization.")
        return None

    if results_df is None or len(results_df) == 0:
        logger.warning("No results to visualize")
        return None

    # Detect lag column
    lag_col = None
    for col in ["lag_days", "best_lag_days", "lag", "delay", "best_lag"]:
        if col in results_df.columns:
            lag_col = col
            break

    if lag_col is None:
        logger.error("No lag column found in results")
        return None

    # Detect p-value column
    p_value_col = None
    for col in ["p_value", "pval", "best_p_value", "q_value"]:
        if col in results_df.columns:
            p_value_col = col
            break

    # Create subplot with dropdowns for source-target selection
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        subplot_titles=("Lag Distribution", "P-value vs Lag"),
        vertical_spacing=0.12,
    )

    # Get significant edges (check if column exists)
    if "is_significant" in results_df.columns:
        significant = results_df[results_df["is_significant"]]
    elif "significant" in results_df.columns:
        significant = results_df[results_df["significant"]]
    else:
        # Try to infer from p-value
        if p_value_col and p_value_col in results_df.columns:
            significant = results_df[results_df[p_value_col] < 0.05]
        else:
            significant = pd.DataFrame()  # Empty dataframe

    # Import numpy locally
    import numpy as np

    # Overall histogram
    fig.add_trace(
        go.Histogram(
            x=results_df[lag_col],
            nbinsx=30,
            name="All edges",
            marker=dict(color="lightgray", line=dict(color="black", width=1)),
            hovertemplate="Lag: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Significant edges histogram
    if len(significant) > 0:
        fig.add_trace(
            go.Histogram(
                x=significant[lag_col],
                nbinsx=30,
                name="Significant",
                marker=dict(color="red", opacity=0.6),
                hovertemplate="Lag: %{x}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Scatter: p-value vs lag (only if p_value column exists)
    if p_value_col:
        fig.add_trace(
            go.Scatter(
                x=results_df[lag_col],
                y=-np.log10(results_df[p_value_col].clip(lower=1e-10)),
                mode="markers",
                marker=dict(
                    size=6,
                    color=results_df[p_value_col],
                    colorscale="Viridis_r",
                    colorbar=dict(title="P-value", x=1.02),
                    line=dict(width=0.5, color="black"),
                ),
                hovertemplate=(
                    "Lag: %{x}<br>P-value: %{customdata}<br><extra></extra>"
                ),
                customdata=results_df[p_value_col],
                name="Edges",
            ),
            row=2,
            col=1,
        )

        # Add significance threshold line
        alpha = 0.05
        fig.add_hline(
            y=-np.log10(alpha),
            line_dash="dash",
            line_color="red",
            annotation_text=f"α={alpha}",
            row=2,
            col=1,
        )

    # Update layout
    plot_title = title or f"Interactive Lag Explorer: {method_name}"
    fig.update_layout(
        title=dict(text=plot_title, x=0.5, xanchor="center", font=dict(size=18)),
        showlegend=True,
        height=height,
        width=width,
        hovermode="closest",
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(title_text="Lag (days)", row=1, col=1)
    fig.update_xaxes(title_text="Lag (days)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    if p_value_col:
        fig.update_yaxes(title_text="-log10(p-value)", row=2, col=1)
    else:
        fig.update_yaxes(title_text="No p-values available", row=2, col=1)

    # Save to HTML if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        logger.info(f"Interactive lag explorer saved to: {output_path}")

    return fig


def create_interactive_dashboard(
    consensus_df: pd.DataFrame,
    results_dict: Dict[str, pd.DataFrame],
    output_dir: Path,
    experiment_name: str = "Causal Discovery",
) -> Dict[str, Path]:
    """
    Create a complete interactive dashboard with multiple views.

    Generates:
    - Main causal network (consensus)
    - Lag explorer per method
    - Method comparison matrix

    Parameters:
        consensus_df (pd.DataFrame): Consensus edges
        results_dict (Dict): Results per method {method_name: results_df}
        output_dir (Path): Directory to save HTML files
        experiment_name (str): Experiment name for titles

    Returns:
        Dict[str, Path]: Mapping of visualization name to file path
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Cannot create interactive dashboard.")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # 1. Main causal network
    logger.info("Creating interactive causal network...")
    network_path = output_dir / "interactive_network.html"
    fig = create_interactive_causal_network(
        consensus_df,
        output_path=network_path,
        title=f"{experiment_name}: Consensus Causal Network",
    )
    if fig:
        saved_files["network"] = network_path

    # 2. Lag explorers per method
    for method_name, results_df in results_dict.items():
        if results_df is None or len(results_df) == 0:
            continue

        logger.info(f"Creating lag explorer for {method_name}...")
        lag_path = output_dir / f"interactive_lags_{method_name.lower()}.html"
        fig = create_interactive_lag_explorer(
            results_df,
            method_name=method_name,
            output_path=lag_path,
            title=f"{experiment_name}: {method_name} Lag Distribution",
        )
        if fig:
            saved_files[f"lags_{method_name}"] = lag_path

    logger.info(f"Interactive dashboard saved to: {output_dir}")
    logger.info(f"Generated {len(saved_files)} visualizations")

    # 3. Create a combined index HTML page
    index_path = output_dir / "index.html"
    _create_dashboard_index(saved_files, index_path, experiment_name)
    saved_files["index"] = index_path

    return saved_files


def _create_dashboard_index(
    saved_files: Dict[str, Path],
    index_path: Path,
    experiment_name: str,
):
    """Create an index.html page that displays all visualizations in a clean layout."""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{experiment_name} - Interactive Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.95;
        }}
        
        .nav-tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
        }}
        
        .nav-tab {{
            padding: 20px 30px;
            cursor: pointer;
            border: none;
            background: transparent;
            font-size: 1em;
            font-weight: 600;
            color: #495057;
            transition: all 0.3s ease;
            white-space: nowrap;
            border-bottom: 3px solid transparent;
        }}
        
        .nav-tab:hover {{
            background: #e9ecef;
            color: #667eea;
        }}
        
        .nav-tab.active {{
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }}
        
        .content {{
            padding: 0;
        }}
        
        .tab-pane {{
            display: none;
            min-height: 600px;
        }}
        
        .tab-pane.active {{
            display: block;
        }}
        
        .iframe-container {{
            width: 100%;
            height: 100%;
            min-height: 800px;
            border: none;
            background: white;
        }}
        
        iframe {{
            width: 100%;
            height: 100%;
            min-height: 800px;
            border: none;
        }}
        
        .info-section {{
            padding: 40px;
            background: #f8f9fa;
        }}
        
        .info-card {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        
        .info-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        
        .info-card ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .info-card li {{
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .info-card li:last-child {{
            border-bottom: none;
        }}
        
        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge-network {{
            background: #28a745;
            color: white;
        }}
        
        .badge-lag {{
            background: #17a2b8;
            color: white;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .nav-tab {{
                padding: 15px 20px;
                font-size: 0.9em;
            }}
            
            iframe {{
                min-height: 600px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 {experiment_name}</h1>
            <p>Interactive Causal Discovery Dashboard</p>
        </div>
        
        <div class="nav-tabs" id="navTabs">
"""

    # Add navigation tabs
    tab_names = {
        "network": ("📊 Causal Network", "network"),
    }

    for key, path in saved_files.items():
        if key == "index":
            continue
        if key.startswith("lags_"):
            method = key.replace("lags_", "")
            tab_names[key] = (f"📈 {method} Lags", "lag")
        elif key not in tab_names:
            tab_names[key] = (f"📊 {key.title()}", "network")

    for idx, (key, (name, badge_type)) in enumerate(tab_names.items()):
        if key in saved_files and key != "index":
            active = "active" if idx == 0 else ""
            html_content += f"""            <button class="nav-tab {active}" onclick="showTab('{key}')">{name}</button>\n"""

    html_content += """        </div>
        
        <div class="content">
"""

    # Add tab panes with iframes
    for idx, key in enumerate(saved_files.keys()):
        if key == "index":
            continue
        active = "active" if idx == 0 else ""
        rel_path = saved_files[key].name
        html_content += f"""            <div id="{key}" class="tab-pane {active}">
                <iframe src="{rel_path}" class="iframe-container"></iframe>
            </div>
"""

    html_content += """        </div>
        
        <div class="info-section">
            <div class="info-card">
                <h3>📖 About This Dashboard</h3>
                <ul>
                    <li><strong>Causal Network:</strong> Interactive visualization of consensus causal relationships</li>
                    <li><strong>Lag Explorers:</strong> Distribution of time lags for each causal discovery method</li>
                    <li><strong>Interactivity:</strong> Hover over elements, zoom, pan, and explore the data</li>
                </ul>
            </div>
            
            <div class="info-card">
                <h3>🔍 How to Use</h3>
                <ul>
                    <li><strong>Switch Tabs:</strong> Click on the tabs above to view different visualizations</li>
                    <li><strong>Zoom:</strong> Use mouse wheel or pinch to zoom in/out on plots</li>
                    <li><strong>Pan:</strong> Click and drag to move around the visualization</li>
                    <li><strong>Hover:</strong> Move your mouse over edges and nodes to see detailed information</li>
                </ul>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {{
            // Hide all tabs
            const panes = document.querySelectorAll('.tab-pane');
            panes.forEach(pane => pane.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.nav-tab');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            const selectedPane = document.getElementById(tabId);
            if (selectedPane) {{
                selectedPane.classList.add('active');
            }}
            
            // Highlight selected button
            const selectedButton = event.target;
            selectedButton.classList.add('active');
        }}
    </script>
</body>
</html>
"""

    # Write the index file
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Dashboard index created: {index_path}")
