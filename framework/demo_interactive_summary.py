"""
Demo: Interactive Causal Visualization and Plain-Text Summaries

Shows how to use the new interactive visualization and summary generation features.

Usage:
    python framework/demo_interactive_summary.py /path/to/experiment/results

    The experiment directory should contain:
    - consensus.csv
    - results_granger.csv (optional)
    - results_pcmci.csv (optional)
"""

import pandas as pd
from pathlib import Path
import sys

# Import framework modules
from framework.plots import (
    create_interactive_causal_network,
    create_interactive_lag_explorer,
    create_interactive_dashboard,
)
from framework.reporting import (
    generate_causal_statement,
    summarize_consensus_edges,
    summarize_method_results,
    generate_full_summary_report,
)

# Example 1: Generate plain-text causal statements
print("=" * 80)
print("EXAMPLE 1: Plain-Text Causal Statements")
print("=" * 80)

# Single statement
statement = generate_causal_statement(
    source="var1",
    target="var2",
    lag_days=35,
    p_value=0.0001,
    strength=0.65,
    strength_metric="correlation",
    n_units=125,
    n_significant=91,
)
print("\nSingle causal relationship:")
print(statement)

# Example 2: Summarize consensus edges
print("\n\n" + "=" * 80)
print("EXAMPLE 2: Summarize Consensus Edges")
print("=" * 80)

# Get experiment directory from command line argument or use current directory
if len(sys.argv) > 1:
    exp_dir = Path(sys.argv[1])
else:
    exp_dir = Path.cwd()
    print(f"\nNo experiment directory provided. Using current directory: {exp_dir}")
    print("Usage: python demo_interactive_summary.py /path/to/experiment/results\n")

if (exp_dir / "consensus.csv").exists():
    consensus_df = pd.read_csv(exp_dir / "consensus.csv")

    print("\nConsensus edges from experiment:")
    summaries = summarize_consensus_edges(consensus_df, alpha=0.05)
    for summary in summaries[:5]:  # Show first 5
        print(f"\n{summary}")
else:
    print("\nNo consensus.csv found. Run experiment first.")

# Example 3: Generate full summary report
print("\n\n" + "=" * 80)
print("EXAMPLE 3: Full Summary Report")
print("=" * 80)

if (exp_dir / "consensus.csv").exists():
    # Load all results
    results_dict = {}

    if (exp_dir / "results_granger.csv").exists():
        results_dict["Granger"] = pd.read_csv(exp_dir / "results_granger.csv")

    if (exp_dir / "results_pcmci.csv").exists():
        results_dict["PCMCI+"] = pd.read_csv(exp_dir / "results_pcmci.csv")

    # Generate report
    output_path = exp_dir / "causal_summary.txt"
    report = generate_full_summary_report(
        consensus_df=consensus_df,
        results_dict=results_dict,
        output_path=output_path,
        experiment_name="Your Experiment",
        alpha=0.05,
        top_n_per_method=5,
    )

    print(f"\n✅ Full summary report saved to: {output_path}")
    print("\nFirst 1000 characters of report:")
    print("-" * 80)
    print(report[:1000] + "...")

# Example 4: Create interactive visualizations
print("\n\n" + "=" * 80)
print("EXAMPLE 4: Interactive Visualizations")
print("=" * 80)

try:
    import plotly

    if (exp_dir / "consensus.csv").exists():
        # Create interactive network
        output_dir = exp_dir / "figures" / "interactive"
        output_dir.mkdir(parents=True, exist_ok=True)

        network_path = output_dir / "causal_network.html"
        fig = create_interactive_causal_network(
            consensus_df=consensus_df,
            output_path=network_path,
            title="Causal Network",
            min_votes=2,
        )

        if fig:
            print(f"\n✅ Interactive network saved to: {network_path}")
            print("   Open in browser to explore!")

        # Create lag explorer for Granger
        if "Granger" in results_dict:
            lag_path = output_dir / "lag_explorer_granger.html"
            fig = create_interactive_lag_explorer(
                results_df=results_dict["Granger"],
                method_name="Granger",
                output_path=lag_path,
            )

            if fig:
                print(f"✅ Lag explorer saved to: {lag_path}")

        # Create full dashboard
        dashboard_files = create_interactive_dashboard(
            consensus_df=consensus_df,
            results_dict=results_dict,
            output_dir=output_dir,
            experiment_name="Your Experiment",
        )

        print("\n✅ Complete interactive dashboard created!")
        print(f"   Files generated: {len(dashboard_files)}")
        for name, path in dashboard_files.items():
            print(f"   - {name}: {path.name}")

except ImportError:
    print("\n⚠️  Plotly not installed. Install with: pip install plotly")
    print("   Interactive visualizations require plotly.")

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
