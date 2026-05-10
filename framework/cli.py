"""
Command-Line Interface for Causal Discovery Framework

Provides easy-to-use CLI for running causal analysis pipelines,
generating visualizations, and producing reports.

Usage:
    python -m framework.cli run --input data.csv --methods granger te pcmci consensus
    python -m framework.cli plot --results results.csv --output plots/
    python -m framework.cli report --experiment_name "Exp1" --results_dir results/
"""

import argparse
import logging
import json
from pathlib import Path
import sys
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Mitigate long Numba JIT compile stalls (e.g., from optional tigramite/numba paths)
# Set before any potential imports of libraries that use numba.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_LOG_LEVEL", "WARNING")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_command(args) -> int:
    """Execute causal discovery run."""
    logger.info(f"Running causal analysis on: {args.input}")

    # Import here to avoid circular imports
    from framework.core.io import load_timeseries_csv
    from framework.core.qc import generate_qc_report
    from framework.core.methods import (
        granger,
        transfer_entropy,
        tigramite_pcmci,
        consensus,
    )
    from framework.core.decision import decide_and_run as auto_decide_and_run

    # Load data
    try:
        df, _meta = load_timeseries_csv(args.input)
        logger.info(f"Loaded data: {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Quality control
    qc_report = generate_qc_report(df)
    logger.info(f"QC Report: {qc_report}")

    # Get variable pairs
    if args.pairs:
        pairs = json.loads(args.pairs)
    else:
        # All combinations
        vars_list = list(df.columns)
        pairs = [
            (vars_list[i], vars_list[j])
            for i in range(len(vars_list))
            for j in range(len(vars_list))
            if i != j
        ]

    logger.info(f"Testing {len(pairs)} variable pairs")

    results_dict = {}

    # Run selected methods
    if "granger" in args.methods:
        logger.info("Running Granger causality...")
        try:
            results_dict["granger"] = granger.batch_granger_causality(
                df, pairs, maxlag=args.maxlag, alpha=args.alpha
            )
        except Exception as e:
            logger.error(f"Granger failed: {e}")

    if "te" in args.methods:
        logger.info("Running Transfer Entropy...")
        try:
            results_dict["transfer_entropy"] = transfer_entropy.batch_transfer_entropy(
                df, pairs, delays=[1, 2, 3], alpha=args.alpha, method="discrete"
            )
        except Exception as e:
            logger.error(f"Transfer Entropy failed: {e}")

    if "pcmci" in args.methods:
        logger.info("Running PCMCI+...")
        try:
            results_dict["pcmci"] = tigramite_pcmci.batch_pcmci(
                df, pairs, tau_max=args.maxlag, alpha=args.alpha
            )
        except Exception as e:
            logger.error(f"PCMCI+ failed: {e}")

    if "auto" in args.methods:
        logger.info("Running AUTO decision engine...")
        try:
            auto_df = auto_decide_and_run(
                df, pairs, window_days=args.maxlag * 5, alpha=args.alpha
            )
            results_dict["auto"] = auto_df
            # Also split by method for downstream compatibility
            if (
                auto_df is not None
                and not auto_df.empty
                and "method" in auto_df.columns
            ):
                g_df = auto_df[auto_df["method"] == "Granger"]
                te_df = auto_df[auto_df["method"] == "TransferEntropy"]
                pc_df = auto_df[auto_df["method"] == "PCMCI+"]
                if not g_df.empty:
                    results_dict["granger"] = g_df
                if not te_df.empty:
                    results_dict["transfer_entropy"] = te_df
                if not pc_df.empty:
                    results_dict["pcmci"] = pc_df
        except Exception as e:
            logger.error(f"AUTO failed: {e}")

    # Save results
    output_dir = Path(args.output) if args.output else Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Consensus if requested
    if "consensus" in args.methods and len(results_dict) > 1:
        logger.info("Computing consensus...")
        try:
            consensus_result = consensus.merge_method_results(
                results_dict.get("granger"),
                results_dict.get("transfer_entropy"),
                results_dict.get("pcmci"),
                min_votes=getattr(args, "min_votes", 2),
                lag_tolerance_steps=getattr(args, "lag_tolerance_steps", 1),
                sampling_days=getattr(args, "sampling_days", 1),
                alpha=args.alpha,
                output_dir=str(output_dir),
            )
            results_dict["consensus"] = consensus_result["consensus_edges"]
            logger.info(
                f"✅ Consensus: {len(consensus_result['consensus_edges'])} edges "
                f"(saved CSV, GraphML, report)"
            )
        except Exception as e:
            logger.error(f"Consensus failed: {e}")

    # Save results

    for method, results in results_dict.items():
        if results is not None:
            output_file = output_dir / f"{method}.csv"
            results.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")

    logger.info(f"✅ Analysis complete. Results saved to: {output_dir}")
    return 0


def plot_command(args) -> int:
    """Generate visualizations."""
    logger.info(f"Generating plots from: {args.results}")

    from framework.plots import (
        plot_pvalue_distribution,
        plot_lag_histogram,
        plot_causal_graph,
        plot_method_comparison,
    )

    output_dir = Path(args.output) if args.output else Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    methods = ["granger", "transfer_entropy", "pcmci", "consensus"]
    results_dict = {}

    for method in methods:
        result_file = (
            Path(args.results) / f"{method}.csv"
            if Path(args.results).is_dir()
            else Path(args.results)
        )
        if result_file.exists():
            results_dict[method] = pd.read_csv(result_file)
            logger.info(f"Loaded: {result_file}")

    if not results_dict:
        logger.warning("No result files found")
        return 1

    # Generate plots
    logger.info("Generating p-value distributions...")
    for method, df in results_dict.items():
        plot_pvalue_distribution(
            df, method=method.upper(), output_path=output_dir / f"pvalues_{method}"
        )

    logger.info("Generating lag histograms...")
    for method, df in results_dict.items():
        lag_col = (
            "delay" if "delay" in df.columns else "lag" if "lag" in df.columns else None
        )
        if lag_col:
            plot_lag_histogram(
                df,
                method=method.upper(),
                lag_column=lag_col,
                output_path=output_dir / f"lags_{method}",
            )

    logger.info("Generating causal graphs...")
    for method, df in results_dict.items():
        if "source" in df.columns and "target" in df.columns:
            plot_causal_graph(
                df, method=method.upper(), output_path=output_dir / f"graph_{method}"
            )

    logger.info("Generating method comparison...")
    plot_method_comparison(results_dict, output_path=output_dir / "method_comparison")

    logger.info(f"✅ Plots saved to: {output_dir}")
    return 0


def diagnose_command(args) -> int:
    """Run diagnostic checks on data or results."""
    logger.info(f"Running diagnostics: {args.experiment_name}")

    from framework.core.io import load_csv
    from framework.core.diagnostics import (
        Diagnostics,
        permutation_negative_control,
        reverse_direction_check,
        method_divergence_alert,
    )
    from framework.plots import (
        plot_pvalue_distribution,
        plot_lag_histogram,
        plot_method_comparison,
    )

    output_dir = Path(args.output) if args.output else Path("diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Load results ===
    results_dict = {}
    if args.results:
        results_path = Path(args.results)
        if results_path.is_dir():
            for csv_file in results_path.glob("*.csv"):
                method = csv_file.stem
                results_dict[method] = pd.read_csv(csv_file)
                logger.info(f"Loaded: {csv_file}")
        elif results_path.is_file():
            results_dict["method"] = pd.read_csv(results_path)
            logger.info(f"Loaded: {results_path}")

    if not results_dict:
        logger.warning("No result files found")
        return 1

    # === Run standard diagnostics ===
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING STANDARD DIAGNOSTICS")
    logger.info("=" * 70)

    all_edges = pd.concat(results_dict.values(), ignore_index=True)
    diag = Diagnostics(all_edges, cadence_days=args.cadence_days)
    diag.run_all()

    # Save standard diagnostics
    diag.save_results(output_dir)

    # === Generate plots ===
    if args.plots:
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING DIAGNOSTIC PLOTS")
        logger.info("=" * 70)

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # P-value distributions per method
        for method, df in results_dict.items():
            if "p_value" in df.columns:
                try:
                    plot_pvalue_distribution(
                        df,
                        method=method.upper(),
                        output_path=figures_dir / f"pvalues_{method}",
                    )
                    logger.info(f"  ✓ P-value distribution: {method}")
                except Exception as e:
                    logger.warning(f"  ✗ P-value plot failed for {method}: {e}")

        # Lag distributions per method
        for method, df in results_dict.items():
            if "lag_days" in df.columns or "lag_steps" in df.columns:
                try:
                    lag_col = "lag_days" if "lag_days" in df.columns else "lag_steps"
                    plot_lag_histogram(
                        df,
                        method=method.upper(),
                        lag_column=lag_col,
                        output_path=figures_dir / f"lags_{method}",
                    )
                    logger.info(f"  ✓ Lag histogram: {method}")
                except Exception as e:
                    logger.warning(f"  ✗ Lag plot failed for {method}: {e}")

        # Method comparison
        try:
            plot_method_comparison(
                results_dict, output_path=figures_dir / "method_comparison"
            )
            logger.info("  ✓ Method comparison")
        except Exception as e:
            logger.warning(f"  ✗ Method comparison failed: {e}")

    # === Run advanced diagnostics ===
    if args.advanced:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING ADVANCED DIAGNOSTICS")
        logger.info("=" * 70)

        advanced_results = {}

        # Permutation negative control
        if args.permutation and args.input:
            logger.info("\n>>> Permutation Negative Control")
            try:
                df = load_csv(args.input)
                # Get pairs from results
                pairs = []
                for _, row in all_edges.iterrows():
                    if "source" in row and "target" in row:
                        pairs.append((row["source"], row["target"]))
                pairs = list(set(pairs))  # unique pairs

                for method in results_dict.keys():
                    perm_result = permutation_negative_control(
                        df=df,
                        pairs=pairs,
                        method_name=method,
                        n_permutations=args.n_permutations,
                        alpha=args.alpha,
                        tolerance=args.tolerance,
                        maxlag=args.maxlag,
                    )
                    advanced_results[f"permutation_{method}"] = perm_result

            except Exception as e:
                logger.error(f"Permutation control failed: {e}")

        # Reverse-direction check
        if args.reverse and args.input:
            logger.info("\n>>> Reverse-Direction Check")
            try:
                df = load_csv(args.input)

                for method, edges_df in results_dict.items():
                    # Filter to significant edges only
                    if "is_significant" in edges_df.columns:
                        sig_edges = edges_df[edges_df["is_significant"] == 1]
                    elif "q_value" in edges_df.columns:
                        sig_edges = edges_df[edges_df["q_value"] < args.alpha]
                    elif "p_value" in edges_df.columns:
                        sig_edges = edges_df[edges_df["p_value"] < args.alpha]
                    else:
                        sig_edges = edges_df

                    if not sig_edges.empty:
                        reverse_result = reverse_direction_check(
                            df=df,
                            detected_edges=sig_edges,
                            method_name=method,
                            alpha=args.alpha,
                            tolerance=args.tolerance,
                            maxlag=args.maxlag,
                        )
                        advanced_results[f"reverse_{method}"] = reverse_result

            except Exception as e:
                logger.error(f"Reverse-direction check failed: {e}")

        # Method divergence alert
        if args.divergence:
            logger.info("\n>>> Method Divergence Alert")
            try:
                divergence_result = method_divergence_alert(
                    results_dict=results_dict,
                    alpha=args.alpha,
                    divergence_threshold=args.divergence_threshold,
                )
                advanced_results["method_divergence"] = divergence_result
            except Exception as e:
                logger.error(f"Method divergence check failed: {e}")

        # Save advanced diagnostics
        if advanced_results:
            import json

            advanced_path = output_dir / "advanced_diagnostics.json"
            with open(advanced_path, "w", encoding="utf-8") as f:
                json.dump(advanced_results, f, indent=2, default=str)
            logger.info(f"\n✅ Advanced diagnostics saved to: {advanced_path}")

    logger.info(f"\n✅ Diagnostics complete. Results saved to: {output_dir}")
    return 0


def report_command(args) -> int:
    """Generate analysis reports."""
    logger.info(f"Generating report for: {args.experiment_name}")

    from framework.reporting import (
        generate_summary_report,
        compare_with_baseline,
        generate_latex_table,
    )

    output_dir = Path(args.output) if args.output else Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_dict = {}
    if args.results:
        results_path = Path(args.results)
        for csv_file in results_path.glob("*.csv"):
            method = csv_file.stem
            results_dict[method] = pd.read_csv(csv_file)
            logger.info(f"Loaded: {csv_file}")

    if not results_dict:
        logger.warning("No result files found")
        return 1

    # Generate report
    logger.info("Generating summary report...")
    generate_summary_report(
        results_dict,
        experiment_name=args.experiment_name,
        output_path=output_dir / "summary.txt",
    )

    # Generate LaTeX table
    logger.info("Generating LaTeX table...")
    comparison_df = compare_with_baseline(results_dict, args.experiment_name)
    generate_latex_table(comparison_df, output_path=output_dir / "results_table.tex")

    logger.info(f"✅ Reports saved to: {output_dir}")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Causal Discovery Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis
  python -m framework.cli run --input data.csv --methods granger te pcmci consensus
  
  # Generate plots
  python -m framework.cli plot --results results/ --output figures/
  
  # Generate report
  python -m framework.cli report --experiment_name "Exp1" --results results/ --output reports/
        """,
    )

    # Global arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # RUN command
    run_parser = subparsers.add_parser("run", help="Run causal analysis")
    run_parser.add_argument("--input", required=True, help="Input CSV file")
    run_parser.add_argument(
        "--methods",
        nargs="+",
        default=["granger"],
        choices=["granger", "te", "pcmci", "consensus", "auto"],
        help="Methods to run",
    )
    run_parser.add_argument("--maxlag", type=int, default=12, help="Maximum lag")
    run_parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level"
    )
    run_parser.add_argument("--pairs", help="JSON string of variable pairs")
    run_parser.add_argument("--output", default="results", help="Output directory")
    run_parser.set_defaults(func=run_command)

    # PLOT command
    plot_parser = subparsers.add_parser("plot", help="Generate visualizations")
    plot_parser.add_argument(
        "--results", required=True, help="Results directory or CSV file"
    )
    plot_parser.add_argument("--output", default="figures", help="Output directory")
    plot_parser.set_defaults(func=plot_command)

    # REPORT command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument(
        "--experiment_name", required=True, help="Experiment name"
    )
    report_parser.add_argument("--results", help="Results directory")
    report_parser.add_argument("--output", default="reports", help="Output directory")
    report_parser.set_defaults(func=report_command)

    # DIAGNOSE command
    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Run diagnostic checks on results"
    )
    diagnose_parser.add_argument(
        "--experiment_name", required=True, help="Experiment name"
    )
    diagnose_parser.add_argument(
        "--results", required=True, help="Results directory or CSV file"
    )
    diagnose_parser.add_argument(
        "--input", help="Input data CSV (required for permutation/reverse checks)"
    )
    diagnose_parser.add_argument(
        "--output", default="diagnostics", help="Output directory"
    )
    diagnose_parser.add_argument(
        "--cadence_days", type=int, default=5, help="Days per timestep"
    )
    diagnose_parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance level"
    )
    diagnose_parser.add_argument(
        "--plots", action="store_true", help="Generate diagnostic plots"
    )
    diagnose_parser.add_argument(
        "--advanced", action="store_true", help="Run advanced diagnostics"
    )
    diagnose_parser.add_argument(
        "--permutation",
        action="store_true",
        help="Run permutation negative control (requires --input)",
    )
    diagnose_parser.add_argument(
        "--reverse",
        action="store_true",
        help="Run reverse-direction check (requires --input)",
    )
    diagnose_parser.add_argument(
        "--divergence",
        action="store_true",
        help="Check for method divergence",
    )
    diagnose_parser.add_argument(
        "--n_permutations", type=int, default=100, help="Number of permutations"
    )
    diagnose_parser.add_argument(
        "--tolerance", type=float, default=0.03, help="Tolerance for FDR deviation"
    )
    diagnose_parser.add_argument(
        "--divergence_threshold",
        type=float,
        default=0.20,
        help="Relative divergence threshold",
    )
    diagnose_parser.add_argument(
        "--maxlag", type=int, default=12, help="Maximum lag for methods"
    )
    diagnose_parser.set_defaults(func=diagnose_command)

    # Parse arguments
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 0

    # Execute command
    try:
        return args.func(args)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
