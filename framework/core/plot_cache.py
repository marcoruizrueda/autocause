"""
Plot Cache & Metadata Management

Provides persistent storage of all intermediate values needed for plot regeneration.
Enables recreating visualizations without rerunning causal discovery.

Features:
- Store method results with full metadata (columns, dtypes, statistics)
- Cache plot-specific intermediate computations
- Support partial method execution (only subset of methods available)
- Track plot generation status per stage and method
- Enable standalone plot regeneration from disk

File Organization:
  results/checkpoints/
    ├── method_results_metadata.json       # Column names, dtypes, shapes per method
    ├── method_results/
    │   ├── granger.pkl                    # Pickled DataFrame (for fast reload)
    │   ├── transfer_entropy.pkl
    │   └── pcmci.pkl
    ├── consensus_results_metadata.json    # Consensus metadata
    ├── consensus_results.pkl              # Consensus DataFrame
    ├── plot_cache/
    │   ├── granger_normalized.pkl         # Cached normalized data per method
    │   ├── te_normalized.pkl
    │   ├── pcmci_normalized.pkl
    │   ├── lag_distribution_data.pkl      # Cross-method plot data
    │   ├── pvalue_comparison_data.pkl
    │   └── method_agreement_matrix.pkl
    └── generation_log.json                # Track which plots were generated
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd

logger = logging.getLogger(__name__)


class PlotCacheManager:
    """Manages persistent storage of results and plot-specific data for regeneration."""

    def __init__(self, output_dir: Path):
        """
        Initialize cache manager.

        Parameters:
            output_dir (Path): Main output directory
        """
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.method_results_dir = self.checkpoint_dir / "method_results"
        self.method_results_dir.mkdir(exist_ok=True)

        self.plot_cache_dir = self.checkpoint_dir / "plot_cache"
        self.plot_cache_dir.mkdir(exist_ok=True)

        self.metadata_path = self.checkpoint_dir / "method_results_metadata.json"
        self.consensus_metadata_path = (
            self.checkpoint_dir / "consensus_results_metadata.json"
        )
        self.generation_log_path = self.checkpoint_dir / "generation_log.json"

    def save_method_results(
        self, results_dict: Dict[str, pd.DataFrame], force: bool = False
    ) -> Dict[str, str]:
        """
        Save method results with metadata for later regeneration.

        Parameters:
            results_dict (dict): Method name -> DataFrame
            force (bool): Overwrite existing files

        Returns:
            dict: Paths to saved files
        """
        saved_paths = {}
        metadata = {}

        for method, df in results_dict.items():
            if df is None or len(df) == 0:
                logger.info(f"  Skipping {method}: empty result")
                continue

            # Save pickled DataFrame (fast reload)
            pkl_path = self.method_results_dir / f"{method}.pkl"
            if not pkl_path.exists() or force:
                with open(pkl_path, "wb") as f:
                    pickle.dump(df, f)
                logger.info(f"  ✓ Cached {method} results: {pkl_path}")
            saved_paths[method] = str(pkl_path)

            # Also save as CSV for manual inspection
            csv_path = (
                self.output_dir / "method" / method / "1-raw" / f"results_{method}.csv"
            )
            if not csv_path.exists() or force:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)

            # Store metadata for structure preservation
            metadata[method] = {
                "shape": list(df.shape),
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "n_edges": len(df),
                "p_value_column": self._detect_pvalue_column(df),
                "lag_column": self._detect_lag_column(df),
                "has_geographic": "latitude" in df.columns
                and "longitude" in df.columns,
            }

        # Save metadata
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved method metadata: {self.metadata_path}")

        return saved_paths

    def save_consensus_results(
        self, consensus_df: pd.DataFrame, consensus_info: Dict[str, Any]
    ) -> None:
        """
        Save consensus results and metadata.

        Parameters:
            consensus_df (DataFrame): Consensus edges
            consensus_info (dict): Additional consensus information
        """
        # Save pickled consensus
        pkl_path = self.checkpoint_dir / "consensus_results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(consensus_df, f)
        logger.info(f"  ✓ Cached consensus results: {pkl_path}")

        # Save metadata
        metadata = {
            "shape": list(consensus_df.shape),
            "columns": list(consensus_df.columns),
            "dtypes": {
                col: str(consensus_df[col].dtype) for col in consensus_df.columns
            },
            "n_edges": len(consensus_df),
            "info": consensus_info,
        }
        with open(self.consensus_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ Saved consensus metadata: {self.consensus_metadata_path}")

    def load_method_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all cached method results from checkpoints.

        Returns:
            dict: Method name -> DataFrame (only those that exist)

        Notes:
            - Gracefully handles partial method execution
            - Returns only methods that were cached
            - Useful for standalone plot regeneration
        """
        results = {}

        if not self.metadata_path.exists():
            logger.warning(f"No metadata found: {self.metadata_path}")
            return results

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        for method in metadata.keys():
            pkl_path = self.method_results_dir / f"{method}.pkl"
            if pkl_path.exists():
                try:
                    with open(pkl_path, "rb") as f:
                        results[method] = pickle.load(f)
                    logger.info(f"  ✓ Loaded {method} from cache")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load {method}: {e}")
            else:
                logger.info(f"  ⊘ {method} not cached (may have been skipped)")

        return results

    def load_consensus_results(self) -> Optional[pd.DataFrame]:
        """
        Load cached consensus results.

        Returns:
            DataFrame or None if not available
        """
        pkl_path = self.checkpoint_dir / "consensus_results.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    consensus_df = pickle.load(f)
                logger.info("  ✓ Loaded consensus from cache")
                return consensus_df
            except Exception as e:
                logger.warning(f"  ✗ Failed to load consensus: {e}")
        return None

    def cache_plot_intermediate_data(
        self, data_name: str, data: Any, method: Optional[str] = None
    ) -> Path:
        """
        Cache intermediate computation results for plot regeneration.

        Parameters:
            data_name (str): Name of the data (e.g., 'normalized', 'lag_distribution')
            data (Any): Data to cache (DataFrame, dict, array, etc.)
            method (str, optional): Method name if method-specific

        Returns:
            Path: Path to cached file
        """
        if method:
            filename = f"{method}_{data_name}.pkl"
        else:
            filename = f"{data_name}.pkl"

        cache_path = self.plot_cache_dir / filename
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

        return cache_path

    def load_plot_cache(self, data_name: str, method: Optional[str] = None) -> Any:
        """
        Load cached intermediate plot data.

        Parameters:
            data_name (str): Name of cached data
            method (str, optional): Method name if method-specific

        Returns:
            Cached data or None if not found
        """
        if method:
            filename = f"{method}_{data_name}.pkl"
        else:
            filename = f"{data_name}.pkl"

        cache_path = self.plot_cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {filename}: {e}")
        return None

    def log_plot_generation(
        self,
        stage: str,
        method: Optional[str],
        plot_name: str,
        status: str = "success",
        error: Optional[str] = None,
    ) -> None:
        """
        Log which plots were generated (for tracking).

        Parameters:
            stage (str): Stage (e.g., '1-raw', '2-adjusted', 'consensus')
            method (str, optional): Method name
            plot_name (str): Plot name
            status (str): 'success', 'skipped', or 'failed'
            error (str, optional): Error message if failed
        """
        if not self.generation_log_path.exists():
            log_data = {}
        else:
            with open(self.generation_log_path, "r") as f:
                log_data = json.load(f)

        key = f"{stage}/{method}/{plot_name}" if method else f"{stage}/{plot_name}"
        log_data[key] = {"status": status, "error": error}

        with open(self.generation_log_path, "w") as f:
            json.dump(log_data, f, indent=2)

    def get_available_methods(self) -> List[str]:
        """
        Get list of methods that have cached results.

        Returns:
            list: Method names
        """
        if not self.metadata_path.exists():
            return []

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return list(metadata.keys())

    def get_method_metadata(self, method: str) -> Dict[str, Any]:
        """Get metadata for a specific method."""
        if not self.metadata_path.exists():
            return {}

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get(method, {})

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get summary of plot generation status."""
        if not self.generation_log_path.exists():
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

        with open(self.generation_log_path, "r") as f:
            log_data = json.load(f)

        stats = {"total": len(log_data), "success": 0, "failed": 0, "skipped": 0}
        for entry in log_data.values():
            stats[entry["status"]] += 1
        return stats

    @staticmethod
    def _detect_pvalue_column(df: pd.DataFrame) -> Optional[str]:
        """Detect which column contains p-values."""
        for col in ["p_value", "best_p_value", "q_value", "pvalue"]:
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _detect_lag_column(df: pd.DataFrame) -> Optional[str]:
        """Detect which column contains lag information."""
        for col in ["delay", "lag", "lag_steps", "tau"]:
            if col in df.columns:
                return col
        return None


def regenerate_plots_from_cache(output_dir: Path, methods: Optional[List[str]] = None):
    """
    Standalone function to regenerate plots from cached results.

    Usage:
        regenerate_plots_from_cache(
            output_dir=Path("/path/to/experiment"),
            methods=["granger", "pcmci"]  # If None, regenerates all cached
        )

    Parameters:
        output_dir (Path): Experiment output directory
        methods (list, optional): Specific methods to regenerate. If None, uses all cached.
    """
    from framework.core.methods.visualize_results import visualize_all_results

    logger.info("\n" + "=" * 70)
    logger.info("REGENERATING PLOTS FROM CACHE")
    logger.info("=" * 70)

    cache_mgr = PlotCacheManager(output_dir)

    # Load available methods
    cached_methods = cache_mgr.get_available_methods()
    if not cached_methods:
        logger.error("No cached results found. Run experiment first.")
        return

    logger.info(f"Available cached methods: {cached_methods}")

    # Filter to requested methods
    if methods:
        methods_to_use = [m for m in methods if m in cached_methods]
        if not methods_to_use:
            logger.error(f"None of the requested methods are cached: {methods}")
            return
    else:
        methods_to_use = cached_methods

    logger.info(f"Regenerating plots for: {methods_to_use}")

    # Load results
    results_dict = cache_mgr.load_method_results()
    results_dict = {m: results_dict[m] for m in methods_to_use if m in results_dict}

    if not results_dict:
        logger.error("Failed to load cached results")
        return

    # Regenerate visualizations
    try:
        visualize_all_results(results_dict, output_dir)
        logger.info("✅ Plot regeneration complete")
    except Exception as e:
        logger.error(f"Plot regeneration failed: {e}")
