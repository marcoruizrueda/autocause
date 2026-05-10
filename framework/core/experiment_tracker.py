#!/usr/bin/env python3
"""
Experiment Tracker for Reproducibility

Lightweight JSON-based logging system that records:
- Experiment parameters
- Data versioning (SHA-256 hashes)
- Timestamps and duration
- Results hashes
- Git commit (if available)
- System information

Enables full reproducibility of causal discovery experiments.
"""

import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import platform
import sys

import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Lightweight experiment tracking for reproducibility.

    Records all information needed to reproduce a causal discovery
    experiment: parameters, data version, code version, timestamps.

    Example:
        >>> tracker = ExperimentTracker('exp1_all_data', 'experiments/exp1/')
        >>> tracker.log_parameters({'tau_max': 12, 'alpha': 0.05})
        >>> tracker.log_data_hash(data_df)
        >>> # ... run experiment ...
        >>> tracker.log_results(consensus_df)
        >>> tracker.save()
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: Path,
        description: Optional[str] = None,
    ):
        """
        Initialize experiment tracker.

        Parameters:
            experiment_name: Unique experiment identifier
            output_dir: Directory for experiment outputs
            description: Optional experiment description
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log = {
            "experiment_name": experiment_name,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "parameters": {},
            "data": {},
            "results": {},
            "system": self._get_system_info(),
            "code_version": self._get_code_version(),
        }

        self.start_timestamp = datetime.now()

    def log_parameters(self, params: Dict[str, Any]):
        """
        Log experiment parameters.

        Parameters:
            params: Dictionary of parameter name -> value
        """
        self.log["parameters"].update(params)
        logger.info(f"Logged {len(params)} parameters")

    def log_data_hash(
        self,
        data: pd.DataFrame,
        data_name: str = "input_data",
    ):
        """
        Log data version hash for reproducibility.

        Computes SHA-256 hash of data content.

        Parameters:
            data: Input DataFrame
            data_name: Identifier for this dataset
        """
        # For large dataframes, use pandas hash function (much faster)
        # Hash based on values, shape, and columns
        import pandas as pd

        if len(data) > 100000:
            # For very large dataframes, hash a sample + metadata
            hasher = hashlib.sha256()
            hasher.update(str(data.shape).encode())
            hasher.update("|".join(data.columns).encode())
            hasher.update(str(data.dtypes.to_dict()).encode())

            # Hash first/last 1000 rows
            sample = pd.concat([data.head(1000), data.tail(1000)])
            hasher.update(pd.util.hash_pandas_object(sample).values.tobytes())
            data_hash = hasher.hexdigest()
        else:
            # For smaller dataframes, use full hash
            data_str = data.to_csv(index=False)
            data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        self.log["data"][data_name] = {
            "shape": data.shape,
            "columns": list(data.columns),
            "hash_sha256": data_hash[:16],  # First 16 chars
            "hash_full": data_hash,
        }

        logger.info(
            f"Logged data '{data_name}': shape={data.shape}, hash={data_hash[:16]}"
        )

    def log_results(
        self,
        results: pd.DataFrame,
        results_name: str = "consensus_edges",
    ):
        """
        Log results hash and summary.

        Parameters:
            results: Results DataFrame or None
            results_name: Identifier for these results
        """
        # Skip if results are None or empty
        if results is None or len(results) == 0:
            self.log["results"][results_name] = {
                "n_rows": 0,
                "columns": [],
                "hash_sha256": None,
            }
            return

        # Hash results efficiently
        if len(results) > 10000:
            # For large results, use pandas hash
            results_hash = hashlib.sha256(
                pd.util.hash_pandas_object(results).values.tobytes()
            ).hexdigest()
        else:
            results_str = results.to_csv(index=False)
            results_hash = hashlib.sha256(results_str.encode()).hexdigest()

        self.log["results"][results_name] = {
            "n_rows": len(results),
            "columns": list(results.columns),
            "hash_sha256": results_hash[:16],
            "hash_full": results_hash,
        }

        # Summary statistics
        if "is_significant" in results.columns:
            n_sig = results["is_significant"].sum()
            self.log["results"][results_name]["n_significant"] = int(n_sig)

        if "method" in results.columns:
            method_counts = results["method"].value_counts().to_dict()
            self.log["results"][results_name]["by_method"] = method_counts

        logger.info(
            f"Logged results '{results_name}': "
            f"n={len(results)}, hash={results_hash[:16]}"
        )

    def log_file_path(
        self,
        filepath: Path,
        file_type: str = "output",
        description: Optional[str] = None,
    ):
        """
        Log path to an output file.

        Parameters:
            filepath: Path to file
            file_type: Type of file ('output', 'figure', 'report', etc.)
            description: Optional description
        """
        if "files" not in self.log:
            self.log["files"] = {}

        if file_type not in self.log["files"]:
            self.log["files"][file_type] = []

        self.log["files"][file_type].append(
            {
                "path": str(filepath),
                "description": description,
                "exists": filepath.exists() if isinstance(filepath, Path) else False,
            }
        )

    def log_metric(
        self,
        metric_name: str,
        value: float,
        description: Optional[str] = None,
    ):
        """
        Log a single metric value.

        Parameters:
            metric_name: Name of metric
            value: Metric value
            description: Optional description
        """
        if "metrics" not in self.log:
            self.log["metrics"] = {}

        self.log["metrics"][metric_name] = {
            "value": value,
            "description": description,
        }

    def log_error(self, error: Exception):
        """
        Log an error that occurred during experiment.

        Parameters:
            error: Exception object
        """
        if "errors" not in self.log:
            self.log["errors"] = []

        self.log["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now().isoformat(),
            }
        )

        logger.error(f"Logged error: {type(error).__name__}: {error}")

    def finalize(self):
        """Mark experiment as complete and compute duration"""
        self.log["end_time"] = datetime.now().isoformat()
        duration = (datetime.now() - self.start_timestamp).total_seconds()
        self.log["duration_seconds"] = duration

        logger.info(f"Experiment finalized: duration={duration:.1f}s")

    def save(self, filename: str = "experiment_log.json"):
        """
        Save experiment log to JSON file.

        Parameters:
            filename: Output filename (relative to output_dir)
        """
        # Finalize if not already done
        if self.log["end_time"] is None:
            self.finalize()

        log_path = self.output_dir / filename

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2, default=str)

        logger.info(f"Experiment log saved to {log_path}")

        return log_path

    def _get_system_info(self) -> Dict:
        """Collect system information"""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

    def _get_code_version(self) -> Dict:
        """Get git commit if available"""
        import subprocess

        code_info = {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
        }

        try:
            # Get git commit
            commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            code_info["git_commit"] = commit[:8]

            # Get branch
            branch = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            code_info["git_branch"] = branch

            # Check if dirty
            status = (
                subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
            code_info["git_dirty"] = len(status) > 0

        except Exception:
            # Not a git repo or git not available
            pass

        return code_info

    def to_dict(self) -> Dict:
        """Return log as dictionary"""
        return self.log.copy()


def load_experiment_log(log_path: Path) -> Dict:
    """
    Load an experiment log from JSON file.

    Parameters:
        log_path: Path to experiment_log.json

    Returns:
        Dictionary with experiment log
    """
    with open(log_path, "r") as f:
        log = json.load(f)

    return log


def compare_experiments(
    log_path1: Path,
    log_path2: Path,
) -> Dict:
    """
    Compare two experiment logs.

    Highlights differences in parameters, data, and results.

    Parameters:
        log_path1: First experiment log
        log_path2: Second experiment log

    Returns:
        Dictionary with comparison results
    """
    log1 = load_experiment_log(log_path1)
    log2 = load_experiment_log(log_path2)

    comparison = {
        "experiments": [
            log1["experiment_name"],
            log2["experiment_name"],
        ],
        "parameter_differences": {},
        "data_matches": True,
        "results_summary": {},
    }

    # Compare parameters
    params1 = log1.get("parameters", {})
    params2 = log2.get("parameters", {})

    all_params = set(params1.keys()) | set(params2.keys())

    for param in all_params:
        val1 = params1.get(param, "N/A")
        val2 = params2.get(param, "N/A")

        if val1 != val2:
            comparison["parameter_differences"][param] = {
                log1["experiment_name"]: val1,
                log2["experiment_name"]: val2,
            }

    # Compare data hashes
    data1_hash = log1.get("data", {}).get("input_data", {}).get("hash_sha256")
    data2_hash = log2.get("data", {}).get("input_data", {}).get("hash_sha256")

    comparison["data_matches"] = data1_hash == data2_hash

    # Compare results
    results1 = log1.get("results", {})
    results2 = log2.get("results", {})

    for result_name in set(results1.keys()) | set(results2.keys()):
        r1 = results1.get(result_name, {})
        r2 = results2.get(result_name, {})

        comparison["results_summary"][result_name] = {
            log1["experiment_name"]: r1.get("n_rows", "N/A"),
            log2["experiment_name"]: r2.get("n_rows", "N/A"),
        }

    return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    import numpy as np

    print("\n" + "=" * 70)
    print("EXPERIMENT TRACKER EXAMPLE")
    print("=" * 70)

    # Create tracker
    tracker = ExperimentTracker(
        "test_experiment",
        Path("/tmp/exp_test"),
        description="Test experiment for causal discovery",
    )

    # Log parameters
    tracker.log_parameters(
        {
            "tau_max": 12,
            "alpha": 0.05,
            "sampling_days": 5,
            "method": "granger",
        }
    )

    # Create fake data
    data = pd.DataFrame(
        {
            "X": np.random.normal(0, 1, 100),
            "Y": np.random.normal(0, 1, 100),
            "Z": np.random.normal(0, 1, 100),
        }
    )

    # Log data hash
    tracker.log_data_hash(data)

    # Simulate results
    results = pd.DataFrame(
        {
            "source": ["X", "Y"],
            "target": ["Y", "Z"],
            "method": ["Granger", "Granger"],
            "is_significant": [True, False],
            "p_value": [0.001, 0.25],
        }
    )

    # Log results
    tracker.log_results(results)

    # Log metrics
    tracker.log_metric("consensus_edges", 5, "Number of consensus causal edges")
    tracker.log_metric("stability_score", 0.85, "Temporal stability score")

    # Save
    log_path = tracker.save()

    print("\n" + "=" * 70)
    print("EXPERIMENT LOG")
    print("=" * 70)

    # Load and display
    log = load_experiment_log(log_path)
    print(json.dumps(log, indent=2, default=str))

    print("\n✅ Example completed!")
