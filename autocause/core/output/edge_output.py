#!/usr/bin/env python3
"""
Edge-Level Output Module

Generates causal discovery results in multiple formats:
- CSV: Tabular edges with all metadata
- GraphML: Network graph format for visualization

Features:
- Combine results from multiple methods
- Include lag information (both timesteps and days)
- Add test statistics and p-values
- Support filtering by significance threshold
- Generate both directed and undirected networks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

logger = logging.getLogger(__name__)


class EdgeCollection:
    """Manages a collection of causal edges from multiple methods"""

    def __init__(self, cadence_days: int = 5):
        """
        Initialize edge collection.

        Parameters:
            cadence_days (int): Days per timestep (for lag conversion)
        """
        self.edges = []
        self.cadence_days = cadence_days

    def add_granger_result(self, result: Dict, alpha: float = 0.05):
        """
        Add Granger causality result.

        Parameters:
            result (Dict): Result from run_granger_causality
            alpha (float): Significance threshold
        """
        if not isinstance(result, dict) or "error" in result:
            logger.debug("Skipping invalid Granger result")
            return

        is_significant = result.get("is_causal", False)
        if not is_significant:
            return

        edge = {
            "source": result.get("cause"),
            "target": result.get("effect"),
            "method": "Granger",
            "lag_steps": result.get("best_lag"),
            "lag_days": result.get("best_lag_days"),
            "p_value": result.get("best_p_value"),
            "q_value": result.get("q_value", np.nan),
            "test_statistic": result.get("test_statistics"),
            "effect_size_method": "granger_beta_std",
            "is_significant": True,
            "n_observations": result.get("n_observations"),
        }
        self.edges.append(edge)

    def add_transfer_entropy_result(self, result: Dict, alpha: float = 0.05):
        """
        Add Transfer Entropy result.

        Parameters:
            result (Dict): Result from run_transfer_entropy
            alpha (float): Significance threshold
        """
        if not isinstance(result, dict) or "error" in result:
            logger.debug("Skipping invalid Transfer Entropy result")
            return

        is_significant = result.get("is_significant", False)
        if not is_significant:
            return

        edge = {
            "source": result.get("source"),
            "target": result.get("target"),
            "method": "TransferEntropy",
            "lag_steps": result.get("delay"),
            "lag_days": result.get("delay_days"),
            "p_value": result.get("p_value"),
            "q_value": result.get("q_value", np.nan),
            "test_statistic": result.get("te_bits"),
            "effect_size_method": "te_bits",
            "is_significant": True,
            "n_observations": result.get("n_observations"),
        }
        self.edges.append(edge)

    def add_pcmci_result(self, result: Dict, alpha: float = 0.05):
        """
        Add PCMCI+ result.

        Parameters:
            result (Dict): Result from run_pcmci_pair
            alpha (float): Significance threshold
        """
        if not isinstance(result, dict) or "error" in result:
            logger.debug("Skipping invalid PCMCI+ result")
            return

        is_causal = result.get("causal", False)
        if not is_causal:
            return

        edge = {
            "source": result.get("source"),
            "target": result.get("target"),
            "method": "PCMCI+",
            "lag_steps": result.get("best_lag"),
            "lag_days": result.get("best_lag_days"),
            "p_value": result.get("best_p_value"),
            "q_value": result.get("q_value", np.nan),
            "test_statistic": result.get("val", result.get("best_p_value")),
            "effect_size_method": "pcmci_val",
            "is_significant": True,
            "n_observations": result.get("n_observations"),
        }
        self.edges.append(edge)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert edges to DataFrame"""
        if not self.edges:
            logger.warning("No edges to export")
            return pd.DataFrame()

        df = pd.DataFrame(self.edges)

        # Sort by p-value (most significant first)
        if "p_value" in df.columns:
            df = df.sort_values("p_value")

        return df

    def to_csv(self, output_path: Path, **kwargs):
        """
        Save edges to CSV file.

        Parameters:
            output_path (Path): Output CSV file path
            **kwargs: Additional arguments for pd.to_csv
        """
        df = self.to_dataframe()

        if df.empty:
            logger.warning(f"No edges to save to {output_path}")
            return

        df.to_csv(output_path, index=False, **kwargs)
        logger.info(f"✅ Saved {len(df)} edges to CSV: {output_path}")

    def to_graphml(
        self, output_path: Path, directed: bool = True, weighted: bool = True
    ):
        """
        Save edges to GraphML format for network visualization.

        Parameters:
            output_path (Path): Output GraphML file path
            directed (bool): Create directed graph (True) or undirected (False)
            weighted (bool): Include edge weights (True) or not (False)
        """
        df = self.to_dataframe()

        if df.empty:
            logger.warning(f"No edges to save to {output_path}")
            return

        # Create GraphML root
        root = ET.Element("graphml")
        root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        # Add graph element
        graph = ET.SubElement(root, "graph")
        graph.set("edgedefault", "directed" if directed else "undirected")
        graph.set("id", "causal_network")

        # Add nodes
        nodes = set()
        for _, edge in df.iterrows():
            nodes.add(edge["source"])
            nodes.add(edge["target"])

        for node_id in sorted(nodes):
            node = ET.SubElement(graph, "node")
            node.set("id", node_id)
            node.set("label", node_id)

        # Add edges
        for idx, (_, edge) in enumerate(df.iterrows()):
            edge_elem = ET.SubElement(graph, "edge")
            edge_elem.set("id", f"e{idx}")
            edge_elem.set("source", edge["source"])
            edge_elem.set("target", edge["target"])

            # Add edge data
            for attr in [
                "method",
                "lag_steps",
                "lag_days",
                "p_value",
                "is_significant",
            ]:
                if attr in edge and pd.notna(edge[attr]):
                    data = ET.SubElement(edge_elem, "data")
                    data.set("key", attr)
                    data.text = str(edge[attr])

            # Add edge weight if requested
            if weighted and "p_value" in edge and pd.notna(edge["p_value"]):
                # Weight inversely proportional to p-value (lower p = higher weight)
                weight = 1.0 - np.clip(edge["p_value"], 0, 1)
                data = ET.SubElement(edge_elem, "data")
                data.set("key", "weight")
                data.text = str(weight)

        # Pretty-print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        # Remove extra blank lines
        xml_str = "\n".join([line for line in xml_str.split("\n") if line.strip()])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(xml_str)

        logger.info(f"✅ Saved {len(df)} edges to GraphML: {output_path}")

    def to_summary_dict(self) -> Dict:
        """Get summary statistics about edges"""
        if not self.edges:
            return {
                "n_edges": 0,
                "n_significant": 0,
                "methods": [],
                "avg_p_value": np.nan,
                "lag_range_steps": (np.nan, np.nan),
                "lag_range_days": (np.nan, np.nan),
            }

        df = self.to_dataframe()

        return {
            "n_edges": len(df),
            "n_significant": df["is_significant"].sum(),
            "methods": df["method"].unique().tolist(),
            "avg_p_value": df["p_value"].mean(),
            "lag_range_steps": (
                df["lag_steps"].min(),
                df["lag_steps"].max(),
            ),
            "lag_range_days": (
                df["lag_days"].min(),
                df["lag_days"].max(),
            ),
            "edges_by_method": df["method"].value_counts().to_dict(),
        }


def save_experiment_results(
    granger_results: List[Dict],
    te_results: List[Dict],
    pcmci_results: List[Dict],
    output_dir: Path,
    experiment_name: str = "experiment",
    cadence_days: int = 5,
    alpha: float = 0.05,
):
    """
    Save all causal discovery results for an experiment.

    Parameters:
        granger_results (List[Dict]): Granger causality results
        te_results (List[Dict]): Transfer Entropy results
        pcmci_results (List[Dict]): PCMCI+ results
        output_dir (Path): Output directory
        experiment_name (str): Name for output files
        cadence_days (int): Days per timestep
        alpha (float): Significance threshold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create edge collection
    edges = EdgeCollection(cadence_days=cadence_days)

    # Add results from all methods
    for result in granger_results:
        if isinstance(result, dict):
            edges.add_granger_result(result, alpha=alpha)

    for result in te_results:
        if isinstance(result, dict):
            edges.add_transfer_entropy_result(result, alpha=alpha)

    for result in pcmci_results:
        if isinstance(result, dict):
            edges.add_pcmci_result(result, alpha=alpha)

    # Save to files
    csv_path = output_dir / f"{experiment_name}_edges.csv"
    graphml_path = output_dir / f"{experiment_name}_network.graphml"

    edges.to_csv(csv_path)
    edges.to_graphml(graphml_path, directed=True, weighted=True)

    # Save summary
    summary = edges.to_summary_dict()
    summary_df = pd.DataFrame([summary])
    summary_path = output_dir / f"{experiment_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"EXPERIMENT SUMMARY: {experiment_name}")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total edges found: {summary['n_edges']}")
    logger.info(f"Significant edges: {summary['n_significant']}")
    logger.info(f"Methods: {', '.join(summary['methods'])}")
    logger.info(f"Average p-value: {summary['avg_p_value']:.4f}")
    logger.info(
        f"Lag range: {summary['lag_range_steps']} steps ({summary['lag_range_days']} days)"
    )
    logger.info(f"Edges by method: {summary['edges_by_method']}")

    return edges


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample results
    granger_result = {
        "cause": "RR",
        "effect": "NDVI",
        "best_lag": 2,
        "best_lag_days": 10,
        "best_p_value": 0.03,
        "is_causal": True,
        "n_observations": 1000,
    }

    te_result = {
        "source": "TG",
        "target": "NDVI",
        "delay": 1,
        "delay_days": 5,
        "p_value": 0.05,
        "is_significant": True,
        "te": 0.12,
        "n_observations": 1000,
    }

    pcmci_result = {
        "source": "RR",
        "target": "TG",
        "best_lag": 0,
        "best_lag_days": 0,
        "best_p_value": 0.001,
        "causal": True,
        "n_observations": 1000,
    }

    # Test save function
    save_experiment_results(
        granger_results=[granger_result],
        te_results=[te_result],
        pcmci_results=[pcmci_result],
        output_dir=Path("/tmp/test_edges"),
        experiment_name="test",
        cadence_days=5,
    )

    print("\n✅ Edge output module test completed!")
