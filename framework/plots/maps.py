"""
Geographic Maps for Causal Relationships

Visualizes spatial distribution of discovered causal relationships on European map,
useful for Earth observation and climate analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import basemap for geographic visualization
try:
    import importlib.util

    basemap_spec = importlib.util.find_spec("mpl_toolkits.basemap")
    BASEMAP_AVAILABLE = basemap_spec is not None
except (ImportError, AttributeError):
    BASEMAP_AVAILABLE = False
    logger.warning("basemap not available. Geographic plots will show basic mapping.")


def plot_europe_map(
    data_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    color_col: Optional[str] = None,
    title: str = "Geographic Distribution",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> Optional[Path]:
    """
    Plot point locations on Europe map.

    Parameters:
        data_df (pd.DataFrame): Data with latitude/longitude columns
        lat_col (str): Latitude column name
        lon_col (str): Longitude column name
        color_col (Optional[str]): Column for coloring points (e.g., p_value)
        title (str): Plot title
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if data_df is None or len(data_df) == 0:
        logger.warning("No data for Europe map")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Extract coordinates
    lats = data_df[lat_col].values
    lons = data_df[lon_col].values

    # Europe bounds (approximately)
    lat_min, lat_max = 35, 72
    lon_min, lon_max = -12, 45

    # Scatter plot
    if color_col and color_col in data_df.columns:
        colors = data_df[color_col].values
        scatter = ax.scatter(
            lons,
            lats,
            c=colors,
            cmap="RdYlGn_r",
            s=100,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label=color_col)
    else:
        ax.scatter(
            lons,
            lats,
            s=100,
            alpha=0.6,
            color="steelblue",
            edgecolors="black",
            linewidth=0.5,
        )

    # Set bounds
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # Grid and labels
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Add region boundaries (simplified)
    ax.axhline(
        y=43, color="gray", linestyle=":", alpha=0.5, label="Mediterranean ~43°N"
    )
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None


def plot_geographic_causality(
    edges_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    loc_id_col: str = "site_id",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> Optional[Path]:
    """
    Plot causal relationships as arrows between geographic locations.

    Parameters:
        edges_df (pd.DataFrame): Causal edges with 'source', 'target' columns
        locations_df (pd.DataFrame): Site locations with ID, lat, lon
        loc_id_col (str): Location ID column name
        lat_col (str): Latitude column name
        lon_col (str): Longitude column name
        output_path (Optional[Path]): Save path
        figsize (tuple): Figure size

    Returns:
        Optional[Path]: Path to saved figure
    """
    if edges_df is None or locations_df is None:
        logger.warning("Missing data for geographic causality plot")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Europe bounds
    lat_min, lat_max = 35, 72
    lon_min, lon_max = -12, 45

    # Create location mapping
    loc_map = dict(
        zip(locations_df[loc_id_col], zip(locations_df[lon_col], locations_df[lat_col]))
    )

    # Plot all locations
    ax.scatter(
        locations_df[lon_col],
        locations_df[lat_col],
        s=150,
        alpha=0.5,
        color="lightblue",
        edgecolors="black",
        linewidth=1,
        label="Sites",
        zorder=2,
    )

    # Plot causal edges
    for _, edge in edges_df.iterrows():
        src = edge.get("source")
        tgt = edge.get("target")

        if src in loc_map and tgt in loc_map:
            src_lon, src_lat = loc_map[src]
            tgt_lon, tgt_lat = loc_map[tgt]

            # Arrow from source to target
            ax.annotate(
                "",
                xy=(tgt_lon, tgt_lat),
                xytext=(src_lon, src_lat),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="darkgreen", alpha=0.6),
            )

    # Formatting
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
    ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Geographic Causality Network ({len(edges_df)} relationships)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        save_path = output_path.parent / f"{output_path.stem}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        logger.info(f"Saved: {save_path}")
        plt.close()
        return save_path

    return None
