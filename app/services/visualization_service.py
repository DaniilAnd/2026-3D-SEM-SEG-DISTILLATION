"""Visualization service for creating Plotly figures from point clouds."""

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from app.utils.color_utils import colorize_by_instance, colorize_by_height
from app.utils.downsampling import random_downsample
from app.config import MAX_POINTS_FOR_VISUALIZATION, DEFAULT_POINT_SIZE


def create_plotly_figure(points: np.ndarray,
                         segments: np.ndarray = None,
                         highlight_instances: List[int] = None,
                         point_size: int = DEFAULT_POINT_SIZE,
                         title: str = "Point Cloud",
                         downsample_ratio: float = 0.2,
                         color_by: str = "instance") -> go.Figure:
    """Create a Plotly 3D scatter plot for point cloud visualization.

    Args:
        points: Point cloud in (D, N) format, first 3 rows are XYZ
        segments: Optional segment/instance labels for each point (N,)
        highlight_instances: List of instance IDs to highlight
        point_size: Size of points in the visualization
        title: Title of the plot
        downsample_ratio: Fraction of points to show
        color_by: 'instance' or 'height'

    Returns:
        Plotly Figure object
    """
    n_points = points.shape[1]
    max_points = int(MAX_POINTS_FOR_VISUALIZATION * downsample_ratio)

    # Downsample if needed
    if n_points > max_points:
        points, segments = random_downsample(points, max_points, segments)

    # Extract coordinates (transpose to N, 3)
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    # Determine colors
    if color_by == "instance" and segments is not None:
        colors = colorize_by_instance(segments)
        # Convert to RGB string for plotly
        color_strings = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})'
                        for c in colors]
        marker_config = dict(
            size=point_size,
            color=color_strings,
            opacity=0.8
        )
    else:
        # Color by height
        colors = colorize_by_height(points)
        marker_config = dict(
            size=point_size,
            color=colors,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Height (Z)')
        )

    # Create hover text
    if segments is not None:
        hover_text = [f'X: {x[i]:.2f}<br>Y: {y[i]:.2f}<br>Z: {z[i]:.2f}<br>Instance: {segments[i]}'
                      for i in range(len(x))]
    else:
        hover_text = [f'X: {x[i]:.2f}<br>Y: {y[i]:.2f}<br>Z: {z[i]:.2f}'
                      for i in range(len(x))]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=marker_config,
        hovertext=hover_text,
        hoverinfo='text'
    )])

    # Layout configuration
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )

    return fig


def create_comparison_figure(original_points: np.ndarray,
                             patched_points: np.ndarray,
                             original_segments: np.ndarray = None,
                             patched_segments: np.ndarray = None,
                             point_size: int = DEFAULT_POINT_SIZE,
                             downsample_ratio: float = 0.2) -> Tuple[go.Figure, go.Figure]:
    """Create two Plotly figures for side-by-side comparison.

    Args:
        original_points: Original point cloud (D, N)
        patched_points: Patched point cloud (D, M)
        original_segments: Original segment labels
        patched_segments: Patched segment labels
        point_size: Size of points
        downsample_ratio: Fraction of points to show

    Returns:
        Tuple of (original_figure, patched_figure)
    """
    orig_fig = create_plotly_figure(
        original_points,
        original_segments,
        point_size=point_size,
        title=f"Original ({original_points.shape[1]:,} points)",
        downsample_ratio=downsample_ratio
    )

    patched_fig = create_plotly_figure(
        patched_points,
        patched_segments,
        point_size=point_size,
        title=f"Patched ({patched_points.shape[1]:,} points)",
        downsample_ratio=downsample_ratio
    )

    return orig_fig, patched_fig


def get_point_cloud_stats(points: np.ndarray,
                          segments: np.ndarray = None) -> dict:
    """Get statistics about a point cloud.

    Args:
        points: Point cloud (D, N)
        segments: Optional segment labels

    Returns:
        Dict with statistics
    """
    stats = {
        'num_points': points.shape[1],
        'x_range': (float(points[0, :].min()), float(points[0, :].max())),
        'y_range': (float(points[1, :].min()), float(points[1, :].max())),
        'z_range': (float(points[2, :].min()), float(points[2, :].max())),
    }

    if segments is not None:
        unique_instances = np.unique(segments)
        stats['num_instances'] = len(unique_instances)
        stats['instance_ids'] = [int(i) for i in unique_instances if i >= 0]

    return stats
