"""Color utilities for point cloud visualization."""

import numpy as np
from typing import List, Tuple


# Predefined color palette for instances (20 distinct colors)
INSTANCE_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Rose
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky Blue
    (255, 128, 128),  # Light Red
    (128, 255, 128),  # Light Green
    (128, 128, 255),  # Light Blue
    (255, 255, 128),  # Light Yellow
    (255, 128, 255),  # Light Magenta
    (128, 255, 255),  # Light Cyan
    (192, 64, 0),     # Brown
    (64, 192, 0),     # Olive
]

# Background/unlabeled color
BACKGROUND_COLOR = (128, 128, 128)  # Gray


def instance_id_to_color(instance_id: int) -> Tuple[int, int, int]:
    """Convert instance ID to RGB color deterministically.

    Args:
        instance_id: Instance ID

    Returns:
        RGB tuple (0-255 range)
    """
    if instance_id < 0:
        return BACKGROUND_COLOR
    return INSTANCE_COLORS[instance_id % len(INSTANCE_COLORS)]


def colorize_by_instance(segments: np.ndarray) -> np.ndarray:
    """Colorize points by their instance/segment ID.

    Args:
        segments: Segment labels of shape (N,)

    Returns:
        RGB colors of shape (N, 3) normalized to 0-1
    """
    n_points = len(segments)
    colors = np.zeros((n_points, 3), dtype=np.float32)

    for i, seg_id in enumerate(segments):
        color = instance_id_to_color(int(seg_id))
        colors[i] = np.array(color) / 255.0

    return colors


def colorize_by_height(points: np.ndarray,
                       z_min: float = None,
                       z_max: float = None) -> np.ndarray:
    """Colorize points by their Z coordinate (height).

    Args:
        points: Point cloud of shape (D, N) or (N, D)
        z_min: Minimum Z for color mapping (auto if None)
        z_max: Maximum Z for color mapping (auto if None)

    Returns:
        Height values normalized to 0-1 of shape (N,)
    """
    # Handle both (D, N) and (N, D) formats
    if points.shape[0] == 3 or points.shape[0] == 4:
        z = points[2, :]
    else:
        z = points[:, 2]

    if z_min is None:
        z_min = z.min()
    if z_max is None:
        z_max = z.max()

    # Normalize to 0-1
    z_range = z_max - z_min
    if z_range == 0:
        return np.zeros_like(z)

    return (z - z_min) / z_range


def get_unique_instances(segments: np.ndarray,
                         excluded_classes: set = None) -> List[int]:
    """Get unique instance IDs excluding specified classes.

    Args:
        segments: Segment labels of shape (N,)
        excluded_classes: Set of class IDs to exclude

    Returns:
        List of unique instance IDs
    """
    if excluded_classes is None:
        excluded_classes = set()

    unique_ids = np.unique(segments)
    return [int(uid) for uid in unique_ids if int(uid) not in excluded_classes]
