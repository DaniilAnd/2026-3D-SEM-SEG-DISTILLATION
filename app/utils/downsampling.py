"""Downsampling utilities for point cloud visualization."""

import numpy as np
from typing import Tuple


def random_downsample(points: np.ndarray,
                      max_points: int,
                      segments: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly downsample point cloud for visualization.

    Args:
        points: Point cloud of shape (D, N) where D is dimensions, N is points
        max_points: Maximum number of points to keep
        segments: Optional segment labels of shape (N,)

    Returns:
        Tuple of (downsampled_points, downsampled_segments or indices)
    """
    n_points = points.shape[1]

    if n_points <= max_points:
        if segments is not None:
            return points, segments
        return points, np.arange(n_points)

    indices = np.random.choice(n_points, size=max_points, replace=False)
    downsampled_points = points[:, indices]

    if segments is not None:
        return downsampled_points, segments[indices]
    return downsampled_points, indices


def voxel_downsample(points: np.ndarray,
                     voxel_size: float = 0.1,
                     segments: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel-based downsampling for more uniform distribution.

    Args:
        points: Point cloud of shape (D, N)
        voxel_size: Size of voxels for downsampling
        segments: Optional segment labels of shape (N,)

    Returns:
        Tuple of (downsampled_points, downsampled_segments or indices)
    """
    coords = points[:3, :].T  # (N, 3)

    # Compute voxel indices
    voxel_indices = np.floor(coords / voxel_size).astype(np.int32)

    # Get unique voxels and keep one point per voxel
    _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)

    downsampled_points = points[:, unique_indices]

    if segments is not None:
        return downsampled_points, segments[unique_indices]
    return downsampled_points, unique_indices
