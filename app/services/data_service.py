"""Data service for loading and caching dataset operations."""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.waymo.pointcept_dataset import PointceptDataset
from src.utils.dataset_helper import group_instances_across_frames
from app.config import EXCLUDED_SEGMENT_CLASSES


@st.cache_resource
def get_dataset(data_root: str) -> PointceptDataset:
    """Get or create cached PointceptDataset instance.

    Args:
        data_root: Root directory of the dataset

    Returns:
        PointceptDataset instance
    """
    return PointceptDataset(data_root)


@st.cache_data
def get_scenes(data_root: str) -> List[str]:
    """Get list of available scenes in the dataset.

    Args:
        data_root: Root directory of the dataset

    Returns:
        List of scene IDs
    """
    dataset = get_dataset(data_root)
    return sorted(dataset.scenes)


@st.cache_data
def get_frames(data_root: str, scene_id: str) -> List[str]:
    """Get list of frames for a given scene.

    Args:
        data_root: Root directory of the dataset
        scene_id: Scene identifier

    Returns:
        List of frame IDs
    """
    dataset = get_dataset(data_root)
    scene_iterator = dataset.get_scene_iterator(scene_id)
    frames = []
    for frame_id, _ in scene_iterator:
        frames.append(frame_id)
    return frames


@st.cache_data
def get_frame_point_cloud(data_root: str,
                          scene_id: str,
                          frame_id: str) -> np.ndarray:
    """Load frame point cloud.

    Args:
        data_root: Root directory of the dataset
        scene_id: Scene identifier
        frame_id: Frame identifier

    Returns:
        Point cloud of shape (D, N)
    """
    dataset = get_dataset(data_root)
    return dataset.get_frame_point_cloud(scene_id, frame_id)


@st.cache_data
def get_frame_segments(data_root: str,
                       scene_id: str,
                       frame_id: str) -> np.ndarray:
    """Load frame segment labels.

    Args:
        data_root: Root directory of the dataset
        scene_id: Scene identifier
        frame_id: Frame identifier

    Returns:
        Segment labels of shape (N,)
    """
    dataset = get_dataset(data_root)
    frame_dir = dataset._get_frame_dir(scene_id, frame_id)
    segment_path = os.path.join(frame_dir, "segment.npy")

    if os.path.exists(segment_path):
        return np.load(segment_path)

    # If no segment file, return -1 for all points
    pc = get_frame_point_cloud(data_root, scene_id, frame_id)
    return np.full(pc.shape[1], -1, dtype=np.int32)


@st.cache_data
def get_frame_instances(data_root: str,
                        scene_id: str,
                        frame_id: str,
                        excluded_classes: Tuple[int, ...] = None) -> Dict[int, int]:
    """Get instances in a frame with their point counts.

    Uses grouped_instances to find which instances are present in this frame,
    then gets point counts from segment data.

    Args:
        data_root: Root directory of the dataset
        scene_id: Scene identifier
        frame_id: Frame identifier
        excluded_classes: Tuple of class IDs to exclude

    Returns:
        Dict mapping instance_id to point_count
    """
    if excluded_classes is None:
        excluded_classes = tuple(EXCLUDED_SEGMENT_CLASSES)

    # Get grouped instances for the scene
    grouped = group_instances_for_scene(data_root, scene_id)

    # Find instances that appear in this frame
    frame_instance_ids = set()
    for instance_id, frame_ids in grouped.items():
        if frame_id in frame_ids:
            frame_instance_ids.add(int(instance_id))

    # Get point counts from segments
    segments = get_frame_segments(data_root, scene_id, frame_id)
    unique_ids, counts = np.unique(segments, return_counts=True)
    segment_counts = {int(uid): int(cnt) for uid, cnt in zip(unique_ids, counts)}

    # Build result: only instances present in this frame (from grouped_instances)
    instances = {}
    for instance_id in frame_instance_ids:
        if instance_id not in excluded_classes:
            # Use segment count if available, otherwise estimate
            point_count = segment_counts.get(instance_id, 0)
            instances[instance_id] = point_count

    return instances


@st.cache_data
def group_instances_for_scene(data_root: str,
                              scene_id: str) -> Dict[str, List[str]]:
    """Group instances across all frames in a scene.

    Args:
        data_root: Root directory of the dataset
        scene_id: Scene identifier

    Returns:
        Dict mapping instance_id to list of frame_ids
    """
    dataset = get_dataset(data_root)
    grouped = group_instances_across_frames(scene_id, dataset)
    # Convert keys to strings for consistency
    return {str(k): v for k, v in grouped.items()}


def clear_cache():
    """Clear all cached data."""
    st.cache_data.clear()
    st.cache_resource.clear()
