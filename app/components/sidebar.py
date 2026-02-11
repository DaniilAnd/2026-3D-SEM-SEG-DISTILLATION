"""Sidebar component for dataset configuration."""

from typing import Dict, List

import streamlit as st

from app.config import ACCUMULATION_STRATEGIES, VIEW_MODES, DEFAULT_DOWNSAMPLING_RATIO, DEFAULT_POINT_SIZE
from app.services.data_service import get_scenes, get_frames, clear_cache


def render_sidebar() -> Dict:
    """Render sidebar controls and return current selections.

    Returns:
        Dict with current configuration
    """
    st.sidebar.header("Dataset Configuration")

    # Dataset path input
    data_root = st.sidebar.text_input(
        "Dataset Root Path",
        value=st.session_state.get('data_root', ''),
        help="Path to Pointcept Waymo dataset (e.g., /data/waymo)"
    )

    # Update session state
    if data_root != st.session_state.get('data_root', ''):
        st.session_state['data_root'] = data_root
        # Clear dependent selections
        st.session_state['scene_id'] = None
        st.session_state['frame_id'] = None
        st.session_state['selected_instances'] = set()
        st.session_state['patched_point_cloud'] = None

    # Scene selector
    scenes = []
    if data_root:
        try:
            scenes = get_scenes(data_root)
        except Exception as e:
            st.sidebar.error(f"Error loading scenes: {e}")

    scene_id = st.sidebar.selectbox(
        "Select Scene",
        options=scenes if scenes else [''],
        index=0,
        disabled=not scenes,
        help="Select a scene from the dataset"
    )

    # Update scene in session state
    if scene_id and scene_id != st.session_state.get('scene_id'):
        st.session_state['scene_id'] = scene_id
        st.session_state['frame_id'] = None
        st.session_state['selected_instances'] = set()
        st.session_state['patched_point_cloud'] = None

    # Frame selector
    frames = []
    if data_root and scene_id:
        try:
            frames = get_frames(data_root, scene_id)
        except Exception as e:
            st.sidebar.error(f"Error loading frames: {e}")

    frame_id = st.sidebar.selectbox(
        "Select Frame",
        options=frames if frames else [''],
        index=0,
        disabled=not frames,
        help="Select a frame from the scene"
    )

    # Update frame in session state
    if frame_id and frame_id != st.session_state.get('frame_id'):
        st.session_state['frame_id'] = frame_id
        st.session_state['selected_instances'] = set()
        st.session_state['patched_point_cloud'] = None

    st.sidebar.divider()

    # Accumulation strategy
    st.sidebar.header("Patching Settings")

    strategy = st.sidebar.selectbox(
        "Accumulation Strategy",
        options=list(ACCUMULATION_STRATEGIES.keys()),
        format_func=lambda x: f"{x}: {ACCUMULATION_STRATEGIES[x]}",
        help="Strategy for merging point clouds across frames"
    )

    st.sidebar.divider()

    # View settings
    st.sidebar.header("View Settings")

    view_mode = st.sidebar.radio(
        "View Mode",
        options=VIEW_MODES,
        horizontal=True,
        help="How to display the point clouds"
    )

    color_by = st.sidebar.radio(
        "Color By",
        options=["instance", "height"],
        horizontal=True,
        help="How to color the points"
    )

    downsample_ratio = st.sidebar.slider(
        "Downsample Ratio",
        min_value=0.05,
        max_value=1.0,
        value=DEFAULT_DOWNSAMPLING_RATIO,
        step=0.05,
        help="Fraction of points to display (lower = faster)"
    )

    point_size = st.sidebar.slider(
        "Point Size",
        min_value=1,
        max_value=10,
        value=DEFAULT_POINT_SIZE,
        help="Size of points in the visualization"
    )

    st.sidebar.divider()

    # Cache management
    if st.sidebar.button("Clear Cache", help="Clear all cached data"):
        clear_cache()
        st.sidebar.success("Cache cleared!")
        st.rerun()

    return {
        'data_root': data_root,
        'scene_id': scene_id if scene_id else None,
        'frame_id': frame_id if frame_id else None,
        'strategy': strategy,
        'view_mode': view_mode,
        'color_by': color_by,
        'downsample_ratio': downsample_ratio,
        'point_size': point_size
    }
