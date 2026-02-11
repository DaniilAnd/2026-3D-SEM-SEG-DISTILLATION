"""Main Streamlit application for 3D point cloud patching visualization."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.components.sidebar import render_sidebar
from app.components.viewer_3d import render_point_cloud_viewer, render_comparison_stats
from app.components.instance_selector import render_instance_selector, render_instance_info
from app.components.save_dialog import render_save_dialog
from app.services.data_service import (
    get_dataset,
    get_frame_point_cloud,
    get_frame_segments,
    get_frame_instances,
    group_instances_for_scene
)
from app.services.patching_service import PatchingService


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_root': '',
        'scene_id': None,
        'frame_id': None,
        'selected_instances': set(),
        'patched_point_cloud': None,
        'patched_segments': None,
        'grouped_instances': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    st.set_page_config(
        page_title="3D Point Cloud Patching",
        page_icon=":point_right:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("3D Point Cloud Patching Visualization")
    st.caption("Visualize and apply multi-frame point cloud accumulation for LiDAR semantic segmentation")

    init_session_state()

    # Render sidebar and get configuration
    config = render_sidebar()

    # Check if we have required configuration
    if not config['data_root']:
        st.info("Please enter the dataset root path in the sidebar to begin.")
        st.markdown("""
        ### Getting Started

        1. Enter the path to your Pointcept Waymo dataset in the sidebar
        2. Select a scene and frame to visualize
        3. Choose instances to patch
        4. Click 'Patch Selected Instances' to apply accumulation
        5. Compare original and patched point clouds
        6. Optionally save the patched result

        **Expected dataset structure:**
        ```
        data_root/
        ├── training/
        │   └── scene_name/
        │       └── frame_id/
        │           ├── coord.npy
        │           ├── strength.npy
        │           ├── pose.npy
        │           └── segment.npy
        ```
        """)
        return

    if not config['scene_id']:
        st.warning("Please select a scene from the sidebar.")
        return

    if not config['frame_id']:
        st.warning("Please select a frame from the sidebar.")
        return

    # Load data
    try:
        with st.spinner("Loading frame data..."):
            original_pc = get_frame_point_cloud(
                config['data_root'],
                config['scene_id'],
                config['frame_id']
            )
            original_segments = get_frame_segments(
                config['data_root'],
                config['scene_id'],
                config['frame_id']
            )
            instances = get_frame_instances(
                config['data_root'],
                config['scene_id'],
                config['frame_id']
            )
    except Exception as e:
        st.error(f"Error loading frame data: {str(e)}")
        return

    # Load grouped instances (cached)
    try:
        grouped_instances = group_instances_for_scene(
            config['data_root'],
            config['scene_id']
        )
        st.session_state['grouped_instances'] = grouped_instances
    except Exception as e:
        st.warning(f"Could not load instance grouping: {e}")
        grouped_instances = {}

    # Statistics
    render_comparison_stats(
        original_pc,
        st.session_state['patched_point_cloud'],
        st.session_state['selected_instances']
    )

    st.divider()

    # Instance selection
    st.session_state['selected_instances'] = render_instance_selector(
        instances,
        st.session_state['selected_instances']
    )

    # Show instance details
    render_instance_info(
        instances,
        st.session_state['selected_instances'],
        grouped_instances
    )

    st.divider()

    # Patching controls
    col1, col2, col3 = st.columns([2, 2, 4])

    with col1:
        patch_disabled = len(st.session_state['selected_instances']) == 0
        patch_button = st.button(
            "Patch Selected Instances",
            type="primary",
            disabled=patch_disabled,
            use_container_width=True,
            help="Apply multi-frame accumulation to selected instances"
        )

    with col2:
        if st.button("Clear Patched", use_container_width=True):
            st.session_state['patched_point_cloud'] = None
            st.session_state['patched_segments'] = None
            st.rerun()

    # Handle patching
    if patch_button:
        progress_bar = st.progress(0, text="Starting patching...")

        def update_progress(progress: float, text: str):
            progress_bar.progress(progress, text=text)

        try:
            dataset = get_dataset(config['data_root'])
            patching_service = PatchingService(dataset, config['strategy'])

            patched_pc, patched_segments = patching_service.patch_frame(
                scene_id=config['scene_id'],
                frame_id=config['frame_id'],
                instance_ids=[str(i) for i in st.session_state['selected_instances']],
                grouped_instances=grouped_instances,
                progress_callback=update_progress
            )

            st.session_state['patched_point_cloud'] = patched_pc
            st.session_state['patched_segments'] = patched_segments

            progress_bar.empty()
            st.success(f"Patching complete! Points: {original_pc.shape[1]:,} → {patched_pc.shape[1]:,}")
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            st.error(f"Error during patching: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    st.divider()

    # 3D Viewer
    render_point_cloud_viewer(
        config,
        original_pc,
        st.session_state['patched_point_cloud'],
        original_segments,
        st.session_state['patched_segments']
    )

    st.divider()

    # Save dialog
    render_save_dialog(
        config['data_root'],
        config['scene_id'],
        config['frame_id'],
        st.session_state['patched_point_cloud'],
        st.session_state['patched_segments']
    )


if __name__ == "__main__":
    main()
