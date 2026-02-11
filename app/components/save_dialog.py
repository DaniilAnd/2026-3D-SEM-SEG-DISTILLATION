"""Save dialog component."""

from typing import Optional

import numpy as np
import streamlit as st

from app.services.data_service import get_dataset


def render_save_dialog(data_root: str,
                       scene_id: str,
                       frame_id: str,
                       patched_pc: Optional[np.ndarray],
                       patched_segments: Optional[np.ndarray]):
    """Render save controls and handle saving.

    Args:
        data_root: Dataset root path
        scene_id: Scene identifier
        frame_id: Frame identifier
        patched_pc: Patched point cloud or None
        patched_segments: Patched segments or None
    """
    with st.expander("Save Patched Result", expanded=False):
        if patched_pc is None:
            st.warning("No patched point cloud to save. Select instances and run patching first.")
            return

        # Show save destination
        st.info(f"Will save to: `{data_root}/patched_v3/<split>/{scene_id}/{frame_id}/`")

        # Show what will be saved
        st.write("**Files to be created:**")
        st.write("- `coord.npy` - Point coordinates")
        st.write("- `strength.npy` - Point intensities")
        st.write("- `segment.npy` - Instance labels")
        st.write("- `pose.npy` - Frame pose (copied from original)")

        st.write(f"**Point count:** {patched_pc.shape[1]:,}")

        col1, col2 = st.columns([1, 3])
        with col1:
            save_button = st.button("Save to Disk", type="primary", use_container_width=True)

        if save_button:
            with st.spinner("Saving patched frame..."):
                try:
                    dataset = get_dataset(data_root)
                    saved_path = dataset.serialise_frame_point_clouds(
                        scene_id=scene_id,
                        frame_id=frame_id,
                        frame_point_cloud=patched_pc
                    )
                    if saved_path:
                        st.success(f"Saved successfully to: `{saved_path}`")
                    else:
                        st.error("Failed to save: unknown error")
                except Exception as e:
                    st.error(f"Error saving: {str(e)}")
