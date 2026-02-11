"""Instance selector component."""

from typing import Dict, Set

import streamlit as st


def render_instance_selector(instances: Dict[int, int],
                             selected_instances: Set[int]) -> Set[int]:
    """Render instance selection UI.

    Args:
        instances: Dict mapping instance_id to point_count
        selected_instances: Currently selected instance IDs

    Returns:
        Updated set of selected instance IDs
    """
    if not instances:
        st.info("No instances found in this frame.")
        return set()

    st.subheader("Instance Selection")

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Select All", use_container_width=True):
            return set(instances.keys())
    with col2:
        if st.button("Clear All", use_container_width=True):
            return set()
    with col3:
        st.caption(f"{len(instances)} instances available, {len(selected_instances)} selected")

    # Instance checkboxes in a grid
    new_selection = set()
    num_cols = 4
    cols = st.columns(num_cols)

    sorted_instances = sorted(instances.items(), key=lambda x: -x[1])  # Sort by point count desc

    for idx, (instance_id, point_count) in enumerate(sorted_instances):
        col = cols[idx % num_cols]
        with col:
            checked = st.checkbox(
                f"ID {instance_id} ({point_count:,} pts)",
                value=instance_id in selected_instances,
                key=f"instance_checkbox_{instance_id}"
            )
            if checked:
                new_selection.add(instance_id)

    return new_selection


def render_instance_info(instances: Dict[int, int],
                         selected_instances: Set[int],
                         grouped_instances: Dict = None):
    """Render detailed information about selected instances.

    Args:
        instances: Dict mapping instance_id to point_count
        selected_instances: Currently selected instance IDs
        grouped_instances: Optional dict with frame info per instance
    """
    if not selected_instances:
        return

    with st.expander("Selected Instance Details", expanded=False):
        for instance_id in sorted(selected_instances):
            if instance_id in instances:
                points = instances[instance_id]
                frames_info = ""
                if grouped_instances and str(instance_id) in grouped_instances:
                    num_frames = len(grouped_instances[str(instance_id)])
                    frames_info = f", appears in {num_frames} frames"
                st.write(f"**Instance {instance_id}**: {points:,} points{frames_info}")
