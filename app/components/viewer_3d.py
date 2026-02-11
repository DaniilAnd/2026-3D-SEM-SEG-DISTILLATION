"""3D point cloud viewer component."""

from typing import Dict, Optional

import numpy as np
import streamlit as st

from app.services.visualization_service import create_plotly_figure, create_comparison_figure


def render_point_cloud_viewer(config: Dict,
                              original_pc: np.ndarray,
                              patched_pc: Optional[np.ndarray],
                              original_segments: np.ndarray,
                              patched_segments: Optional[np.ndarray]):
    """Render 3D point cloud based on view mode.

    Args:
        config: Configuration dict from sidebar
        original_pc: Original point cloud (D, N)
        patched_pc: Patched point cloud (D, M) or None
        original_segments: Original segment labels
        patched_segments: Patched segment labels or None
    """
    view_mode = config['view_mode']
    point_size = config['point_size']
    downsample_ratio = config['downsample_ratio']
    color_by = config['color_by']

    if view_mode == "Original":
        fig = create_plotly_figure(
            original_pc,
            original_segments,
            point_size=point_size,
            title=f"Original Point Cloud ({original_pc.shape[1]:,} points)",
            downsample_ratio=downsample_ratio,
            color_by=color_by
        )
        st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Patched":
        if patched_pc is None:
            st.warning("No patched point cloud available. Select instances and click 'Patch' first.")
            # Show original as fallback
            fig = create_plotly_figure(
                original_pc,
                original_segments,
                point_size=point_size,
                title=f"Original Point Cloud ({original_pc.shape[1]:,} points)",
                downsample_ratio=downsample_ratio,
                color_by=color_by
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = create_plotly_figure(
                patched_pc,
                patched_segments,
                point_size=point_size,
                title=f"Patched Point Cloud ({patched_pc.shape[1]:,} points)",
                downsample_ratio=downsample_ratio,
                color_by=color_by
            )
            st.plotly_chart(fig, use_container_width=True)

    elif view_mode == "Side-by-Side":
        col1, col2 = st.columns(2)

        with col1:
            fig1 = create_plotly_figure(
                original_pc,
                original_segments,
                point_size=point_size,
                title=f"Original ({original_pc.shape[1]:,} pts)",
                downsample_ratio=downsample_ratio,
                color_by=color_by
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            if patched_pc is not None:
                fig2 = create_plotly_figure(
                    patched_pc,
                    patched_segments,
                    point_size=point_size,
                    title=f"Patched ({patched_pc.shape[1]:,} pts)",
                    downsample_ratio=downsample_ratio,
                    color_by=color_by
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Run patching to see results")

    elif view_mode == "Toggle":
        show_patched = st.toggle("Show Patched", value=False, disabled=patched_pc is None)

        if show_patched and patched_pc is not None:
            fig = create_plotly_figure(
                patched_pc,
                patched_segments,
                point_size=point_size,
                title=f"Patched Point Cloud ({patched_pc.shape[1]:,} points)",
                downsample_ratio=downsample_ratio,
                color_by=color_by
            )
        else:
            fig = create_plotly_figure(
                original_pc,
                original_segments,
                point_size=point_size,
                title=f"Original Point Cloud ({original_pc.shape[1]:,} points)",
                downsample_ratio=downsample_ratio,
                color_by=color_by
            )

        st.plotly_chart(fig, use_container_width=True)


def render_comparison_stats(original_pc: np.ndarray,
                            patched_pc: Optional[np.ndarray],
                            selected_instances: set):
    """Render statistics comparing original and patched point clouds.

    Args:
        original_pc: Original point cloud
        patched_pc: Patched point cloud or None
        selected_instances: Set of selected instance IDs
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Points", f"{original_pc.shape[1]:,}")

    with col2:
        if patched_pc is not None:
            delta = patched_pc.shape[1] - original_pc.shape[1]
            st.metric(
                "Patched Points",
                f"{patched_pc.shape[1]:,}",
                delta=f"{delta:+,}"
            )
        else:
            st.metric("Patched Points", "—")

    with col3:
        st.metric("Selected Instances", len(selected_instances))

    with col4:
        if patched_pc is not None:
            change_pct = ((patched_pc.shape[1] - original_pc.shape[1]) / original_pc.shape[1]) * 100
            st.metric("Change", f"{change_pct:+.1f}%")
        else:
            st.metric("Change", "—")
