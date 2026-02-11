"""Patching service for orchestrating point cloud accumulation and patching."""

import os
import sys
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.waymo.pointcept_dataset import PointceptDataset
from src.accumulation.point_cloud_accumulator import PointCloudAccumulator
from src.accumulation.accumulation_strategy import AccumulationStrategy
from src.accumulation.default_accumulator_strategy import DefaultAccumulatorStrategy
from src.accumulation.greedy_grid_accumulator_strategy import GreedyGridAccumulatorStrategy
from app.config import ACCUMULATION_STEP


class PatchingService:
    """Service for patching point clouds with accumulated instances."""

    def __init__(self, dataset: PointceptDataset, strategy: str = 'default'):
        """Initialize patching service.

        Args:
            dataset: PointceptDataset instance
            strategy: Accumulation strategy name ('default' or 'greedy_grid')
        """
        self.dataset = dataset
        self.strategy = self._get_strategy(strategy)
        self.strategy_name = strategy

    def _get_strategy(self, strategy_name: str) -> AccumulationStrategy:
        """Get accumulation strategy by name.

        Args:
            strategy_name: Name of the strategy

        Returns:
            AccumulationStrategy instance
        """
        if strategy_name == 'greedy_grid':
            return GreedyGridAccumulatorStrategy()
        return DefaultAccumulatorStrategy()

    def patch_frame(self,
                    scene_id: str,
                    frame_id: str,
                    instance_ids: List[str],
                    grouped_instances: Dict[str, List[str]],
                    progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Patch a frame with accumulated instances.

        Args:
            scene_id: Scene identifier
            frame_id: Frame identifier
            instance_ids: List of instance IDs to patch
            grouped_instances: Dict mapping instance_id to list of frame_ids
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (patched_point_cloud, patched_segments)
        """
        # Create accumulator
        accumulator = PointCloudAccumulator(
            step=ACCUMULATION_STEP,
            grouped_instances=grouped_instances,
            dataset=self.dataset
        )

        # Load frame patcher
        patcher = self.dataset.load_frame_patcher(scene_id, frame_id)

        # Patch each selected instance
        total = len(instance_ids)
        for i, instance_id in enumerate(instance_ids):
            if progress_callback:
                progress_callback(i / total, f"Patching instance {instance_id}...")

            # Check if instance exists in grouped_instances
            if instance_id not in grouped_instances:
                continue

            # Accumulate point cloud for this instance
            accumulated_pc = accumulator.merge(
                scene_id=scene_id,
                instance_id=instance_id,
                accumulation_strategy=self.strategy
            )

            # Patch the frame with accumulated point cloud
            # Important: copy to avoid state carryover
            patcher.patch_instance(instance_id, np.copy(accumulated_pc))

        if progress_callback:
            progress_callback(1.0, "Patching complete!")

        # Return patched point cloud and segments
        return patcher.frame, patcher._segments

    def save_patched_frame(self,
                           scene_id: str,
                           frame_id: str,
                           patched_pc: np.ndarray) -> Optional[str]:
        """Save patched frame to disk.

        Args:
            scene_id: Scene identifier
            frame_id: Frame identifier
            patched_pc: Patched point cloud

        Returns:
            Path where the frame was saved, or None if failed
        """
        return self.dataset.serialise_frame_point_clouds(
            scene_id=scene_id,
            frame_id=frame_id,
            frame_point_cloud=patched_pc
        )

    def get_instance_info(self,
                          scene_id: str,
                          instance_id: str,
                          grouped_instances: Dict[str, List[str]]) -> Dict:
        """Get information about an instance.

        Args:
            scene_id: Scene identifier
            instance_id: Instance identifier
            grouped_instances: Dict mapping instance_id to list of frame_ids

        Returns:
            Dict with instance information
        """
        if instance_id not in grouped_instances:
            return {'frames': 0, 'exists': False}

        frames = grouped_instances[instance_id]
        return {
            'frames': len(frames),
            'first_frame': frames[0] if frames else None,
            'last_frame': frames[-1] if frames else None,
            'exists': True
        }
