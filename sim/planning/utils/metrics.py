"""
Evaluation Metrics for MPPI Planning

Chamfer distance and task-specific reward functions
"""

import torch
import numpy as np
from typing import Dict, Tuple


def batch_chamfer_dist(xyz_batch: torch.Tensor, xyz_target: torch.Tensor) -> torch.Tensor:
    """
    Compute batch Chamfer distance

    Args:
        xyz_batch: (B, N, 3) - Batch of point clouds
        xyz_target: (M, 3) - Target point cloud

    Returns:
        chamfer: (B,) - Chamfer distance for each sample
    """
    # NaN/Inf detection
    if torch.isnan(xyz_batch).any():
        print(f"[WARNING] NaN detected in xyz_batch")
        has_nan = torch.isnan(xyz_batch).any(dim=(1, 2))
        penalty = torch.full(
            (xyz_batch.shape[0],), 1e6,
            device=xyz_batch.device, dtype=xyz_batch.dtype
        )
        return penalty

    if torch.isnan(xyz_target).any():
        print(f"[WARNING] NaN detected in xyz_target")
        return torch.full(
            (xyz_batch.shape[0],), 1e6,
            device=xyz_batch.device, dtype=xyz_batch.dtype
        )

    # Check for empty point clouds
    if xyz_batch.shape[1] == 0 or xyz_target.shape[0] == 0:
        print(f"[WARNING] Empty point cloud")
        return torch.full(
            (xyz_batch.shape[0],), 1e6,
            device=xyz_batch.device, dtype=xyz_batch.dtype
        )

    # Compute bidirectional Chamfer distance
    xyz_target = xyz_target[None]  # (1, M, 3)

    # xyz_batch -> xyz_target direction
    dist_xy = torch.cdist(xyz_batch, xyz_target)  # (B, N, M)
    min_dist_xy = dist_xy.min(dim=2)[0]  # (B, N)

    # xyz_target -> xyz_batch direction
    dist_yx = torch.cdist(xyz_target, xyz_batch)  # (B, M, N)
    min_dist_yx = dist_yx.min(dim=2)[0]  # (B, M)

    # Bidirectional average
    chamfer = min_dist_xy.mean(dim=1) + min_dist_yx.mean(dim=1)

    # Final NaN check
    if torch.isnan(chamfer).any():
        nan_mask = torch.isnan(chamfer)
        print(f"[WARNING] NaN in computed Chamfer distance for {nan_mask.sum().item()} samples")
        chamfer = torch.where(
            nan_mask,
            torch.tensor(1e6, device=chamfer.device, dtype=chamfer.dtype),
            chamfer
        )

    return chamfer


def rope_routing_reward(
    final_states: torch.Tensor,
    target_pts: torch.Tensor,
    trajectories: Dict[str, torch.Tensor],
    config
) -> torch.Tensor:
    """
    Rope routing task-specific reward function

    Args:
        final_states: (B, N, 3) - Final point cloud states
        target_pts: (M, 3) - Target point cloud
        trajectories: dict - Trajectory information with keys:
            'eef_xyz': (B, T, n_grippers, 3)
            'eef_rot': (B, T, n_grippers, 3, 3)
            'eef_gripper': (B, T, n_grippers, 1)
        config: Configuration object

    Returns:
        rewards: (B,) - Reward for each trajectory
    """
    # Primary objective: minimize Chamfer distance
    chamfer = batch_chamfer_dist(final_states, target_pts)

    # Penalty terms
    penalty = torch.zeros(final_states.shape[0], device=final_states.device)

    # 1. End-effector height constraint (avoid table collision)
    eef_xyz = trajectories['eef_xyz']  # (B, T, n_grippers, 3)
    max_eef_height = eef_xyz[:, :, :, 2].max(dim=1)[0]  # (B, n_grippers)

    # Penalize if any gripper exceeds -0.02m (close to table)
    penalty += (max_eef_height.max(dim=1)[0] > -0.02).float() * 100.0

    # 2. Bimanual distance constraint (if dual-arm)
    if config.env.robot.n_grippers == 2:
        eef_left = eef_xyz[:, :, 0]  # (B, T, 3)
        eef_right = eef_xyz[:, :, 1]  # (B, T, 3)
        dist = torch.norm(eef_left - eef_right, dim=-1)  # (B, T)

        # Arms should not exceed 0.4m apart
        penalty += (dist.max(dim=1)[0] > 0.4).float() * 50.0

    # Final reward = negative Chamfer distance - penalties
    reward = -chamfer - penalty

    return reward


def compute_success_rate_rope(
    final_pts: torch.Tensor,
    target_pts: torch.Tensor,
    threshold: float = 0.02
) -> Tuple[bool, float]:
    """
    Compute success rate for rope routing task

    Success criterion: Chamfer distance < threshold

    Args:
        final_pts: (N, 3) - Final point cloud
        target_pts: (M, 3) - Target point cloud
        threshold: Success threshold (meters)

    Returns:
        success: bool - Whether task succeeded
        chamfer: float - Chamfer distance value
    """
    chamfer = batch_chamfer_dist(final_pts[None], target_pts)[0].item()
    success = chamfer < threshold

    return success, chamfer


def fps_sample(pts: torch.Tensor, n_samples: int, device: str = 'cuda', random_start: bool = False) -> torch.Tensor:
    """
    Farthest Point Sampling

    Args:
        pts: (N, 3) - Input point cloud
        n_samples: Number of points to sample
        device: Device for computation
        random_start: Whether to use random start point

    Returns:
        indices: (n_samples,) - Sampled indices
    """
    try:
        from dgl.geometry import farthest_point_sampler
        import random

        if random_start:
            start_idx = random.randint(0, pts.shape[0] - 1)
        else:
            start_idx = 0

        fps_idx = farthest_point_sampler(
            pts[None], n_samples, start_idx=start_idx
        )[0]

        return fps_idx.to(device)

    except ImportError:
        print("[WARNING] DGL not available, using random sampling")
        indices = torch.randperm(pts.shape[0])[:n_samples]
        return indices.to(device)
