#!/usr/bin/env python3
"""
Reference Trajectory Generation for MPPI

Generate reference trajectories for MPPI optimization from demonstrations.
MVP: Direct copy with translation offset
"""

import torch
import numpy as np
from typing import Dict, Optional


def generate_reference_trajectory_simple(
    demo_trajectory: Dict[str, torch.Tensor],
    t_start: int,
    horizon: int,
    translation_offset: Optional[torch.Tensor] = None,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate reference trajectory by copying demonstration (MVP version)

    Args:
        demo_trajectory: Demonstration trajectory
            'eef_xyz': (T, n_grippers, 3)
            'eef_rot': (T, n_grippers, 3, 3)
            'eef_gripper': (T, n_grippers, 1)
        t_start: Start timestep in demo (typically grasp time)
        horizon: Planning horizon (number of steps)
        translation_offset: Optional translation to apply (3,)
        verbose: Print debug info

    Returns:
        reference_traj: Reference trajectory for MPPI
            'eef_xyz': (horizon, n_grippers, 3)
            'eef_rot': (horizon, n_grippers, 3, 3)
            'eef_gripper': (horizon, n_grippers, 1)
    """
    T_demo = demo_trajectory['eef_xyz'].shape[0]
    device = demo_trajectory['eef_xyz'].device

    # Extract segment from demo
    t_end = min(t_start + horizon, T_demo)
    actual_horizon = t_end - t_start

    reference_traj = {
        'eef_xyz': demo_trajectory['eef_xyz'][t_start:t_end].clone(),
        'eef_rot': demo_trajectory['eef_rot'][t_start:t_end].clone(),
        'eef_gripper': demo_trajectory['eef_gripper'][t_start:t_end].clone(),
    }

    # If demo is shorter than horizon, repeat last pose
    if actual_horizon < horizon:
        n_repeat = horizon - actual_horizon
        last_xyz = reference_traj['eef_xyz'][-1:].repeat(n_repeat, 1, 1)
        last_rot = reference_traj['eef_rot'][-1:].repeat(n_repeat, 1, 1, 1)
        last_gripper = reference_traj['eef_gripper'][-1:].repeat(n_repeat, 1, 1)

        reference_traj['eef_xyz'] = torch.cat([reference_traj['eef_xyz'], last_xyz], dim=0)
        reference_traj['eef_rot'] = torch.cat([reference_traj['eef_rot'], last_rot], dim=0)
        reference_traj['eef_gripper'] = torch.cat([reference_traj['eef_gripper'], last_gripper], dim=0)

    # Apply translation offset if provided
    if translation_offset is not None:
        reference_traj['eef_xyz'] = reference_traj['eef_xyz'] + translation_offset.to(device)

        if verbose:
            print(f"  Applied translation offset: {translation_offset.cpu().numpy()}")

    if verbose:
        print(f"  Generated reference trajectory:")
        print(f"    Horizon: {horizon}")
        print(f"    Actual length: {reference_traj['eef_xyz'].shape[0]}")
        print(f"    Start position: {reference_traj['eef_xyz'][0, 0].cpu().numpy()}")
        print(f"    End position: {reference_traj['eef_xyz'][-1, 0].cpu().numpy()}")

    return reference_traj


def compute_trajectory_offset(
    demo_object_centroid: torch.Tensor,
    new_object_centroid: torch.Tensor
) -> torch.Tensor:
    """
    Compute translation offset between demo and new scene

    Args:
        demo_object_centroid: Demo object centroid (3,)
        new_object_centroid: New object centroid (3,)

    Returns:
        offset: Translation offset (3,)
    """
    return new_object_centroid - demo_object_centroid


def resample_trajectory(
    trajectory: Dict[str, torch.Tensor],
    new_length: int,
    method: str = 'linear'
) -> Dict[str, torch.Tensor]:
    """
    Resample trajectory to different length

    Args:
        trajectory: Input trajectory
        new_length: Desired length
        method: Interpolation method ('linear' or 'nearest')

    Returns:
        resampled_traj: Resampled trajectory
    """
    import torch.nn.functional as F

    T_old = trajectory['eef_xyz'].shape[0]
    n_grippers = trajectory['eef_xyz'].shape[1]

    # Create time indices
    t_old = torch.linspace(0, 1, T_old)
    t_new = torch.linspace(0, 1, new_length)

    # Interpolate positions
    xyz_flat = trajectory['eef_xyz'].reshape(T_old, -1).T.unsqueeze(0)  # (1, n_grippers*3, T)
    xyz_interp = F.interpolate(xyz_flat, size=new_length, mode='linear', align_corners=True)
    xyz_new = xyz_interp.squeeze(0).T.reshape(new_length, n_grippers, 3)

    # Gripper (simple linear)
    gripper_flat = trajectory['eef_gripper'].reshape(T_old, -1).T.unsqueeze(0)
    gripper_interp = F.interpolate(gripper_flat, size=new_length, mode='linear', align_corners=True)
    gripper_new = gripper_interp.squeeze(0).T.reshape(new_length, n_grippers, 1)

    # Rotation (nearest neighbor for simplicity in MVP)
    indices = (t_new.unsqueeze(1) - t_old.unsqueeze(0)).abs().argmin(dim=1)
    rot_new = trajectory['eef_rot'][indices]

    resampled_traj = {
        'eef_xyz': xyz_new,
        'eef_rot': rot_new,
        'eef_gripper': gripper_new
    }

    return resampled_traj


def visualize_reference_trajectory(
    demo_traj: Dict[str, torch.Tensor],
    reference_traj: Dict[str, torch.Tensor],
    save_path: str = 'reference_traj_vis.png'
):
    """
    Visualize demo vs reference trajectory

    Args:
        demo_traj: Demo trajectory
        reference_traj: Generated reference trajectory
        save_path: Save path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # Demo trajectory
    ax1 = fig.add_subplot(121, projection='3d')
    demo_xyz = demo_traj['eef_xyz'][:, 0].cpu().numpy()
    ax1.plot(demo_xyz[:, 0], demo_xyz[:, 1], demo_xyz[:, 2],
             'b-o', markersize=2, alpha=0.6, label='Demo')
    ax1.scatter([demo_xyz[0, 0]], [demo_xyz[0, 1]], [demo_xyz[0, 2]],
                c='green', s=100, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax1.set_title('Demo Trajectory')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # Reference trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ref_xyz = reference_traj['eef_xyz'][:, 0].cpu().numpy()
    ax2.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
             'r-o', markersize=2, alpha=0.6, label='Reference')
    ax2.scatter([ref_xyz[0, 0]], [ref_xyz[0, 1]], [ref_xyz[0, 2]],
                c='green', s=100, marker='*', label='Start', edgecolors='black', linewidths=2)
    ax2.set_title('Reference Trajectory (Transferred)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()

    # Match axis limits
    all_xyz = np.concatenate([demo_xyz, ref_xyz], axis=0)
    for ax in [ax1, ax2]:
        ax.set_xlim(all_xyz[:, 0].min() - 0.02, all_xyz[:, 0].max() + 0.02)
        ax.set_ylim(all_xyz[:, 1].min() - 0.02, all_xyz[:, 1].max() + 0.02)
        ax.set_zlim(all_xyz[:, 2].min() - 0.02, all_xyz[:, 2].max() + 0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ Reference trajectory visualization saved to: {save_path}")


if __name__ == '__main__':
    # Test reference trajectory generation
    print("Testing reference trajectory generation...")

    # Create dummy demo trajectory
    T_demo = 100
    demo_traj = {
        'eef_xyz': torch.randn(T_demo, 1, 3) * 0.01 + torch.tensor([0.3, 0.0, 0.15]),
        'eef_rot': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(T_demo, 1, 1, 1),
        'eef_gripper': torch.linspace(0, 0.8, T_demo).reshape(-1, 1, 1)
    }

    # Generate reference (starting from t=30, horizon=20)
    t_start = 30
    horizon = 20

    ref_traj = generate_reference_trajectory_simple(
        demo_traj, t_start, horizon, verbose=True
    )

    # With translation offset
    offset = torch.tensor([0.05, -0.03, 0.02])
    ref_traj_offset = generate_reference_trajectory_simple(
        demo_traj, t_start, horizon, translation_offset=offset, verbose=True
    )

    # Test resampling
    ref_resampled = resample_trajectory(ref_traj, new_length=50)
    print(f"\n  Resampled trajectory length: {ref_resampled['eef_xyz'].shape[0]}")

    # Visualize
    visualize_reference_trajectory(demo_traj, ref_traj_offset, 'test_reference_traj.png')
