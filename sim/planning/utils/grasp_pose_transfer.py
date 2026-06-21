#!/usr/bin/env python3
"""
Grasp Pose Transfer for Trajectory Generalization

Extract grasp pose from demonstration and transfer to new initial states.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


def detect_grasp_moment(
    gripper_values: torch.Tensor,
    threshold: float = 0.01,
    window: int = 3,
    min_close_value: float = 0.01,
) -> int:
    """
    Detect when grasp happens (gripper closing ends).

    Args:
        gripper_values: Gripper openness values (T,)
        threshold: Threshold for considering the change small (plateau).
        window: Number of consecutive steps with small change to accept.
        min_close_value: Minimum gripper value to consider closing started.

    Returns:
        t_grasp: Time step when grasp happens
    """
    if gripper_values.numel() < 2:
        return len(gripper_values) // 2

    start_mask = gripper_values > float(min_close_value)
    if not bool(start_mask.any()):
        return len(gripper_values) // 2

    start_idx = int(torch.where(start_mask)[0][0].item())
    diffs = torch.abs(gripper_values[1:] - gripper_values[:-1])
    window = max(int(window), 1)
    max_idx = len(gripper_values) - 1

    for idx in range(start_idx, max_idx):
        end = idx + window
        if end > diffs.numel():
            break
        if bool((diffs[idx:end] < float(threshold)).all()):
            return min(idx + 1, max_idx)

    return max_idx

    return t_grasp


def extract_grasp_pose_relative(
    demo_trajectory: Dict[str, torch.Tensor],
    demo_init_pts: torch.Tensor,
    grasp_radius: float = 0.05,
    verbose: bool = False
) -> Dict:
    """
    Extract grasp pose in object-relative coordinates (Simplified MVP version)

    Args:
        demo_trajectory: Demonstration trajectory
            'eef_xyz': (T, n_grippers, 3)
            'eef_rot': (T, n_grippers, 3, 3)
            'eef_gripper': (T, n_grippers, 1)
        demo_init_pts: Initial object point cloud (N, 3)
        grasp_radius: Radius to define grasp region (meters)
        verbose: Print debug info

    Returns:
        relative_grasp: Dict containing:
            - 'delta_position': Relative position from object centroid
            - 'absolute_rotation': EEF rotation (simplified - not relative)
            - 'grasp_time': Time step of grasp
            - 'object_centroid': Object centroid at grasp time
    """
    # 1. Detect grasp moment
    gripper_values = demo_trajectory['eef_gripper'][:, 0, 0]  # (T,)
    t_grasp = detect_grasp_moment(gripper_values)

    if verbose:
        print(f"  Detected grasp at timestep: {t_grasp}")

    # 2. Get EEF pose at grasp
    p_eef_demo = demo_trajectory['eef_xyz'][t_grasp, 0]  # (3,)
    R_eef_demo = demo_trajectory['eef_rot'][t_grasp, 0]  # (3, 3)

    # 3. Compute object centroid (simplified - using initial state)
    # In full version, should use object state at time t_grasp
    # Ensure same device
    c_object = demo_init_pts.mean(dim=0).to(p_eef_demo.device)

    # 4. Compute relative pose (simplified MVP)
    delta_position = p_eef_demo - c_object

    if verbose:
        print(f"  EEF position: {p_eef_demo.cpu().numpy()}")
        print(f"  Object centroid: {c_object.cpu().numpy()}")
        print(f"  Relative offset: {delta_position.cpu().numpy()}")

    relative_grasp = {
        'delta_position': delta_position,
        'absolute_rotation': R_eef_demo,  # Simplified: not making it relative
        'gripper_value': demo_trajectory['eef_gripper'][t_grasp, 0, 0],
        'grasp_time': t_grasp,
        'object_centroid': c_object,
    }

    return relative_grasp


def compute_new_grasp_pose(
    new_init_pts: torch.Tensor,
    relative_grasp: Dict,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Compute grasp pose for new initial state (Simplified MVP version)

    Args:
        new_init_pts: New initial object point cloud (N, 3)
        relative_grasp: Relative grasp information from extract_grasp_pose_relative
        verbose: Print debug info

    Returns:
        new_grasp_pose: Dict containing:
            - 'eef_xyz': New EEF position (3,)
            - 'eef_rot': New EEF rotation (3, 3)
            - 'gripper': Gripper value
    """
    # 1. Compute new object centroid
    c_new = new_init_pts.mean(dim=0)

    # Get device from delta_position
    device = relative_grasp['delta_position'].device

    # 2. Transfer grasp pose (simplified - only translate, no rotation adaptation)
    p_eef_new = c_new.to(device) + relative_grasp['delta_position']
    R_eef_new = relative_grasp['absolute_rotation']

    if verbose:
        print(f"  New object centroid: {c_new.cpu().numpy()}")
        print(f"  New EEF position: {p_eef_new.cpu().numpy()}")
        print(f"  Translation: {(c_new.to(device) - relative_grasp['object_centroid']).cpu().numpy()}")

    new_grasp_pose = {
        'eef_xyz': p_eef_new,
        'eef_rot': R_eef_new,
        'gripper': relative_grasp['gripper_value'],
        'grasp_time': relative_grasp['grasp_time']
    }

    return new_grasp_pose


def extract_manipulation_trajectory(
    demo_trajectory: Dict[str, torch.Tensor],
    t_grasp: int,
    horizon: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract manipulation portion of trajectory (after grasp)

    Args:
        demo_trajectory: Full demonstration trajectory
        t_grasp: Grasp timestep
        horizon: Max length to extract (if None, use all remaining)

    Returns:
        manipulation_traj: Trajectory starting from grasp
    """
    T = demo_trajectory['eef_xyz'].shape[0]

    # Extract from grasp to end (or horizon)
    end_idx = min(t_grasp + horizon, T) if horizon else T

    manipulation_traj = {
        'eef_xyz': demo_trajectory['eef_xyz'][t_grasp:end_idx],
        'eef_rot': demo_trajectory['eef_rot'][t_grasp:end_idx],
        'eef_gripper': demo_trajectory['eef_gripper'][t_grasp:end_idx],
    }

    return manipulation_traj


def visualize_grasp_transfer(
    demo_init_pts: torch.Tensor,
    demo_grasp_pose: Dict,
    new_init_pts: torch.Tensor,
    new_grasp_pose: Dict,
    save_path: str = 'grasp_transfer_vis.png'
):
    """
    Visualize grasp pose transfer

    Args:
        demo_init_pts: Demo object point cloud
        demo_grasp_pose: Dict with original grasp 'eef_xyz'
        new_init_pts: New object point cloud
        new_grasp_pose: Dict with new grasp 'eef_xyz'
        save_path: Save path
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 6))

    # Demo state
    ax1 = fig.add_subplot(121, projection='3d')
    pts = demo_init_pts.cpu().numpy()
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, alpha=0.5, label='Object')

    # Demo grasp
    eef_pos = demo_grasp_pose['eef_xyz'].cpu().numpy() if 'eef_xyz' in demo_grasp_pose else demo_grasp_pose['delta_position'].cpu().numpy() + demo_init_pts.mean(dim=0).cpu().numpy()
    ax1.scatter([eef_pos[0]], [eef_pos[1]], [eef_pos[2]],
                c='red', s=100, marker='*', label='Grasp', edgecolors='black', linewidths=2)

    ax1.set_title('Demo Grasp')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # New state
    ax2 = fig.add_subplot(122, projection='3d')
    pts_new = new_init_pts.cpu().numpy()
    ax2.scatter(pts_new[:, 0], pts_new[:, 1], pts_new[:, 2], c='green', s=1, alpha=0.5, label='Object')

    # New grasp
    eef_new = new_grasp_pose['eef_xyz'].cpu().numpy()
    ax2.scatter([eef_new[0]], [eef_new[1]], [eef_new[2]],
                c='red', s=100, marker='*', label='Transferred Grasp', edgecolors='black', linewidths=2)

    ax2.set_title('Transferred Grasp')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.legend()

    # Match axis limits
    all_pts = np.concatenate([pts, pts_new], axis=0)
    all_eef = np.stack([eef_pos, eef_new], axis=0)
    all_coords = np.concatenate([all_pts, all_eef], axis=0)

    for ax in [ax1, ax2]:
        ax.set_xlim(all_coords[:, 0].min() - 0.02, all_coords[:, 0].max() + 0.02)
        ax.set_ylim(all_coords[:, 1].min() - 0.02, all_coords[:, 1].max() + 0.02)
        ax.set_zlim(all_coords[:, 2].min() - 0.02, all_coords[:, 2].max() + 0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ Grasp transfer visualization saved to: {save_path}")


if __name__ == '__main__':
    # Test grasp pose transfer
    print("Testing grasp pose transfer...")

    # Create dummy data
    demo_init_pts = torch.randn(1000, 3) * 0.05 + torch.tensor([0.3, 0.0, 0.05])
    T = 100

    demo_traj = {
        'eef_xyz': torch.randn(T, 1, 3) * 0.01 + torch.tensor([0.3, 0.0, 0.15]),
        'eef_rot': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(T, 1, 1, 1),
        'eef_gripper': torch.cat([
            torch.zeros(30, 1, 1),
            torch.linspace(0, 0.8, 70).reshape(-1, 1, 1)
        ], dim=0)
    }

    # Extract relative grasp
    relative_grasp = extract_grasp_pose_relative(demo_traj, demo_init_pts, verbose=True)

    # Create perturbed state
    new_init_pts = demo_init_pts + torch.tensor([0.03, -0.02, 0.01])

    # Compute new grasp
    new_grasp = compute_new_grasp_pose(new_init_pts, relative_grasp, verbose=True)

    # Visualize
    demo_grasp_vis = {
        'eef_xyz': demo_traj['eef_xyz'][relative_grasp['grasp_time'], 0]
    }
    visualize_grasp_transfer(demo_init_pts, demo_grasp_vis, new_init_pts, new_grasp, 'test_grasp_transfer.png')
