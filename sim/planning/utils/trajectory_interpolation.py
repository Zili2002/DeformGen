#!/usr/bin/env python3
"""
Trajectory Interpolation for Manipulation Phase

Generate manipulation trajectories using interpolation instead of MPPI.
This provides a simpler baseline for trajectory generalization.

Interpolation methods:
1. Linear interpolation: Straight line from grasp to target
2. Cubic spline: Smooth curve through waypoints
3. Demo-based: Interpolate from demonstration trajectory
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Dict, Optional, Literal


def interpolate_linear(
    start_pose: torch.Tensor,
    end_pose: torch.Tensor,
    num_steps: int,
    device: str = 'cuda:0'
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between start and end poses.

    Args:
        start_pose: (13,) tensor [xyz(3), rot_mat(9), gripper(1)]
        end_pose: (13,) tensor
        num_steps: Number of interpolation steps
        device: Device for tensors

    Returns:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
    """
    # Extract components
    start_xyz = start_pose[:3]
    start_rot = start_pose[3:12].reshape(3, 3)
    start_gripper = start_pose[12:13]

    end_xyz = end_pose[:3]
    end_rot = end_pose[3:12].reshape(3, 3)
    end_gripper = end_pose[12:13]

    # Linear interpolation for position
    t = torch.linspace(0, 1, num_steps, device=device)
    xyz_traj = start_xyz[None, :] + t[:, None] * (end_xyz - start_xyz)[None, :]

    # Linear interpolation for rotation (simple, not geodesic)
    rot_traj = torch.zeros(num_steps, 3, 3, device=device)
    for i in range(num_steps):
        rot_traj[i] = start_rot + t[i] * (end_rot - start_rot)
        # Normalize to maintain orthogonality (approximate)
        U, _, Vt = torch.linalg.svd(rot_traj[i])
        rot_traj[i] = U @ Vt

    # Linear interpolation for gripper
    gripper_traj = start_gripper + t[:, None] * (end_gripper - start_gripper)

    return {
        'eef_xyz': xyz_traj.unsqueeze(1),  # (T, 1, 3)
        'eef_rot': rot_traj.unsqueeze(1),  # (T, 1, 3, 3)
        'eef_gripper': gripper_traj.unsqueeze(1),  # (T, 1, 1)
    }


def interpolate_cubic_spline(
    waypoints: torch.Tensor,
    num_steps: int,
    device: str = 'cuda:0'
) -> Dict[str, torch.Tensor]:
    """
    Cubic spline interpolation through waypoints.

    Args:
        waypoints: (N, 13) tensor of waypoint poses
        num_steps: Number of interpolation steps
        device: Device for tensors

    Returns:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
    """
    N = waypoints.shape[0]

    # Extract components
    xyz_waypoints = waypoints[:, :3].cpu().numpy()
    rot_waypoints = waypoints[:, 3:12].reshape(N, 3, 3).cpu().numpy()
    gripper_waypoints = waypoints[:, 12:13].cpu().numpy()

    # Parameter values for waypoints
    t_waypoints = np.linspace(0, 1, N)
    t_interp = np.linspace(0, 1, num_steps)

    # Cubic spline for position
    cs_xyz = CubicSpline(t_waypoints, xyz_waypoints)
    xyz_traj = torch.from_numpy(cs_xyz(t_interp)).float().to(device)

    # Linear interpolation for rotation (cubic spline on rotation matrices is complex)
    rot_traj = torch.zeros(num_steps, 3, 3, device=device)
    for i in range(num_steps):
        # Find surrounding waypoints
        idx = np.searchsorted(t_waypoints, t_interp[i])
        if idx == 0:
            rot_traj[i] = torch.from_numpy(rot_waypoints[0]).float().to(device)
        elif idx >= N:
            rot_traj[i] = torch.from_numpy(rot_waypoints[-1]).float().to(device)
        else:
            # Linear interpolation between waypoints
            alpha = (t_interp[i] - t_waypoints[idx-1]) / (t_waypoints[idx] - t_waypoints[idx-1])
            rot_start = torch.from_numpy(rot_waypoints[idx-1]).float().to(device)
            rot_end = torch.from_numpy(rot_waypoints[idx]).float().to(device)
            rot_traj[i] = rot_start + alpha * (rot_end - rot_start)
            # Normalize
            U, _, Vt = torch.linalg.svd(rot_traj[i])
            rot_traj[i] = U @ Vt

    # Cubic spline for gripper
    cs_gripper = CubicSpline(t_waypoints, gripper_waypoints)
    gripper_traj = torch.from_numpy(cs_gripper(t_interp)).float().to(device)

    return {
        'eef_xyz': xyz_traj.unsqueeze(1),  # (T, 1, 3)
        'eef_rot': rot_traj.unsqueeze(1),  # (T, 1, 3, 3)
        'eef_gripper': gripper_traj.unsqueeze(1),  # (T, 1, 1)
    }


def interpolate_from_demo(
    demo_trajectory: Dict[str, torch.Tensor],
    start_idx: int,
    num_steps: int,
    translation_offset: Optional[torch.Tensor] = None,
    time_scaling: float = 1.0,
    device: str = 'cuda:0'
) -> Dict[str, torch.Tensor]:
    """
    Interpolate trajectory from demonstration with time scaling.

    Args:
        demo_trajectory: Demo trajectory dict
        start_idx: Start index in demo trajectory
        num_steps: Number of steps to generate
        translation_offset: Optional translation to apply
        time_scaling: Time scaling factor (>1 = slower, <1 = faster)
        device: Device for tensors

    Returns:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
    """
    demo_eef_xyz = demo_trajectory['eef_xyz'][:, 0]  # (T, 3)
    demo_eef_rot = demo_trajectory['eef_rot'][:, 0]  # (T, 3, 3)
    demo_eef_gripper = demo_trajectory['eef_gripper'][:, 0]  # (T, 1)

    T_demo = demo_eef_xyz.shape[0]

    # Calculate demo segment length
    demo_length = int((T_demo - start_idx) * time_scaling)

    # Source indices in demo (with time scaling)
    if demo_length >= num_steps:
        # Demo is longer, sample from it
        src_indices = torch.linspace(start_idx, T_demo - 1, num_steps, device=device)
    else:
        # Demo is shorter, extend by repeating last pose
        src_indices = torch.cat([
            torch.linspace(start_idx, T_demo - 1, demo_length, device=device),
            torch.full((num_steps - demo_length,), T_demo - 1, device=device)
        ])

    # Interpolate using indices
    src_indices_floor = src_indices.long()
    src_indices_ceil = torch.clamp(src_indices_floor + 1, max=T_demo - 1)
    alpha = (src_indices - src_indices_floor.float()).unsqueeze(-1)

    # Move demo tensors to device if needed
    demo_eef_xyz = demo_eef_xyz.to(device)
    demo_eef_rot = demo_eef_rot.to(device)
    demo_eef_gripper = demo_eef_gripper.to(device)

    # Interpolate position
    xyz_floor = demo_eef_xyz[src_indices_floor]
    xyz_ceil = demo_eef_xyz[src_indices_ceil]
    xyz_traj = xyz_floor + alpha * (xyz_ceil - xyz_floor)

    # Apply translation offset
    if translation_offset is not None:
        xyz_traj = xyz_traj + translation_offset.to(device)

    # Interpolate rotation
    rot_floor = demo_eef_rot[src_indices_floor]
    rot_ceil = demo_eef_rot[src_indices_ceil]
    rot_traj = torch.zeros(num_steps, 3, 3, device=device)
    for i in range(num_steps):
        rot_traj[i] = rot_floor[i] + alpha[i, 0] * (rot_ceil[i] - rot_floor[i])
        # Normalize
        U, _, Vt = torch.linalg.svd(rot_traj[i])
        rot_traj[i] = U @ Vt

    # Interpolate gripper
    gripper_floor = demo_eef_gripper[src_indices_floor]
    gripper_ceil = demo_eef_gripper[src_indices_ceil]
    gripper_traj = gripper_floor + alpha * (gripper_ceil - gripper_floor)

    return {
        'eef_xyz': xyz_traj.unsqueeze(1),  # (T, 1, 3)
        'eef_rot': rot_traj.unsqueeze(1),  # (T, 1, 3, 3)
        'eef_gripper': gripper_traj.unsqueeze(1),  # (T, 1, 1)
    }


def generate_manipulation_trajectory(
    method: Literal['linear', 'cubic', 'demo'],
    num_steps: int,
    device: str = 'cuda:0',
    # For linear interpolation
    start_pose: Optional[torch.Tensor] = None,
    end_pose: Optional[torch.Tensor] = None,
    # For cubic spline
    waypoints: Optional[torch.Tensor] = None,
    # For demo-based
    demo_trajectory: Optional[Dict[str, torch.Tensor]] = None,
    start_idx: Optional[int] = None,
    translation_offset: Optional[torch.Tensor] = None,
    time_scaling: float = 1.0,
    verbose: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate manipulation trajectory using interpolation.

    Args:
        method: Interpolation method ('linear', 'cubic', 'demo')
        num_steps: Number of steps to generate
        device: Device for tensors
        start_pose: Start pose for linear interpolation
        end_pose: End pose for linear interpolation
        waypoints: Waypoints for cubic spline
        demo_trajectory: Demo trajectory for demo-based interpolation
        start_idx: Start index in demo
        translation_offset: Translation offset for demo-based
        time_scaling: Time scaling for demo-based
        verbose: Print debug info

    Returns:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
    """
    if verbose:
        print(f"Generating trajectory using {method} interpolation...")
        print(f"  Target steps: {num_steps}")

    if method == 'linear':
        assert start_pose is not None and end_pose is not None, \
            "Linear interpolation requires start_pose and end_pose"
        trajectory = interpolate_linear(start_pose, end_pose, num_steps, device)

    elif method == 'cubic':
        assert waypoints is not None, \
            "Cubic spline requires waypoints"
        trajectory = interpolate_cubic_spline(waypoints, num_steps, device)

    elif method == 'demo':
        assert demo_trajectory is not None and start_idx is not None, \
            "Demo-based interpolation requires demo_trajectory and start_idx"
        trajectory = interpolate_from_demo(
            demo_trajectory, start_idx, num_steps,
            translation_offset, time_scaling, device
        )

    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    if verbose:
        print(f"  Generated trajectory:")
        print(f"    Shape: {trajectory['eef_xyz'].shape}")
        print(f"    Start position: {trajectory['eef_xyz'][0, 0].cpu().numpy()}")
        print(f"    End position: {trajectory['eef_xyz'][-1, 0].cpu().numpy()}")
        print(f"    Gripper range: [{trajectory['eef_gripper'].min():.3f}, "
              f"{trajectory['eef_gripper'].max():.3f}]")

    return trajectory


if __name__ == '__main__':
    # Test interpolation methods
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Testing interpolation methods...")
    print("=" * 70)

    # Test 1: Linear interpolation
    print("\n1. Linear Interpolation")
    print("-" * 70)
    start_pose = torch.tensor([
        0.3, 0.0, 0.2,  # xyz
        1, 0, 0, 0, 1, 0, 0, 0, 1,  # rotation (identity)
        0.0  # gripper (open)
    ], device=device)

    end_pose = torch.tensor([
        0.3, 0.0, 0.1,  # xyz (moved down)
        1, 0, 0, 0, 1, 0, 0, 0, 1,  # rotation (identity)
        1.0  # gripper (closed)
    ], device=device)

    traj_linear = generate_manipulation_trajectory(
        method='linear',
        num_steps=10,
        start_pose=start_pose,
        end_pose=end_pose,
        device=device,
        verbose=True
    )

    # Test 2: Cubic spline
    print("\n2. Cubic Spline Interpolation")
    print("-" * 70)
    waypoints = torch.stack([
        start_pose,
        torch.tensor([0.3, 0.05, 0.15, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.5], device=device),
        end_pose
    ])

    traj_cubic = generate_manipulation_trajectory(
        method='cubic',
        num_steps=10,
        waypoints=waypoints,
        device=device,
        verbose=True
    )

    # Test 3: Demo-based (simulated)
    print("\n3. Demo-based Interpolation")
    print("-" * 70)
    demo_traj = {
        'eef_xyz': torch.randn(20, 1, 3, device=device) * 0.1 + torch.tensor([0.3, 0.0, 0.15], device=device),
        'eef_rot': torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(20, 1, 1, 1),
        'eef_gripper': torch.linspace(0, 1, 20, device=device).unsqueeze(1).unsqueeze(1)
    }

    traj_demo = generate_manipulation_trajectory(
        method='demo',
        num_steps=15,
        demo_trajectory=demo_traj,
        start_idx=5,
        translation_offset=torch.tensor([0.0, 0.05, 0.0], device=device),
        time_scaling=1.2,
        device=device,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
