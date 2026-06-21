#!/usr/bin/env python3
"""
State Perturbation for Generalization

Generate perturbed initial object states for testing trajectory generalization.
"""

import torch
import numpy as np
from typing import Literal, Optional


def generate_perturbed_state(
    init_pts: torch.Tensor,
    method: Literal['spatial', 'rotation', 'deformation', 'combined'] = 'spatial',
    translation_std: float = 0.02,  # 2cm standard deviation
    rotation_range: float = 15.0,   # ±15 degrees
    deformation_std: float = 0.005, # 5mm deformation
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Generate perturbed initial state from demonstration initial state

    Args:
        init_pts: Initial point cloud (N, 3)
        method: Perturbation method
            - 'spatial': Translation only
            - 'rotation': Rotation around Z-axis
            - 'deformation': Local deformation
            - 'combined': All three
        translation_std: Standard deviation for translation (meters)
        rotation_range: Rotation range in degrees (±)
        deformation_std: Standard deviation for deformation (meters)
        seed: Random seed for reproducibility

    Returns:
        perturbed_pts: Perturbed point cloud (N, 3)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = init_pts.device
    perturbed_pts = init_pts.clone()

    # Spatial translation
    if method in ['spatial', 'combined']:
        delta_xyz = torch.randn(3, device=device) * translation_std
        perturbed_pts = perturbed_pts + delta_xyz

    # Rotation around Z-axis
    if method in ['rotation', 'combined']:
        centroid = perturbed_pts.mean(dim=0)
        angle = (torch.rand(1, device=device) * 2 - 1) * (rotation_range * np.pi / 180)

        # Rotation matrix around Z-axis
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        R_z = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], device=device).squeeze()

        perturbed_pts = (perturbed_pts - centroid) @ R_z.T + centroid

    # Local deformation
    if method in ['deformation', 'combined']:
        noise = torch.randn_like(perturbed_pts) * deformation_std
        perturbed_pts = perturbed_pts + noise

    return perturbed_pts


def physics_stabilize(
    env,
    init_pts: torch.Tensor,
    steps: int = 30,
    verbose: bool = False
) -> torch.Tensor:
    """
    Stabilize perturbed state using physics simulation

    Args:
        env: Environment instance
        init_pts: Perturbed point cloud (N, 3)
        steps: Number of stabilization steps
        verbose: Print progress

    Returns:
        stabilized_pts: Stabilized point cloud (N, 3)
    """
    # Set physics state
    env.physics.dynamics_module.current_points = init_pts.clone()

    # Get current gripper state
    state = env.renderer.get_state()
    eef_xyz = state['eef_xyz']
    eef_quat = state['eef_quat']
    eef_gripper = state['eef_gripper']

    # Convert quaternion to rotation matrix
    import kornia
    eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)

    # Create hold-in-place action
    action = torch.cat([
        eef_xyz,
        eef_rot.reshape(eef_rot.shape[0], -1),
        eef_gripper
    ], dim=1)

    # Run stabilization
    if verbose:
        print(f"    Stabilizing physics for {steps} steps...")

    for _ in range(steps):
        env.step({'action': action, 'do_velocity_control': False})

    # Get stabilized state
    stabilized_pts = env.physics.dynamics_module.current_points.clone()

    return stabilized_pts


def visualize_perturbation(
    original_pts: torch.Tensor,
    perturbed_pts: torch.Tensor,
    save_path: str = 'perturbation_vis.png'
):
    """
    Visualize original and perturbed states

    Args:
        original_pts: Original point cloud (N, 3)
        perturbed_pts: Perturbed point cloud (N, 3)
        save_path: Save path for visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    pts = original_pts.cpu().numpy()
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='blue', s=1, alpha=0.6)
    ax1.set_title('Original State')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')

    # Perturbed
    ax2 = fig.add_subplot(122, projection='3d')
    pts_pert = perturbed_pts.cpu().numpy()
    ax2.scatter(pts_pert[:, 0], pts_pert[:, 1], pts_pert[:, 2], c='red', s=1, alpha=0.6)
    ax2.set_title('Perturbed State')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')

    # Match axis limits
    all_pts = np.concatenate([pts, pts_pert], axis=0)
    for ax in [ax1, ax2]:
        ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
        ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
        ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  ✓ Visualization saved to: {save_path}")


if __name__ == '__main__':
    # Test perturbation
    print("Testing state perturbation...")

    # Create dummy point cloud
    init_pts = torch.randn(1000, 3) * 0.05 + torch.tensor([0.3, 0.0, 0.05])

    # Test different methods
    methods = ['spatial', 'rotation', 'deformation', 'combined']

    for method in methods:
        perturbed = generate_perturbed_state(init_pts, method=method, seed=42)

        # Compute statistics
        diff = perturbed - init_pts
        trans = perturbed.mean(dim=0) - init_pts.mean(dim=0)

        print(f"\n{method.upper()}:")
        print(f"  Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}] m")
        print(f"  Mean displacement: {diff.norm(dim=1).mean():.4f} m")
        print(f"  Max displacement: {diff.norm(dim=1).max():.4f} m")

    # Visualize
    perturbed = generate_perturbed_state(init_pts, method='combined', seed=42)
    visualize_perturbation(init_pts, perturbed, 'test_perturbation.png')
