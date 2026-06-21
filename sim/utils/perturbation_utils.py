"""
Utility functions for applying perturbations (translation and rotation) to soft bodies in simulation.

This module provides tools to:
1. Apply custom perturbations to soft body objects
2. Save and load perturbation parameters
3. Verify perturbation consistency between rendering and physics
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import kornia


def create_perturbation_matrix(
    translation: Union[np.ndarray, list, tuple],
    rotation_deg: float,
    rotation_axis: str = 'z'
) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from translation and rotation.

    Args:
        translation: Translation offset [dx, dy, dz] in meters
        rotation_deg: Rotation angle in degrees
        rotation_axis: Rotation axis ('x', 'y', or 'z'). Default is 'z'

    Returns:
        4x4 transformation matrix

    Example:
        >>> T = create_perturbation_matrix([0.05, -0.03, 0.0], 15.0, 'z')
        >>> print(T.shape)
        (4, 4)
    """
    translation = np.array(translation, dtype=np.float32)
    assert translation.shape == (3,), "Translation must be [dx, dy, dz]"

    # Convert degrees to radians
    angle_rad = rotation_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Create rotation matrix based on axis
    if rotation_axis.lower() == 'z':
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
    elif rotation_axis.lower() == 'y':
        rot_matrix = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0    ],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)
    elif rotation_axis.lower() == 'x':
        rot_matrix = np.array([
            [1, 0,      0     ],
            [0, cos_a, -sin_a],
            [0, sin_a,  cos_a]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Invalid rotation axis: {rotation_axis}. Must be 'x', 'y', or 'z'")

    # Create 4x4 homogeneous transformation matrix
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rot_matrix
    T[:3, 3] = translation

    return T


def apply_perturbation_to_pose(
    base_pose: np.ndarray,
    translation: Union[np.ndarray, list, tuple],
    rotation_deg: float,
    rotation_axis: str = 'z'
) -> np.ndarray:
    """
    Apply perturbation (translation + rotation) to a base pose matrix.

    The perturbation is applied in world frame:
    1. First apply translation offset
    2. Then apply rotation offset (left multiplication)

    Args:
        base_pose: Base 4x4 pose matrix
        translation: Translation offset [dx, dy, dz] in meters
        rotation_deg: Rotation angle in degrees
        rotation_axis: Rotation axis ('x', 'y', or 'z'). Default is 'z'

    Returns:
        Perturbed 4x4 pose matrix

    Example:
        >>> base_pose = np.eye(4)
        >>> base_pose[:3, 3] = [0.4, 0.0, 0.05]
        >>> perturbed_pose = apply_perturbation_to_pose(base_pose, [0.05, 0.0, 0.0], 10.0)
    """
    base_pose = np.array(base_pose, dtype=np.float32).reshape(4, 4)
    translation = np.array(translation, dtype=np.float32)

    # Create perturbed pose
    perturbed_pose = base_pose.copy()

    # Step 1: Apply translation offset
    perturbed_pose[:3, 3] += translation

    # Step 2: Apply rotation offset (left multiplication for world frame rotation)
    angle_rad = rotation_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if rotation_axis.lower() == 'z':
        rot_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=np.float32)
    elif rotation_axis.lower() == 'y':
        rot_matrix = np.array([
            [cos_a,  0, sin_a],
            [0,      1, 0    ],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)
    elif rotation_axis.lower() == 'x':
        rot_matrix = np.array([
            [1, 0,      0     ],
            [0, cos_a, -sin_a],
            [0, sin_a,  cos_a]
        ], dtype=np.float32)
    else:
        raise ValueError(f"Invalid rotation axis: {rotation_axis}")

    perturbed_pose[:3, :3] = rot_matrix @ perturbed_pose[:3, :3]

    return perturbed_pose


def apply_perturbation_to_env(
    env,
    translation: Union[np.ndarray, list, tuple],
    rotation_deg: float,
    rotation_axis: str = 'z'
) -> Dict[str, np.ndarray]:
    """
    Apply perturbation to the soft body in an existing environment.

    This function modifies the environment's renderer state to apply
    the specified translation and rotation perturbation to the soft body.

    Args:
        env: Gymnasium environment instance (BaseEnv)
        translation: Translation offset [dx, dy, dz] in meters
        rotation_deg: Rotation angle in degrees
        rotation_axis: Rotation axis ('x', 'y', or 'z'). Default is 'z'

    Returns:
        Dictionary containing:
            - 'original_pose': Original pose matrix (4x4)
            - 'perturbed_pose': Perturbed pose matrix (4x4)
            - 'translation': Applied translation [dx, dy, dz]
            - 'rotation_deg': Applied rotation in degrees
            - 'rotation_axis': Rotation axis

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('BaseEnv-v0', cfg=cfg, ...)
        >>> obs, _ = env.reset(seed=0)
        >>>
        >>> # Apply 5cm translation in x and 10 degree rotation around z
        >>> info = apply_perturbation_to_env(env, [0.05, 0.0, 0.0], 10.0, 'z')
        >>> print(f"Applied translation: {info['translation']}")
        >>> print(f"Applied rotation: {info['rotation_deg']}°")
    """
    translation = np.array(translation, dtype=np.float32)
    device = env.renderer.device

    # Get original pose from renderer
    original_pose = env.renderer.pose_obj.cpu().numpy() if isinstance(env.renderer.pose_obj, torch.Tensor) else env.renderer.pose_obj.copy()

    # Apply perturbation to pose
    perturbed_pose = apply_perturbation_to_pose(original_pose, translation, rotation_deg, rotation_axis)

    # Convert to torch tensor
    perturbed_pose_torch = torch.from_numpy(perturbed_pose).to(torch.float32).to(device)

    # Update renderer's pose
    env.renderer.pose_obj = perturbed_pose_torch

    # Get current state
    state = env.renderer.get_state()
    original_x = state['x'].clone()

    # Create perturbation transformation matrix
    perturbation_matrix = create_perturbation_matrix(translation, rotation_deg, rotation_axis)
    perturbation_matrix_torch = torch.from_numpy(perturbation_matrix).to(torch.float32).to(device)

    # Apply perturbation to physics points
    # Transform: x_new = R @ x_old + t
    R = perturbation_matrix_torch[:3, :3]
    t = perturbation_matrix_torch[:3, 3]
    perturbed_x = original_x @ R.T + t

    # Update state with perturbed points
    state['x'] = perturbed_x
    state['v'] = torch.zeros_like(perturbed_x)

    # Update renderer state (this synchronizes GS rendering points via LBS)
    env.renderer.update_state(state)

    # Re-initialize physics with perturbed state
    state_for_physics = env.renderer.get_state()
    phystwin_pts = env.physics.reset(
        state_for_physics,
        init_meshes_dict=env.renderer.meshes,
        robot=env.renderer.robot,
        eef_pts_func=env.renderer.eef_pts_func,
        kin_helper=env.renderer.kin_helper,
        init_eef_xyz=env.renderer.init_eef_xyz,
        pose_obj=env.renderer.pose_obj
    )
    env.renderer.update_phystwin_pts(phystwin_pts)

    # Return perturbation info
    return {
        'original_pose': original_pose,
        'perturbed_pose': perturbed_pose,
        'translation': translation,
        'rotation_deg': rotation_deg,
        'rotation_axis': rotation_axis
    }


def apply_perturbation_to_renderer(
    renderer,
    translation: Union[np.ndarray, list, tuple],
    rotation_deg: float,
    rotation_axis: str = 'z',
    update_physics_points: bool = True
) -> Dict[str, np.ndarray]:
    """
    Apply perturbation directly to a renderer object.

    This is a lower-level function that operates on the renderer directly,
    useful when you need more control or are working outside of the env wrapper.

    Args:
        renderer: GSRenderer instance
        translation: Translation offset [dx, dy, dz] in meters
        rotation_deg: Rotation angle in degrees
        rotation_axis: Rotation axis ('x', 'y', or 'z'). Default is 'z'
        update_physics_points: Whether to update physics points (state['x'])

    Returns:
        Dictionary containing perturbation information

    Example:
        >>> info = apply_perturbation_to_renderer(
        ...     env.renderer,
        ...     translation=[0.05, -0.03, 0.0],
        ...     rotation_deg=15.0,
        ...     rotation_axis='z'
        ... )
    """
    translation = np.array(translation, dtype=np.float32)

    # Get original pose
    original_pose = renderer.pose_obj.cpu().numpy() if isinstance(renderer.pose_obj, torch.Tensor) else renderer.pose_obj.copy()

    # Apply perturbation
    perturbed_pose = apply_perturbation_to_pose(original_pose, translation, rotation_deg, rotation_axis)

    # Convert to torch
    device = renderer.device
    perturbed_pose_torch = torch.from_numpy(perturbed_pose).to(torch.float32).to(device)

    # Update pose
    renderer.pose_obj = perturbed_pose_torch

    # Get current Gaussian points
    xyz = renderer.rendervar['means3D']
    quat = renderer.rendervar['rotations']

    # Apply transformation to positions
    rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat)
    xyz_new = xyz @ perturbed_pose_torch[:3, :3].T + perturbed_pose_torch[:3, 3]

    # Apply transformation to rotations
    rot_new = perturbed_pose_torch[:3, :3] @ rot
    quat_new = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot_new)

    # Update rendervar
    renderer.rendervar['means3D'] = xyz_new
    renderer.rendervar['rotations'] = torch.nn.functional.normalize(quat_new, dim=-1)

    # Update physics points if requested
    if update_physics_points and renderer.state['x'] is not None:
        # Extract the physics points (first N points)
        n_physics = renderer.state['x'].shape[0]
        renderer.state['x'] = xyz_new[:n_physics].clone()

    return {
        'original_pose': original_pose,
        'perturbed_pose': perturbed_pose,
        'translation': translation,
        'rotation_deg': rotation_deg,
        'rotation_axis': rotation_axis
    }


def save_perturbation_params(
    save_path: Union[str, Path],
    translation: Union[np.ndarray, list, tuple],
    rotation_deg: float,
    rotation_axis: str = 'z',
    metadata: Optional[Dict] = None
) -> None:
    """
    Save perturbation parameters to a JSON file.

    Args:
        save_path: Path to save the JSON file
        translation: Translation offset [dx, dy, dz]
        rotation_deg: Rotation angle in degrees
        rotation_axis: Rotation axis
        metadata: Optional additional metadata to save

    Example:
        >>> save_perturbation_params(
        ...     'perturbation.json',
        ...     translation=[0.05, 0.0, 0.0],
        ...     rotation_deg=10.0,
        ...     metadata={'episode_id': 0, 'seed': 42}
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    translation = np.array(translation, dtype=np.float32)

    data = {
        'translation': translation.tolist(),
        'rotation_deg': float(rotation_deg),
        'rotation_axis': rotation_axis,
        'metadata': metadata or {}
    }

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Perturbation parameters saved to: {save_path}")


def load_perturbation_params(load_path: Union[str, Path]) -> Dict:
    """
    Load perturbation parameters from a JSON file.

    Args:
        load_path: Path to the JSON file

    Returns:
        Dictionary containing:
            - 'translation': Translation offset [dx, dy, dz]
            - 'rotation_deg': Rotation angle in degrees
            - 'rotation_axis': Rotation axis
            - 'metadata': Additional metadata

    Example:
        >>> params = load_perturbation_params('perturbation.json')
        >>> apply_perturbation_to_env(env, params['translation'], params['rotation_deg'])
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Perturbation file not found: {load_path}")

    with open(load_path, 'r') as f:
        data = json.load(f)

    return {
        'translation': np.array(data['translation'], dtype=np.float32),
        'rotation_deg': float(data['rotation_deg']),
        'rotation_axis': data.get('rotation_axis', 'z'),
        'metadata': data.get('metadata', {})
    }


def compute_perturbation_from_poses(
    original_pose: np.ndarray,
    perturbed_pose: np.ndarray,
    rotation_axis: str = 'z'
) -> Tuple[np.ndarray, float]:
    """
    Compute translation and rotation perturbation from two pose matrices.

    This is useful for extracting perturbation parameters from existing poses.

    Args:
        original_pose: Original 4x4 pose matrix
        perturbed_pose: Perturbed 4x4 pose matrix
        rotation_axis: Assumed rotation axis for angle extraction

    Returns:
        Tuple of (translation, rotation_deg):
            - translation: [dx, dy, dz] translation offset
            - rotation_deg: Rotation angle in degrees

    Example:
        >>> translation, rotation = compute_perturbation_from_poses(pose1, pose2)
        >>> print(f"Translation: {translation}, Rotation: {rotation}°")
    """
    original_pose = np.array(original_pose, dtype=np.float32).reshape(4, 4)
    perturbed_pose = np.array(perturbed_pose, dtype=np.float32).reshape(4, 4)

    # Compute translation difference
    translation = perturbed_pose[:3, 3] - original_pose[:3, 3]

    # Compute rotation difference
    # R_perturbed = R_delta @ R_original
    # R_delta = R_perturbed @ R_original^T
    R_delta = perturbed_pose[:3, :3] @ original_pose[:3, :3].T

    # Extract rotation angle based on axis
    if rotation_axis.lower() == 'z':
        # For Z-axis rotation: angle = atan2(R[1,0], R[0,0])
        rotation_rad = np.arctan2(R_delta[1, 0], R_delta[0, 0])
    elif rotation_axis.lower() == 'y':
        # For Y-axis rotation: angle = atan2(-R[2,0], R[0,0])
        rotation_rad = np.arctan2(-R_delta[2, 0], R_delta[0, 0])
    elif rotation_axis.lower() == 'x':
        # For X-axis rotation: angle = atan2(R[2,1], R[1,1])
        rotation_rad = np.arctan2(R_delta[2, 1], R_delta[1, 1])
    else:
        raise ValueError(f"Invalid rotation axis: {rotation_axis}")

    rotation_deg = rotation_rad * 180.0 / np.pi

    return translation, rotation_deg


def verify_perturbation_consistency(
    env,
    tolerance_translation: float = 1e-3,
    tolerance_rotation: float = 1e-2
) -> Dict[str, bool]:
    """
    Verify that rendering and physics have consistent perturbations.

    Args:
        env: Gymnasium environment instance
        tolerance_translation: Tolerance for translation difference (meters)
        tolerance_rotation: Tolerance for rotation difference (degrees)

    Returns:
        Dictionary containing verification results

    Example:
        >>> result = verify_perturbation_consistency(env)
        >>> if result['consistent']:
        ...     print("Rendering and physics are consistent!")
    """
    # Get renderer pose
    renderer_pose = env.renderer.pose_obj.cpu().numpy() if isinstance(env.renderer.pose_obj, torch.Tensor) else env.renderer.pose_obj

    # Get physics pose (if available)
    # Note: This assumes physics module stores the pose
    physics_pose = None
    if hasattr(env.physics, 'reset_metadata') and 'pose_obj' in env.physics.reset_metadata:
        physics_pose = env.physics.reset_metadata['pose_obj']
        if isinstance(physics_pose, torch.Tensor):
            physics_pose = physics_pose.cpu().numpy()

    if physics_pose is None:
        return {
            'consistent': None,
            'message': 'Physics pose not available for comparison'
        }

    # Compare poses
    translation_diff = np.linalg.norm(renderer_pose[:3, 3] - physics_pose[:3, 3])
    rotation_diff = np.linalg.norm(renderer_pose[:3, :3] - physics_pose[:3, :3])

    translation_ok = translation_diff < tolerance_translation
    rotation_ok = rotation_diff < tolerance_rotation

    return {
        'consistent': translation_ok and rotation_ok,
        'translation_diff': float(translation_diff),
        'rotation_diff': float(rotation_diff),
        'translation_ok': translation_ok,
        'rotation_ok': rotation_ok,
        'tolerance_translation': tolerance_translation,
        'tolerance_rotation': tolerance_rotation
    }


def apply_smooth_deformation_to_env(
    env,
    deformation_std: float,
    k_neighbors: int = 5,
    smoothing_iterations: int = 1
) -> Dict:
    """
    Apply smooth deformation perturbation using K-NN weighted averaging.

    This function creates spatially-coherent smooth deformations by:
    1. Generating base random perturbations
    2. Applying K-NN weighted smoothing to create spatial coherence
    3. Synchronizing physics and rendering via LBS

    Args:
        env: Gymnasium environment instance (BaseEnv)
        deformation_std: Standard deviation of base random noise (meters)
        k_neighbors: Number of nearest neighbors for smoothing (default: 5)
        smoothing_iterations: Number of smoothing passes (default: 1)

    Returns:
        Dictionary containing:
            - 'deformation_std': Applied deformation standard deviation
            - 'k_neighbors': Number of neighbors used
            - 'smoothing_iterations': Number of smoothing passes
            - 'mean_displacement': Mean displacement magnitude
            - 'max_displacement': Maximum displacement magnitude
            - 'displacement_variance': Smoothness metric (lower = smoother)
            - 'original_center': Original center of mass
            - 'deformed_center': Deformed center of mass
            - 'n_physics_points': Number of physics points

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('BaseEnv-v0', cfg=cfg, ...)
        >>> obs, _ = env.reset(seed=0)
        >>>
        >>> # Apply smooth deformation with K=5 neighbors
        >>> info = apply_smooth_deformation_to_env(env, 0.005, k_neighbors=5)
        >>> print(f"Mean displacement: {info['mean_displacement']*1000:.2f}mm")
        >>> print(f"Smoothness (variance): {info['displacement_variance']:.6f}")
    """
    device = env.renderer.device

    # 1. Get current state
    state = env.renderer.get_state()
    original_x = state['x'].clone()  # (N, 3)
    original_center = original_x.mean(dim=0)

    # 2. Generate base random perturbations
    torch.manual_seed(42)  # Reproducibility
    base_noise = torch.randn_like(original_x) * deformation_std  # (N, 3)

    # 3. Compute K-NN relations and weights
    # Reuse existing K-NN infrastructure pattern from gs_renderer.py
    dist = torch.norm(original_x[:, None] - original_x[None, :], dim=-1)  # (N, N)
    _, indices = torch.topk(dist, k_neighbors, dim=-1, largest=False)  # (N, k)

    # Compute inverse distance weights
    neighbors = original_x[indices]  # (N, k, 3)
    neighbor_dist = torch.norm(neighbors - original_x[:, None], dim=-1)  # (N, k)
    weights = 1.0 / (neighbor_dist + 1e-6)  # (N, k)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize

    # 4. Apply K-NN weighted smoothing
    smoothed_noise = base_noise.clone()
    for iteration in range(smoothing_iterations):
        # Gather neighbor perturbations
        neighbor_noise = smoothed_noise[indices]  # (N, k, 3)

        # Weighted average
        smoothed_noise = (weights[:, :, None] * neighbor_noise).sum(dim=1)  # (N, 3)

    # 5. Apply smoothed deformation
    deformed_x = original_x + smoothed_noise
    deformed_center = deformed_x.mean(dim=0)

    # 6. Compute statistics
    displacement = smoothed_noise.norm(dim=1)
    mean_displacement = displacement.mean().item()
    max_displacement = displacement.max().item()

    # Compute smoothness metric (displacement variance among neighbors)
    neighbor_displacement = displacement[indices]  # (N, k)
    displacement_variance = neighbor_displacement.var(dim=1).mean().item()

    # 7. Update state (same as random deformation)
    state['x'] = deformed_x
    state['v'] = torch.zeros_like(deformed_x)
    env.renderer.update_state(state)

    # 8. Re-initialize physics with deformed state
    state_for_physics = env.renderer.get_state()
    phystwin_pts = env.physics.reset(
        state_for_physics,
        init_meshes_dict=env.renderer.meshes,
        robot=env.renderer.robot,
        eef_pts_func=env.renderer.eef_pts_func,
        kin_helper=env.renderer.kin_helper,
        init_eef_xyz=env.renderer.init_eef_xyz,
        pose_obj=env.renderer.pose_obj
    )
    env.renderer.update_phystwin_pts(phystwin_pts)

    # 9. Return deformation info
    return {
        'deformation_std': deformation_std,
        'k_neighbors': k_neighbors,
        'smoothing_iterations': smoothing_iterations,
        'mean_displacement': mean_displacement,
        'max_displacement': max_displacement,
        'displacement_variance': displacement_variance,  # Smoothness measure
        'original_center': original_center.cpu().numpy(),
        'deformed_center': deformed_center.cpu().numpy(),
        'n_physics_points': original_x.shape[0]
    }


def apply_cylindrical_bend_to_env(
    env,
    bend_axis: str = 'z',
    bend_angle: float = np.pi/4,
    scale: float = 0.5
) -> Dict:
    """
    Apply cylindrical coordinate bending deformation to soft body.

    Uses the algorithm from deformation_base.py to create smooth geometric
    bending along a specified axis through cylindrical coordinate transformation.

    Args:
        env: Gymnasium environment instance (BaseEnv)
        bend_axis: Bending axis ('x', 'y', or 'z')
        bend_angle: Total bending angle in radians (e.g., π/4, π/2)
        scale: Curvature scaling coefficient (0.0-1.0, default 0.5)

    Returns:
        Dictionary containing:
            - 'bend_axis': Axis used for bending
            - 'bend_angle': Bending angle in radians
            - 'bend_angle_deg': Bending angle in degrees
            - 'scale': Curvature scale factor
            - 'mean_displacement': Mean displacement magnitude (meters)
            - 'max_displacement': Maximum displacement magnitude (meters)
            - 'original_center': Original center of mass
            - 'deformed_center': Deformed center of mass
            - 'n_physics_points': Number of physics points

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make('BaseEnv-v0', cfg=cfg, ...)
        >>> obs, _ = env.reset(seed=0)
        >>>
        >>> # Bend rope along Z-axis by 45 degrees
        >>> info = apply_cylindrical_bend_to_env(env, 'z', np.pi/4, 0.5)
        >>> print(f"Mean displacement: {info['mean_displacement']*1000:.2f}mm")
    """
    device = env.renderer.device

    # Step 1: Get current state (standard pattern)
    state = env.renderer.get_state()
    original_x = state['x'].clone()  # (N, 3) torch.Tensor on CUDA
    original_center = original_x.mean(dim=0)

    # Step 2: Convert PyTorch tensor to NumPy (CPU transfer)
    points_np = original_x.cpu().numpy()  # (N, 3) float32 -> float64

    # Step 3: Apply cylindrical bending algorithm (adapted from deformation_base.py)
    # Normalize coordinates to [-1, 1]
    min_p = points_np.min(axis=0)
    max_p = points_np.max(axis=0)
    points_norm = (points_np - min_p) / (max_p - min_p + 1e-8) * 2 - 1
    xn, yn, zn = points_norm[:, 0], points_norm[:, 1], points_norm[:, 2]

    # Apply cylindrical coordinate transformation based on axis
    if bend_axis.lower() == 'z':
        # Bend along Z-axis: XY plane polar angle gradient
        r = np.sqrt(xn ** 2 + yn ** 2)
        theta = np.arctan2(yn, xn)
        theta_delta = bend_angle * zn * scale
        theta_new = theta + theta_delta
        xn_new = r * np.cos(theta_new)
        yn_new = r * np.sin(theta_new)
        zn_new = zn
    elif bend_axis.lower() == 'x':
        # Bend along X-axis: YZ plane polar angle gradient
        r = np.sqrt(yn ** 2 + zn ** 2)
        theta = np.arctan2(zn, yn)
        theta_delta = bend_angle * xn * scale
        theta_new = theta + theta_delta
        yn_new = r * np.cos(theta_new)
        zn_new = r * np.sin(theta_new)
        xn_new = xn
    elif bend_axis.lower() == 'y':
        # Bend along Y-axis: XZ plane polar angle gradient
        r = np.sqrt(xn ** 2 + zn ** 2)
        theta = np.arctan2(zn, xn)
        theta_delta = bend_angle * yn * scale
        theta_new = theta + theta_delta
        xn_new = r * np.cos(theta_new)
        zn_new = r * np.sin(theta_new)
        yn_new = yn
    else:
        raise ValueError(f"Invalid bend_axis: {bend_axis}. Must be 'x', 'y', or 'z'")

    # Denormalize back to original scale
    bent_points_np = np.zeros_like(points_np)
    bent_points_np[:, 0] = (xn_new + 1) / 2 * (max_p[0] - min_p[0]) + min_p[0]
    bent_points_np[:, 1] = (yn_new + 1) / 2 * (max_p[1] - min_p[1]) + min_p[1]
    bent_points_np[:, 2] = (zn_new + 1) / 2 * (max_p[2] - min_p[2]) + min_p[2]

    # Step 4: Convert back to PyTorch tensor (GPU transfer)
    bent_x = torch.from_numpy(bent_points_np).to(
        dtype=torch.float32,
        device=device
    )

    # Step 5: Update state (standard synchronization pattern)
    state['x'] = bent_x
    state['v'] = torch.zeros_like(bent_x)
    env.renderer.update_state(state)  # Triggers LBS synchronization

    # Step 6: Re-initialize physics (CRITICAL for consistency)
    state_for_physics = env.renderer.get_state()
    phystwin_pts = env.physics.reset(
        state_for_physics,
        init_meshes_dict=env.renderer.meshes,
        robot=env.renderer.robot,
        eef_pts_func=env.renderer.eef_pts_func,
        kin_helper=env.renderer.kin_helper,
        init_eef_xyz=env.renderer.init_eef_xyz,
        pose_obj=env.renderer.pose_obj
    )
    env.renderer.update_phystwin_pts(phystwin_pts)

    # Step 7: Compute statistics
    deformed_center = bent_x.mean(dim=0)
    displacement = (bent_x - original_x).norm(dim=1)
    mean_displacement = displacement.mean().item()
    max_displacement = displacement.max().item()

    # Step 8: Return metadata
    return {
        'bend_axis': bend_axis,
        'bend_angle': bend_angle,
        'bend_angle_deg': float(np.degrees(bend_angle)),
        'scale': scale,
        'mean_displacement': mean_displacement,
        'max_displacement': max_displacement,
        'original_center': original_center.cpu().numpy(),
        'deformed_center': deformed_center.cpu().numpy(),
        'n_physics_points': original_x.shape[0]
    }


if __name__ == '__main__':
    # Example usage
    print("Perturbation Utils - Example Usage")
    print("=" * 80)

    # Example 1: Create perturbation matrix
    print("\n1. Create perturbation matrix:")
    T = create_perturbation_matrix([0.05, -0.03, 0.0], 15.0, 'z')
    print(f"Translation: [0.05, -0.03, 0.0] m")
    print(f"Rotation: 15° around Z-axis")
    print(f"Transformation matrix:\n{T}")

    # Example 2: Apply perturbation to pose
    print("\n2. Apply perturbation to base pose:")
    base_pose = np.eye(4)
    base_pose[:3, 3] = [0.4, 0.0, 0.05]
    perturbed_pose = apply_perturbation_to_pose(base_pose, [0.05, 0.0, 0.0], 10.0)
    print(f"Base position: {base_pose[:3, 3]}")
    print(f"Perturbed position: {perturbed_pose[:3, 3]}")

    # Example 3: Save and load parameters
    print("\n3. Save and load perturbation parameters:")
    save_perturbation_params(
        '/tmp/test_perturbation.json',
        translation=[0.05, -0.03, 0.0],
        rotation_deg=15.0,
        metadata={'test': True}
    )
    params = load_perturbation_params('/tmp/test_perturbation.json')
    print(f"Loaded parameters: {params}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
