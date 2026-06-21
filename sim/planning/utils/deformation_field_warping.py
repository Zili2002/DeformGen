from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import transforms3d
from transforms3d import _gohlketransforms as gt

from sim.planning.utils.rigid_transformation import get_object_points_from_env, points_to_world
from sim.utils.deformation_bridge import DeformationBridge


def compute_deformation_field(
    p_orig: torch.Tensor,
    p_def: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-point deformation vectors.

    Args:
        p_orig: Original point cloud (N, 3) in world frame.
        p_def: Deformed point cloud (N, 3) in world frame.

    Returns:
        Deformation vectors delta (N, 3).
    """
    if p_orig.shape != p_def.shape:
        raise ValueError(f"Point shape mismatch: {p_orig.shape} vs {p_def.shape}")
    return p_def - p_orig


def interpolate_deformation_at_points(
    query_points: torch.Tensor,
    p_orig: torch.Tensor,
    delta: torch.Tensor,
    k_neighbors: int = 5,
) -> torch.Tensor:
    """
    Interpolate deformation vectors at query points using KNN inverse-distance weights.

    Args:
        query_points: Query points (M, 3) in world frame.
        p_orig: Original point cloud (N, 3) in world frame.
        delta: Deformation vectors for p_orig (N, 3).
        k_neighbors: Number of neighbors for interpolation.

    Returns:
        Interpolated deformation vectors (M, 3).
    """
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(f"Query points must be (M, 3), got {query_points.shape}")
    if p_orig.shape != delta.shape:
        raise ValueError(f"delta shape mismatch: {delta.shape} vs {p_orig.shape}")

    k = min(int(k_neighbors), int(p_orig.shape[0]))
    dist = torch.cdist(query_points, p_orig)
    knn_dist, knn_idx = torch.topk(dist, k, dim=1, largest=False)
    weights = 1.0 / (knn_dist + 1e-6)
    weights = weights / weights.sum(dim=1, keepdim=True)

    neighbor_delta = delta[knn_idx]  # (M, k, 3)
    return (weights[:, :, None] * neighbor_delta).sum(dim=1)


def compute_local_deformation_gradient(
    query_points: torch.Tensor,
    p_orig: torch.Tensor,
    delta: torch.Tensor,
    k_neighbors: int = 8,
) -> torch.Tensor:
    """
    Compute local deformation gradients (Jacobians) via least squares.

    Args:
        query_points: Query points (M, 3) in world frame.
        p_orig: Original point cloud (N, 3) in world frame.
        delta: Deformation vectors for p_orig (N, 3).
        k_neighbors: Number of neighbors for local fitting.

    Returns:
        Local Jacobians (M, 3, 3).
    """
    if p_orig.shape != delta.shape:
        raise ValueError(f"delta shape mismatch: {delta.shape} vs {p_orig.shape}")

    k = min(int(k_neighbors), int(p_orig.shape[0]))
    dist = torch.cdist(query_points, p_orig)
    _, knn_idx = torch.topk(dist, k, dim=1, largest=False)

    jacobians = []
    for i in range(query_points.shape[0]):
        idx = knn_idx[i]
        query = query_points[i]
        local_orig = p_orig[idx] - query
        local_def = local_orig + delta[idx]

        x_orig = local_orig.T
        x_def = local_def.T
        mat = x_orig @ x_orig.T
        mat_pinv = torch.linalg.pinv(mat)
        j = x_def @ x_orig.T @ mat_pinv
        jacobians.append(j)

    return torch.stack(jacobians, dim=0)


def _orthogonalize_rotations(rotations: torch.Tensor) -> torch.Tensor:
    """
    Orthonormalize rotation matrices using SVD.

    Args:
        rotations: (M, 3, 3) rotation matrices.

    Returns:
        Orthonormalized rotations (M, 3, 3).
    """
    u, _, vt = torch.linalg.svd(rotations)
    r = u @ vt
    det = torch.det(r)
    if torch.any(det < 0):
        u[det < 0, :, -1] *= -1
        r = u @ vt
    return r


def warp_orientation_with_jacobian(
    rot_mats: torch.Tensor,
    jacobians: torch.Tensor,
) -> torch.Tensor:
    """
    Warp orientation matrices using local Jacobians and orthogonalize.

    Args:
        rot_mats: Rotation matrices (M, 3, 3).
        jacobians: Local Jacobians (M, 3, 3).

    Returns:
        Warped and orthogonalized rotations (M, 3, 3).
    """
    warped = jacobians @ rot_mats
    return _orthogonalize_rotations(warped)


def _decay_factor(
    step: int,
    total_steps: int,
    decay_mode: str,
    decay_rate: float,
) -> float:
    if decay_mode == "none":
        return 1.0
    if total_steps <= 1:
        return 1.0
    if decay_mode == "linear":
        return 1.0 - float(step) / float(total_steps - 1)
    if decay_mode == "exponential":
        return float(np.exp(-decay_rate * step))
    raise ValueError(f"Unknown decay_mode: {decay_mode}")


def warp_trajectory_with_decay(
    eef_xyz: torch.Tensor,
    eef_rot: Optional[torch.Tensor],
    p_orig: torch.Tensor,
    p_def: torch.Tensor,
    k_neighbors: int = 5,
    decay_mode: str = "none",
    decay_rate: float = 0.1,
    adapt_orientation: bool = False,
    disable_z_warp: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Warp a trajectory using a deformation field with optional decay and orientation adaptation.

    Args:
        eef_xyz: End-effector positions (T, 1, 3) in world frame.
        eef_rot: End-effector rotations (T, 1, 3, 3) in world frame.
        p_orig: Original point cloud (N, 3) in world frame.
        p_def: Deformed point cloud (N, 3) in world frame.
        k_neighbors: Number of neighbors for deformation interpolation.
        decay_mode: "none", "linear", or "exponential".
        decay_rate: Exponential decay rate (only used for decay_mode="exponential").
        adapt_orientation: Whether to adapt orientations using local Jacobians.
        disable_z_warp: If True, remove Z-axis deformation from warping.
        jacobian_query_point:
            Optional query point (3,) in world frame for Jacobian estimation.
            If None, use warped grasp position.

    Returns:
        Dict with warped 'eef_xyz' and 'eef_rot' (if provided).
    """
    if eef_xyz.ndim != 3 or eef_xyz.shape[-1] != 3:
        raise ValueError(f"eef_xyz must be (T, 1, 3), got {eef_xyz.shape}")

    t_steps = eef_xyz.shape[0]
    eef_xyz_flat = eef_xyz[:, 0]
    delta = compute_deformation_field(p_orig, p_def)
    if disable_z_warp:
        delta = delta.clone()
        delta[:, 2] = 0.0
    delta_query = interpolate_deformation_at_points(eef_xyz_flat, p_orig, delta, k_neighbors)

    warped_xyz = torch.empty_like(eef_xyz_flat)
    for i in range(t_steps):
        alpha = _decay_factor(i, t_steps, decay_mode, decay_rate)
        warped_xyz[i] = eef_xyz_flat[i] + delta_query[i] * alpha

    warped_rot = None
    if eef_rot is not None:
        rot_flat = eef_rot[:, 0]
        if adapt_orientation:
            jacobians = compute_local_deformation_gradient(
                eef_xyz_flat, p_orig, delta, k_neighbors=max(4, k_neighbors)
            )
            warped_full = warp_orientation_with_jacobian(rot_flat, jacobians)
            warped_rot = torch.empty_like(rot_flat)
            for i in range(t_steps):
                alpha = _decay_factor(i, t_steps, decay_mode, decay_rate)
                quat_start = transforms3d.quaternions.mat2quat(rot_flat[i].detach().cpu().numpy())
                quat_end = transforms3d.quaternions.mat2quat(warped_full[i].detach().cpu().numpy())
                interp_quat = gt.quaternion_slerp(
                    quat_start, quat_end, alpha, shortestpath=True
                )
                warped_rot[i] = torch.from_numpy(
                    transforms3d.quaternions.quat2mat(interp_quat)
                ).to(rot_flat.device, dtype=rot_flat.dtype)
        else:
            warped_rot = rot_flat.clone()

    out = {"eef_xyz": warped_xyz.unsqueeze(1)}
    if warped_rot is not None:
        out["eef_rot"] = warped_rot.unsqueeze(1)
    return out


def warp_grasp_pose(
    grasp_xyz: torch.Tensor,
    grasp_rot: torch.Tensor,
    p_orig: torch.Tensor,
    p_def: torch.Tensor,
    k_neighbors: int = 5,
    adapt_orientation: bool = False,
    disable_z_warp: bool = False,
    jacobian_query_point: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp a single grasp pose using the deformation field.

    Args:
        grasp_xyz: Grasp position (3,) in world frame.
        grasp_rot: Grasp rotation (3, 3) in world frame.
        p_orig: Original point cloud (N, 3) in world frame.
        p_def: Deformed point cloud (N, 3) in world frame.
        k_neighbors: Number of neighbors for interpolation.
        adapt_orientation: Whether to adapt orientation using local Jacobian.
        disable_z_warp: If True, remove Z-axis deformation from warping.

    Returns:
        Warped (grasp_xyz, grasp_rot).
    """
    delta = compute_deformation_field(p_orig, p_def)
    if disable_z_warp:
        delta = delta.clone()
        delta[:, 2] = 0.0
    delta_query = interpolate_deformation_at_points(
        grasp_xyz[None, :], p_orig, delta, k_neighbors
    )[0]
    warped_xyz = grasp_xyz + delta_query

    warped_rot = grasp_rot
    if adapt_orientation:
        jacobian_query = warped_xyz if jacobian_query_point is None else jacobian_query_point
        jac = compute_local_deformation_gradient(
            jacobian_query[None, :], p_orig, delta, k_neighbors=max(4, k_neighbors)
        )[0]
        warped_rot = warp_orientation_with_jacobian(grasp_rot[None, ...], jac[None, ...])[0]
    return warped_xyz, warped_rot


def load_deformation_for_warping(
    deformation_path: str,
    env,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load original and deformed point clouds in world frame.

    Args:
        deformation_path: Path to deformation .npy/.npz file.
        env: Gym environment.

    Returns:
        (p_orig, p_def) in world frame.
    """
    p_orig = get_object_points_from_env(env)

    raw_state = np.load(deformation_path, allow_pickle=True)
    if isinstance(raw_state, np.lib.npyio.NpzFile):
        raw_state = {key: raw_state[key] for key in raw_state.files}
    elif isinstance(raw_state, np.ndarray) and raw_state.shape == ():
        raw_state = raw_state.item()

    validated = DeformationBridge.validate_state(raw_state)
    points_np = validated["points"]
    metadata = validated["metadata"]

    points = torch.from_numpy(points_np).to(torch.float32).to(p_orig.device)
    frame = str(metadata.get("frame", "object"))
    pose_applied = bool(metadata.get("pose_applied", False))

    p_def = points_to_world(
        points=points,
        frame=frame,
        pose_obj=env.renderer.pose_obj,
        table_height=float(env.cfg.physics.table_height),
        pose_applied=pose_applied,
    )
    return p_orig, p_def
