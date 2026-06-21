from __future__ import annotations

from typing import Dict, Optional, Tuple, Any

import torch


def compute_optimal_rigid_transform(
    p_orig: torch.Tensor,
    p_def: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_grasp_region: bool = False,
    grasp_center: Optional[torch.Tensor] = None,
    grasp_radius: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the optimal rigid transform (R, t) using the Kabsch algorithm.

    Args:
        p_orig: Original point cloud (N, 3).
        p_def: Deformed point cloud (N, 3).
        weights: Optional point weights (N,).
        use_grasp_region: If True, only use points within grasp_radius.
        grasp_center: Grasp center in the same frame as p_orig/p_def.
        grasp_radius: Radius for selecting grasp region points (meters).

    Returns:
        R: Rotation matrix (3, 3).
        t: Translation vector (3,).
    """
    if p_orig.shape != p_def.shape:
        raise ValueError(f"Point shape mismatch: {p_orig.shape} vs {p_def.shape}")
    if p_orig.ndim != 2 or p_orig.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {p_orig.shape}")

    if use_grasp_region and grasp_center is not None:
        dist = torch.norm(p_orig - grasp_center, dim=1)
        mask = dist < grasp_radius
        if mask.sum().item() < 3:
            raise ValueError("Not enough points in grasp region to estimate transform.")
        p_orig = p_orig[mask]
        p_def = p_def[mask]
        if weights is not None:
            weights = weights[mask]

    if weights is None:
        weights = torch.ones(p_orig.shape[0], device=p_orig.device)
    weights = weights / (weights.sum() + 1e-8)

    centroid_orig = (weights[:, None] * p_orig).sum(dim=0)
    centroid_def = (weights[:, None] * p_def).sum(dim=0)

    P = p_orig - centroid_orig
    Q = p_def - centroid_def

    H = (P.T * weights) @ Q
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.T

    R = V @ U.T
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    t = centroid_def - R @ centroid_orig
    return R, t


def compute_transformation_error(
    p_orig: torch.Tensor,
    p_def: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute alignment error statistics for a rigid transform.

    Args:
        p_orig: Original point cloud (N, 3).
        p_def: Target point cloud (N, 3).
        R: Rotation matrix (3, 3).
        t: Translation vector (3,).

    Returns:
        Error statistics dict.
    """
    p_transformed = (p_orig @ R.T) + t
    errors = torch.norm(p_transformed - p_def, dim=1)

    return {
        "rmse": torch.sqrt((errors ** 2).mean()).item(),
        "max_error": errors.max().item(),
        "mean_error": errors.mean().item(),
        "median_error": errors.median().item(),
        "std_error": errors.std().item(),
    }


def apply_rigid_transform_to_trajectory(
    trajectory: Dict[str, torch.Tensor],
    R: torch.Tensor,
    t: torch.Tensor,
    transform_orientation: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Apply rigid transform to an end-effector trajectory.

    Args:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'.
        R: Rotation matrix (3, 3).
        t: Translation vector (3,).
        transform_orientation: If True, transform eef_rot by R.

    Returns:
        Transformed trajectory dict.
    """
    eef_xyz = trajectory["eef_xyz"]
    eef_rot = trajectory["eef_rot"]
    eef_gripper = trajectory["eef_gripper"]

    R = R.to(eef_xyz.device)
    t = t.to(eef_xyz.device)

    transformed_xyz = torch.matmul(eef_xyz, R.T) + t

    if transform_orientation:
        transformed_rot = torch.matmul(R, eef_rot)
    else:
        transformed_rot = eef_rot.clone()

    return {
        "eef_xyz": transformed_xyz,
        "eef_rot": transformed_rot,
        "eef_gripper": eef_gripper.clone(),
    }


def points_to_world(
    points: torch.Tensor,
    frame: str,
    pose_obj: torch.Tensor,
    table_height: float,
    pose_applied: bool,
) -> torch.Tensor:
    """
    Convert points to world frame following DeformationBridge semantics.

    Args:
        points: Point cloud (N, 3).
        frame: Frame name ('object', 'model', or 'world').
        pose_obj: Object pose in world frame (4, 4).
        table_height: Table height used in the renderer (meters).
        pose_applied: Whether pose_obj has already been applied.

    Returns:
        Points in world frame.
    """
    frame_norm = frame.lower()
    if frame_norm not in {"object", "model", "world"}:
        raise ValueError(f"Unknown frame: {frame}")

    world_points = points
    if frame_norm == "model":
        global_translation = torch.tensor(
            [0.0, 0.0, -table_height],
            dtype=points.dtype,
            device=points.device,
        )
        world_points = points - global_translation

    if frame_norm in {"model", "object"} and not pose_applied:
        world_points = world_points @ pose_obj[:3, :3].T + pose_obj[:3, 3]

    return world_points


def get_object_points_from_env(env: Any) -> torch.Tensor:
    """
    Extract object points (world frame) from a real2sim environment state.

    Args:
        env: Gym environment with renderer and physics modules.

    Returns:
        Object point cloud (N, 3).
    """
    if hasattr(env, "physics") and hasattr(env.physics, "dynamics_module"):
        return env.physics.dynamics_module.current_points.clone()

    state = env.renderer.get_state()
    if "v" in state:
        num_object_points = state["v"].shape[0]
    else:
        simulator = env.physics.dynamics_module.simulator
        num_object_points = int(
            getattr(simulator, "num_object_points", state["x"].shape[0])
        )

    return state["x"][:num_object_points].clone()
