from typing import Dict, List, Optional

import numpy as np


def generate_planar_perturbation_trajectory(
    grasp_eef_xyz: np.ndarray,
    grasp_eef_rot: np.ndarray,
    config: Dict,
    rng: np.random.Generator,
) -> List[Dict]:
    """Generate planar perturbation trajectory from a grasp pose.

    Args:
        grasp_eef_xyz: Grasp end-effector position (3,).
        grasp_eef_rot: Grasp end-effector rotation (3, 3).
        config: Perturbation config containing translation_xy, rotation_z, num_waypoints, gripper_state.
        rng: Random number generator (unused; kept for API symmetry).

    Returns:
        List of waypoints with keys: xyz, rot, gripper. Uses teleop-style incremental updates
        (per-step delta in XY and Z rotation) to match keyboard control behavior.
    """
    translation_xy = np.array(config["translation_xy"], dtype=np.float32).reshape(2)
    rotation_z = float(config["rotation_z"])
    num_waypoints = int(config.get("num_waypoints", 10))
    gripper_state = float(config.get("gripper_state", 1.0))

    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1.")

    delta_xy = translation_xy / float(num_waypoints)
    delta_angle = np.deg2rad(rotation_z / float(num_waypoints))
    rot_z_delta = np.array(
        [
            [np.cos(delta_angle), -np.sin(delta_angle), 0.0],
            [np.sin(delta_angle), np.cos(delta_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    waypoints: List[Dict] = []
    current_xyz = grasp_eef_xyz.astype(np.float32).copy()
    current_rot = grasp_eef_rot.astype(np.float32).copy()
    for _ in range(num_waypoints):
        current_xyz = current_xyz.copy()
        current_xyz[0] += delta_xy[0]
        current_xyz[1] += delta_xy[1]
        current_rot = current_rot @ rot_z_delta
        waypoints.append({
            "xyz": current_xyz,
            "rot": current_rot,
            "gripper": gripper_state,
        })

    return waypoints


def generate_batch_perturbations(
    grasp_pose: Dict,
    config: Dict,
    num_samples: int,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Generate multiple planar perturbation parameter sets.

    Args:
        grasp_pose: Dict with grasp pose info (unused; kept for extensibility).
        config: Perturbation config with translation_range_xy, rotation_range_z,
            num_waypoints, and optional min_translation_norm/min_rotation_abs.
        num_samples: Number of samples to generate.
        seed: Optional random seed.

    Returns:
        List of perturbation parameter dicts.
    """
    rng = np.random.default_rng(seed)
    trans_min, trans_max = config["translation_range_xy"]
    rot_min, rot_max = config["rotation_range_z"]
    num_waypoints = int(config.get("num_waypoints", 10))
    gripper_state = float(config.get("gripper_state", 1.0))
    min_translation_norm = float(config.get("min_translation_norm", 0.0))
    min_rotation_abs = float(config.get("min_rotation_abs", 0.0))
    max_attempts = int(config.get("max_sampling_attempts", 1000))

    perturbations: List[Dict] = []
    for _ in range(num_samples):
        attempts = 0
        while True:
            translation_xy = rng.uniform(trans_min, trans_max, size=2).astype(np.float32)
            rotation_z = float(rng.uniform(rot_min, rot_max))
            translation_norm = float(np.linalg.norm(translation_xy))
            if translation_norm >= min_translation_norm and abs(rotation_z) >= min_rotation_abs:
                break
            attempts += 1
            if attempts >= max_attempts:
                raise ValueError(
                    "Failed to sample perturbation meeting minimum magnitude constraints."
                )
        perturbations.append({
            "translation_xy": translation_xy,
            "rotation_z": rotation_z,
            "num_waypoints": num_waypoints,
            "gripper_state": gripper_state,
        })

    return perturbations
