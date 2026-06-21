from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import kornia


def estimate_object_center(env) -> np.ndarray:
    """Estimate object center from renderer physics points.

    Args:
        env: Environment or renderer with a get_state() method.

    Returns:
        Object center in world coordinates as (3,) float32 numpy array.
    """
    renderer = getattr(env, "renderer", env)
    state = renderer.get_state()
    points = state.get("x", None)
    if points is None:
        raise ValueError("Renderer state does not include 'x' points.")

    if torch.is_tensor(points):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = np.asarray(points)

    if points_np.size == 0:
        raise ValueError("Renderer state 'x' is empty.")
    return points_np.mean(axis=0).astype(np.float32)


def generate_random_collision_trajectory(
    current_eef_xyz: np.ndarray,
    current_eef_rot: np.ndarray,
    object_center: np.ndarray,
    config: Dict[str, Union[float, List[float]]],
    rng: Optional[np.random.Generator] = None,
) -> List[Dict[str, Union[np.ndarray, float]]]:
    """Generate a random collision trajectory toward the object.

    Args:
        current_eef_xyz: Current end-effector position (3,).
        current_eef_rot: Current end-effector rotation matrix (3, 3).
        object_center: Object center in world coordinates (3,).
        config: Collision config dictionary.
        rng: Optional numpy random generator for reproducibility.

    Returns:
        List of waypoint dicts with keys: xyz, rot, gripper.
    """
    if rng is None:
        rng = np.random.default_rng()

    direction = object_center.astype(np.float32) - current_eef_xyz.astype(np.float32)
    direction = _safe_normalize(direction, "object-center minus eef")

    approach_randomness = float(config.get("approach_randomness", 0.0))
    if approach_randomness > 0:
        random_offset = rng.normal(size=3).astype(np.float32) * approach_randomness
        direction = _safe_normalize(direction + random_offset, "randomized approach")

    collision_distance = float(config.get("collision_distance", 0.08))
    distance_range = config.get("collision_distance_range", None)
    if distance_range is not None:
        min_d = float(distance_range[0])
        max_d = float(distance_range[1])
        collision_distance = max(min_d, min(max_d, collision_distance))

    num_waypoints = int(config.get("num_waypoints", 8))
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be >= 1.")

    retract_distance = float(config.get("retract_distance", 0.08))
    gripper = float(config.get("gripper", 1.0))

    trajectory: List[Dict[str, Union[np.ndarray, float]]] = []
    for i in range(num_waypoints):
        t = (i + 1) / num_waypoints
        xyz = current_eef_xyz + direction * collision_distance * t
        trajectory.append({
            "xyz": xyz.astype(np.float32),
            "rot": current_eef_rot.astype(np.float32),
            "gripper": gripper,
        })

    n_retract = num_waypoints // 2
    if retract_distance > 0 and n_retract > 0:
        final_xyz = trajectory[-1]["xyz"]
        for i in range(n_retract):
            t = (i + 1) / n_retract
            xyz = final_xyz - direction * retract_distance * t
            trajectory.append({
                "xyz": xyz.astype(np.float32),
                "rot": current_eef_rot.astype(np.float32),
                "gripper": gripper,
            })

    return trajectory


def execute_collision_sequence(
    env,
    trajectory: List[Dict[str, Union[np.ndarray, float]]],
    stabilization_steps: int = 20,
    do_velocity_control: bool = False,
    step_callback: Optional[Callable[[Dict, torch.Tensor], None]] = None,
) -> Dict[str, Union[float, int, bool, List[float]]]:
    """Execute a collision trajectory in simulation and stabilize.

    Args:
        env: Unwrapped environment with step(), cfg, and physics.device.
        trajectory: Collision trajectory waypoints.
        stabilization_steps: Number of post-collision stabilization steps.
        do_velocity_control: Whether to use velocity control in env.step.
        step_callback: Optional callback invoked per step with (obs, action).

    Returns:
        Dict with collision metadata.
    """
    object_center_before = estimate_object_center(env)
    last_action = None
    last_obs = None

    for waypoint in trajectory:
        action = _waypoint_to_action(env, waypoint)
        last_action = action
        last_obs, _, _, _, _ = env.step({
            "action": action,
            "do_velocity_control": do_velocity_control,
        })
        if step_callback is not None:
            step_callback(last_obs, action)

    for _ in range(max(int(stabilization_steps), 0)):
        if last_action is None:
            break
        last_obs, _, _, _, _ = env.step({
            "action": last_action,
            "do_velocity_control": do_velocity_control,
        })
        if step_callback is not None:
            step_callback(last_obs, last_action)

    object_center_after = estimate_object_center(env)
    displacement = float(np.linalg.norm(object_center_after - object_center_before))

    return {
        "trajectory_length": len(trajectory),
        "stabilization_steps": int(stabilization_steps),
        "collision_detected": displacement > 0.005,
        "displacement": displacement,
        "final_object_center": object_center_after.tolist(),
    }


def _safe_normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        raise ValueError(f"Cannot normalize near-zero vector for {label}.")
    return vector / norm


def _waypoint_to_action(env, waypoint: Dict[str, Union[np.ndarray, float]]) -> torch.Tensor:
    cfg = getattr(env, "cfg", None)
    n_grippers = int(cfg.env.robot.n_grippers) if cfg is not None else 1
    use_pusher = bool(cfg.env.robot.use_pusher) if cfg is not None else False
    device = env.physics.device if hasattr(env, "physics") else torch.device("cpu")

    eef_xyz = np.asarray(waypoint["xyz"], dtype=np.float32)
    if eef_xyz.ndim == 1:
        eef_xyz = eef_xyz.reshape(1, 3)
    eef_rot = np.asarray(waypoint["rot"], dtype=np.float32)
    if eef_rot.ndim == 2:
        eef_rot = eef_rot.reshape(1, 3, 3)
    gripper_val = float(waypoint.get("gripper", 1.0))
    eef_gripper = np.full((eef_xyz.shape[0], 1), gripper_val, dtype=np.float32)

    if eef_xyz.shape[0] != n_grippers:
        if n_grippers == 1:
            eef_xyz = eef_xyz[:1]
            eef_rot = eef_rot[:1]
            eef_gripper = eef_gripper[:1]
        else:
            raise ValueError("Waypoint gripper count does not match env cfg.")

    if use_pusher:
        pos_z = 0.22
        eef_xyz = np.concatenate([
            eef_xyz[:, :2],
            pos_z * np.ones((n_grippers, 1), dtype=np.float32),
        ], axis=1)
        quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 4)
        quat = np.repeat(quat, n_grippers, axis=0)
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.from_numpy(quat)
        ).cpu().numpy()
        eef_gripper = np.zeros((n_grippers, 1), dtype=np.float32)

    action_np = np.concatenate([
        eef_xyz.reshape(n_grippers, 3),
        eef_rot.reshape(n_grippers, 9),
        eef_gripper,
    ], axis=1)
    return torch.from_numpy(action_np).to(device=device, dtype=torch.float32)
