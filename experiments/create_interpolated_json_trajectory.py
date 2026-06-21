#!/usr/bin/env python3
"""
Create Interpolated JSON Trajectory from Demo

This script creates an interpolated trajectory in JSON format (compatible with replay.py)
directly from the original demo JSON files, including the approach phase.

The output format is identical to episode_0001/robot/*.json, so it can be directly
used with experiments/replay.py.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import OmegaConf
import torch
import json
import numpy as np
import glob
from tqdm import tqdm
import gymnasium as gym
import transforms3d
from transforms3d import _gohlketransforms as gt

import sim.envs
from sim.planning.utils.deformation_field_warping import (
    compute_deformation_field,
    compute_local_deformation_gradient,
    load_deformation_for_warping,
    warp_orientation_with_jacobian,
    warp_grasp_pose,
    warp_trajectory_with_decay,
)
from sim.planning.utils.trajectory_loader import load_replay_trajectory, extract_states_from_replay
from sim.planning.utils.rigid_transformation import (
    apply_rigid_transform_to_trajectory,
    compute_optimal_rigid_transform,
    compute_transformation_error,
    get_object_points_from_env,
    points_to_world,
)
from sim.utils.deformation_bridge import DeformationBridge


def interpolate_trajectory(demo_trajectory, num_steps=118, start_idx=None, perturbation_offset=None):
    """
    Interpolate demo trajectory to specified number of steps

    Args:
        demo_trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
        num_steps: Number of steps for interpolated trajectory
        start_idx: Starting index in demo (if None, auto-detect grasp point)
        perturbation_offset: [dx, dy, dz] offset for perturbation (grasp point offset, final point unchanged)

    Returns:
        interpolated: Dict with interpolated trajectory
        start_idx: The starting index used
    """
    demo_xyz = demo_trajectory['eef_xyz']  # (T_demo, 1, 3)
    demo_rot = demo_trajectory['eef_rot']  # (T_demo, 1, 3, 3)
    demo_gripper = demo_trajectory['eef_gripper']  # (T_demo, 1, 1)

    T_demo = demo_xyz.shape[0]

    # Auto-detect grasp point if not provided
    if start_idx is None:
        # Find the last frame where gripper is continuously closing
        # Strategy: Find the last frame in the continuous closing sequence
        gripper_values = demo_gripper[:, 0, 0].cpu().numpy()

        # Calculate gripper velocity (rate of change)
        gripper_velocity = np.diff(gripper_values)

        # Find frames where gripper is closing (velocity > threshold)
        closing_threshold = 0.01  # Gripper is actively closing
        closing_frames = np.where(gripper_velocity > closing_threshold)[0]

        if len(closing_frames) > 0:
            # Find continuous closing sequences
            # A sequence is continuous if frame indices are consecutive
            sequences = []
            current_seq = [closing_frames[0]]

            for i in range(1, len(closing_frames)):
                if closing_frames[i] == closing_frames[i-1] + 1:
                    # Consecutive frame, extend current sequence
                    current_seq.append(closing_frames[i])
                else:
                    # Gap detected, start new sequence
                    sequences.append(current_seq)
                    current_seq = [closing_frames[i]]
            sequences.append(current_seq)  # Add last sequence

            # Find the longest continuous closing sequence
            longest_seq = max(sequences, key=len)

            # Use the last frame of the longest closing sequence as grasp point
            start_idx = longest_seq[-1] + 1  # +1 because velocity is computed with diff

            print(f"  Auto-detected grasp point at frame {start_idx}")
            print(f"    Gripper value: {gripper_values[start_idx]:.4f}")
            print(f"    Last frame of continuous closing sequence (length: {len(longest_seq)} frames)")
            print(f"    Closing sequence: frames {longest_seq[0]} to {longest_seq[-1]}")
        else:
            # No closing detected, use beginning
            start_idx = 0
            print(f"  Warning: No gripper closing detected, using frame 0")

    # Extract manipulation phase (from start_idx to end)
    manip_xyz = demo_xyz[start_idx:]
    manip_rot = demo_rot[start_idx:]
    manip_gripper = demo_gripper[start_idx:]

    T_manip = manip_xyz.shape[0]
    print(f"  Manipulation phase: {T_manip} frames (demo[{start_idx}:{T_demo}])")

    # Interpolate to num_steps
    indices = torch.linspace(0, T_manip - 1, num_steps)

    # Interpolate position
    interp_xyz = torch.zeros((num_steps, 1, 3), dtype=demo_xyz.dtype, device=demo_xyz.device)
    for i in range(num_steps):
        idx = indices[i]
        idx_low = int(torch.floor(idx))
        idx_high = min(int(torch.ceil(idx)), T_manip - 1)
        alpha = idx - idx_low

        if idx_low == idx_high:
            interp_xyz[i] = manip_xyz[idx_low]
        else:
            interp_xyz[i] = (1 - alpha) * manip_xyz[idx_low] + alpha * manip_xyz[idx_high]

    # Use original rotations (no interpolation) for manipulation phase
    interp_rot = torch.zeros((num_steps, 1, 3, 3), dtype=demo_rot.dtype, device=demo_rot.device)
    for i in range(num_steps):
        idx = indices[i]
        idx_low = int(torch.floor(idx))
        interp_rot[i] = manip_rot[idx_low]

    # Interpolate gripper
    interp_gripper = torch.zeros((num_steps, 1, 1), dtype=demo_gripper.dtype, device=demo_gripper.device)
    for i in range(num_steps):
        idx = indices[i]
        idx_low = int(torch.floor(idx))
        idx_high = min(int(torch.ceil(idx)), T_manip - 1)
        alpha = idx - idx_low

        if idx_low == idx_high:
            interp_gripper[i] = manip_gripper[idx_low]
        else:
            interp_gripper[i] = (1 - alpha) * manip_gripper[idx_low] + alpha * manip_gripper[idx_high]

    # Apply perturbation offset with linear decay (full at start, zero at end)
    if perturbation_offset is not None:
        offset = torch.tensor(perturbation_offset, dtype=interp_xyz.dtype, device=interp_xyz.device).reshape(1, 1, 3)
        print(f"  Applying perturbation offset: {perturbation_offset}")
        print(f"    Start position (before): {interp_xyz[0, 0].cpu().numpy()}")
        print(f"    End position (unchanged): {interp_xyz[-1, 0].cpu().numpy()}")

        for i in range(num_steps):
            # alpha goes from 0 (start) to 1 (end)
            alpha = i / (num_steps - 1) if num_steps > 1 else 1.0
            # Offset decays from full at start to zero at end
            decay = 1.0 - alpha
            interp_xyz[i] = interp_xyz[i] + offset * decay

        print(f"    Start position (after): {interp_xyz[0, 0].cpu().numpy()}")
        print(f"    End position (after): {interp_xyz[-1, 0].cpu().numpy()}")

    return {
        'eef_xyz': interp_xyz,
        'eef_rot': interp_rot,
        'eef_gripper': interp_gripper,
    }, start_idx


def load_grasp_override_from_sample(
    sample_dir: Path,
    frame_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Load grasp pose override from a sample directory.

    Args:
        sample_dir: Directory containing metadata.json and episode_0000/robot/*.json.
        frame_idx: Optional explicit frame index. If None, use metadata grasp_timestep.

    Returns:
        Dict with 'eef_xyz' and 'eef_rot' tensors shaped (1, 1, 3) and (1, 1, 3, 3).
    """
    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {sample_dir}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if frame_idx is None:
        frame_idx = metadata.get("grasp_timestep", None)
    if frame_idx is None:
        raise ValueError(f"grasp_timestep missing in {metadata_path}")

    robot_path = sample_dir / "episode_0000" / "robot" / f"{int(frame_idx):06d}.json"
    if not robot_path.exists():
        raise FileNotFoundError(f"robot frame not found: {robot_path}")
    with open(robot_path, "r") as f:
        robot = json.load(f)

    if "action.ee_pos" in robot and "action.ee_quat" in robot:
        pos = np.array(robot["action.ee_pos"], dtype=np.float32)
        quat = np.array(robot["action.ee_quat"], dtype=np.float32)
    elif "obs.ee_pos" in robot and "obs.ee_quat" in robot:
        pos = np.array(robot["obs.ee_pos"], dtype=np.float32)
        quat = np.array(robot["obs.ee_quat"], dtype=np.float32)
    else:
        raise KeyError(f"Missing ee pose in {robot_path}")

    rot = transforms3d.quaternions.quat2mat(quat)
    return {
        "eef_xyz": torch.from_numpy(pos).reshape(1, 1, 3),
        "eef_rot": torch.from_numpy(rot).reshape(1, 1, 3, 3),
    }


def restrict_rotation_to_gripper_row(
    warped_rot: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    """Restrict rotation to row/roll around gripper's local X axis.

    Args:
        warped_rot: Warped rotation matrix (3, 3).
        reference_rot: Reference rotation matrix (3, 3) to define local axes.

    Returns:
        Rotation matrix (3, 3) with only local X-axis rotation preserved.
    """
    r_ref = reference_rot.detach().cpu().numpy()
    r_warp = warped_rot.detach().cpu().numpy()
    # Delta rotation in gripper local frame
    r_delta_local = r_ref.T @ r_warp
    row = float(np.arctan2(r_delta_local[2, 1], r_delta_local[1, 1]))
    cos_row = float(np.cos(row))
    sin_row = float(np.sin(row))
    r_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_row, -sin_row],
            [0.0, sin_row, cos_row],
        ],
        dtype=np.float32,
    )
    r_new = r_ref @ r_x
    return torch.from_numpy(r_new).to(device=warped_rot.device, dtype=warped_rot.dtype)


def restrict_rotation_to_gripper_yaw(
    warped_rot: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    """Restrict rotation to yaw around gripper's local Z axis."""
    r_ref = reference_rot.detach().cpu().numpy()
    r_warp = warped_rot.detach().cpu().numpy()
    # Delta rotation in gripper local frame
    r_delta_local = r_ref.T @ r_warp
    yaw = float(np.arctan2(r_delta_local[1, 0], r_delta_local[0, 0]))
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    r_z = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    r_new = r_ref @ r_z
    return torch.from_numpy(r_new).to(device=warped_rot.device, dtype=warped_rot.dtype)


def restrict_rotation_to_gripper_pitch(
    warped_rot: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    """Restrict rotation to pitch around gripper's local Y axis."""
    r_ref = reference_rot.detach().cpu().numpy()
    r_warp = warped_rot.detach().cpu().numpy()
    # Delta rotation in gripper local frame
    r_delta_local = r_ref.T @ r_warp
    pitch = float(np.arctan2(r_delta_local[0, 2], r_delta_local[0, 0]))
    cos_pitch = float(np.cos(pitch))
    sin_pitch = float(np.sin(pitch))
    r_y = np.array(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float32,
    )
    r_new = r_ref @ r_y
    return torch.from_numpy(r_new).to(device=warped_rot.device, dtype=warped_rot.dtype)


def restrict_rotation_to_gripper_axis(
    warped_rot: torch.Tensor,
    reference_rot: torch.Tensor,
    axis_mode: str,
) -> torch.Tensor:
    """Restrict warped rotation to a single local gripper axis."""
    if axis_mode == "row":
        return restrict_rotation_to_gripper_row(warped_rot, reference_rot)
    if axis_mode == "yaw":
        return restrict_rotation_to_gripper_yaw(warped_rot, reference_rot)
    if axis_mode == "pitch":
        return restrict_rotation_to_gripper_pitch(warped_rot, reference_rot)
    raise ValueError(f"Unsupported axis_mode: {axis_mode}")


def _build_world_z_rotation(
    yaw: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a world-frame Z-axis rotation matrix."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    mat = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )
    return mat


def _compute_decay_factor(
    step: int,
    total_steps: int,
    decay_mode: str,
    decay_rate: float,
) -> float:
    """Compute trajectory warp decay factor."""
    if decay_mode == "none":
        return 1.0
    if total_steps <= 1:
        return 1.0
    if decay_mode == "linear":
        return 1.0 - float(step) / float(total_steps - 1)
    if decay_mode == "exponential":
        return float(np.exp(-decay_rate * step))
    raise ValueError(f"Unknown decay_mode: {decay_mode}")


def add_approach_phase(
    interpolated_trajectory: Dict[str, torch.Tensor],
    demo_trajectory: Dict[str, torch.Tensor],
    start_idx: int,
    num_approach_steps: int = 30,
    num_grasp_steps: int = 100,
    num_rotate_steps: int = 30,
    perturbation_offset: Optional[List[float]] = None,
    grasp_override: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Add interpolated approach phase, grasp phase, and rotation recovery phase.

    Args:
        interpolated_trajectory: Interpolated manipulation trajectory
        demo_trajectory: Original demo trajectory
        start_idx: Starting index in demo where manipulation begins
        num_approach_steps: Number of steps for approach phase
        num_grasp_steps: Number of steps for grasp phase (gripper closing)
        num_rotate_steps: Number of steps for rotating back to demo orientation
        perturbation_offset: [dx, dy, dz] offset for perturbation (applied to grasp point)
        grasp_override: Optional dict with 'eef_xyz' and/or 'eef_rot' overrides

    Returns:
        complete_trajectory: Trajectory with approach + grasp + manipulation
    """
    def _compute_interpolated_approach(
        init_xyz: np.ndarray,
        init_rot: np.ndarray,
        grasp_xyz: np.ndarray,
        grasp_rot: np.ndarray,
        steps: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alphas = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        eef_xyz = init_xyz[None, :] + alphas[:, None] * (grasp_xyz - init_xyz)[None, :]

        quat_init = transforms3d.quaternions.mat2quat(init_rot)
        quat_grasp = transforms3d.quaternions.mat2quat(grasp_rot)
        eef_rot = np.zeros((steps, 3, 3), dtype=np.float32)
        for i, alpha in enumerate(alphas):
            if alpha <= 0.0:
                interp_quat = quat_init
            elif alpha >= 1.0:
                interp_quat = quat_grasp
            else:
                interp_quat = gt.quaternion_slerp(
                    quat_init, quat_grasp, float(alpha), shortestpath=True
                )
            eef_rot[i] = transforms3d.quaternions.quat2mat(interp_quat)

        eef_xyz = torch.from_numpy(eef_xyz).to(device).unsqueeze(1)
        eef_rot = torch.from_numpy(eef_rot).to(device).unsqueeze(1)
        return eef_xyz, eef_rot
    # Get initial pose from demo
    init_xyz = demo_trajectory['eef_xyz'][0:1]  # (1, 1, 3)
    init_rot = demo_trajectory['eef_rot'][0:1]  # (1, 1, 3, 3)
    init_gripper = demo_trajectory['eef_gripper'][0:1]  # (1, 1, 1)

    # Get grasp pose (from demo at start_idx, or first frame of interpolated trajectory)
    if start_idx < demo_trajectory['eef_xyz'].shape[0]:
        grasp_xyz = demo_trajectory['eef_xyz'][start_idx:start_idx+1].clone()
        grasp_rot = demo_trajectory['eef_rot'][start_idx:start_idx+1]
        grasp_gripper_open = demo_trajectory['eef_gripper'][start_idx:start_idx+1]
    else:
        grasp_xyz = interpolated_trajectory['eef_xyz'][0:1].clone()
        grasp_rot = interpolated_trajectory['eef_rot'][0:1]
        grasp_gripper_open = interpolated_trajectory['eef_gripper'][0:1]

    if grasp_override is not None:
        if "eef_xyz" in grasp_override:
            grasp_xyz = grasp_override["eef_xyz"].clone()
        if "eef_rot" in grasp_override:
            grasp_rot = grasp_override["eef_rot"].clone()

    # Apply perturbation offset to grasp point
    if perturbation_offset is not None:
        offset = torch.tensor(perturbation_offset, dtype=grasp_xyz.dtype, device=grasp_xyz.device).reshape(1, 1, 3)
        grasp_xyz = grasp_xyz + offset
        print(f"  Grasp point with perturbation: {grasp_xyz[0, 0].cpu().numpy()}")

    # Create approach trajectory (Cartesian interpolation, gripper stays OPEN)
    approach_xyz, approach_rot = _compute_interpolated_approach(
        init_xyz[0, 0].cpu().numpy(),
        init_rot[0, 0].cpu().numpy(),
        grasp_xyz[0, 0].cpu().numpy(),
        grasp_rot[0, 0].cpu().numpy(),
        num_approach_steps,
        init_xyz.device,
    )
    approach_gripper = init_gripper.repeat(num_approach_steps, 1, 1)

    # Create grasp trajectory (gripper closes from open to closed)
    grasp_xyz_traj = grasp_xyz.repeat(num_grasp_steps, 1, 1)  # Stay at grasp point
    grasp_rot_traj = grasp_rot.repeat(num_grasp_steps, 1, 1, 1)  # Stay at grasp orientation
    grasp_gripper_traj = torch.zeros((num_grasp_steps, 1, 1), dtype=init_gripper.dtype, device=init_gripper.device)

    # Interpolate gripper from open to closed
    gripper_open = init_gripper[0, 0, 0].item()
    gripper_closed = grasp_gripper_open[0, 0, 0].item()

    for i in range(num_grasp_steps):
        alpha = (i + 1) / num_grasp_steps  # 0 -> 1
        grasp_gripper_traj[i, 0, 0] = (1 - alpha) * gripper_open + alpha * gripper_closed

    # Create rotation recovery phase (rotate from grasp to manipulation orientation)
    rotate_xyz_traj = grasp_xyz.repeat(num_rotate_steps, 1, 1)
    rotate_gripper_traj = grasp_gripper_traj[-1:].repeat(num_rotate_steps, 1, 1)
    rotate_rot_traj = torch.zeros((num_rotate_steps, 1, 3, 3), dtype=grasp_rot.dtype, device=grasp_rot.device)
    rotate_start = grasp_rot[0, 0].detach().cpu().numpy()
    target_idx = min(start_idx, demo_trajectory["eef_rot"].shape[0] - 1)
    rotate_end = demo_trajectory["eef_rot"][target_idx, 0].detach().cpu().numpy()
    quat_start = transforms3d.quaternions.mat2quat(rotate_start)
    quat_end = transforms3d.quaternions.mat2quat(rotate_end)
    for i in range(num_rotate_steps):
        if num_rotate_steps == 1:
            alpha = 1.0
        else:
            alpha = float(i) / float(num_rotate_steps - 1)
        interp_quat = gt.quaternion_slerp(quat_start, quat_end, alpha, shortestpath=True)
        rotate_rot_traj[i, 0] = torch.from_numpy(
            transforms3d.quaternions.quat2mat(interp_quat)
        ).to(rotate_rot_traj.device, dtype=rotate_rot_traj.dtype)

    # Concatenate approach + grasp + rotate + manipulation
    complete_xyz = torch.cat(
        [approach_xyz, grasp_xyz_traj, rotate_xyz_traj, interpolated_trajectory["eef_xyz"]], dim=0
    )
    complete_rot = torch.cat(
        [approach_rot, grasp_rot_traj, rotate_rot_traj, interpolated_trajectory["eef_rot"]], dim=0
    )
    complete_gripper = torch.cat(
        [approach_gripper, grasp_gripper_traj, rotate_gripper_traj, interpolated_trajectory["eef_gripper"]], dim=0
    )

    return {
        'eef_xyz': complete_xyz,
        'eef_rot': complete_rot,
        'eef_gripper': complete_gripper,
    }


def load_deformation_points(load_path: str, env) -> torch.Tensor:
    """
    Load deformation points and convert to world frame.

    Args:
        load_path: Path to deformation .npy/.npz file.
        env: Gym environment for pose/table metadata.

    Returns:
        Deformed points in world frame (N, 3).
    """
    raw_state = np.load(load_path, allow_pickle=True)
    if isinstance(raw_state, np.lib.npyio.NpzFile):
        raw_state = {key: raw_state[key] for key in raw_state.files}
    elif isinstance(raw_state, np.ndarray) and raw_state.shape == ():
        raw_state = raw_state.item()

    validated = DeformationBridge.validate_state(raw_state)
    points_np = validated["points"]
    metadata = validated["metadata"]

    device = env.renderer.device
    points = torch.from_numpy(points_np).to(torch.float32).to(device)

    frame = str(metadata.get("frame", "object"))
    pose_applied = bool(metadata.get("pose_applied", False))

    world_points = points_to_world(
        points=points,
        frame=frame,
        pose_obj=env.renderer.pose_obj,
        table_height=float(env.cfg.physics.table_height),
        pose_applied=pose_applied,
    )
    return world_points


def save_trajectory_as_json(trajectory, output_dir, episode_name="episode_0000"):
    """
    Save trajectory in JSON format compatible with replay.py

    Args:
        trajectory: Dict with 'eef_xyz', 'eef_rot', 'eef_gripper'
        output_dir: Output directory
        episode_name: Episode name
    """
    import kornia

    episode_dir = Path(output_dir) / episode_name
    robot_dir = episode_dir / 'robot'
    robot_dir.mkdir(parents=True, exist_ok=True)

    eef_xyz = trajectory['eef_xyz'].cpu()  # (T, 1, 3)
    eef_rot = trajectory['eef_rot'].cpu()  # (T, 1, 3, 3)
    eef_gripper = trajectory['eef_gripper'].cpu()  # (T, 1, 1)

    T = eef_xyz.shape[0]

    print(f"Saving {T} frames to JSON format...")

    for t in tqdm(range(T), desc="Saving JSON"):
        # Extract data
        xyz = eef_xyz[t, 0].numpy()  # (3,)
        rot = eef_rot[t, 0]  # (3, 3)
        gripper_sim = eef_gripper[t, 0, 0].item()  # scalar

        # Convert rotation matrix to quaternion (wxyz)
        quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot.unsqueeze(0))  # (1, 4)
        quat = quat[0].numpy()  # (4,) wxyz

        # Convert gripper: sim space (0=open, 1=closed) -> policy space (0=closed, 1=open)
        #gripper_policy = 1.0 - gripper_sim
        gripper_policy = gripper_sim

        # Create JSON data (same format as episode_0001)
        robot_data = {
            "obs.ee_pos": xyz.tolist(),
            "obs.ee_quat": quat.tolist(),
            "obs.gripper_qpos": gripper_policy,
            "action.ee_pos": xyz.tolist(),
            "action.ee_quat": quat.tolist(),
            "action.gripper_qpos": [gripper_policy],
        }

        # Save JSON
        json_path = robot_dir / f'{t:06d}.json'
        with open(json_path, 'w') as f:
            json.dump(robot_data, f, indent=4)

    return episode_dir


@hydra.main(version_base=None, config_path="../cfg", config_name="replay")
def main(cfg):
    OmegaConf.resolve(cfg)

    print("=" * 80)
    print("Create Interpolated JSON Trajectory from Demo")
    print("=" * 80)
    print()

    # Configuration
    demo_dir = Path(cfg.gt_dir)  # e.g., log/policy_rollouts/rope_act_7000
    episode_id = cfg.get('episode_id', 1)
    num_interp_steps = cfg.get('num_interp_steps', 118)
    num_approach_steps = cfg.get('num_approach_steps', 30)
    num_grasp_steps = cfg.get('num_grasp_steps', 100)  # Number of frames for gripper closing
    num_rotate_steps = cfg.get('num_rotate_steps', 30)  # Number of frames to rotate back to demo orientation
    start_idx = cfg.get('start_idx', None)  # Manual override for grasp point
    output_dir = Path(cfg.get('output_dir', 'log/experiments/interpolation_json'))
    use_rigid_transform = bool(cfg.get('use_rigid_transform', False))
    deformation_path = cfg.get('deformation_path', None)
    use_deformation_warping = bool(cfg.get('use_deformation_warping', False))
    warp_decay = str(cfg.get('warp_decay', 'none')).lower()
    decay_rate = float(cfg.get('decay_rate', 0.1))
    k_neighbors = int(cfg.get('k_neighbors', 5))
    manip_k_neighbors_cfg = cfg.get('manip_k_neighbors', None)
    if manip_k_neighbors_cfg is None:
        manip_k_neighbors = k_neighbors
    else:
        manip_k_neighbors = int(manip_k_neighbors_cfg)
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be >= 1.")
    if manip_k_neighbors < 1:
        raise ValueError("manip_k_neighbors must be >= 1.")
    use_local_rigid_baseline = bool(cfg.get('use_local_rigid_baseline', False))
    local_rigid_k = int(cfg.get('local_rigid_k', 5))
    local_rigid_decay_mode_cfg = cfg.get('local_rigid_decay_mode', None)
    if local_rigid_decay_mode_cfg is None:
        local_rigid_decay_mode = warp_decay
    else:
        local_rigid_decay_mode = str(local_rigid_decay_mode_cfg).lower()
    local_rigid_decay_rate_cfg = cfg.get('local_rigid_decay_rate', None)
    if local_rigid_decay_rate_cfg is None:
        local_rigid_decay_rate = decay_rate
    else:
        local_rigid_decay_rate = float(local_rigid_decay_rate_cfg)
    if use_local_rigid_baseline:
        if local_rigid_k < 3:
            raise ValueError("local_rigid_k must be >= 3.")
        if local_rigid_decay_mode not in ("none", "linear", "exponential"):
            raise ValueError(
                "local_rigid_decay_mode must be 'none', 'linear', or 'exponential'."
            )
    adapt_orientation = bool(cfg.get('adapt_orientation', False))
    grasp_row_only = bool(cfg.get('grasp_row_only', False))
    grasp_yaw_only = bool(cfg.get('grasp_yaw_only', False))
    grasp_pitch_only = bool(cfg.get('grasp_pitch_only', False))
    enabled_axis_modes = []
    if grasp_row_only:
        enabled_axis_modes.append("row")
    if grasp_yaw_only:
        enabled_axis_modes.append("yaw")
    if grasp_pitch_only:
        enabled_axis_modes.append("pitch")
    if len(enabled_axis_modes) > 1:
        raise ValueError(
            "grasp_row_only, grasp_yaw_only, and grasp_pitch_only are mutually exclusive."
        )
    grasp_axis_only_mode = enabled_axis_modes[0] if enabled_axis_modes else None
    if use_local_rigid_baseline and grasp_axis_only_mode not in (None, "yaw"):
        raise ValueError(
            "use_local_rigid_baseline only supports yaw axis constraint."
        )
    disable_z_warp = bool(cfg.get('disable_z_warp', False))
    grasp_warp_mode = str(cfg.get('grasp_warp_mode', 'knn')).lower()
    use_grasp_local_warp = bool(cfg.get('use_grasp_local_warp', False))
    grasp_local_radius = float(cfg.get('grasp_local_radius', 0.1))
    grasp_local_min_points = int(cfg.get('grasp_local_min_points', 50))
    grasp_local_k = int(cfg.get('grasp_local_k', 0))
    use_grasp_region = bool(cfg.get('use_grasp_region_only', False))
    grasp_radius = float(cfg.get('grasp_radius', 0.1))
    grasp_center_cfg = cfg.get('grasp_center', None)
    transform_orientation = bool(cfg.get('transform_orientation', True))
    preserve_grasp_rotation = bool(cfg.get('preserve_grasp_rotation', False))
    undo_rigid_rotation_after_grasp = bool(cfg.get('undo_rigid_rotation_after_grasp', False))
    decay_rigid_transform_in_manip = bool(cfg.get('decay_rigid_transform_in_manip', False))
    grasp_override_sample_dir = cfg.get('grasp_override_sample_dir', None)
    grasp_override_frame = cfg.get('grasp_override_frame', None)

    # Perturbation parameters - support both list and individual components
    perturbation_offset = cfg.get('perturbation_offset', None)  # [dx, dy, dz] in meters
    if perturbation_offset is not None:
        perturbation_offset = list(perturbation_offset)  # Convert from OmegaConf list
    else:
        # Alternative: use individual components
        px = cfg.get('perturbation_x', None)
        py = cfg.get('perturbation_y', None)
        pz = cfg.get('perturbation_z', None)
        if px is not None or py is not None or pz is not None:
            perturbation_offset = [
                float(px) if px is not None else 0.0,
                float(py) if py is not None else 0.0,
                float(pz) if pz is not None else 0.0
            ]

    if use_deformation_warping and use_rigid_transform:
        raise ValueError("Cannot enable both use_deformation_warping and use_rigid_transform.")
    if grasp_warp_mode not in ("knn", "centroid", "gs_centroid"):
        raise ValueError("grasp_warp_mode must be 'knn', 'centroid', or 'gs_centroid'.")

    print(f"Demo directory: {demo_dir}")
    print(f"Episode ID: {episode_id}")
    print(f"Interpolation steps: {num_interp_steps}")
    print(f"Approach steps: {num_approach_steps}")
    print(f"Grasp steps: {num_grasp_steps}")
    print(f"Rotate steps: {num_rotate_steps}")
    print(f"Output directory: {output_dir}")
    if perturbation_offset is not None:
        print(f"Perturbation offset: {perturbation_offset} (meters)")
    if use_rigid_transform:
        print("Rigid transform: enabled")
        if deformation_path is not None:
            print(f"Deformation path: {deformation_path}")
        if preserve_grasp_rotation:
            print("Rigid transform: preserve grasp rotation")
        if undo_rigid_rotation_after_grasp:
            print("Rigid transform: undo rotation after grasp")
        if decay_rigid_transform_in_manip:
            print("Rigid transform: decay transform in manipulation")
    if use_deformation_warping:
        print("Deformation warping: enabled")
        print(f"Warp decay: {warp_decay} (rate={decay_rate})")
        print(f"K neighbors (grasp): {k_neighbors}")
        print(f"K neighbors (manip): {manip_k_neighbors}")
        print(f"Adapt orientation: {adapt_orientation}")
        print(f"Grasp row-only  (local X axis): {grasp_row_only}")
        print(f"Grasp yaw-only  (local Z axis): {grasp_yaw_only}")
        print(f"Grasp pitch-only(local Y axis): {grasp_pitch_only}")
        print(f"Grasp axis-only mode: {grasp_axis_only_mode}")
        print(f"Disable Z warp: {disable_z_warp}")
        print(f"Grasp warp mode: {grasp_warp_mode}")
        if use_local_rigid_baseline:
            print("Local rigid baseline: enabled")
            print(f"Local rigid k: {local_rigid_k}")
            print(
                f"Local rigid decay: {local_rigid_decay_mode} "
                f"(rate={local_rigid_decay_rate})"
            )
            print("Local rigid constraints: XY translation + world Z yaw rotation")
        if use_grasp_local_warp:
            if grasp_local_k > 0:
                print(f"Grasp local warp: enabled (k={grasp_local_k})")
            else:
                print(
                    "Grasp local warp: enabled "
                    f"(radius={grasp_local_radius}, min_points={grasp_local_min_points})"
                )
    print()

    # Load demo trajectory
    print("Loading demo trajectory...")
    demo_trajectory = load_replay_trajectory(demo_dir, episode_id)
    T_demo = demo_trajectory['eef_xyz'].shape[0]
    print(f"  Demo trajectory: {T_demo} frames")
    print()

    # Interpolate trajectory
    print("Interpolating trajectory...")
    if start_idx is not None:
        print(f"  Using manual grasp point: frame {start_idx}")
    interpolated, actual_start_idx = interpolate_trajectory(
        demo_trajectory,
        num_steps=num_interp_steps,
        start_idx=start_idx,  # Use configured value or auto-detect
        perturbation_offset=perturbation_offset  # Apply perturbation with decay
    )
    print(f"  Interpolated trajectory: {num_interp_steps} frames")
    print()

    grasp_override = None
    if use_deformation_warping:
        if deformation_path is None:
            raise ValueError("use_deformation_warping=True requires deformation_path.")

        print("Applying deformation field warping...")
        env = gym.make(
            cfg.env_name,
            max_episode_steps=10,
            cfg=cfg,
            obs_mode=cfg.obs_mode,
            exp_root=cfg.exp_root,
            local_rank=0,
            randomize=True,
        )
        env.reset(seed=episode_id)

        p_orig, p_def = load_deformation_for_warping(deformation_path, env)
        for key in ("eef_xyz", "eef_rot", "eef_gripper"):
            demo_trajectory[key] = demo_trajectory[key].to(p_orig.device)
        interpolated["eef_xyz"] = interpolated["eef_xyz"].to(p_orig.device)
        interpolated["eef_rot"] = interpolated["eef_rot"].to(p_orig.device)
        interpolated["eef_gripper"] = interpolated["eef_gripper"].to(p_orig.device)
        grasp_xyz_demo = demo_trajectory["eef_xyz"][actual_start_idx, 0].to(p_orig.device)
        grasp_rot_demo = demo_trajectory["eef_rot"][actual_start_idx, 0].to(p_orig.device)
        if use_local_rigid_baseline:
            dists = torch.norm(p_orig - grasp_xyz_demo[None, :], dim=1)
            k_local = min(local_rigid_k, int(p_orig.shape[0]))
            knn_idx = torch.topk(dists, k_local, largest=False).indices
            p_orig_local = p_orig[knn_idx]
            p_def_local = p_def[knn_idx]

            r_local, t_local = compute_optimal_rigid_transform(p_orig_local, p_def_local)
            yaw = float(torch.atan2(r_local[1, 0], r_local[0, 0]).item())
            r_yaw = _build_world_z_rotation(yaw, p_orig.device, p_orig.dtype)
            centroid_orig = p_orig_local.mean(dim=0)
            centroid_def = p_def_local.mean(dim=0)
            t_xy = (centroid_def - (centroid_orig @ r_yaw.T)).clone()
            t_xy[2] = 0.0
            centroid_fit = centroid_orig @ r_yaw.T + t_xy
            centroid_xy_residual = torch.norm((centroid_fit - centroid_def)[:2]).item()

            eef_xyz_flat = interpolated["eef_xyz"][:, 0]
            eef_rot_flat = interpolated["eef_rot"][:, 0]
            t_steps = int(eef_xyz_flat.shape[0])
            warped_xyz_flat = torch.empty_like(eef_xyz_flat)
            warped_rot_flat = torch.empty_like(eef_rot_flat)
            for i in range(t_steps):
                alpha = _compute_decay_factor(
                    i, t_steps, local_rigid_decay_mode, local_rigid_decay_rate
                )
                yaw_alpha = yaw * alpha
                r_alpha = _build_world_z_rotation(yaw_alpha, p_orig.device, p_orig.dtype)
                warped_xyz_flat[i] = eef_xyz_flat[i] @ r_alpha.T + (t_xy * alpha)
                warped_rot_flat[i] = r_alpha @ eef_rot_flat[i]

            interpolated["eef_xyz"] = warped_xyz_flat.unsqueeze(1)
            interpolated["eef_rot"] = warped_rot_flat.unsqueeze(1)

            warped_grasp_xyz = grasp_xyz_demo @ r_yaw.T + t_xy
            warped_grasp_rot = r_yaw @ grasp_rot_demo
            print(
                "  Local rigid baseline applied: "
                f"k={k_local}, yaw={np.rad2deg(yaw):.3f}deg, "
                f"t_xy={t_xy.detach().cpu().numpy()}, "
                f"centroid_xy_residual={centroid_xy_residual:.6f}"
            )
        else:
            warped = warp_trajectory_with_decay(
                interpolated["eef_xyz"],
                interpolated["eef_rot"],
                p_orig,
                p_def,
                k_neighbors=manip_k_neighbors,
                decay_mode=warp_decay,
                decay_rate=decay_rate,
                adapt_orientation=adapt_orientation,
                disable_z_warp=disable_z_warp,
            )
            interpolated["eef_xyz"] = warped["eef_xyz"]
            if "eef_rot" in warped:
                interpolated["eef_rot"] = warped["eef_rot"]

            p_orig_grasp = p_orig
            p_def_grasp = p_def
            idx_local = None
            if use_grasp_local_warp:
                dists = torch.norm(p_orig - grasp_xyz_demo[None, :], dim=1)
                if grasp_local_k > 0:
                    k_local = min(grasp_local_k, int(p_orig.shape[0]))
                    knn_idx = torch.topk(dists, k_local, largest=False).indices
                    p_orig_grasp = p_orig[knn_idx]
                    p_def_grasp = p_def[knn_idx]
                    idx_local = knn_idx
                    print(f"  Grasp local warp: using {k_local} nearest points.")
                    if k_local < grasp_local_min_points:
                        print(
                            "  Warning: grasp local warp has "
                            f"{k_local} points (<{grasp_local_min_points})."
                        )
                else:
                    mask = dists <= grasp_local_radius
                    num_local = int(mask.sum().item())
                    if num_local < grasp_local_min_points:
                        print(
                            "  Warning: grasp local warp has "
                            f"{num_local} points (<{grasp_local_min_points}); using global."
                        )
                    else:
                        p_orig_grasp = p_orig[mask]
                        p_def_grasp = p_def[mask]
                        idx_local = mask.nonzero(as_tuple=False).squeeze(1)
                        print(
                            f"  Grasp local warp: {num_local} points within "
                            f"{grasp_local_radius}m."
                        )
            if grasp_warp_mode == "centroid":
                centroid_orig = p_orig_grasp.mean(dim=0)
                centroid_def = p_def_grasp.mean(dim=0)
                translation = centroid_def - centroid_orig
                if disable_z_warp:
                    translation = translation.clone()
                    translation[2] = 0.0
                warped_grasp_xyz = grasp_xyz_demo + translation
                warped_grasp_rot = grasp_rot_demo
                if adapt_orientation:
                    delta = compute_deformation_field(p_orig_grasp, p_def_grasp)
                    if disable_z_warp:
                        delta = delta.clone()
                        delta[:, 2] = 0.0
                    grasp_jacobian_query = (
                        grasp_xyz_demo if grasp_axis_only_mode == "yaw" else warped_grasp_xyz
                    )
                    jac = compute_local_deformation_gradient(
                        grasp_jacobian_query[None, :],
                        p_orig_grasp,
                        delta,
                        k_neighbors=max(4, k_neighbors),
                    )[0]
                    warped_grasp_rot = warp_orientation_with_jacobian(
                        grasp_rot_demo[None, ...], jac[None, ...]
                    )[0]
                    if grasp_axis_only_mode is not None:
                        warped_grasp_rot = restrict_rotation_to_gripper_axis(
                            warped_grasp_rot, grasp_rot_demo, grasp_axis_only_mode
                        )
            elif grasp_warp_mode == "gs_centroid":
                if idx_local is None:
                    idx_local = torch.arange(p_orig.shape[0], device=p_orig.device)
                env.renderer.update_phystwin_pts(p_orig)
                env.renderer.update_rendervar(x_pred=p_orig)
                gs_orig = env.renderer.rendervar["means3D"].clone()
                if cfg.physics.use_lbs:
                    weights_info = env.renderer.weights
                    if weights_info is None:
                        raise ValueError("Expected LBS weights to be computed.")
                    _, weights_indices = weights_info
                    gs_bind = weights_indices[:, 0]
                else:
                    relations = env.renderer.relations
                    if relations is None:
                        raise ValueError("Expected KNN relations to be computed.")
                    gs_bind = relations[:, 0]
                gs_mask = torch.isin(gs_bind, idx_local)
                if int(gs_mask.sum().item()) == 0:
                    raise ValueError("No GS points bound to selected physical points.")

                env.renderer.update_rendervar(x_pred=p_def)
                gs_def = env.renderer.rendervar["means3D"].clone()

                centroid_orig = gs_orig[gs_mask].mean(dim=0)
                centroid_def = gs_def[gs_mask].mean(dim=0)
                translation = centroid_def - centroid_orig
                if disable_z_warp:
                    translation = translation.clone()
                    translation[2] = 0.0
                warped_grasp_xyz = grasp_xyz_demo + translation
                warped_grasp_rot = grasp_rot_demo
                if adapt_orientation:
                    delta = compute_deformation_field(p_orig_grasp, p_def_grasp)
                    if disable_z_warp:
                        delta = delta.clone()
                        delta[:, 2] = 0.0
                    grasp_jacobian_query = (
                        grasp_xyz_demo if grasp_axis_only_mode == "yaw" else warped_grasp_xyz
                    )
                    jac = compute_local_deformation_gradient(
                        grasp_jacobian_query[None, :],
                        p_orig_grasp,
                        delta,
                        k_neighbors=max(4, k_neighbors),
                    )[0]
                    warped_grasp_rot = warp_orientation_with_jacobian(
                        grasp_rot_demo[None, ...], jac[None, ...]
                    )[0]
                    if grasp_axis_only_mode is not None:
                        warped_grasp_rot = restrict_rotation_to_gripper_axis(
                            warped_grasp_rot, grasp_rot_demo, grasp_axis_only_mode
                        )
            else:
                warped_grasp_xyz, warped_grasp_rot = warp_grasp_pose(
                    grasp_xyz_demo,
                    grasp_rot_demo,
                    p_orig_grasp,
                    p_def_grasp,
                    k_neighbors=k_neighbors,
                    adapt_orientation=adapt_orientation,
                    disable_z_warp=disable_z_warp,
                    jacobian_query_point=(
                        grasp_xyz_demo if grasp_axis_only_mode == "yaw" else None
                    ),
                )
                if adapt_orientation and grasp_axis_only_mode is not None:
                    warped_grasp_rot = restrict_rotation_to_gripper_axis(
                        warped_grasp_rot, grasp_rot_demo, grasp_axis_only_mode
                    )
        grasp_override = {"eef_xyz": warped_grasp_xyz.reshape(1, 1, 3)}
        if adapt_orientation or use_local_rigid_baseline:
            grasp_override["eef_rot"] = warped_grasp_rot.reshape(1, 1, 3, 3)

        env.close()
        print("  Deformation warping applied.")
        print()

    if grasp_override_sample_dir is not None:
        sample_dir = Path(grasp_override_sample_dir)
        print(f"Using grasp override from sample: {sample_dir}")
        grasp_override = load_grasp_override_from_sample(
            sample_dir, frame_idx=grasp_override_frame
        )
        target_device = interpolated["eef_xyz"].device
        grasp_override["eef_xyz"] = grasp_override["eef_xyz"].to(target_device)
        grasp_override["eef_rot"] = grasp_override["eef_rot"].to(target_device)
        print("  Grasp override applied.")
        print()

    # Add approach phase, grasp phase, and rotation recovery phase
    print("Adding approach, grasp, and rotation recovery phases...")
    complete_trajectory = add_approach_phase(
        interpolated,
        demo_trajectory,
        actual_start_idx,
        num_approach_steps=num_approach_steps,
        num_grasp_steps=num_grasp_steps,
        num_rotate_steps=num_rotate_steps,
        perturbation_offset=perturbation_offset,  # Apply perturbation to grasp point
        grasp_override=grasp_override,
    )
    original_xyz = complete_trajectory["eef_xyz"].clone()
    original_rotations = complete_trajectory["eef_rot"].clone()
    T_complete = complete_trajectory['eef_xyz'].shape[0]
    print(
        f"  Complete trajectory: {T_complete} frames ("
        f"{num_approach_steps} approach + {num_grasp_steps} grasp + "
        f"{num_rotate_steps} rotate + {num_interp_steps} manipulation)"
    )
    print()

    if use_rigid_transform:
        if deformation_path is None:
            raise ValueError("use_rigid_transform=True requires deformation_path.")

        print("Applying rigid transform from deformation...")
        env = gym.make(
            cfg.env_name,
            max_episode_steps=10,
            cfg=cfg,
            obs_mode=cfg.obs_mode,
            exp_root=cfg.exp_root,
            local_rank=0,
            randomize=True,
        )
        obs, _ = env.reset(seed=episode_id)

        p_orig = get_object_points_from_env(env)
        p_def = load_deformation_points(deformation_path, env)

        grasp_center = None
        if grasp_center_cfg is not None:
            grasp_center = torch.tensor(
                grasp_center_cfg, dtype=torch.float32, device=p_orig.device
            )
        elif use_grasp_region:
            grasp_center = demo_trajectory["eef_xyz"][actual_start_idx, 0].to(p_orig.device)
            print(f"  Using grasp center from demo frame {actual_start_idx}.")

        R, t = compute_optimal_rigid_transform(
            p_orig,
            p_def,
            use_grasp_region=use_grasp_region,
            grasp_center=grasp_center,
            grasp_radius=grasp_radius,
        )

        errors = compute_transformation_error(p_orig, p_def, R, t)
        print(
            "  Transform error: "
            f"RMSE={errors['rmse']:.4f}m, Max={errors['max_error']:.4f}m"
        )

        complete_trajectory = apply_rigid_transform_to_trajectory(
            complete_trajectory,
            R,
            t,
            transform_orientation=transform_orientation,
        )
        if preserve_grasp_rotation:
            prefix_len = num_approach_steps + num_grasp_steps
            complete_trajectory["eef_rot"][:prefix_len] = original_rotations[:prefix_len]
        if undo_rigid_rotation_after_grasp and decay_rigid_transform_in_manip:
            print("  Warning: undo rotation disabled due to decay_rigid_transform_in_manip")
        if undo_rigid_rotation_after_grasp and transform_orientation and not decay_rigid_transform_in_manip:
            rotate_start = num_approach_steps + num_grasp_steps
            rotate_end = rotate_start + num_rotate_steps
            manip_start = rotate_end
            total_steps = complete_trajectory["eef_rot"].shape[0]
            if num_rotate_steps < 1:
                print("  Warning: undo rotation requested but num_rotate_steps=0, skipping")
            elif manip_start >= total_steps:
                print("  Warning: rotate/manip indices out of range, skipping undo rotation")
            else:
                if rotate_start == 0:
                    start_rot = complete_trajectory["eef_rot"][0, 0].detach().cpu().numpy()
                else:
                    start_rot = complete_trajectory["eef_rot"][rotate_start - 1, 0].detach().cpu().numpy()
                end_rot = original_rotations[manip_start, 0].detach().cpu().numpy()

                quat_start = transforms3d.quaternions.mat2quat(start_rot)
                quat_end = transforms3d.quaternions.mat2quat(end_rot)
                for i in range(num_rotate_steps):
                    if num_rotate_steps == 1:
                        alpha = 1.0
                    else:
                        alpha = float(i) / float(num_rotate_steps - 1)
                    interp_quat = gt.quaternion_slerp(
                        quat_start, quat_end, alpha, shortestpath=True
                    )
                    complete_trajectory["eef_rot"][rotate_start + i, 0] = torch.from_numpy(
                        transforms3d.quaternions.quat2mat(interp_quat)
                    ).to(complete_trajectory["eef_rot"].device, dtype=complete_trajectory["eef_rot"].dtype)

                complete_trajectory["eef_rot"][manip_start:] = original_rotations[manip_start:]
        if decay_rigid_transform_in_manip:
            rotate_start = num_approach_steps + num_grasp_steps
            manip_start = rotate_start + num_rotate_steps
            total_steps = complete_trajectory["eef_rot"].shape[0]
            if manip_start >= total_steps:
                print("  Warning: manipulation start out of range, skipping decay")
            else:
                num_manip = total_steps - manip_start
                R_np = R.detach().cpu().numpy()
                t_np = t.detach().cpu().numpy()
                quat_R = transforms3d.quaternions.mat2quat(R_np)
                quat_I = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                device = complete_trajectory["eef_xyz"].device
                dtype = complete_trajectory["eef_xyz"].dtype
                t_torch = torch.from_numpy(t_np).to(device, dtype=dtype)

                for i in range(num_manip):
                    if num_manip == 1:
                        alpha = 1.0
                    else:
                        alpha = 1.0 - float(i) / float(num_manip - 1)
                    interp_quat = gt.quaternion_slerp(
                        quat_I, quat_R, alpha, shortestpath=True
                    )
                    R_alpha = transforms3d.quaternions.quat2mat(interp_quat)
                    R_alpha_t = torch.from_numpy(R_alpha).to(device, dtype=dtype)

                    x_orig = original_xyz[manip_start + i, 0]
                    rot_orig = original_rotations[manip_start + i, 0]
                    complete_trajectory["eef_xyz"][manip_start + i, 0] = (
                        x_orig @ R_alpha_t.T + (t_torch * alpha)
                    )
                    if transform_orientation:
                        complete_trajectory["eef_rot"][manip_start + i, 0] = (
                            R_alpha_t @ rot_orig
                        )
                    else:
                        complete_trajectory["eef_rot"][manip_start + i, 0] = rot_orig

        env.close()

    # Save as JSON
    print("Saving trajectory as JSON...")
    episode_dir = save_trajectory_as_json(
        complete_trajectory,
        output_dir,
        episode_name=f"episode_{episode_id:04d}"
    )
    print(f"  Saved to: {episode_dir}")
    print()

    # Save perturbed soft body state if perturbation is applied (Method 2)
    if perturbation_offset is not None:
        print("Saving perturbed soft body state (Method 2)...")

        # Create environment to get initial soft body state
        env = gym.make(
            cfg.env_name,
            max_episode_steps=10,
            cfg=cfg,
            obs_mode=cfg.obs_mode,
            exp_root=cfg.exp_root,
            local_rank=0,
            randomize=True
        )

        # Reset environment
        obs, reset_info = env.reset(seed=episode_id)

        # Get current state
        state = env.unwrapped.renderer.get_state()
        original_x = state['x'].clone()

        # Apply perturbation offset (Method 2: replace point cloud)
        offset_tensor = torch.tensor(
            perturbation_offset,
            dtype=torch.float32,
            device=env.unwrapped.physics.device
        )
        perturbed_x = original_x + offset_tensor.reshape(1, 3)

        print(f"    Original center: {original_x.mean(dim=0).cpu().numpy()}")
        print(f"    Perturbed center: {perturbed_x.mean(dim=0).cpu().numpy()}")

        # Save perturbed soft body state
        soft_body_state = {
            'x': perturbed_x.cpu(),
            'v': torch.zeros_like(perturbed_x).cpu(),
            'perturbation_offset': perturbation_offset,
        }

        soft_body_path = episode_dir / 'soft_body_state.pt'
        torch.save(soft_body_state, soft_body_path)
        print(f"    ✓ Saved perturbed soft body state to: {soft_body_path}")

        # Clean up
        env.close()
        del env
        print()

    # Print usage instructions
    print("=" * 80)
    print("JSON Trajectory Created Successfully")
    print("=" * 80)
    print()
    print("Output structure:")
    print(f"  {episode_dir}/")
    print(f"    robot/")
    print(f"      000000.json  (approach frame 0)")
    print(f"      ...")
    print(f"      {num_approach_steps-1:06d}.json  (approach frame {num_approach_steps-1})")
    grasp_start = num_approach_steps
    grasp_end = num_approach_steps + num_grasp_steps - 1
    rotate_start = grasp_end + 1
    rotate_end = rotate_start + num_rotate_steps - 1
    manip_start = rotate_end + 1
    print(f"      {grasp_start:06d}.json  (grasp frame 0)")
    print(f"      {grasp_end:06d}.json  (grasp frame {num_grasp_steps-1})")
    print(f"      {rotate_start:06d}.json  (rotate frame 0)")
    print(f"      {rotate_end:06d}.json  (rotate frame {num_rotate_steps-1})")
    print(f"      {manip_start:06d}.json  (manipulation frame 0)")
    print(f"      ...")
    print(f"      {T_complete-1:06d}.json  (manipulation frame {num_interp_steps-1})")
    print()
    print("Data format:")
    print("  - Rotation: Quaternion (wxyz)")
    print("  - Gripper: Policy space (0=closed, 1=open)")
    print("  - Compatible with: experiments/replay.py")
    print()
    print("Usage:")
    print(f"  python experiments/replay.py \\")
    print(f"    gs=rope env=xarm_gripper \\")
    print(f"    physics.ckpt_path=log/phystwin/rope \\")
    print(f"    physics.case_name=rope_0001 \\")
    print(f"    gt_dir={output_dir} \\")
    print(f"    use_qpos=False")
    print("=" * 80)


if __name__ == '__main__':
    main()
