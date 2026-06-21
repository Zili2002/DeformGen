"""
Trajectory Loader for MPPI Planning

Load initial/target states and trajectories from replay data
"""

import json
import numpy as np
import torch
from pathlib import Path
import gymnasium as gym
import kornia
from typing import Dict, Tuple
import sys
sys.path.append(str(Path(__file__).parents[3]))
import sim.envs  # Register environments


def load_replay_trajectory(replay_dir: Path, episode_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Load robot trajectory from replay data

    Args:
        replay_dir: Path to replay directory (e.g., log/policy_rollouts/rope_act_7000)
        episode_id: Episode number

    Returns:
        dict{
            'eef_xyz': (T, n_grippers, 3),
            'eef_rot': (T, n_grippers, 3, 3),
            'eef_gripper': (T, n_grippers, 1)
        }
    """
    episode_dir = replay_dir / f'episode_{episode_id:04d}'
    robot_dir = episode_dir / 'robot'

    if not robot_dir.exists():
        raise FileNotFoundError(f"Robot directory not found: {robot_dir}")

    robot_files = sorted(robot_dir.glob('*.json'))

    if len(robot_files) == 0:
        raise ValueError(f"No robot files found in {robot_dir}")

    trajectories = {
        'eef_xyz': [],
        'eef_rot': [],
        'eef_gripper': []
    }

    for robot_file in robot_files:
        with open(robot_file, 'r') as f:
            data = json.load(f)

        # Extract action (not obs)
        eef_xyz = np.array(data['action.ee_pos']).reshape(1, 3)
        eef_quat = np.array(data['action.ee_quat']).reshape(1, 4)  # wxyz
        eef_gripper = np.array(data['action.gripper_qpos']).reshape(1, 1)

        # Convert quaternion to rotation matrix
        eef_quat_torch = torch.from_numpy(eef_quat).float()
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            eef_quat_torch
        ).numpy()

        trajectories['eef_xyz'].append(eef_xyz)
        trajectories['eef_rot'].append(eef_rot)
        trajectories['eef_gripper'].append(eef_gripper)

    # Stack and convert to tensors
    eef_xyz = torch.from_numpy(np.concatenate(trajectories['eef_xyz'])).float()  # (T, 3)
    eef_rot = torch.from_numpy(np.concatenate(trajectories['eef_rot'])).float()  # (T, 3, 3)
    eef_gripper = torch.from_numpy(np.concatenate(trajectories['eef_gripper'])).float()  # (T, 1)

    # Add gripper dimension: (T, 3) -> (T, 1, 3)
    eef_xyz = eef_xyz.unsqueeze(1)  # (T, 1, 3)
    eef_rot = eef_rot.unsqueeze(1)  # (T, 1, 3, 3)
    eef_gripper = eef_gripper.unsqueeze(1)  # (T, 1, 1)

    return {
        'eef_xyz': eef_xyz,
        'eef_rot': eef_rot,
        'eef_gripper': eef_gripper
    }


def extract_states_from_replay(
    cfg,
    replay_dir: Path,
    episode_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Run replay and extract physical states

    Args:
        cfg: Hydra configuration
        replay_dir: Path to replay directory
        episode_id: Episode number

    Returns:
        dict{
            'init_pts': Initial point cloud (N, 3),
            'target_pts': Target point cloud (N, 3),
            'init_trajectory': Robot trajectory dict,
            'n_frames': Trajectory length
        }
    """
    print(f"  Loading trajectory from episode {episode_id}...")

    # 1. Load trajectory data
    trajectory = load_replay_trajectory(replay_dir, episode_id)
    n_frames = trajectory['eef_xyz'].shape[0]
    print(f"    Loaded {n_frames} frames")

    # 2. Initialize environment
    print(f"  Initializing environment...")
    env = gym.make(
        cfg.env_name,
        max_episode_steps=n_frames + 100,
        cfg=cfg,
        obs_mode=cfg.obs_mode,
        exp_root=cfg.exp_root,
        local_rank=0,
        randomize=True
    )

    # 3. Reset environment (use same seed for consistency)
    obs, _ = env.reset(seed=episode_id)
    print(f"    Environment reset complete")

    # 4. Stabilize initial state
    eef_xyz = obs['robot']['eef_xyz']
    eef_quat = obs['robot']['eef_quat']
    eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)
    eef_gripper = obs['robot']['eef_gripper']

    action = torch.cat([
        eef_xyz,
        eef_rot.reshape(eef_rot.shape[0], -1),
        eef_gripper
    ], dim=1)

    print(f"    Stabilizing initial state...")
    for _ in range(30):  # Stabilize for 1 second
        env.step({'action': action, 'do_velocity_control': False})

    # 5. Record initial point cloud
    init_pts = env.physics.dynamics_module.current_points.clone()
    print(f"    Initial points: {init_pts.shape[0]}")

    # 6. Execute full trajectory
    print(f"    Executing trajectory ({n_frames} steps)...")
    device = env.physics.device

    for t in range(n_frames):
        eef_xyz = trajectory['eef_xyz'][t].to(device)  # (1, 3)
        eef_rot = trajectory['eef_rot'][t].to(device)  # (1, 3, 3)
        eef_gripper = 1.0 - trajectory['eef_gripper'][t].to(device)  # (1, 1) Convert to sim space

        action = torch.cat([
            eef_xyz,  # (1, 3)
            eef_rot.reshape(1, -1),  # (1, 9)
            eef_gripper  # (1, 1)
        ], dim=1)  # (1, 13)

        env.step({
            'action': action,
            'do_velocity_control': cfg.env.robot.do_velocity_control
        })

        if (t + 1) % 50 == 0:
            print(f"      Step {t+1}/{n_frames}")

    # 7. Record final point cloud
    target_pts = env.physics.dynamics_module.current_points.clone()
    print(f"    Final points: {target_pts.shape[0]}")

    # 8. Cleanup
    env.close()

    return {
        'init_pts': init_pts,
        'target_pts': target_pts,
        'init_trajectory': trajectory,
        'n_frames': n_frames
    }
