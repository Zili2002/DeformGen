"""
MPPI (Model Predictive Path Integral) Planner

Sampling-based trajectory optimization for robotic manipulation
"""

import numpy as np
import torch
import torch.nn.functional as F
import kornia
from typing import Dict, Any, Optional

from .utils.metrics import batch_chamfer_dist, rope_routing_reward, fps_sample


class MPPIPlanner:
    """
    Model Predictive Path Integral (MPPI) trajectory planner

    Uses sampling-based optimization to find optimal robot trajectories:
    1. Sample noisy trajectories around current best
    2. Rollout all trajectories in dynamics model
    3. Evaluate each trajectory with reward function
    4. Update best trajectory using weighted average
    """

    def __init__(self, config, dynamics_module, device='cuda'):
        """
        Initialize MPPI planner

        Args:
            config: Configuration object
            dynamics_module: Dynamics module for rollouts
            device: Device to use ('cuda' or 'cpu')
        """
        self.cfg = config
        self.dynamics_module = dynamics_module
        self.torch_device = torch.device(device)

        # MPPI parameters
        mppi_cfg = config.mppi
        self.n_look_ahead = mppi_cfg.n_look_ahead
        self.n_sample = mppi_cfg.n_sample
        self.n_sample_chunk = mppi_cfg.n_sample_chunk
        self.n_chunk = int(np.ceil(self.n_sample / self.n_sample_chunk))
        self.n_update_iter = mppi_cfg.n_update_iter
        self.reward_weight = mppi_cfg.reward_weight
        self.repeated_action = mppi_cfg.repeated_action

        # Noise levels
        self.xyz_noise_level = mppi_cfg.xyz_noise_level
        self.quat_noise_level = mppi_cfg.quat_noise_level
        self.gripper_noise_level = mppi_cfg.gripper_noise_level

        # Workspace bounds
        bbox = config.workspace.bbox_bimanual if config.env.robot.n_grippers > 1 else config.workspace.bbox_single
        self.bbox = torch.tensor(bbox, dtype=torch.float32, device=self.torch_device)

        # State
        self.target_state = torch.empty(0, device=self.torch_device)
        self.pts = torch.empty(0, device=self.torch_device)

        print(f"MPPIPlanner initialized")
        print(f"  Planning horizon: {self.n_look_ahead}")
        print(f"  Samples per iteration: {self.n_sample}")
        print(f"  Optimization iterations: {self.n_update_iter}")
        print(f"  Reward weight: {self.reward_weight}")

    def set_target(self, target_pts: torch.Tensor, n_samples: int = 1000):
        """
        Set target point cloud for planning

        Args:
            target_pts: Target point cloud (N, 3)
            n_samples: Number of points to downsample to
        """
        if len(target_pts) == 0:
            print('[WARNING] Target state is empty')
            return

        # Downsample target
        if target_pts.shape[0] > n_samples:
            fps_idx = fps_sample(target_pts, n_samples, device=self.torch_device, random_start=False)
            target_pts = target_pts[fps_idx]

        self.target_state = target_pts.clone()
        self.dynamics_module.set_target_state(target_pts)

        print(f"  Target set: {self.target_state.shape[0]} points")

    def plan(
        self,
        init_pts: torch.Tensor,
        init_traj: Dict[str, torch.Tensor],
        target_pts: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Plan trajectory using MPPI optimization

        Args:
            init_pts: Initial point cloud (N, 3)
            init_traj: Initial trajectory dict with keys:
                'eef_xyz': (T, n_grippers, 3)
                'eef_rot': (T, n_grippers, 3, 3)
                'eef_gripper': (T, n_grippers, 1)
            target_pts: Target point cloud (M, 3)

        Returns:
            optimized_traj: Optimized trajectory (same format as init_traj)
        """
        self.pts = init_pts.to(self.torch_device)

        # Convert trajectory to action sequence format
        n_grippers = init_traj['eef_xyz'].shape[1]
        act_seq = self._traj_to_action_seq(init_traj)

        # MPPI optimization loop
        best_act_seq = act_seq
        best_reward = float('-inf')

        for iter_idx in range(self.n_update_iter):
            print(f'\nMPPI iteration: {iter_idx+1}/{self.n_update_iter}')

            with torch.no_grad():
                # 1. Sample action sequences
                act_seqs = self.sample_action_seq(act_seq, iter_idx)

                # 2. Rollout dynamics
                final_states = self.model_rollout(act_seqs)

                # 3. Evaluate trajectories
                trajs_dict = self._action_seq_to_traj_dict(act_seqs)
                reward_seqs = self.evaluate_traj(final_states, trajs_dict)

                # 4. Optimize using MPPI
                act_seq = self.optimize_action_mppi(act_seqs, reward_seqs)

                # Track best
                best_idx = torch.argmax(reward_seqs)
                if reward_seqs[best_idx] > best_reward:
                    best_act_seq = act_seqs[best_idx]
                    best_reward = reward_seqs[best_idx]

                print(f'  Best reward so far: {best_reward:.4f}')

        # Convert back to trajectory format
        optimized_traj = self._action_seq_to_traj(best_act_seq)

        return optimized_traj

    def sample_action_seq(self, act_seq: torch.Tensor, iter_index: int = 0) -> torch.Tensor:
        """
        Sample noisy action sequences around base trajectory

        Args:
            act_seq: Base action sequence (T, n_grippers * 13)
            iter_index: Current iteration index (for adaptive noise)

        Returns:
            act_seqs: Sampled action sequences (n_sample, T, n_grippers * 13)
        """
        n_grippers = self.cfg.env.robot.n_grippers
        n_sample = self.n_sample_chunk

        # Parse current action
        eef_xyz = act_seq[:, :n_grippers * 3].reshape(self.n_look_ahead, n_grippers, 3)
        eef_rot = act_seq[:, n_grippers * 3:n_grippers * 12].reshape(
            self.n_look_ahead, n_grippers, 3, 3
        )
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
        eef_gripper = act_seq[:, n_grippers * 12:].reshape(self.n_look_ahead, n_grippers, 1)

        # Generate position noise (segmented repeated)
        if self.repeated_action:
            # Single noise repeated over horizon
            xyz_delta = torch.randn(
                (n_sample, n_grippers, 3), device=self.torch_device
            ) * self.xyz_noise_level
            xyz_delta = xyz_delta[:, None].repeat(1, self.n_look_ahead, 1, 1)
        else:
            # Segmented repeated noise (4 segments)
            n_parts = 4
            delta_list = []
            for p in range(n_parts):
                p_len = self.n_look_ahead // n_parts if p < n_parts - 1 else \
                    self.n_look_ahead - (n_parts - 1) * (self.n_look_ahead // n_parts)

                noise_scale = 1.0 / (iter_index + 1)  # Adaptive decay
                delta = torch.randn(
                    (n_sample, n_grippers, 3), device=self.torch_device
                ) * self.xyz_noise_level * noise_scale
                delta = delta[:, None].repeat(1, p_len, 1, 1)
                delta_list.append(delta)
            xyz_delta = torch.cat(delta_list, dim=1)

        # Cumulative noise for continuity
        xyz_delta_cum = torch.cumsum(xyz_delta, dim=1)

        # Quaternion sampling
        noise_scale = 1.0 / (iter_index + 1)
        quat_noise = torch.randn(
            (n_sample, self.n_look_ahead, n_grippers, 4), device=self.torch_device
        ) * self.quat_noise_level * noise_scale

        # Apply noise
        eef_xyz_sampled = eef_xyz[None] + xyz_delta_cum
        eef_quat_sampled = eef_quat[None] + quat_noise

        # Normalize quaternions
        quat_norm = torch.norm(eef_quat_sampled, dim=-1, keepdim=True)
        eef_quat_sampled = eef_quat_sampled / (quat_norm + 1e-8)

        # Convert to rotation matrices
        eef_rot_sampled = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat_sampled)

        # Orthogonalize rotation matrices
        eef_rot_sampled = self._orthogonalize_rotations(eef_rot_sampled)

        eef_gripper_sampled = eef_gripper[None].repeat(n_sample, 1, 1, 1)

        # Package actions
        act_seqs = torch.zeros(
            (n_sample, self.n_look_ahead, n_grippers * 13), device=self.torch_device
        )
        act_seqs[:, :, :n_grippers * 3] = eef_xyz_sampled.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * 3:n_grippers * 12] = eef_rot_sampled.reshape(
            n_sample, self.n_look_ahead, -1
        )
        act_seqs[:, :, n_grippers * 12:] = eef_gripper_sampled.reshape(n_sample, self.n_look_ahead, -1)

        # Clip to workspace
        act_seqs = self.clip_actions(act_seqs)

        return act_seqs

    def _orthogonalize_rotations(self, rot_matrices: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize rotation matrices using QR decomposition

        Args:
            rot_matrices: (..., 3, 3) rotation matrices

        Returns:
            Orthogonalized rotation matrices with det=1
        """
        original_shape = rot_matrices.shape
        rot_flat = rot_matrices.reshape(-1, 3, 3)

        # QR decomposition
        Q, R = torch.linalg.qr(rot_flat)

        # Ensure det(Q) = 1 (proper rotation, not reflection)
        det_Q = torch.det(Q)
        Q[:, :, 2] = Q[:, :, 2] * det_Q.sign().unsqueeze(-1)

        Q = Q.reshape(original_shape)
        return Q

    def model_rollout(self, act_seqs: torch.Tensor) -> torch.Tensor:
        """
        Rollout dynamics model with action sequences

        Args:
            act_seqs: (n_sample, T, n_grippers * 13)

        Returns:
            final_states: (n_sample, N, 3) - Final point clouds
        """
        n_sample = act_seqs.shape[0]
        T = act_seqs.shape[1]
        n_grippers = self.cfg.env.robot.n_grippers

        # Reshape to (n_sample, T, n_grippers, 13) for dynamics rollout
        act_seqs_reshaped = act_seqs.reshape(n_sample, T, n_grippers, 13)

        # Rollout dynamics
        final_states = self.dynamics_module.rollout(self.pts, act_seqs_reshaped)

        return final_states

    def evaluate_traj(
        self,
        final_states: torch.Tensor,
        trajectories: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Evaluate sampled trajectories

        Args:
            final_states: (n_sample, N, 3)
            trajectories: dict with trajectory information

        Returns:
            rewards: (n_sample,)
        """
        # Compute rewards
        rewards = rope_routing_reward(
            final_states,
            self.target_state,
            trajectories,
            self.cfg
        )

        # Debug info
        chamfer = batch_chamfer_dist(final_states, self.target_state)
        print(f'  Chamfer range: [{chamfer.min().item():.6f}, {chamfer.max().item():.6f}]')
        print(f'  Reward range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]')

        return rewards

    def optimize_action_mppi(
        self,
        act_seqs: torch.Tensor,
        reward_seqs: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimize action using MPPI weighted average

        Args:
            act_seqs: (n_sample, T, n_grippers * 13)
            reward_seqs: (n_sample,)

        Returns:
            opt_act_seq: (T, n_grippers * 13)
        """
        # Softmax weights
        weight_seqs = F.softmax(reward_seqs * self.reward_weight, dim=0)

        n_sample = self.n_sample_chunk
        n_grippers = self.cfg.env.robot.n_grippers

        # Parse actions
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(
            n_sample, self.n_look_ahead, n_grippers, 3
        )
        eef_rot = act_seqs[:, :, n_grippers * 3:n_grippers * 12].reshape(
            n_sample, self.n_look_ahead, n_grippers, 3, 3
        )
        eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
        eef_gripper = act_seqs[:, :, n_grippers * 12:].reshape(
            n_sample, self.n_look_ahead, n_grippers, 1
        )

        # Weighted average
        eef_xyz = torch.sum(weight_seqs[:, None, None, None] * eef_xyz, dim=0)
        eef_quat = torch.sum(weight_seqs[:, None, None, None] * eef_quat, dim=0)
        eef_gripper = torch.sum(weight_seqs[:, None, None, None] * eef_gripper, dim=0)

        # Normalize quaternion
        eef_quat = eef_quat / (torch.norm(eef_quat, dim=-1, keepdim=True) + 1e-6)
        eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)

        # Package
        act_seq = torch.zeros(
            (self.n_look_ahead, n_grippers * 13), device=self.torch_device
        )
        act_seq[:, :n_grippers * 3] = eef_xyz.reshape(self.n_look_ahead, -1)
        act_seq[:, n_grippers * 3:n_grippers * 12] = eef_rot.reshape(self.n_look_ahead, -1)
        act_seq[:, n_grippers * 12:] = eef_gripper.reshape(self.n_look_ahead, -1)

        act_seq = self.clip_actions(act_seq[None])[0]
        return act_seq

    def clip_actions(self, act_seqs: torch.Tensor) -> torch.Tensor:
        """Clip actions to workspace bounds"""
        if len(act_seqs.shape) == 2:
            act_seqs = act_seqs[None]
            squeeze_output = True
        else:
            squeeze_output = False

        n_sample = act_seqs.shape[0]
        n_grippers = self.cfg.env.robot.n_grippers

        # Parse
        eef_xyz = act_seqs[:, :, :n_grippers * 3].reshape(
            n_sample, self.n_look_ahead, n_grippers, 3
        )
        eef_gripper = act_seqs[:, :, n_grippers * 12:].reshape(
            n_sample, self.n_look_ahead, n_grippers, 1
        )

        # Clip position
        eef_xyz = torch.clamp(eef_xyz, self.bbox[:, 0], self.bbox[:, 1])

        # Clip gripper
        eef_gripper = torch.clamp(eef_gripper, 0.0, 1.0)

        # Repack
        act_seqs[:, :, :n_grippers * 3] = eef_xyz.reshape(n_sample, self.n_look_ahead, -1)
        act_seqs[:, :, n_grippers * 12:] = eef_gripper.reshape(n_sample, self.n_look_ahead, -1)

        if squeeze_output:
            return act_seqs[0]
        return act_seqs

    def _traj_to_action_seq(self, traj: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert trajectory dict to action sequence"""
        n_grippers = traj['eef_xyz'].shape[1]
        T = traj['eef_xyz'].shape[0]

        act_seq = torch.zeros((T, n_grippers * 13), device=self.torch_device)
        act_seq[:, :n_grippers * 3] = traj['eef_xyz'].reshape(T, -1).to(self.torch_device)
        act_seq[:, n_grippers * 3:n_grippers * 12] = traj['eef_rot'].reshape(T, -1).to(self.torch_device)
        act_seq[:, n_grippers * 12:] = traj['eef_gripper'].reshape(T, -1).to(self.torch_device)

        return act_seq

    def _action_seq_to_traj(self, act_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert action sequence to trajectory dict"""
        n_grippers = self.cfg.env.robot.n_grippers

        return {
            'eef_xyz': act_seq[:, :n_grippers * 3].reshape(self.n_look_ahead, n_grippers, 3),
            'eef_rot': act_seq[:, n_grippers * 3:n_grippers * 12].reshape(
                self.n_look_ahead, n_grippers, 3, 3
            ),
            'eef_gripper': act_seq[:, n_grippers * 12:].reshape(self.n_look_ahead, n_grippers, 1)
        }

    def _action_seq_to_traj_dict(self, act_seqs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert batch action sequences to trajectory dict"""
        n_sample = act_seqs.shape[0]
        n_grippers = self.cfg.env.robot.n_grippers

        return {
            'eef_xyz': act_seqs[:, :, :n_grippers * 3].reshape(
                n_sample, self.n_look_ahead, n_grippers, 3
            ),
            'eef_rot': act_seqs[:, :, n_grippers * 3:n_grippers * 12].reshape(
                n_sample, self.n_look_ahead, n_grippers, 3, 3
            ),
            'eef_gripper': act_seqs[:, :, n_grippers * 12:].reshape(
                n_sample, self.n_look_ahead, n_grippers, 1
            )
        }
