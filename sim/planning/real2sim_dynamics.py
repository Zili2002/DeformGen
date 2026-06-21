"""
Real2Sim Dynamics Adapter

Wraps Real2Sim-Eval BaseEnv as a dynamics model for MPPI planning
"""

import torch
import warp as wp
import copy
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class Real2SimDynamics:
    """
    Dynamics adapter for Real2Sim-Eval environment

    Wraps BaseEnv to provide rollout functionality for MPPI planning.
    Handles environment state save/restore for batch trajectory simulation.
    """

    def __init__(self, env, config):
        """
        Initialize dynamics adapter

        Args:
            env: BaseEnv instance from Real2Sim-Eval
            config: Configuration object
        """
        self.env = env
        self.physics = env.physics  # PhysTwinDynamics
        self.renderer = env.renderer  # GSRenderer
        self.cfg = config

        # State management
        self.init_state = None
        self.target_state = None

        # Device
        self.device = env.physics.device

        print(f"Real2SimDynamics initialized")
        print(f"  Device: {self.device}")

    def set_target_state(self, target_pts: torch.Tensor):
        """Set target point cloud"""
        self.target_state = target_pts.clone()

    def rollout(
        self,
        init_pts: torch.Tensor,
        traj_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Batch rollout trajectories

        Args:
            init_pts: Initial point cloud (N, 3)
            traj_batch: Trajectory batch (batch_size, T, n_grippers, 13)
                       Each action contains [xyz(3), rot_matrix(9), gripper(1)]

        Returns:
            final_states: (batch_size, N, 3) - Final point cloud for each trajectory
        """
        batch_size, T, n_grippers, action_dim = traj_batch.shape
        assert action_dim == 13, f"Expected action_dim=13, got {action_dim}"

        final_states = []

        # Save current environment state
        if self.init_state is None:
            self.init_state = self._save_state()

        # Rollout each trajectory
        for b in tqdm(range(batch_size), desc="Rollout", disable=(batch_size <= 4)):
            # Reset to initial state
            self._reset_to_state(self.init_state)

            # Execute trajectory
            state = self.renderer.get_state()
            for t in range(T):
                action = traj_batch[b, t]  # (n_grippers, 13)
                state = self.physics.step(state, action)
                self.renderer.update_state(state)

            # Collect final point cloud
            final_pts = self.physics.dynamics_module.current_points.clone()
            final_states.append(final_pts)

        # Restore to initial state
        self._reset_to_state(self.init_state)

        return torch.stack(final_states)

    def _save_state(self) -> Dict:
        """
        Save complete environment state

        Returns:
            state: Dict containing physics and renderer state
        """
        # Deep copy Warp physics state
        wp_state_copy = self._deep_copy_wp_state(
            self.physics.dynamics_module.simulator.wp_state
        )

        return {
            'physics': {
                'x': self.physics.dynamics_module.current_points.clone(),
                'v': self.physics.dynamics_module.current_velocities.clone(),
                'wp_state': wp_state_copy,
                'current_openness': self.physics.dynamics_module.current_openness,
                'grasped': self.physics.dynamics_module.grasped,
            },
            'renderer': {
                'x': self.renderer.get_state()['x'].clone(),
            }
        }

    def _reset_to_state(self, state: Dict):
        """
        Restore environment to saved state

        Args:
            state: State dict from _save_state()
        """
        # Restore physics state
        wp.copy(
            state['physics']['wp_state'].wp_x,
            self.physics.dynamics_module.simulator.wp_state.wp_x
        )
        wp.copy(
            state['physics']['wp_state'].wp_v,
            self.physics.dynamics_module.simulator.wp_state.wp_v
        )

        self.physics.dynamics_module.current_openness = state['physics']['current_openness']
        self.physics.dynamics_module.grasped = state['physics']['grasped']

        # Restore renderer state
        self.renderer.update_phystwin_pts(state['renderer']['x'])

    def _deep_copy_wp_state(self, wp_state):
        """
        Deep copy Warp state

        Args:
            wp_state: Warp state object

        Returns:
            Copied state
        """
        # Create new state with copied arrays
        class WPStateCopy:
            def __init__(self, original_state):
                # Copy position and velocity
                self.wp_x = wp.clone(original_state.wp_x)
                self.wp_v = wp.clone(original_state.wp_v)

                # Copy other attributes if they exist
                if hasattr(original_state, 'wp_f'):
                    self.wp_f = wp.clone(original_state.wp_f)
                if hasattr(original_state, 'wp_inv_mass'):
                    self.wp_inv_mass = wp.clone(original_state.wp_inv_mass)

        return WPStateCopy(wp_state)

    def reset_preprocess_meta(self, pts: torch.Tensor):
        """Reset preprocessing metadata (for compatibility)"""
        pass

    def reset_downsample_indices(self, pts: torch.Tensor):
        """Reset downsampling indices (for compatibility)"""
        pass
