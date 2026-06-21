from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import time

import numpy as np
import torch
import warp as wp


class DeformationBridge:
    """
    Bridge deformation states between PhysTwin/qqtt and real2sim-eval.

    Exported data is assumed to be in the PhysTwin model frame by default.
    The real2sim-eval renderer operates in world frame, and the physics
    module converts to model frame using table height translation.
    """

    @staticmethod
    def export_from_qqtt(
        simulator: Any,
        save_path: str | Path,
        frame: str = "model",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Export the current PhysTwin/qqtt simulator state to a .npy file.

        Args:
            simulator: qqtt simulator instance with wp_state/wp_states.
            save_path: Output file path.
            frame: Coordinate frame of exported points ("model" or "world").
            extra_metadata: Optional metadata to merge into the export.

        Returns:
            The state dictionary that was saved.
        """
        state = DeformationBridge._extract_sim_state(simulator, frame, extra_metadata)
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, state)
        return state

    @staticmethod
    def import_to_real2sim(env: Any, load_path: str | Path) -> Dict[str, Any]:
        """
        Import a saved deformation state into a real2sim-eval environment.

        Args:
            env: BaseEnv instance (or compatible wrapper with renderer/physics).
            load_path: Path to .npy/.npz file storing deformation state.

        Returns:
            Metadata associated with the imported state.
        """
        raw_state = DeformationBridge._load_state(load_path)
        validated = DeformationBridge.validate_state(raw_state)
        points_np = validated["points"]
        velocities_np = validated["velocities"]
        metadata = validated["metadata"]

        device = env.renderer.device
        points = torch.from_numpy(points_np).to(torch.float32).to(device)
        velocities = None
        if velocities_np is not None:
            velocities = torch.from_numpy(velocities_np).to(torch.float32).to(device)

        frame = str(metadata.get("frame", "object")).lower()
        pose_applied = bool(metadata.get("pose_applied", False))
        global_translation = torch.tensor(
            [0.0, 0.0, -env.cfg.physics.table_height],
            dtype=points.dtype,
            device=device,
        )
        if frame == "model":
            world_points = points - global_translation
        elif frame == "world":
            world_points = points
        elif frame == "object":
            world_points = points
        else:
            raise ValueError(f"Unknown deformation frame: {frame}")

        if frame in {"model", "object"} and not pose_applied:
            pose_obj = env.renderer.pose_obj
            world_points = world_points @ pose_obj[:3, :3].T + pose_obj[:3, 3]
            if velocities is not None:
                velocities = velocities @ pose_obj[:3, :3].T

        if velocities is None:
            velocities = torch.zeros_like(world_points)

        # Initialize physics to the default state to align renderer/physics sizing.
        base_state = env.renderer.get_state()
        phystwin_pts = env.physics.reset(
            base_state,
            init_meshes_dict=env.renderer.meshes,
            robot=env.renderer.robot,
            eef_pts_func=env.renderer.eef_pts_func,
            kin_helper=env.renderer.kin_helper,
            init_eef_xyz=env.renderer.init_eef_xyz,
            pose_obj=env.renderer.pose_obj,
        )
        env.renderer.update_phystwin_pts(phystwin_pts)

        expected_points = phystwin_pts.shape[0]
        if world_points.shape[0] != expected_points:
            raise ValueError(
                f"Expected {expected_points} points, got {world_points.shape[0]}."
            )

        renderer_state = env.renderer.get_state()
        renderer_state["x"] = world_points
        renderer_state["v"] = velocities
        env.renderer.update_state(renderer_state)

        simulator = env.physics.dynamics_module.simulator
        model_points = world_points + global_translation
        simulator.set_init_state(model_points, velocities)

        metadata_out = dict(metadata)
        metadata_out["source_path"] = str(load_path)
        metadata_out["num_object_points"] = int(points.shape[0])
        metadata_out["table_height"] = float(env.cfg.physics.table_height)
        return metadata_out

    @staticmethod
    def validate_state(
        state: Dict[str, Any],
        expected_num_points: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Validate and normalize a deformation state dict.

        Args:
            state: Raw state dictionary loaded from disk.
            expected_num_points: Expected number of points to validate against.

        Returns:
            Normalized state with keys: points, velocities, metadata.
        """
        points = state.get("points", state.get("x"))
        if points is None:
            raise ValueError("Deformation state missing 'points' field.")
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points.shape}.")

        velocities = state.get("velocities", state.get("v", None))
        if velocities is not None:
            velocities = np.asarray(velocities)
            if velocities.shape != points.shape:
                raise ValueError(
                    f"Velocities shape {velocities.shape} does not match points {points.shape}."
                )

        num_object_points = int(state.get("num_object_points", points.shape[0]))
        if num_object_points != points.shape[0]:
            raise ValueError(
                f"num_object_points {num_object_points} does not match points {points.shape[0]}."
            )
        if expected_num_points is not None and expected_num_points != points.shape[0]:
            raise ValueError(
                f"Expected {expected_num_points} points, got {points.shape[0]}."
            )

        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {"metadata": metadata}

        return {
            "points": points,
            "velocities": velocities,
            "metadata": metadata,
        }

    @staticmethod
    def _extract_sim_state(
        simulator: Any,
        frame: str,
        extra_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        wp_state = DeformationBridge._get_wp_state(simulator)
        points = DeformationBridge._to_numpy(wp_state.wp_x)
        velocities = None
        if hasattr(wp_state, "wp_v"):
            velocities = DeformationBridge._to_numpy(wp_state.wp_v)

        num_object_points = int(getattr(simulator, "num_object_points", points.shape[0]))
        metadata: Dict[str, Any] = {
            "timestamp": time.time(),
            "frame": frame,
        }
        if hasattr(simulator, "wp_spring_Y"):
            metadata["spring_Y"] = DeformationBridge._to_numpy(simulator.wp_spring_Y)
        if extra_metadata:
            metadata.update(extra_metadata)

        return {
            "points": points,
            "velocities": velocities,
            "num_object_points": num_object_points,
            "metadata": metadata,
        }

    @staticmethod
    def _get_wp_state(simulator: Any) -> Any:
        if hasattr(simulator, "wp_states") and simulator.wp_states:
            return simulator.wp_states[0]
        if hasattr(simulator, "wp_state"):
            return simulator.wp_state
        raise ValueError("Simulator does not expose wp_state/wp_states.")

    @staticmethod
    def _load_state(load_path: str | Path) -> Dict[str, Any]:
        path = Path(load_path)
        if not path.exists():
            raise FileNotFoundError(f"Deformation state not found: {path}")

        if path.suffix == ".npz":
            loaded = np.load(path, allow_pickle=True)
            return {key: loaded[key] for key in loaded.files}

        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            return {key: loaded[key] for key in loaded.files}
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            return loaded.item()
        if isinstance(loaded, dict):
            return loaded
        raise ValueError(f"Unsupported deformation state format: {path}")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        try:
            return wp.to_torch(value).detach().cpu().numpy()
        except Exception:
            return np.asarray(value)
