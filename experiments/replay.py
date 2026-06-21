from pathlib import Path
import hydra
from omegaconf import OmegaConf
import os
import shutil
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import json
import kornia
import glob
import time
import sys
import transforms3d
from transforms3d import _gohlketransforms as gt
from typing import Any, Dict, Optional, Tuple
sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

from experiments.utils.dir_utils import mkdir
from experiments.utils.ffmpeg import make_video
from experiments.utils.calculate_success_cloth3 import (
    evaluate_cloth3_last_frame_triangle,
    save_cloth3_triangle_debug_image,
)
import sim.envs
from sim.utils.robot.kinematics_utils import KinHelper, set_ik_log_path, set_ik_log_frame

kin_helper = KinHelper('xarm7')

# utils for transforming qpos to cartesian
def compute_fk(qpos):                
    eef_xyz = []
    eef_rot = []
    assert kin_helper is not None
    for i in range(qpos.shape[0]):
        e2b = kin_helper.compute_fk_sapien_links(qpos[i][:7], [kin_helper.sapien_eef_idx])[0]  # (4, 4)
        eef_xyz_base = e2b[:3, 3]  # (3,)
        eef_rot_base = e2b[:3, :3]  # (3, 3)
        eef_xyz.append(eef_xyz_base)
        eef_rot.append(eef_rot_base)
    eef_xyz = np.array(eef_xyz).astype(np.float32).reshape(-1, 3)
    eef_rot = np.array(eef_rot).astype(np.float32).reshape(-1, 3, 3)
    return eef_xyz, eef_rot

# utils for loading robot json
def load_robot_json(path, use_qpos=True, prefix='action'):
    with open(path, 'r') as f:
        robot = json.load(f)

    action_qpos = None
    if f'{prefix}.qpos' in robot.keys():
        action_qpos = np.array(robot[f'{prefix}.qpos']).reshape(1, -1)

    if f'{prefix}.xy' in robot.keys():  # planar pushing
        if use_qpos:
            robot_trans, robot_rot = compute_fk(np.array(robot[f'{prefix}.qpos']).reshape(1, -1))
        else:
            xy = np.array(robot[f'{prefix}.xy']).reshape(-1, 2)  # (1, 2)
            robot_trans = np.zeros((1, 3), dtype=np.float32)
            robot_trans[:, :2] = xy
            robot_trans[:, 2] = 0.22  # fixed height
            robot_rot = np.eye(3, dtype=np.float32)
            robot_rot[1, 1] *= -1
            robot_rot[2, 2] *= -1
            robot_rot = robot_rot[None]  # (1, 3, 3)
        gripper = np.array([1.0], dtype=np.float32).reshape(-1, 1)

    else:  # full 6-DoF
        if use_qpos:
            robot_trans, robot_rot = compute_fk(np.array(robot[f'{prefix}.qpos']).reshape(1, -1))
        else:
            if f'{prefix}.cartesian' in robot:
                e2b = np.array(robot[f'{prefix}.cartesian']).reshape(4, 4)
                robot_rot = e2b[:3, :3][None]  # (1, 3, 3)
                robot_trans = e2b[:3, 3]  # (1, 3)
            else:
                assert f'{prefix}.ee_pos' in robot and f'{prefix}.ee_quat' in robot
                eef_xyz = np.array(robot[f'{prefix}.ee_pos']).reshape(1, 3)  # (1, 3)
                eef_quat = np.array(robot[f'{prefix}.ee_quat']).reshape(1, 4)  # (1, 4) wxyz
                robot_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                    torch.from_numpy(eef_quat).to(torch.float32)
                ).numpy()  # (1, 3, 3)
                robot_trans = eef_xyz  # (1, 3)
        gripper = 1.0 - np.array(robot[f'{prefix}.gripper_qpos']).reshape(-1)  # (1,)

    return robot_trans, robot_rot, gripper, action_qpos


def tcp_to_eef(
    tcp_xyz: torch.Tensor,
    tcp_rot: torch.Tensor,
    tcp_offset: torch.Tensor,
    tcp_rot_offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert TCP pose to end-effector pose using a fixed offset in the EEF frame."""
    eef_rot = torch.matmul(tcp_rot, tcp_rot_offset.transpose(0, 1))
    offset = tcp_offset.reshape(1, 3, 1)
    eef_xyz = tcp_xyz - torch.matmul(eef_rot, offset).squeeze(-1)
    return eef_xyz, eef_rot


def _compute_qpos_from_eef(
    eef_xyz: np.ndarray,
    eef_rot: np.ndarray,
    qpos_seed: np.ndarray,
) -> np.ndarray:
    cur_xyzrpy = np.zeros(6, dtype=np.float32)
    cur_xyzrpy[:3] = eef_xyz
    cur_xyzrpy[3:] = transforms3d.euler.mat2euler(eef_rot)
    return kin_helper.compute_ik_sapien(qpos_seed, cur_xyzrpy)


def _parse_runtime_perturbation(cfg) -> Optional[Dict[str, Any]]:
    runtime_cfg = cfg.get("runtime_perturbation", None)
    if runtime_cfg is None or not bool(runtime_cfg.get("enabled", False)):
        return None

    mode = str(runtime_cfg.get("mode", "fixed")).lower()
    translation_xy = np.array(
        runtime_cfg.get("translation_xy", [0.0, 0.0]),
        dtype=np.float32,
    ).reshape(2)
    rotation_z = float(runtime_cfg.get("rotation_z", 0.0))
    num_waypoints = int(runtime_cfg.get("num_waypoints", 0))
    gripper_state = float(runtime_cfg.get("gripper_state", 1.0))

    start_step = runtime_cfg.get("start_step", None)
    if start_step is not None:
        start_step = int(start_step)

    release_cfg = runtime_cfg.get("release", None) or {}
    release_enabled = bool(release_cfg.get("enabled", False))
    release_steps = int(release_cfg.get("num_waypoints", 0)) if release_enabled else 0
    if release_enabled and release_steps < 1:
        raise ValueError("runtime_perturbation.release.num_waypoints must be >= 1.")
    release_gripper_open = float(release_cfg.get("gripper_open", 0.0))

    steps: list[Dict[str, np.ndarray]] = []
    if mode == "fixed":
        if num_waypoints < 1:
            raise ValueError("runtime_perturbation.num_waypoints must be >= 1.")
        delta_xy = translation_xy / float(num_waypoints)
        delta_angle = np.deg2rad(rotation_z / float(num_waypoints))
        rot_delta = np.array(
            [
                [np.cos(delta_angle), -np.sin(delta_angle), 0.0],
                [np.sin(delta_angle), np.cos(delta_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for _ in range(num_waypoints):
            steps.append({
                "delta_xy": delta_xy.astype(np.float32),
                "rot_delta": rot_delta,
            })
        num_steps = num_waypoints
    elif mode == "teleop_random":
        num_steps = int(runtime_cfg.get("random_steps", 0))
        if num_steps < 1:
            raise ValueError("runtime_perturbation.random_steps must be >= 1.")
        translation_step_sizes = runtime_cfg.get("translation_step_sizes", [0.005, 0.001])
        translation_step_sizes = [float(x) for x in translation_step_sizes]
        rotation_step_deg = float(runtime_cfg.get("rotation_step_deg", 2.0))
        rotation_prob = float(runtime_cfg.get("rotation_prob", 0.5))
        seed = runtime_cfg.get("seed", None)
        rng = np.random.default_rng(seed)

        directions = np.array([
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        rot_angle = np.deg2rad(rotation_step_deg)
        rot_pos = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle), 0.0],
                [np.sin(rot_angle), np.cos(rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rot_neg = np.array(
            [
                [np.cos(-rot_angle), -np.sin(-rot_angle), 0.0],
                [np.sin(-rot_angle), np.cos(-rot_angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        for _ in range(num_steps):
            if rng.random() < rotation_prob:
                rot_delta = rot_pos if rng.random() < 0.5 else rot_neg
                steps.append({
                    "delta_xy": np.zeros(2, dtype=np.float32),
                    "rot_delta": rot_delta,
                })
            else:
                step_size = float(rng.choice(translation_step_sizes))
                delta_xy = directions[rng.integers(0, len(directions))] * step_size
                steps.append({
                    "delta_xy": delta_xy.astype(np.float32),
                    "rot_delta": np.eye(3, dtype=np.float32),
                })
    else:
        raise ValueError(f"Unknown runtime_perturbation.mode: {mode}")

    return {
        "mode": mode,
        "start_step": start_step,
        "num_steps": int(num_steps),
        "steps": steps,
        "gripper_state": gripper_state,
        "release_steps": release_steps,
        "release_gripper_open": release_gripper_open,
        "total_steps": int(num_steps) + release_steps,
    }


def _init_runtime_state(obs) -> Tuple[np.ndarray, np.ndarray]:
    eef_xyz = obs["robot"]["eef_xyz"].detach().cpu().numpy().astype(np.float32)
    eef_quat = obs["robot"]["eef_quat"].detach().cpu().numpy().astype(np.float32)
    eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
        torch.from_numpy(eef_quat).to(torch.float32)
    ).cpu().numpy()
    return eef_xyz, eef_rot.astype(np.float32)


def _apply_runtime_perturbation_step(
    state_xyz: np.ndarray,
    state_rot: np.ndarray,
    runtime_cfg: Dict[str, Any],
    runtime_idx: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if runtime_idx < runtime_cfg["num_steps"]:
        step = runtime_cfg["steps"][runtime_idx]
        state_xyz = state_xyz.copy()
        state_xyz[:, 0] += step["delta_xy"][0]
        state_xyz[:, 1] += step["delta_xy"][1]
        state_rot = state_rot @ step["rot_delta"]
        gripper = runtime_cfg["gripper_state"]
    else:
        if runtime_cfg["release_steps"] > 0:
            t = (runtime_idx - runtime_cfg["num_steps"] + 1) / runtime_cfg["release_steps"]
            gripper = runtime_cfg["gripper_state"] + (
                runtime_cfg["release_gripper_open"] - runtime_cfg["gripper_state"]
            ) * t
        else:
            gripper = runtime_cfg["gripper_state"]
    return state_xyz, state_rot, float(gripper)


def _write_timestamps(path: Path, timestamps: list[tuple[float, float]]) -> None:
    with open(path, "w") as f:
        for ts0, ts1 in timestamps:
            f.write(f"{ts0:.7f} {ts1:.7f}\n")


def _extract_robot_render_part(renderer) -> Dict[str, torch.Tensor]:
    table_n = int(renderer.total_mask_full.shape[0])
    full_n = int(renderer.rendervar_full["means3D"].shape[0])
    prefix_n = full_n - table_n
    table_data = {k: v[prefix_n:].clone() for k, v in renderer.rendervar_full.items() if k != "means2D"}
    total_mask = renderer.total_mask_full.to(torch.long)

    if renderer.cfg.env["robot"]["use_pusher"]:
        link_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10]
    else:
        link_ids = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16]

    robot_mask = torch.zeros_like(total_mask, dtype=torch.bool)
    for link_id in link_ids:
        robot_mask |= total_mask == int(link_id)

    out: Dict[str, torch.Tensor] = {}
    for key in ["means3D", "shs", "rotations", "opacities", "scales"]:
        out[key] = table_data[key][robot_mask].clone()
    out["means2D"] = torch.zeros_like(out["means3D"])
    return out


def _merge_render_parts(parts: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in ["means3D", "shs", "rotations", "opacities", "scales"]:
        out[key] = torch.cat([p[key] for p in parts], dim=0)
    out["means2D"] = torch.zeros_like(out["means3D"])
    return out


def _build_arm_soft_mesh_render_data(renderer) -> Dict[str, torch.Tensor]:
    object_part = {k: renderer.rendervar[k].clone() for k in ["means3D", "shs", "rotations", "opacities", "scales"]}
    object_part["means2D"] = torch.zeros_like(object_part["means3D"])

    mesh_parts = []
    for _, mesh in renderer.params_meshes.items():
        mesh_part = {k: mesh[k].clone() for k in ["means3D", "shs", "rotations", "opacities", "scales"]}
        mesh_part["means2D"] = torch.zeros_like(mesh_part["means3D"])
        mesh_parts.append(mesh_part)

    robot_part = _extract_robot_render_part(renderer)
    return _merge_render_parts([object_part] + mesh_parts + [robot_part])


def _save_arm_soft_mesh_rgba_frame(
    env,
    cfg,
    out_path: str,
    run_name: str,
    episode_id: int,
    frame_idx: int,
    dirname: str,
) -> None:
    renderer = env.unwrapped.renderer
    render_data = _build_arm_soft_mesh_render_data(renderer)

    index_side = 0
    index_wrist = 0
    for cam_id in range(len(cfg.env.cameras)):
        camera = cfg.env.cameras[cam_id]
        if camera["type"] == "side":
            rgba, _ = renderer.render_rgba(render_data=render_data, camera=renderer.cameras[index_side])
            index_side += 1
        elif camera["type"] == "wrist":
            rgba, _ = renderer.render_wrist_rgba(
                render_data=render_data, camera=renderer.wrist_cameras[index_wrist]
            )
            index_wrist += 1
        else:
            raise ValueError(f"Unknown camera type {camera['type']}")

        rgba_np = (rgba.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
        rgba_bgra = cv2.cvtColor(rgba_np, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(
            f"{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/{dirname}/{frame_idx:06d}.png",
            rgba_bgra,
        )


def _import_lerobot_dataset():
    """Import LeRobotDataset with local submodule fallback."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except ImportError:
        lerobot_root = Path(__file__).parents[1] / "policy" / "third_party" / "lerobot"
        if str(lerobot_root) not in sys.path:
            sys.path.insert(0, str(lerobot_root))
        # The environment may contain another `lerobot` package without `lerobot.common`.
        # Remove it from import cache so local submodule takes precedence.
        stale_modules = [k for k in sys.modules.keys() if k == "lerobot" or k.startswith("lerobot.")]
        for key in stale_modules:
            sys.modules.pop(key, None)
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            return LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobot export requires the LeRobot package. Install policy dependencies "
                "or ensure policy/third_party/lerobot is available."
            ) from exc


class LeRobotReplayExporter:
    """Export replay trajectories into a LeRobot-format dataset."""

    def __init__(
        self,
        dataset_root: Path,
        repo_id: str,
        task_name: str,
        fps: int,
        use_pusher: bool,
        front_shape: Optional[Tuple[int, int]],
        wrist_shape: Optional[Tuple[int, int]],
        use_videos: bool,
        image_writer_processes: int,
        image_writer_threads: int,
        video_backend: Optional[str],
        robot_type: str,
    ) -> None:
        LeRobotDataset = _import_lerobot_dataset()

        features: Dict[str, Dict[str, Any]] = {
            "action": {
                "dtype": "float32",
                "shape": (2,) if use_pusher else (8,),
                "names": {
                    "waypoint": ["x_pos", "y_pos"] if use_pusher else [
                        "x_pos", "y_pos", "z_pos", "q_w", "q_x", "q_y", "q_z", "gripper_qpos"
                    ]
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,) if use_pusher else (8,),
                "names": {
                    "waypoint": ["x_pos", "y_pos"] if use_pusher else [
                        "x_pos", "y_pos", "z_pos", "q_w", "q_x", "q_y", "q_z", "gripper_qpos"
                    ]
                },
            },
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
                "names": None,
            },
        }

        if front_shape is not None:
            front_h, front_w = front_shape
            features["observation.images.front"] = {
                "dtype": "video" if use_videos else "image",
                "shape": (front_h, front_w, 3),
                "names": ["height", "width", "channel"],
            }
        if wrist_shape is not None:
            wrist_h, wrist_w = wrist_shape
            features["observation.images.wrist"] = {
                "dtype": "video" if use_videos else "image",
                "shape": (wrist_h, wrist_w, 3),
                "names": ["height", "width", "channel"],
            }
        if "observation.images.front" not in features and "observation.images.wrist" not in features:
            raise ValueError("LeRobot export requires at least one camera stream (side or wrist).")

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_root,
            robot_type=robot_type,
            fps=int(fps),
            features=features,
            use_videos=use_videos,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
            video_backend=video_backend,
        )
        self.task_name = task_name
        self.fps = int(fps)
        self.use_pusher = use_pusher
        self.has_front = front_shape is not None
        self.has_wrist = wrist_shape is not None

    @staticmethod
    def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
        image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
        image_np = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
        return image_np

    def add_frame(
        self,
        image_list: list[torch.Tensor],
        image_list_wrist: list[torch.Tensor],
        obs_pos: torch.Tensor,
        obs_quat_wxyz: torch.Tensor,
        obs_gripper_qpos: torch.Tensor,
        action_pos: torch.Tensor,
        action_quat_wxyz: torch.Tensor,
        action_gripper_qpos: torch.Tensor,
        step_idx: int,
        n_steps: int,
    ) -> None:
        if self.use_pusher:
            observation_state = obs_pos[0, :2].detach().cpu().numpy().astype(np.float32)
            action = action_pos[0, :2].detach().cpu().numpy().astype(np.float32)
        else:
            observation_state = np.concatenate(
                [
                    obs_pos[0].detach().cpu().numpy(),
                    obs_quat_wxyz[0].detach().cpu().numpy(),
                    obs_gripper_qpos[0].detach().cpu().numpy().reshape(-1),
                ],
                axis=0,
            ).astype(np.float32)
            action = np.concatenate(
                [
                    action_pos[0].detach().cpu().numpy(),
                    action_quat_wxyz[0].detach().cpu().numpy(),
                    action_gripper_qpos[0].detach().cpu().numpy().reshape(-1),
                ],
                axis=0,
            ).astype(np.float32)

        frame: Dict[str, Any] = {
            "action": action,
            "observation.state": observation_state,
            "next.done": np.array([step_idx == n_steps - 1], dtype=bool),
        }
        if self.has_front:
            frame["observation.images.front"] = self._to_uint8_hwc(image_list[0])
        if self.has_wrist:
            frame["observation.images.wrist"] = self._to_uint8_hwc(image_list_wrist[0])

        self.dataset.add_frame(
            frame=frame,
            task=self.task_name,
            timestamp=float(step_idx) / float(self.fps),
        )

    def save_episode(self) -> None:
        self.dataset.save_episode()


def _build_lerobot_exporter(
    cfg,
    run_dir: Path,
    run_name: str,
    frame_rate: int,
    obs: Dict[str, Any],
    overwrite_output: bool,
) -> Optional[LeRobotReplayExporter]:
    export_cfg = cfg.get("lerobot_export", None)
    if export_cfg is None or not bool(export_cfg.get("enabled", False)):
        return None

    output_dir_cfg = export_cfg.get("output_dir", None)
    dataset_root = Path(output_dir_cfg) if output_dir_cfg is not None else run_dir / "lerobot_dataset"
    if dataset_root.exists():
        if overwrite_output:
            shutil.rmtree(dataset_root)
        else:
            raise FileExistsError(
                f"LeRobot output directory already exists: {dataset_root}. "
                "Set overwrite_output=true or choose lerobot_export.output_dir."
            )

    repo_id = export_cfg.get("repo_id", None)
    if repo_id is None:
        repo_id = f"replay/{run_name}"
    task_name = str(export_cfg.get("task_name", "replay"))
    use_videos = bool(export_cfg.get("use_videos", True))
    image_writer_processes = int(export_cfg.get("image_writer_processes", 0))
    image_writer_threads = int(export_cfg.get("image_writer_threads", 0))
    video_backend = export_cfg.get("video_backend", None)

    front_shape: Optional[Tuple[int, int]] = None
    wrist_shape: Optional[Tuple[int, int]] = None
    if len(obs.get("image_list", [])) > 0:
        _, h, w = obs["image_list"][0].shape
        front_shape = (int(h), int(w))
    if len(obs.get("image_wrist_list", [])) > 0:
        _, h, w = obs["image_wrist_list"][0].shape
        wrist_shape = (int(h), int(w))

    exporter = LeRobotReplayExporter(
        dataset_root=dataset_root,
        repo_id=str(repo_id),
        task_name=task_name,
        fps=int(frame_rate),
        use_pusher=bool(cfg.env.robot.use_pusher),
        front_shape=front_shape,
        wrist_shape=wrist_shape,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        video_backend=video_backend,
        robot_type=str(cfg.env.robot.type),
    )
    print(f"LeRobot export enabled: {dataset_root}")
    return exporter


def _as_matrix(pose_list: list) -> np.ndarray:
    pose = np.array(pose_list, dtype=np.float32)
    return pose.reshape(4, 4)


def _find_mesh_cfg(cfg, name: str) -> Optional[Dict]:
    meshes = cfg.gs.get("meshes", None)
    if meshes is None:
        return None
    for mesh_cfg in meshes:
        if mesh_cfg.get("name", None) == name:
            return mesh_cfg
    return None


def _load_clip_mesh(cfg) -> Tuple["trimesh.Trimesh", np.ndarray]:
    import trimesh

    clip_mesh_path = cfg.get("clip_mesh_path", None)
    clip_pose_cfg = cfg.get("clip_pose", None)
    clip_mesh_name = cfg.get("clip_mesh_name", "clip")

    mesh_cfg = None
    if clip_mesh_path is None or clip_pose_cfg is None:
        mesh_cfg = _find_mesh_cfg(cfg, clip_mesh_name)
        if mesh_cfg is None:
            raise ValueError(f"Could not find mesh '{clip_mesh_name}' in cfg.gs.meshes.")

    if clip_mesh_path is None:
        clip_mesh_path = mesh_cfg.get("mesh_path", None)
    if clip_mesh_path is None:
        raise ValueError("clip_mesh_path is required (no mesh_path in cfg.gs.meshes).")

    if clip_pose_cfg is None:
        clip_pose_cfg = mesh_cfg.get("pose", None)
    if clip_pose_cfg is None:
        raise ValueError("clip_pose is required (no pose in cfg.gs.meshes).")

    clip_pose = _as_matrix(clip_pose_cfg)
    mesh = trimesh.load(clip_mesh_path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Failed to load mesh from {clip_mesh_path}.")
    return mesh, clip_pose


def _compute_groove_bounds(points_lwh: np.ndarray, cfg) -> Dict[str, float]:
    min_vals = points_lwh.min(axis=0)
    max_vals = points_lwh.max(axis=0)
    length_range = max_vals[0] - min_vals[0]
    height_range = max_vals[2] - min_vals[2]

    length_margin_ratio = float(cfg.get("length_margin_ratio", 0.1))
    height_bottom_margin_ratio = float(cfg.get("height_bottom_margin_ratio", 0.05))
    height_top_margin_ratio = float(cfg.get("height_top_margin_ratio", 0.05))

    length_margin = length_range * length_margin_ratio
    height_bottom = min_vals[2] + height_range * height_bottom_margin_ratio
    height_top = max_vals[2] - height_range * height_top_margin_ratio

    w = points_lwh[:, 1]
    median_w = np.median(w)
    left_mask = w <= median_w
    right_mask = w > median_w

    left_pct = float(cfg.get("width_inner_percentile_left", 95))
    right_pct = float(cfg.get("width_inner_percentile_right", 5))

    if left_mask.sum() < 10 or right_mask.sum() < 10:
        left_inner = np.percentile(w, 40)
        right_inner = np.percentile(w, 60)
    else:
        left_inner = np.percentile(w[left_mask], left_pct)
        right_inner = np.percentile(w[right_mask], right_pct)

    if left_inner >= right_inner:
        left_inner = np.percentile(w, 40)
        right_inner = np.percentile(w, 60)

    width_margin = float(cfg.get("width_margin", 0.0))
    width_min = left_inner + width_margin
    width_max = right_inner - width_margin

    length_min = min_vals[0] + length_margin
    length_max = max_vals[0] - length_margin

    height_inner_percentile = cfg.get("height_inner_percentile", None)
    height_inner_margin = float(cfg.get("height_inner_margin", 0.0))
    if height_inner_percentile is not None:
        mask = (
            (points_lwh[:, 0] >= length_min)
            & (points_lwh[:, 0] <= length_max)
            & (points_lwh[:, 1] >= width_min)
            & (points_lwh[:, 1] <= width_max)
        )
        if int(mask.sum()) >= 10:
            height_bottom = np.percentile(points_lwh[mask, 2], float(height_inner_percentile))
            height_bottom += height_inner_margin

    height_min_override = cfg.get("height_min_override", None)
    if height_min_override is not None:
        height_bottom = float(height_min_override)

    return {
        "length_min": length_min,
        "length_max": length_max,
        "width_min": width_min,
        "width_max": width_max,
        "height_min": height_bottom,
        "height_max": height_top,
    }


def _compute_groove_bounds_from_normals(
    points_lwh: np.ndarray,
    normals_lwh: np.ndarray,
    cfg,
) -> Dict[str, float]:
    up_axis = str(cfg.get("normal_up_axis", "z")).lower()
    up_idx = {"x": 0, "y": 1, "z": 2}.get(up_axis, 2)
    up_thresh = float(cfg.get("normal_up_threshold", 0.9))

    axis_map = {"x": 0, "y": 1, "z": 2}
    center_axis = str(cfg.get("center_axis", "x")).lower()
    center_idx = axis_map.get(center_axis, 0)
    center_fraction = float(cfg.get("center_fraction", 0.5))
    center_fraction = max(min(center_fraction, 1.0), 0.0)

    min_vals = points_lwh.min(axis=0)
    max_vals = points_lwh.max(axis=0)
    center = (min_vals + max_vals) * 0.5
    axis_range = max_vals[center_idx] - min_vals[center_idx]
    half_span = axis_range * center_fraction * 0.5

    up_mask = normals_lwh[:, up_idx] > up_thresh
    center_mask = np.abs(points_lwh[:, center_idx] - center[center_idx]) < half_span
    groove_mask = up_mask & center_mask
    groove_points = points_lwh[groove_mask]

    min_points = int(cfg.get("min_groove_points", 50))
    if groove_points.shape[0] < min_points:
        return _compute_groove_bounds(points_lwh, cfg)

    pad = float(cfg.get("groove_bounds_padding", 0.0))
    gmin = groove_points.min(axis=0) - pad
    gmax = groove_points.max(axis=0) + pad

    height_min_override = cfg.get("height_min_override", None)
    if height_min_override is not None:
        gmin[2] = float(height_min_override)

    return {
        "length_min": gmin[0],
        "length_max": gmax[0],
        "width_min": gmin[1],
        "width_max": gmax[1],
        "height_min": gmin[2],
        "height_max": gmax[2],
    }


def _get_soft_points_world(env) -> np.ndarray:
    if hasattr(env, "physics") and hasattr(env.physics, "dynamics_module"):
        points = env.physics.dynamics_module.current_points
        return points.detach().cpu().numpy()
    state = env.renderer.get_state()
    points = state["x"]
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    return points


def _count_points_in_groove(points_local: np.ndarray, bounds: Dict[str, float]) -> int:
    in_mask = (
        (points_local[:, 0] >= bounds["length_min"])
        & (points_local[:, 0] <= bounds["length_max"])
        & (points_local[:, 1] >= bounds["width_min"])
        & (points_local[:, 1] <= bounds["width_max"])
        & (points_local[:, 2] >= bounds["height_min"])
        & (points_local[:, 2] <= bounds["height_max"])
    )
    return int(in_mask.sum())


def _segment_plane_intersections_xy(
    p0: np.ndarray,
    p1: np.ndarray,
    z_plane: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Check segment intersections with plane z=z_plane and xy rectangle bounds.

    Returns:
        Boolean mask (M,) indicating whether each segment intersects.
    """
    z0 = p0[:, 2]
    z1 = p1[:, 2]
    dz = z1 - z0

    parallel = np.isclose(dz, 0.0, atol=eps)
    t = np.zeros_like(dz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t[~parallel] = (z_plane - z0[~parallel]) / dz[~parallel]
    on_segment = (~parallel) & (t >= -eps) & (t <= 1.0 + eps)

    xi = p0[:, 0] + t * (p1[:, 0] - p0[:, 0])
    yi = p0[:, 1] + t * (p1[:, 1] - p0[:, 1])
    inside_rect = (
        (xi >= x_min - eps)
        & (xi <= x_max + eps)
        & (yi >= y_min - eps)
        & (yi <= y_max + eps)
    )
    hits_crossing = on_segment & inside_rect

    coplanar = parallel & np.isclose(z0 - z_plane, 0.0, atol=eps)
    end0_in = (
        (p0[:, 0] >= x_min - eps)
        & (p0[:, 0] <= x_max + eps)
        & (p0[:, 1] >= y_min - eps)
        & (p0[:, 1] <= y_max + eps)
    )
    end1_in = (
        (p1[:, 0] >= x_min - eps)
        & (p1[:, 0] <= x_max + eps)
        & (p1[:, 1] >= y_min - eps)
        & (p1[:, 1] <= y_max + eps)
    )
    hits_coplanar = coplanar & (end0_in | end1_in)
    return hits_crossing | hits_coplanar


def _count_rope_groove_plane_intersections(
    points_local: np.ndarray,
    springs_idx: np.ndarray,
    bounds: Dict[str, float],
) -> Dict[str, int]:
    if springs_idx.ndim != 2 or springs_idx.shape[1] != 2:
        raise ValueError(f"springs_idx must be (M,2), got {springs_idx.shape}")
    if points_local.ndim != 2 or points_local.shape[1] != 3:
        raise ValueError(f"points_local must be (N,3), got {points_local.shape}")

    if np.any(springs_idx < 0) or np.any(springs_idx >= points_local.shape[0]):
        raise ValueError("springs_idx has out-of-range indices for points_local.")

    p0 = points_local[springs_idx[:, 0]]
    p1 = points_local[springs_idx[:, 1]]

    length_min = float(bounds["length_min"])
    length_max = float(bounds["length_max"])
    width_min = float(bounds["width_min"])
    width_max = float(bounds["width_max"])
    z_min = float(bounds["height_min"])
    z_max = float(bounds["height_max"])

    hits_bottom = _segment_plane_intersections_xy(
        p0, p1, z_min, length_min, length_max, width_min, width_max
    )
    hits_top = _segment_plane_intersections_xy(
        p0, p1, z_max, length_min, length_max, width_min, width_max
    )
    return {
        "bottom_count": int(np.count_nonzero(hits_bottom)),
        "top_count": int(np.count_nonzero(hits_top)),
    }


def _init_clip_success_tracker(env, cfg) -> Dict[str, Any]:
    mesh, clip_pose = _load_clip_mesh(cfg)
    groove_mode = str(cfg.get("groove_detection_mode", "bbox")).lower()
    if groove_mode == "normal":
        if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
            mesh.compute_vertex_normals()
        bounds = _compute_groove_bounds_from_normals(mesh.vertices, mesh.vertex_normals, cfg)
    else:
        bounds = _compute_groove_bounds(mesh.vertices, cfg)

    state = env.unwrapped.get_state()
    springs = state.get("physics", {}).get("init_springs", None)
    if springs is None:
        raise ValueError("clip_success_mode=rope_routed requires physics.init_springs in env state.")
    if isinstance(springs, torch.Tensor):
        springs_idx = springs.detach().cpu().numpy().astype(np.int64)
    else:
        springs_idx = np.asarray(springs, dtype=np.int64)

    return {
        "clip_pose": clip_pose,
        "bounds": bounds,
        "springs_idx": springs_idx,
        "routed_flags": [],
        "frame_counts": [],
    }


def _update_clip_success_tracker(tracker: Dict[str, Any], env, cfg) -> None:
    points_world = _get_soft_points_world(env)
    rot = tracker["clip_pose"][:3, :3]
    trans = tracker["clip_pose"][:3, 3]
    points_local = (points_world - trans) @ rot

    counts = _count_rope_groove_plane_intersections(
        points_local=points_local,
        springs_idx=tracker["springs_idx"],
        bounds=tracker["bounds"],
    )
    min_bottom = int(cfg.get("clip_success_plane_min_bottom", 100))
    min_top = int(cfg.get("clip_success_plane_min_top", 100))
    routed = bool(counts["bottom_count"] >= min_bottom and counts["top_count"] >= min_top)

    tracker["frame_counts"].append(counts)
    tracker["routed_flags"].append(routed)
    tracker["last_points_in_groove"] = _count_points_in_groove(points_local, tracker["bounds"])


def _evaluate_clip_success(
    env,
    cfg,
    output_dir: Path,
    episode_id: int,
    tracker: Optional[Dict[str, Any]] = None,
    n_steps: Optional[int] = None,
) -> Dict[str, Any]:
    mode = str(cfg.get("clip_success_mode", "rope_routed")).lower()
    if mode == "rope_routed":
        if tracker is None:
            tracker = _init_clip_success_tracker(env, cfg)
            _update_clip_success_tracker(tracker, env, cfg)

        routed_flags = list(tracker.get("routed_flags", []))
        frame_counts = list(tracker.get("frame_counts", []))
        total_steps = int(n_steps if n_steps is not None else len(routed_flags))
        if total_steps <= 0:
            total_steps = len(routed_flags)

        tail_steps_cfg = cfg.get("clip_success_tail_steps", None)
        if tail_steps_cfg is not None:
            tail_steps = max(1, min(int(tail_steps_cfg), len(routed_flags)))
        else:
            tail_ratio = float(cfg.get("clip_success_tail_ratio", 0.1111111111))
            tail_steps = max(1, int(np.ceil(max(total_steps, 1) * tail_ratio)))
            tail_steps = min(tail_steps, len(routed_flags))

        if tail_steps > 0:
            tail_flags = routed_flags[-tail_steps:]
        else:
            tail_flags = []
        routed_count_tail = int(sum(1 for flag in tail_flags if flag))
        routed_count_total = int(sum(1 for flag in routed_flags if flag))

        required_steps_cfg = cfg.get("clip_success_required_routed_steps", None)
        if required_steps_cfg is not None:
            required_routed = max(1, int(required_steps_cfg))
        else:
            required_ratio = float(cfg.get("clip_success_required_routed_ratio", 0.3))
            required_routed = max(1, int(np.ceil(tail_steps * required_ratio)))

        success = bool(routed_count_tail >= required_routed)
        last_counts = frame_counts[-1] if frame_counts else {"bottom_count": 0, "top_count": 0}

        result = {
            "episode_id": int(episode_id),
            "mode": mode,
            "success": success,
            "total_steps": int(total_steps),
            "tail_steps": int(tail_steps),
            "routed_true_frames_total": int(routed_count_total),
            "routed_true_frames_tail": int(routed_count_tail),
            "required_routed_frames": int(required_routed),
            "plane_min_bottom_threshold": int(cfg.get("clip_success_plane_min_bottom", 100)),
            "plane_min_top_threshold": int(cfg.get("clip_success_plane_min_top", 100)),
            "last_bottom_count": int(last_counts.get("bottom_count", 0)),
            "last_top_count": int(last_counts.get("top_count", 0)),
            # Keep these keys for compatibility with existing summary scripts.
            "points_in_groove": int(tracker.get("last_points_in_groove", 0)),
            "success_min_points": int(cfg.get("clip_success_min_points", 5)),
            "groove_bounds": tracker["bounds"],
        }
    else:
        mesh, clip_pose = _load_clip_mesh(cfg)
        groove_mode = str(cfg.get("groove_detection_mode", "bbox")).lower()
        if groove_mode == "normal":
            if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
                mesh.compute_vertex_normals()
            bounds = _compute_groove_bounds_from_normals(mesh.vertices, mesh.vertex_normals, cfg)
        else:
            bounds = _compute_groove_bounds(mesh.vertices, cfg)

        points_world = _get_soft_points_world(env)
        rot = clip_pose[:3, :3]
        trans = clip_pose[:3, 3]
        points_local = (points_world - trans) @ rot

        count_in = _count_points_in_groove(points_local, bounds)
        success_thresh = int(cfg.get("clip_success_min_points", 5))
        success = bool(count_in >= success_thresh)

        result = {
            "episode_id": int(episode_id),
            "mode": "points_in_groove",
            "success": success,
            "points_in_groove": int(count_in),
            "success_min_points": int(success_thresh),
            "groove_bounds": bounds,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"clip_success_ep{episode_id:04d}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _save_final_state(
    env: Any,
    output_dir: Path,
    episode_id: int,
    n_steps: int,
    action: Optional[torch.Tensor],
    stabilization_steps: int,
    do_velocity_control: bool,
    dir_name: str,
) -> None:
    """Save the final soft-body state after optional stabilization.

    Args:
        env: Environment with renderer/physics access.
        output_dir: Root output directory for the run.
        episode_id: Episode index.
        n_steps: Number of replay steps executed.
        action: Last action tensor used in replay.
        stabilization_steps: Number of steps to stabilize before saving.
        do_velocity_control: Whether to use velocity control when stepping.
        dir_name: Directory name for saving final state files.
    """
    if action is None:
        return

    for _ in range(max(int(stabilization_steps), 0)):
        env.step({
            "action": action,
            "do_velocity_control": do_velocity_control,
        })

    state_dir = output_dir / f"episode_{episode_id:04d}" / dir_name
    state_dir.mkdir(parents=True, exist_ok=True)

    state = env.unwrapped.get_state()
    with open(state_dir / "state.pkl", "wb") as f:
        import pickle as pkl

        pkl.dump(state, f)

    renderer_state = env.unwrapped.renderer.get_state()
    points_world = renderer_state["x"].detach().cpu().numpy()

    state_npy = {
        "points": points_world,
        "points_world": points_world,
        "num_object_points": points_world.shape[0],
        "metadata": {
            "frame": "world",
            "pose_applied": True,
            "episode_id": episode_id,
        },
    }
    np.save(state_dir / "state.npy", state_npy)

    metadata = {
        "episode_id": episode_id,
        "n_steps": n_steps,
        "final_frame": n_steps - 1,
        "stabilization_steps": int(stabilization_steps),
        "timestamp": time.time(),
    }
    with open(state_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


@hydra.main(version_base='1.2', config_path='../cfg', config_name="replay")
def main(cfg):
    OmegaConf.resolve(cfg)

    # robot traj to replay
    gt_dir = Path(cfg.gt_dir)
    assert gt_dir.exists(), f"GT directory {cfg.gt_dir} does not exist"

    if (gt_dir / 'episode_0000').exists():  # there are multiple episodes
        use_episodes = True
        n_episodes = len(sorted(glob.glob(str(gt_dir / 'episode_*'))))
    else:  # there is only one episode
        use_episodes = False
        n_episodes = 1

    # unique run name
    if cfg.timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = cfg.timestamp
    run_name = f"{timestamp}"
    save_depth = bool(cfg.get("save_depth", True))
    save_arm_soft_mesh_rgba = bool(cfg.get("save_arm_soft_mesh_rgba", False))
    arm_soft_mesh_rgba_dirname = str(cfg.get("arm_soft_mesh_rgba_dirname", "arm_soft_mesh_rgba"))

    policy_rollout_format = bool(cfg.get("policy_rollout_format", False))
    output_root = cfg.get("output_root", None)
    if output_root is None:
        if policy_rollout_format:
            output_root = "log/policy_rollouts_replay"
        else:
            output_root = str(Path(cfg.exp_root) / "output_replay")
    out_path = str(Path(output_root))
    run_dir = Path(out_path) / run_name
    overwrite_output = bool(cfg.get("overwrite_output", False))
    mkdir(run_dir, resume=False, overwrite=overwrite_output)
    save_hydra_config = cfg.get("save_hydra_config", None)
    if save_hydra_config is None:
        save_hydra_config = not policy_rollout_format
    if save_hydra_config:
        OmegaConf.save(cfg, run_dir / "hydra.yaml", resolve=True)
    ik_log_path = cfg.get("ik_log_path", None)
    if ik_log_path is not None:
        set_ik_log_path(ik_log_path)
    randomize = bool(cfg.get("randomize", True))
    save_final_state = bool(cfg.get("save_final_state", False))
    final_state_stabilization_steps = int(cfg.get("final_state_stabilization_steps", 0))
    final_state_dir_name = str(cfg.get("final_state_dir_name", "final_state"))
    lerobot_exporter: Optional[LeRobotReplayExporter] = None

    cloth3_success_eval_enabled = bool(cfg.get("cloth3_success_eval", False))
    cloth3_success_list: list[int] = []

    for episode_id in range(n_episodes):
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}', exist_ok=True)
        if use_episodes:
            episode_gt_dir = gt_dir / f'episode_{episode_id:04d}'
        else:
            episode_gt_dir = gt_dir
        if not (episode_gt_dir / "robot").exists():
            print(f"Episode directory {episode_gt_dir} does not exist")
            continue

        robot_paths = sorted(glob.glob(str(episode_gt_dir / "robot" / "*.json")))
        n_frames = len(robot_paths)

        robot_traj_list = []
        robot_rot_list = []
        robot_gripper_list = []
        robot_qpos_list = []
        for frame_id in range(n_frames):
            robot_traj, robot_rot, robot_gripper, robot_qpos = load_robot_json(
                robot_paths[frame_id], use_qpos=cfg.use_qpos
            )
            robot_traj_list.append(robot_traj)
            robot_rot_list.append(robot_rot)
            robot_gripper_list.append(robot_gripper)
            robot_qpos_list.append(robot_qpos)

        robot_traj_list = np.stack(robot_traj_list)  # (n, n_grippers, 3)
        robot_rot_list = np.stack(robot_rot_list)  # (n, n_grippers, 3, 3)
        robot_gripper_list = np.stack(robot_gripper_list)  # (n, n_grippers)

        recorded_steps = len(robot_traj_list)
        start_idx = cfg.get("start_idx", None)
        init_from_trajectory = bool(cfg.get("init_from_trajectory", False))
        init_frame_idx = cfg.get("init_frame_idx", None)
        init_pose = None
        if init_from_trajectory:
            if init_frame_idx is None:
                init_frame_idx = 0
            init_frame_idx = int(init_frame_idx)
            init_frame_idx = max(0, min(init_frame_idx, recorded_steps - 1))
            init_pose = (
                robot_traj_list[init_frame_idx],
                robot_rot_list[init_frame_idx],
                robot_gripper_list[init_frame_idx],
            )
            if start_idx is None:
                start_idx = init_frame_idx
        if start_idx is not None:
            start_idx = int(start_idx)
            start_idx = max(0, min(start_idx, recorded_steps - 1))
            robot_traj_list = robot_traj_list[start_idx:]
            robot_rot_list = robot_rot_list[start_idx:]
            robot_gripper_list = robot_gripper_list[start_idx:]
            robot_qpos_list = robot_qpos_list[start_idx:]
            recorded_steps = len(robot_traj_list)
        runtime_cfg = _parse_runtime_perturbation(cfg)
        runtime_start_step = None
        runtime_steps = 0
        if runtime_cfg is not None:
            runtime_start_step = runtime_cfg["start_step"]
            if runtime_start_step is None:
                runtime_start_step = recorded_steps
            runtime_start_step = max(0, min(int(runtime_start_step), recorded_steps))
            runtime_steps = int(runtime_cfg["total_steps"])

        robot_cutoff = recorded_steps
        n_steps = recorded_steps
        if runtime_cfg is not None:
            robot_cutoff = runtime_start_step
            if runtime_start_step < recorded_steps:
                n_steps = runtime_start_step + runtime_steps
            else:
                n_steps = recorded_steps + runtime_steps

        frame_rate = cfg.physics.fps
        duration = n_steps // frame_rate  # seconds
        print(f"Replaying {n_steps} steps, duration {duration}s")
        if runtime_cfg is not None:
            print(
                "Runtime perturbation enabled: "
                f"mode={runtime_cfg['mode']}, "
                f"start_step={runtime_start_step}, "
                f"num_steps={runtime_cfg['num_steps']}, "
                f"release_steps={runtime_cfg['release_steps']}"
            )

        deformed_state_path = cfg.get("deformed_state_path", None)
        use_tcp = bool(cfg.get("use_tcp", False))
        tcp_offset_cfg = cfg.get("tcp_offset", None)
        tcp_rot_cfg = cfg.get("tcp_rot", None)
        if tcp_offset_cfg is not None and not use_tcp:
            use_tcp = True
        # random reset
        env = gym.make(cfg.env_name, max_episode_steps=frame_rate * duration, cfg=cfg, 
            obs_mode=cfg.obs_mode, exp_root=cfg.exp_root, local_rank=0, randomize=randomize)
        reset_seed_cfg = cfg.get("reset_seed", None)
        reset_seed = episode_id if reset_seed_cfg is None else int(reset_seed_cfg)
        if deformed_state_path is not None:
            obs, reset_info = env.reset(seed=reset_seed, deformed_state_path=deformed_state_path)
        else:
            obs, reset_info = env.reset(seed=reset_seed)

        tcp_offset = None
        tcp_rot_offset = None
        if use_tcp:
            if tcp_offset_cfg is None:
                raise ValueError("use_tcp=True requires tcp_offset=[x,y,z].")
            tcp_offset = torch.tensor(
                list(tcp_offset_cfg),
                dtype=torch.float32,
                device=env.unwrapped.physics.device,
            )
            if tcp_rot_cfg is None:
                tcp_rot_offset = torch.eye(3, dtype=torch.float32, device=env.unwrapped.physics.device)
            else:
                tcp_rot_arr = np.array(tcp_rot_cfg, dtype=np.float32).reshape(-1)
                if tcp_rot_arr.size == 9:
                    tcp_rot_offset = torch.from_numpy(tcp_rot_arr.reshape(3, 3)).to(
                        env.unwrapped.physics.device
                    )
                elif tcp_rot_arr.size == 4:
                    tcp_rot_offset = kornia.geometry.conversions.quaternion_to_rotation_matrix(
                        torch.from_numpy(tcp_rot_arr.reshape(1, 4)).to(
                            torch.float32
                        ).to(env.unwrapped.physics.device)
                    )[0]
                elif tcp_rot_arr.size == 3:
                    tcp_rot_offset = torch.from_numpy(
                        transforms3d.euler.euler2mat(
                            tcp_rot_arr[0], tcp_rot_arr[1], tcp_rot_arr[2], axes="sxyz"
                        ).astype(np.float32)
                    ).to(env.unwrapped.physics.device)
                else:
                    raise ValueError("tcp_rot must be length 3 (rpy), 4 (quat wxyz), or 9 (rotation matrix).")

        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_0/rgb', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_1/rgb', exist_ok=True)
        if save_depth:
            os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_0/depth', exist_ok=True)
            os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_1/depth', exist_ok=True)
        if save_arm_soft_mesh_rgba:
            for cam_id in range(len(cfg.env.cameras)):
                os.makedirs(
                    f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/{arm_soft_mesh_rgba_dirname}',
                    exist_ok=True,
                )
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration', exist_ok=True)
        os.makedirs(f'{out_path}/{run_name}/episode_{episode_id:04d}/robot', exist_ok=True)
        save_start_final = cfg.get("save_start_final_images", None)
        if save_start_final is None:
            save_start_final = not policy_rollout_format
        if save_start_final:
            os.makedirs(f'{out_path}/{run_name}/start_images', exist_ok=True)
            os.makedirs(f'{out_path}/{run_name}/final_images', exist_ok=True)

        # save calibration data
        rvecs = []
        tvecs = []
        for cam_id in range(len(cfg.env.cameras)):
            camera = cfg.env.cameras[cam_id]
            if 'c2w' in camera:
                trans_mat = np.array(camera['c2w']).reshape(4, 4).astype(np.float32)
                trans_mat = np.linalg.inv(trans_mat)  # w2c
            else:
                assert 'w2c' in camera
                trans_mat = np.array(camera['w2c']).reshape(4, 4).astype(np.float32)
            rvec = R.from_matrix(trans_mat[:3, :3]).as_rotvec()
            tvec = trans_mat[:3, 3]
            rvecs.append(rvec)
            tvecs.append(tvec)
        rvecs_save_npy = np.stack(rvecs).reshape(-1, 3, 1)
        tvecs_save_npy = np.stack(tvecs).reshape(-1, 3, 1)

        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/rvecs.npy', rvecs_save_npy)
        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/tvecs.npy', tvecs_save_npy)

        intrs = []
        for cam_id in range(len(cfg.env.cameras)):
            camera = cfg.env.cameras[cam_id]
            intr_mat = np.array(camera['intr']).reshape(3, 3).astype(np.float32)
            intrs.append(intr_mat)
        intrs_save = np.stack(intrs).reshape(-1, 3, 3)
        np.save(f'{out_path}/{run_name}/episode_{episode_id:04d}/calibration/intrinsics.npy', intrs_save)

        print("Resetting robot initial state")
        init_stabilization_steps = int(cfg.get("init_stabilization_steps", 30))
        n_grippers = int(cfg.env.robot.n_grippers)
        if init_pose is not None:
            init_xyz, init_rot, init_gripper = init_pose
            eef_xyz = torch.from_numpy(init_xyz).to(torch.float32).to(env.unwrapped.physics.device)
            eef_rot = torch.from_numpy(init_rot).to(torch.float32).to(env.unwrapped.physics.device)
            eef_gripper = torch.from_numpy(init_gripper).to(torch.float32).to(env.unwrapped.physics.device)
            if eef_gripper.ndim == 1:
                eef_gripper = eef_gripper.reshape(n_grippers, 1)
            if use_tcp:
                eef_xyz, eef_rot = tcp_to_eef(eef_xyz, eef_rot, tcp_offset, tcp_rot_offset)
            action = torch.cat([
                eef_xyz.reshape(n_grippers, 3),
                eef_rot.reshape(n_grippers, -1),
                eef_gripper.reshape(n_grippers, 1),
            ], dim=1)
            for _ in range(init_stabilization_steps):
                env.step({'action': action, 'do_velocity_control': False})
            obs = env.unwrapped.get_obs()
            print(f"Initialized robot to trajectory frame {init_frame_idx}")
        else:
            eef_xyz = obs['robot']['eef_xyz']  # (n_grippers, 3)
            eef_quat = obs['robot']['eef_quat']  # (n_grippers, 4)
            eef_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(eef_quat)  # (n_grippers, 3, 3)
            eef_gripper = obs['robot']['eef_gripper']  # (n_grippers, 1)
            action = torch.cat([
                eef_xyz,
                eef_rot.reshape(eef_rot.shape[0], -1),
                eef_gripper
            ], dim=1)
            for _ in range(init_stabilization_steps):
                env.step({'action': action, 'do_velocity_control': False})
            obs = env.unwrapped.get_obs()

        if lerobot_exporter is None:
            lerobot_exporter = _build_lerobot_exporter(
                cfg=cfg,
                run_dir=run_dir,
                run_name=run_name,
                frame_rate=frame_rate,
                obs=obs,
                overwrite_output=overwrite_output,
            )

        timestamps = []
        timestamp_offset_s = float(cfg.get("timestamp_offset_s", 0.021))
        base_time = time.time()
        last_action = None
        n_grippers = robot_traj_list.shape[1]
        runtime_state_xyz = None
        runtime_state_rot = None
        clip_success_eval_enabled = bool(cfg.get("clip_success_eval", False))
        clip_success_mode = str(cfg.get("clip_success_mode", "rope_routed")).lower()
        clip_success_tracker: Optional[Dict[str, Any]] = None
        if clip_success_eval_enabled and clip_success_mode == "rope_routed":
            clip_success_tracker = _init_clip_success_tracker(env, cfg)
        for cnt in range(n_steps):
            if ik_log_path is not None:
                set_ik_log_frame(cnt)
            torch.cuda.synchronize()
            tt0 = time.perf_counter()

            image_list = obs['image_list']  # list of (3, H, W) tensors
            image_list_wrist = obs['image_wrist_list']  # list of (3, H, W) tensors

            index_side = 0
            index_wrist = 0
            for cam_id in range(len(cfg.env.cameras)):
                camera = cfg.env.cameras[cam_id]
                if camera['type'] == 'side':
                    image = image_list[index_side]
                    index_side += 1
                elif camera['type'] == 'wrist':
                    image = image_list_wrist[index_wrist]
                    index_wrist += 1
                else:
                    raise ValueError(f"Unknown camera type {camera['type']}")

                image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt:06d}.jpg', image)

                if cnt == 0 and save_start_final:
                    cv2.imwrite(
                        f'{out_path}/{run_name}/start_images/episode_{episode_id:04d}_camera_{cam_id}.jpg',
                        image,
                    )
            if save_arm_soft_mesh_rgba:
                _save_arm_soft_mesh_rgba_frame(
                    env=env,
                    cfg=cfg,
                    out_path=out_path,
                    run_name=run_name,
                    episode_id=episode_id,
                    frame_idx=cnt,
                    dirname=arm_soft_mesh_rgba_dirname,
                )

            if save_depth:
                # Save depth maps
                depth_list = obs['depth_list']  # list of (H, W) or (1, H, W) tensors
                depth_list_wrist = obs['depth_wrist_list']

                index_side = 0
                index_wrist = 0
                for cam_id in range(len(cfg.env.cameras)):
                    camera = cfg.env.cameras[cam_id]
                    if camera['type'] == 'side':
                        depth = depth_list[index_side]
                        index_side += 1
                    elif camera['type'] == 'wrist':
                        depth = depth_list_wrist[index_wrist]
                        index_wrist += 1
                    else:
                        raise ValueError(f"Unknown camera type {camera['type']}")

                    # Convert to numpy and ensure 2D
                    if depth.ndim == 3:  # (1, H, W)
                        depth_np = depth[0].cpu().numpy()
                    else:  # (H, W)
                        depth_np = depth.cpu().numpy()

                    # Save as float32 NPY (preserves precision)
                    np.save(
                        f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/depth/{cnt:06d}.npy',
                        depth_np.astype(np.float32)
                    )

            # get observations
            pos = obs['robot']['eef_xyz']  # (n_grippers, 3)
            quat_wxyz = obs['robot']['eef_quat']  # (n_grippers, 4)
            gripper_qpos = 1.0 - obs['robot']['eef_gripper']  # (n_grippers, 1); in policy space, 1 is closed, 0 is open
            obs_rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat_wxyz)

            # get action
            assert n_grippers == pos.shape[0] and n_grippers == quat_wxyz.shape[0] and n_grippers == gripper_qpos.shape[0]
            assert n_grippers == cfg.env.robot.n_grippers

            use_runtime = runtime_cfg is not None and cnt >= robot_cutoff
            action_qpos = None
            if use_runtime:
                if runtime_state_xyz is None or runtime_state_rot is None:
                    runtime_state_xyz, runtime_state_rot = _init_runtime_state(obs)
                runtime_idx = cnt - robot_cutoff
                runtime_state_xyz, runtime_state_rot, runtime_gripper = _apply_runtime_perturbation_step(
                    runtime_state_xyz,
                    runtime_state_rot,
                    runtime_cfg,
                    runtime_idx,
                )
                eef_xyz = runtime_state_xyz.reshape(n_grippers, 3)
                eef_rot = runtime_state_rot.reshape(n_grippers, -1)
                eef_gripper = np.full(
                    (n_grippers, 1),
                    1.0 - float(runtime_gripper),
                    dtype=np.float32,
                )  # sim space
            else:
                eef_xyz = robot_traj_list[cnt].reshape(n_grippers, 3)
                eef_rot = robot_rot_list[cnt].reshape(n_grippers, -1)
                eef_gripper = robot_gripper_list[cnt].reshape(n_grippers, 1)  # sim space

            eef_xyz = torch.from_numpy(eef_xyz).to(torch.float32).to(env.unwrapped.physics.device)
            eef_rot = torch.from_numpy(eef_rot).to(torch.float32).to(env.unwrapped.physics.device).reshape(n_grippers, 3, 3)
            eef_gripper = torch.from_numpy(eef_gripper).to(torch.float32).to(env.unwrapped.physics.device)
            tcp_xyz = None
            tcp_quat = None
            if use_tcp:
                tcp_xyz = eef_xyz.clone()
                tcp_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)
                eef_xyz, eef_rot = tcp_to_eef(eef_xyz, eef_rot, tcp_offset, tcp_rot_offset)
            eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)

            eef_gripper = 1.0 - eef_gripper  # convert to policy space (0 open, 1 close)

            relax_eef_rot = cfg.get("relax_eef_rot", False)
            if relax_eef_rot:
                max_rot_deg = float(cfg.get("max_eef_rot_delta_deg", 10.0))
                max_rot_rad = np.deg2rad(max_rot_deg)
                obs_quat = quat_wxyz.detach().cpu().numpy()
                target_quat = eef_quat.detach().cpu().numpy()
                eef_rot_np = eef_rot.detach().cpu().numpy()
                for gi in range(n_grippers):
                    dot = float(np.dot(obs_quat[gi], target_quat[gi]))
                    dot = max(min(dot, 1.0), -1.0)
                    angle = 2.0 * np.arccos(abs(dot))
                    if angle > max_rot_rad:
                        t = max_rot_rad / angle
                        adj_quat = gt.quaternion_slerp(
                            obs_quat[gi], target_quat[gi], float(t), shortestpath=True
                        )
                        eef_rot_np[gi] = transforms3d.quaternions.quat2mat(adj_quat)
                eef_rot = torch.from_numpy(eef_rot_np).to(eef_rot.device, dtype=eef_rot.dtype)
                eef_quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(eef_rot)

            obs_qpos = _compute_qpos_from_eef(
                pos[0].cpu().numpy(),
                obs_rot[0].cpu().numpy(),
                env.renderer.qpos_curr_xarm,
            )
            if not use_runtime:
                action_qpos = robot_qpos_list[cnt]
            if action_qpos is None:
                action_qpos = _compute_qpos_from_eef(
                    eef_xyz[0].cpu().numpy(),
                    eef_rot[0].cpu().numpy(),
                    env.renderer.qpos_curr_xarm,
                ).reshape(1, -1)

            # save robot data
            robot_save = {
                "obs.ee_pos": pos[0].cpu().numpy().tolist(),  # [3] 
                "obs.ee_quat": quat_wxyz[0].cpu().numpy().tolist(),  # [4]
                "obs.gripper_qpos": float(gripper_qpos[0].cpu().numpy().reshape(-1)[0]),  # (0 open, 1 close)
                "obs.qpos": obs_qpos.reshape(-1).tolist(),
                "action.ee_pos": eef_xyz[0].cpu().numpy().tolist(),  # [3]
                "action.ee_quat": eef_quat[0].cpu().numpy().tolist(),  # [4]
                "action.gripper_qpos": eef_gripper[0].cpu().numpy().tolist(),  # [1] (0 open, 1 close)
                "action.qpos": action_qpos.reshape(-1).tolist(),
            }
            if use_tcp and tcp_xyz is not None and tcp_quat is not None:
                robot_save["action.tcp_pos"] = tcp_xyz[0].cpu().numpy().tolist()
                robot_save["action.tcp_quat"] = tcp_quat[0].cpu().numpy().tolist()

            with open(f'{out_path}/{run_name}/episode_{episode_id:04d}/robot/{cnt:06d}.json', 'w') as f:
                json.dump(robot_save, f, indent=4)

            if lerobot_exporter is not None:
                lerobot_exporter.add_frame(
                    image_list=image_list,
                    image_list_wrist=image_list_wrist,
                    obs_pos=pos,
                    obs_quat_wxyz=quat_wxyz,
                    obs_gripper_qpos=gripper_qpos,
                    action_pos=eef_xyz,
                    action_quat_wxyz=eef_quat,
                    action_gripper_qpos=eef_gripper,
                    step_idx=cnt,
                    n_steps=n_steps,
                )

            eef_gripper = 1.0 - eef_gripper  # convert to sim tradition (1 open, 0 close)

            action = torch.cat([
                eef_xyz,
                eef_rot.reshape(n_grippers, -1),
                eef_gripper
            ], dim=1)
            last_action = action
            _, _, done, truncated, _ = env.step({
                'action': action, 
                'do_velocity_control': cfg.env.robot.do_velocity_control
            })
            obs = env.unwrapped.get_obs()
            if clip_success_tracker is not None:
                _update_clip_success_tracker(clip_success_tracker, env, cfg)
            ts0 = base_time + cnt / frame_rate
            ts1 = ts0 - timestamp_offset_s
            timestamps.append((ts0, ts1))

            if cnt == n_steps - 1:
                image_list = obs['image_list']  # list of (3, H, W) tensors
                image_list_wrist = obs['image_wrist_list']  # list of (3, H, W) tensors

                index_side = 0
                index_wrist = 0
                for cam_id in range(len(cfg.env.cameras)):
                    camera = cfg.env.cameras[cam_id]
                    if camera['type'] == 'side':
                        image = image_list[index_side]
                        index_side += 1
                    elif camera['type'] == 'wrist':
                        image = image_list_wrist[index_wrist]
                        index_wrist += 1
                    else:
                        raise ValueError(f"Unknown camera type {camera['type']}")
                    
                    image = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb/{cnt + 1:06d}.jpg', image)
                    if save_start_final:
                        cv2.imwrite(
                            f'{out_path}/{run_name}/final_images/episode_{episode_id:04d}_camera_{cam_id}.jpg',
                            image,
                        )
                if save_arm_soft_mesh_rgba:
                    _save_arm_soft_mesh_rgba_frame(
                        env=env,
                        cfg=cfg,
                        out_path=out_path,
                        run_name=run_name,
                        episode_id=episode_id,
                        frame_idx=cnt + 1,
                        dirname=arm_soft_mesh_rgba_dirname,
                    )

                if save_depth:
                    # Save final frame depth
                    depth_list = obs['depth_list']
                    depth_list_wrist = obs['depth_wrist_list']

                    index_side = 0
                    index_wrist = 0
                    for cam_id in range(len(cfg.env.cameras)):
                        camera = cfg.env.cameras[cam_id]
                        if camera['type'] == 'side':
                            depth = depth_list[index_side]
                            index_side += 1
                        elif camera['type'] == 'wrist':
                            depth = depth_list_wrist[index_wrist]
                            index_wrist += 1
                        else:
                            raise ValueError(f"Unknown camera type {camera['type']}")

                        if depth.ndim == 3:
                            depth_np = depth[0].cpu().numpy()
                        else:
                            depth_np = depth.cpu().numpy()

                        np.save(
                            f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/depth/{cnt + 1:06d}.npy',
                            depth_np.astype(np.float32)
                        )

            torch.cuda.synchronize()
            tt1 = time.perf_counter()
            print(f"Episode: {episode_id}, step: {cnt - 1}, time: {tt1 - tt0:.4f}, fps: {1 / (tt1 - tt0):.2f}")

        if lerobot_exporter is not None:
            lerobot_exporter.save_episode()

        if save_final_state:
            _save_final_state(
                env,
                Path(f"{out_path}/{run_name}"),
                episode_id,
                n_steps,
                last_action,
                final_state_stabilization_steps,
                cfg.env.robot.do_velocity_control,
                final_state_dir_name,
            )

        if policy_rollout_format:
            _write_timestamps(
                Path(f'{out_path}/{run_name}/episode_{episode_id:04d}/timestamps.txt'),
                timestamps,
            )
        make_videos = cfg.get("make_videos", None)
        if make_videos is None:
            make_videos = not policy_rollout_format
        if make_videos:
            for cam_id in range(len(cfg.env.cameras)):
                make_video(
                    Path(f'{out_path}/{run_name}/episode_{episode_id:04d}/camera_{cam_id}/rgb'),
                    Path(f'{out_path}/{run_name}/episode_{episode_id:04d}_camera_{cam_id}.mp4'),
                    '%06d.jpg',
                    frame_rate=frame_rate,
                )
        if clip_success_eval_enabled:
            eval_dir = Path(f'{out_path}/{run_name}/episode_{episode_id:04d}')
            result = _evaluate_clip_success(
                env,
                cfg,
                eval_dir,
                episode_id,
                tracker=clip_success_tracker,
                n_steps=n_steps,
            )
            print("Clip success eval:", result)

        if cloth3_success_eval_enabled:
            eval_dir = Path(f'{out_path}/{run_name}/episode_{episode_id:04d}')
            cloth3_mode = str(cfg.get("cloth3_success_mode", "last_frame_triangle")).lower()
            if cloth3_mode != "last_frame_triangle":
                raise ValueError(f"Unknown cloth3_success_mode: {cloth3_mode}")
            points_world = _get_soft_points_world(env)
            cloth3_result = evaluate_cloth3_last_frame_triangle(points_world, cfg=cfg)
            cloth3_result["episode_id"] = int(episode_id)
            with open(eval_dir / f"cloth3_success_ep{episode_id:04d}.json", "w") as f:
                json.dump(cloth3_result, f, indent=2)
            if bool(cfg.get("cloth3_success_save_debug_image", True)):
                save_cloth3_triangle_debug_image(
                    points_world,
                    cloth3_result,
                    eval_dir / f"cloth3_success_ep{episode_id:04d}.png",
                )
            cloth3_success_list.append(int(bool(cloth3_result.get("success", False))))
            print("Cloth3 success eval:", cloth3_result)

    if cloth3_success_eval_enabled and len(cloth3_success_list) > 0:
        cloth3_success_arr = np.zeros((len(cloth3_success_list) + 2), dtype=int)
        cloth3_success_arr[:-2] = np.asarray(cloth3_success_list, dtype=int)
        cloth3_success_arr[-2] = int(cloth3_success_arr[:-2].sum())
        cloth3_success_arr[-1] = int(np.round(cloth3_success_arr[:-2].mean() * 100.0))
        np.savetxt(Path(out_path) / run_name / "success.txt", cloth3_success_arr, fmt="%d")


if __name__ == '__main__':
    main()
