from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import shutil
import subprocess
import sys

import numpy as np
import torch
import kornia
from omegaconf import OmegaConf
import hydra

sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

from experiments.utils.dir_utils import mkdir
from sim.planning.utils.grasp_pose_transfer import detect_grasp_moment
from sim.planning.utils.trajectory_loader import load_replay_trajectory
from sim.utils.planar_perturbation_utils import (
    generate_batch_perturbations,
)


def _trajectory_to_waypoints(
    demo_trajectory: Dict[str, torch.Tensor],
    end_idx: int,
) -> List[Dict[str, Any]]:
    waypoints: List[Dict[str, Any]] = []
    for t in range(end_idx + 1):
        eef_xyz = demo_trajectory["eef_xyz"][t, 0].detach().cpu().numpy()
        eef_rot = demo_trajectory["eef_rot"][t, 0].detach().cpu().numpy()
        eef_gripper = float(demo_trajectory["eef_gripper"][t, 0, 0].detach().cpu().numpy())
        waypoints.append({
            "xyz": eef_xyz,
            "rot": eef_rot,
            "gripper": eef_gripper,
        })
    return waypoints


def _waypoints_to_replay_dir(waypoints: List[Dict[str, Any]], output_dir: Path) -> None:
    robot_dir = output_dir / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)

    for frame_id, waypoint in enumerate(waypoints):
        rot = torch.from_numpy(waypoint["rot"]).to(torch.float32).unsqueeze(0)
        quat = kornia.geometry.conversions.rotation_matrix_to_quaternion(rot)[0].cpu().numpy()
        robot_data = {
            "action.ee_pos": waypoint["xyz"].tolist(),
            "action.ee_quat": quat.tolist(),
            "action.gripper_qpos": [float(waypoint["gripper"])],
        }
        with open(robot_dir / f"{frame_id:06d}.json", "w") as f:
            json.dump(robot_data, f, indent=2)


def _load_executed_trajectory(robot_dir: Path) -> Dict[str, torch.Tensor]:
    robot_paths = sorted(robot_dir.glob("*.json"))
    if not robot_paths:
        raise ValueError(f"No robot json files found in {robot_dir}.")

    xyz_list: List[np.ndarray] = []
    rot_list: List[np.ndarray] = []
    gripper_list: List[float] = []
    for path in robot_paths:
        with open(path, "r") as f:
            robot = json.load(f)
        xyz = np.array(robot["action.ee_pos"], dtype=np.float32)
        quat = np.array(robot["action.ee_quat"], dtype=np.float32).reshape(1, 4)
        rot = kornia.geometry.conversions.quaternion_to_rotation_matrix(
            torch.from_numpy(quat).to(torch.float32)
        )[0].cpu().numpy()
        gripper_raw = robot["action.gripper_qpos"]
        if isinstance(gripper_raw, list):
            gripper_val = float(gripper_raw[0]) if gripper_raw else 0.0
        else:
            gripper_val = float(gripper_raw)
        xyz_list.append(xyz)
        rot_list.append(rot)
        gripper_list.append(gripper_val)

    eef_xyz = torch.from_numpy(np.stack(xyz_list)).float().unsqueeze(1)
    eef_rot = torch.from_numpy(np.stack(rot_list)).float().unsqueeze(1)
    eef_gripper = torch.tensor([[g] for g in gripper_list]).float().unsqueeze(1)
    return {
        "eef_xyz": eef_xyz,
        "eef_rot": eef_rot,
        "eef_gripper": eef_gripper,
    }


def _run_replay(
    cfg,
    gt_dir: Path,
    output_root: Path,
    run_name: str,
    runtime_perturbation: Optional[Dict[str, Any]] = None,
    replay_init: Optional[Dict[str, Any]] = None,
) -> None:
    cmd = [
        sys.executable,
        "experiments/replay.py",
        f"gs={cfg.gs_config}",
        f"env={cfg.env_config}",
        f"physics.ckpt_path={cfg.physics.ckpt_path}",
        f"physics.case_name={cfg.physics.case_name}",
        f"gt_dir={str(gt_dir)}",
        "use_qpos=false",
        f"randomize={'true' if bool(cfg.get('replay_randomize', False)) else 'false'}",
        "save_final_state=true",
        f"final_state_stabilization_steps={int(cfg.output.final_state_stabilization_steps)}",
        f"timestamp={run_name}",
        f"output_root={str(output_root)}",
        "overwrite_output=true",
    ]
    collide_fric_override = cfg.physics.get("collide_fric_override", None)
    if collide_fric_override is not None:
        cmd.append(f"physics.collide_fric_override={float(collide_fric_override)}")

    reset_seed = cfg.get("reset_seed", None)
    if reset_seed is not None:
        cmd.append(f"+reset_seed={int(reset_seed)}")

    # Optional: force replay reset from a fixed deformed soft-body state.
    # This allows perturbation synthesis to start from an explicit world-frame
    # state instead of the simulator default reset state.
    fixed_init_state = cfg.get("initial_deformed_state_path", None)
    if fixed_init_state is not None and str(fixed_init_state).strip():
        cmd.append(f"+deformed_state_path={str(fixed_init_state)}")
    if runtime_perturbation is not None:
        cmd += [
            "runtime_perturbation.enabled=true",
            f"runtime_perturbation.mode={runtime_perturbation['mode']}",
            f"runtime_perturbation.start_step={int(runtime_perturbation['start_step'])}",
            f"runtime_perturbation.gripper_state={float(runtime_perturbation['gripper_state'])}",
            f"runtime_perturbation.release.enabled={str(runtime_perturbation['release']['enabled']).lower()}",
            f"runtime_perturbation.release.num_waypoints={int(runtime_perturbation['release']['num_waypoints'])}",
            f"runtime_perturbation.release.gripper_open={float(runtime_perturbation['release']['gripper_open'])}",
        ]
        if runtime_perturbation["mode"] == "fixed":
            translation_xy = runtime_perturbation["translation_xy"]
            cmd += [
                f"runtime_perturbation.num_waypoints={int(runtime_perturbation['num_waypoints'])}",
                f"runtime_perturbation.translation_xy=[{float(translation_xy[0])},{float(translation_xy[1])}]",
                f"runtime_perturbation.rotation_z={float(runtime_perturbation['rotation_z'])}",
            ]
        else:
            step_sizes = runtime_perturbation["translation_step_sizes"]
            step_sizes_str = ",".join(str(float(x)) for x in step_sizes)
            cmd += [
                f"runtime_perturbation.random_steps={int(runtime_perturbation['random_steps'])}",
                f"runtime_perturbation.translation_step_sizes=[{step_sizes_str}]",
                f"runtime_perturbation.rotation_step_deg={float(runtime_perturbation['rotation_step_deg'])}",
                f"runtime_perturbation.rotation_prob={float(runtime_perturbation['rotation_prob'])}",
                f"runtime_perturbation.seed={int(runtime_perturbation['seed'])}",
            ]
    if replay_init is not None and replay_init.get("enabled", False):
        cmd += [
            "init_from_trajectory=true",
            f"init_frame_idx={int(replay_init['frame_idx'])}",
            f"init_stabilization_steps={int(replay_init['stabilization_steps'])}",
        ]
        if replay_init.get("start_idx", None) is not None:
            cmd.append(f"start_idx={int(replay_init['start_idx'])}")
    print("Running replay:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).parents[1]))


def _copy_sample_outputs(
    sample_dir: Path,
    replay_dir: Path,
    cfg,
    metadata: Dict[str, Any],
    trajectory_tensor: Dict[str, torch.Tensor],
) -> None:
    if cfg.output.save_soft_body_state:
        src = replay_dir / "final_state" / "state.npy"
        if src.exists():
            shutil.copyfile(src, sample_dir / "soft_body_state.npy")
        else:
            print(f"[WARNING] Missing final state npy: {src}")

    if cfg.output.save_robot_state:
        robot_dir = replay_dir / "robot"
        robot_files = sorted(robot_dir.glob("*.json"))
        if robot_files:
            last_robot = robot_files[-1]
            with open(last_robot, "r") as f:
                robot_json = json.load(f)
            with open(sample_dir / "robot_state.json", "w") as f:
                json.dump(robot_json, f, indent=2)
            with open(sample_dir / "robot_state.pkl", "wb") as f:
                import pickle as pkl

                pkl.dump(robot_json, f)
        else:
            print(f"[WARNING] No robot json files in {robot_dir}")

    if cfg.output.save_rgb_images:
        for cam_id in [0, 1]:
            cam_dir = replay_dir / f"camera_{cam_id}" / "rgb"
            if not cam_dir.exists():
                continue
            frames = sorted(cam_dir.glob("*.jpg"))
            if frames:
                shutil.copyfile(frames[-1], sample_dir / f"camera_{cam_id}.jpg")

    if cfg.output.save_trajectory:
        torch.save(trajectory_tensor, sample_dir / "trajectory.pt")

    if cfg.output.save_metadata:
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)


@hydra.main(version_base="1.2", config_path="../cfg", config_name="augmentation")
def main(cfg) -> None:
    OmegaConf.resolve(cfg)

    if cfg.physics.ckpt_path is None or cfg.physics.case_name is None:
        raise ValueError("physics.ckpt_path and physics.case_name must be set.")

    output_root = Path(cfg.output.root_dir) / f"demo_episode_{int(cfg.demo.episode_id):04d}"
    mkdir(output_root, resume=False, overwrite=bool(cfg.output.overwrite))

    replay_output_root = Path(cfg.output.replay_output_root) if cfg.output.replay_output_root else output_root
    replay_output_root.mkdir(parents=True, exist_ok=True)

    input_root = output_root / "replay_inputs"
    input_root.mkdir(parents=True, exist_ok=True)

    demo_trajectory = load_replay_trajectory(Path(cfg.demo.gt_dir), int(cfg.demo.episode_id))
    gripper_values = demo_trajectory["eef_gripper"][:, 0, 0]
    t_grasp = detect_grasp_moment(gripper_values)
    print(f"Detected grasp at timestep: {t_grasp}")

    replay_init_enabled = bool(cfg.replay_init.enabled)
    init_frame_idx = None
    if replay_init_enabled:
        frame_mode = str(cfg.replay_init.frame).lower()
        if frame_mode == "grasp":
            init_frame_idx = int(t_grasp)
        elif frame_mode == "start":
            init_frame_idx = 0
        elif frame_mode == "index":
            if cfg.replay_init.index is None:
                raise ValueError("replay_init.index must be set when frame='index'.")
            init_frame_idx = int(cfg.replay_init.index)
        else:
            raise ValueError(f"Unknown replay_init.frame: {cfg.replay_init.frame}")

    demo_end_idx = int(t_grasp)
    if replay_init_enabled and init_frame_idx is not None:
        demo_end_idx = max(demo_end_idx, int(init_frame_idx))
    demo_waypoints = _trajectory_to_waypoints(demo_trajectory, demo_end_idx)
    grasp_xyz = demo_trajectory["eef_xyz"][t_grasp, 0].detach().cpu().numpy()
    grasp_rot = demo_trajectory["eef_rot"][t_grasp, 0].detach().cpu().numpy()

    mode = str(cfg.perturbation.get("mode", "fixed")).lower()
    if mode == "fixed":
        perturbations = generate_batch_perturbations(
            grasp_pose={"xyz": grasp_xyz, "rot": grasp_rot},
            config=cfg.perturbation,
            num_samples=int(cfg.batch.num_samples),
            seed=cfg.batch.seed,
        )
    elif mode == "teleop_random":
        perturbations = [{"seed": int(cfg.batch.seed) + idx} for idx in range(int(cfg.batch.num_samples))]
    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")
    release_enabled = bool(cfg.release.enabled)

    summary = {
        "demo_episode": int(cfg.demo.episode_id),
        "grasp_timestep": int(t_grasp),
        "num_samples": int(cfg.batch.num_samples),
        "mode": mode,
        "replay_init": {
            "enabled": replay_init_enabled,
            "frame": str(cfg.replay_init.frame),
            "index": int(cfg.replay_init.index) if cfg.replay_init.index is not None else None,
            "frame_idx": int(init_frame_idx) if init_frame_idx is not None else None,
            "stabilization_steps": int(cfg.replay_init.stabilization_steps),
            "run_demo_tail": bool(cfg.replay_init.get("run_demo_tail", False)),
        },
        "release": {
            "enabled": release_enabled,
            "num_waypoints": int(cfg.release.num_waypoints),
            "gripper_open": float(cfg.release.gripper_open),
        },
        "samples": [],
    }

    for idx, perturb in enumerate(perturbations):
        sample_name = f"sample_{idx:04d}"
        sample_dir = output_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        input_dir = input_root / sample_name / "episode_0000"
        _waypoints_to_replay_dir(demo_waypoints, input_dir)

        run_demo_tail = bool(cfg.replay_init.get("run_demo_tail", False))
        if replay_init_enabled and init_frame_idx is not None:
            # Enforce flow: init frame -> run demo to grasp(close gripper) -> runtime perturbation -> release.
            runtime_start_step = max(0, len(demo_waypoints) - int(init_frame_idx))
        elif replay_init_enabled:
            runtime_start_step = 0
        else:
            runtime_start_step = len(demo_waypoints)
        runtime_perturbation = {
            "start_step": runtime_start_step,
            "mode": mode,
            "gripper_state": float(cfg.perturbation.gripper_state),
            "release": {
                "enabled": release_enabled,
                "num_waypoints": int(cfg.release.num_waypoints),
                "gripper_open": float(cfg.release.gripper_open),
            },
        }
        if mode == "fixed":
            runtime_perturbation.update({
                "num_waypoints": int(cfg.perturbation.num_waypoints),
                "translation_xy": perturb["translation_xy"],
                "rotation_z": float(perturb["rotation_z"]),
            })
        else:
            runtime_perturbation.update({
                "random_steps": int(cfg.perturbation.random_steps),
                "translation_step_sizes": list(cfg.perturbation.translation_step_sizes),
                "rotation_step_deg": float(cfg.perturbation.rotation_step_deg),
                "rotation_prob": float(cfg.perturbation.rotation_prob),
                "seed": int(perturb["seed"]),
            })
        replay_init = None
        if replay_init_enabled and init_frame_idx is not None:
            replay_init = {
                "enabled": True,
                "frame_idx": int(init_frame_idx),
                "stabilization_steps": int(cfg.replay_init.stabilization_steps),
                "start_idx": int(init_frame_idx),
            }
        _run_replay(
            cfg,
            input_root / sample_name,
            replay_output_root,
            sample_name,
            runtime_perturbation=runtime_perturbation,
            replay_init=replay_init,
        )

        replay_dir = replay_output_root / sample_name / "episode_0000"
        trajectory_tensor = _load_executed_trajectory(replay_dir / "robot")

        metadata = {
            "demo_episode": int(cfg.demo.episode_id),
            "grasp_timestep": int(t_grasp),
            "perturbation_id": idx,
            "mode": mode,
            "runtime_perturbation": True,
            "runtime_start_step": int(runtime_start_step),
            "replay_init": {
                "enabled": replay_init_enabled,
                "frame": str(cfg.replay_init.frame),
                "index": int(cfg.replay_init.index) if cfg.replay_init.index is not None else None,
                "frame_idx": int(init_frame_idx) if init_frame_idx is not None else None,
                "stabilization_steps": int(cfg.replay_init.stabilization_steps),
                "run_demo_tail": bool(cfg.replay_init.get("run_demo_tail", False)),
            },
            "translation_xy": perturb["translation_xy"].tolist() if "translation_xy" in perturb else None,
            "rotation_z": float(perturb["rotation_z"]) if "rotation_z" in perturb else None,
            "seed": int(perturb["seed"]) if "seed" in perturb else None,
            "release": {
                "enabled": release_enabled,
                "num_waypoints": int(cfg.release.num_waypoints),
                "gripper_open": float(cfg.release.gripper_open),
            },
            "config": {
                "translation_range_xy": [float(x) for x in cfg.perturbation.translation_range_xy],
                "rotation_range_z": [float(x) for x in cfg.perturbation.rotation_range_z],
                "num_waypoints": int(cfg.perturbation.num_waypoints),
                "min_translation_norm": float(cfg.perturbation.get("min_translation_norm", 0.0)),
                "min_rotation_abs": float(cfg.perturbation.get("min_rotation_abs", 0.0)),
                "random_steps": int(cfg.perturbation.get("random_steps", 0)),
                "translation_step_sizes": [float(x) for x in cfg.perturbation.translation_step_sizes],
                "rotation_step_deg": float(cfg.perturbation.rotation_step_deg),
                "rotation_prob": float(cfg.perturbation.rotation_prob),
            },
        }

        _copy_sample_outputs(
            sample_dir,
            replay_dir,
            cfg,
            metadata,
            trajectory_tensor,
        )

        summary["samples"].append({
            "sample": sample_name,
            "output_dir": str(sample_dir),
        })
        print(f"Completed sample {sample_name} ({idx + 1}/{len(perturbations)})")

    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
