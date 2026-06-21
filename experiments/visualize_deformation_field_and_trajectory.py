#!/usr/bin/env python3
"""
Visualize deformation field and trajectory changes after warping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json
import sys

import gymnasium as gym
import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

import sim.envs
from sim.planning.utils.deformation_field_warping import load_deformation_for_warping
from sim.planning.utils.trajectory_loader import load_replay_trajectory


def find_grasp_frame(gripper: torch.Tensor, closing_threshold: float = 0.01) -> int:
    """Find grasp frame using positive gripper-closing velocity."""
    values = gripper[:, 0, 0].detach().cpu().numpy()
    velocity = np.diff(values)
    closing_frames = np.where(velocity > closing_threshold)[0]
    if len(closing_frames) == 0:
        return 0

    sequences = []
    current_seq = [int(closing_frames[0])]
    for i in range(1, len(closing_frames)):
        if closing_frames[i] == closing_frames[i - 1] + 1:
            current_seq.append(int(closing_frames[i]))
        else:
            sequences.append(current_seq)
            current_seq = [int(closing_frames[i])]
    sequences.append(current_seq)
    longest_seq = max(sequences, key=len)
    return int(longest_seq[-1] + 1)


def resolve_episode_dir(path: Path, episode_id: int) -> Path:
    """Resolve either <root>/episode_xxxx or direct episode directory."""
    if (path / "robot").exists():
        return path

    candidate = path / f"episode_{episode_id:04d}"
    if (candidate / "robot").exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find episode directory under {path} for episode_id={episode_id}."
    )


def load_trajectory_from_episode_dir(episode_dir: Path) -> Dict[str, torch.Tensor]:
    """Load trajectory from resolved episode directory."""
    episode_name = episode_dir.name
    if not episode_name.startswith("episode_"):
        raise ValueError(
            f"Episode directory must be named episode_xxxx, got: {episode_name}"
        )

    episode_id = int(episode_name.split("_")[1])
    return load_replay_trajectory(episode_dir.parent, episode_id)


def sample_indices(n_points: int, max_points: int) -> np.ndarray:
    """Uniformly sample indices for plotting."""
    n = int(max(1, min(n_points, max_points)))
    if n == n_points:
        return np.arange(n_points, dtype=np.int64)
    return np.linspace(0, n_points - 1, num=n, dtype=np.int64)


def resample_curve(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Resample a 3D curve by normalized arclength."""
    if points.shape[0] <= 1:
        return np.repeat(points[:1], max(1, n_samples), axis=0)

    n = int(max(2, n_samples))
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return np.repeat(points[:1], n, axis=0)
    cum = cum / total

    query = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3), dtype=np.float64)
    for dim in range(3):
        out[:, dim] = np.interp(query, cum, points[:, dim])
    return out


def set_axes_equal_3d(ax: plt.Axes, points: np.ndarray, pad_ratio: float = 0.1) -> None:
    """Set equal scales for 3D axes."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float((maxs - mins).max()), 1e-6)
    radius = span * (0.5 + pad_ratio)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def compute_trajectory_length(points: np.ndarray) -> float:
    """Compute polyline length."""
    if points.shape[0] <= 1:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


@hydra.main(version_base="1.2", config_path="../cfg", config_name="deformation_warping")
def main(cfg) -> None:
    """Visualize deformation field and trajectory change on top of warped results."""
    OmegaConf.resolve(cfg)

    episode_id = int(cfg.get("episode_id", 1))
    deformation_path_raw = cfg.get("deformation_path", None)
    original_gt_dir_raw = cfg.get("original_gt_dir", None)
    warped_gt_dir_raw = cfg.get("warped_gt_dir", cfg.get("gt_dir", None))

    if deformation_path_raw is None:
        raise ValueError("Missing deformation_path.")
    if original_gt_dir_raw is None:
        raise ValueError("Missing original_gt_dir.")
    if warped_gt_dir_raw is None:
        raise ValueError("Missing warped_gt_dir or gt_dir.")

    deformation_path = Path(str(deformation_path_raw))
    original_gt_dir = Path(str(original_gt_dir_raw))
    warped_gt_dir = Path(str(warped_gt_dir_raw))

    original_episode_dir = resolve_episode_dir(original_gt_dir, episode_id)
    warped_episode_dir = resolve_episode_dir(warped_gt_dir, episode_id)

    output_dir = Path(
        cfg.get(
            "deformation_vis_output_dir",
            "log/experiments/visualizations/deformation_field",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    max_points = int(cfg.get("deformation_vis_max_points", 4000))
    max_vectors = int(cfg.get("deformation_vis_max_vectors", 350))
    trajectory_compare_samples = int(cfg.get("trajectory_compare_samples", 36))
    plot_stride = int(max(1, cfg.get("trajectory_plot_stride", 1)))
    output_tag = cfg.get("deformation_vis_tag", None)

    print("Loading trajectories...")
    print(f"  Original episode: {original_episode_dir}")
    print(f"  Warped episode:   {warped_episode_dir}")
    original_traj = load_trajectory_from_episode_dir(original_episode_dir)
    warped_traj = load_trajectory_from_episode_dir(warped_episode_dir)

    original_xyz = original_traj["eef_xyz"][:, 0, :].detach().cpu().numpy()
    warped_xyz = warped_traj["eef_xyz"][:, 0, :].detach().cpu().numpy()
    original_grasp_idx = find_grasp_frame(original_traj["eef_gripper"])
    warped_grasp_idx = find_grasp_frame(warped_traj["eef_gripper"])

    original_xyz_plot = original_xyz[::plot_stride]
    warped_xyz_plot = warped_xyz[::plot_stride]
    original_cmp = resample_curve(original_xyz, trajectory_compare_samples)
    warped_cmp = resample_curve(warped_xyz, trajectory_compare_samples)

    print("Loading deformation field point clouds...")
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
    p_orig_t, p_def_t = load_deformation_for_warping(str(deformation_path), env)
    env.close()

    p_orig = p_orig_t.detach().cpu().numpy()
    p_def = p_def_t.detach().cpu().numpy()
    delta = p_def - p_orig
    delta_norm = np.linalg.norm(delta, axis=1)

    point_idx = sample_indices(p_orig.shape[0], max_points)
    vector_idx = sample_indices(p_orig.shape[0], max_vectors)
    p_orig_plot = p_orig[point_idx]
    p_def_plot = p_def[point_idx]
    p_vec = p_orig[vector_idx]
    d_vec = delta[vector_idx]

    fig = plt.figure(figsize=(19, 6))
    ax_obj = fig.add_subplot(131, projection="3d")
    ax_traj = fig.add_subplot(132, projection="3d")
    ax_mix = fig.add_subplot(133, projection="3d")

    ax_obj.scatter(
        p_orig_plot[:, 0],
        p_orig_plot[:, 1],
        p_orig_plot[:, 2],
        c="#4c72b0",
        s=4,
        alpha=0.28,
        label="object (before)",
    )
    ax_obj.scatter(
        p_def_plot[:, 0],
        p_def_plot[:, 1],
        p_def_plot[:, 2],
        c="#dd8452",
        s=4,
        alpha=0.28,
        label="object (after)",
    )
    ax_obj.quiver(
        p_vec[:, 0],
        p_vec[:, 1],
        p_vec[:, 2],
        d_vec[:, 0],
        d_vec[:, 1],
        d_vec[:, 2],
        color="#2b8cbe",
        linewidth=0.7,
        alpha=0.70,
        arrow_length_ratio=0.12,
    )
    ax_obj.set_title("Deformation Field on Object")
    ax_obj.set_xlabel("X")
    ax_obj.set_ylabel("Y")
    ax_obj.set_zlabel("Z")
    ax_obj.legend(loc="upper left", fontsize=8)

    ax_traj.plot(
        original_xyz_plot[:, 0],
        original_xyz_plot[:, 1],
        original_xyz_plot[:, 2],
        color="#4c72b0",
        linewidth=2.0,
        label="trajectory (before)",
    )
    ax_traj.plot(
        warped_xyz_plot[:, 0],
        warped_xyz_plot[:, 1],
        warped_xyz_plot[:, 2],
        color="#dd8452",
        linewidth=2.0,
        label="trajectory (after warp)",
    )
    cmp_delta = warped_cmp - original_cmp
    ax_traj.quiver(
        original_cmp[:, 0],
        original_cmp[:, 1],
        original_cmp[:, 2],
        cmp_delta[:, 0],
        cmp_delta[:, 1],
        cmp_delta[:, 2],
        color="#777777",
        linewidth=0.8,
        alpha=0.8,
        arrow_length_ratio=0.16,
    )
    ax_traj.scatter(
        original_xyz[0, 0],
        original_xyz[0, 1],
        original_xyz[0, 2],
        c="#4c72b0",
        marker="o",
        s=45,
        label="start (before)",
    )
    ax_traj.scatter(
        warped_xyz[0, 0],
        warped_xyz[0, 1],
        warped_xyz[0, 2],
        c="#dd8452",
        marker="o",
        s=45,
        label="start (after)",
    )
    ax_traj.scatter(
        original_xyz[min(original_grasp_idx, original_xyz.shape[0] - 1), 0],
        original_xyz[min(original_grasp_idx, original_xyz.shape[0] - 1), 1],
        original_xyz[min(original_grasp_idx, original_xyz.shape[0] - 1), 2],
        c="#1f77b4",
        marker="*",
        s=120,
        label="grasp (before)",
    )
    ax_traj.scatter(
        warped_xyz[min(warped_grasp_idx, warped_xyz.shape[0] - 1), 0],
        warped_xyz[min(warped_grasp_idx, warped_xyz.shape[0] - 1), 1],
        warped_xyz[min(warped_grasp_idx, warped_xyz.shape[0] - 1), 2],
        c="#c44e52",
        marker="*",
        s=120,
        label="grasp (after)",
    )
    ax_traj.set_title("Trajectory Change (Before vs After)")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_zlabel("Z")
    ax_traj.legend(loc="upper left", fontsize=8)

    ax_mix.scatter(
        p_orig_plot[:, 0],
        p_orig_plot[:, 1],
        p_orig_plot[:, 2],
        c="#4c72b0",
        s=2,
        alpha=0.12,
        label="object before",
    )
    ax_mix.scatter(
        p_def_plot[:, 0],
        p_def_plot[:, 1],
        p_def_plot[:, 2],
        c="#dd8452",
        s=2,
        alpha=0.12,
        label="object after",
    )
    ax_mix.plot(
        original_xyz_plot[:, 0],
        original_xyz_plot[:, 1],
        original_xyz_plot[:, 2],
        color="#1f77b4",
        linewidth=2.2,
        label="traj before",
    )
    ax_mix.plot(
        warped_xyz_plot[:, 0],
        warped_xyz_plot[:, 1],
        warped_xyz_plot[:, 2],
        color="#c44e52",
        linewidth=2.2,
        label="traj after",
    )
    ax_mix.set_title("Object Morphology + Trajectory Overlay")
    ax_mix.set_xlabel("X")
    ax_mix.set_ylabel("Y")
    ax_mix.set_zlabel("Z")
    ax_mix.legend(loc="upper left", fontsize=8)

    all_points = np.concatenate(
        [p_orig_plot, p_def_plot, original_xyz_plot, warped_xyz_plot], axis=0
    )
    set_axes_equal_3d(ax_obj, all_points)
    set_axes_equal_3d(ax_traj, all_points)
    set_axes_equal_3d(ax_mix, all_points)
    ax_obj.view_init(elev=20.0, azim=35.0)
    ax_traj.view_init(elev=20.0, azim=35.0)
    ax_mix.view_init(elev=20.0, azim=35.0)

    traj_shift = np.linalg.norm(cmp_delta, axis=1)
    metrics = {
        "episode_id": int(episode_id),
        "object_points_total": int(p_orig.shape[0]),
        "deformation_norm_mean": float(delta_norm.mean()),
        "deformation_norm_median": float(np.median(delta_norm)),
        "deformation_norm_p95": float(np.percentile(delta_norm, 95)),
        "deformation_norm_max": float(delta_norm.max()),
        "trajectory_before_frames": int(original_xyz.shape[0]),
        "trajectory_after_frames": int(warped_xyz.shape[0]),
        "trajectory_before_length": float(compute_trajectory_length(original_xyz)),
        "trajectory_after_length": float(compute_trajectory_length(warped_xyz)),
        "trajectory_shift_mean": float(traj_shift.mean()),
        "trajectory_shift_p95": float(np.percentile(traj_shift, 95)),
        "trajectory_shift_max": float(traj_shift.max()),
        "grasp_frame_before": int(original_grasp_idx),
        "grasp_frame_after": int(warped_grasp_idx),
        "deformation_path": str(deformation_path),
        "original_episode_dir": str(original_episode_dir),
        "warped_episode_dir": str(warped_episode_dir),
    }

    if output_tag is None:
        output_tag = (
            f"ep{episode_id:04d}_{deformation_path.stem}_"
            f"{warped_episode_dir.parent.name}_{warped_episode_dir.name}"
        )
    output_tag = str(output_tag).replace("/", "_")
    png_path = output_dir / f"{output_tag}_deformation_trajectory.png"
    json_path = output_dir / f"{output_tag}_deformation_trajectory_metrics.json"

    fig.suptitle(
        "Deformation-Field Visualization (object morphology + trajectory change)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved figure:  {png_path}")
    print(f"Saved metrics: {json_path}")
    print(
        "Summary: "
        f"deformation_mean={metrics['deformation_norm_mean']:.6f}, "
        f"traj_shift_mean={metrics['trajectory_shift_mean']:.6f}"
    )


if __name__ == "__main__":
    main()
