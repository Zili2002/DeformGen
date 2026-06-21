#!/usr/bin/env python3
"""
Visualize local rigid baseline principal axes before/after deformation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import gymnasium as gym
import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.append(str(Path(__file__).parents[1]))
OmegaConf.register_new_resolver("eval", eval, replace=True)

import sim.envs
from sim.planning.utils.deformation_field_warping import load_deformation_for_warping
from sim.planning.utils.rigid_transformation import compute_optimal_rigid_transform
from sim.planning.utils.trajectory_loader import load_replay_trajectory


def find_grasp_frame(gripper: torch.Tensor, closing_threshold: float = 0.01) -> int:
    """Find grasp frame using the same heuristic as interpolation script."""
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


def build_world_z_rotation(yaw: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build world-frame Z-axis rotation matrix."""
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid, principal axes and axis lengths from a point cloud."""
    centroid = points.mean(axis=0)
    centered = points - centroid[None, :]
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    proj = centered @ eigvecs
    lengths = np.percentile(np.abs(proj), 90.0, axis=0)
    return centroid, eigvecs, lengths


def align_axis_signs(axes: np.ndarray, ref_axes: np.ndarray) -> np.ndarray:
    """Align principal axis directions to reference axes for visual consistency."""
    aligned = axes.copy()
    for i in range(3):
        if float(np.dot(aligned[:, i], ref_axes[:, i])) < 0.0:
            aligned[:, i] *= -1.0
    return aligned


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_val = float(np.dot(v1, v2) / (n1 * n2))
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.rad2deg(np.arccos(cos_val)))


def draw_principal_axes(
    ax: plt.Axes,
    centroid: np.ndarray,
    axes: np.ndarray,
    lengths: np.ndarray,
    axis_scale: float,
    min_axis_len: float,
    linestyle: str,
    linewidth: float,
    alpha: float,
) -> None:
    """Draw three principal axes as line segments."""
    axis_colors = ["#d62728", "#2ca02c", "#1f77b4"]  # x, y, z
    for i in range(3):
        seg_len = max(float(lengths[i]) * axis_scale, min_axis_len)
        direction = axes[:, i] * seg_len
        p0 = centroid - direction
        p1 = centroid + direction
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=axis_colors[i],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )


def save_visualization(
    p_orig_local: np.ndarray,
    p_def_local: np.ndarray,
    p_rigid_local: np.ndarray,
    out_path: Path,
    title: str,
    axis_scale: float,
    min_axis_len: float,
) -> Dict[str, float]:
    """Save local cloud and principal-axis visualization."""
    c_orig, a_orig, l_orig = pca_axes(p_orig_local)
    c_def, a_def_raw, l_def = pca_axes(p_def_local)
    c_rigid, a_rigid_raw, l_rigid = pca_axes(p_rigid_local)
    a_def = align_axis_signs(a_def_raw, a_orig)
    a_rigid = align_axis_signs(a_rigid_raw, a_orig)

    major_angle_orig_def = angle_deg(a_orig[:, 0], a_def[:, 0])
    major_angle_orig_rigid = angle_deg(a_orig[:, 0], a_rigid[:, 0])
    major_angle_def_rigid = angle_deg(a_def[:, 0], a_rigid[:, 0])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        p_orig_local[:, 0], p_orig_local[:, 1], p_orig_local[:, 2],
        c="#4c72b0", s=22, alpha=0.8, label="orig local points",
    )
    ax.scatter(
        p_def_local[:, 0], p_def_local[:, 1], p_def_local[:, 2],
        c="#dd8452", s=22, alpha=0.7, label="deformed local points",
    )
    ax.scatter(
        p_rigid_local[:, 0], p_rigid_local[:, 1], p_rigid_local[:, 2],
        c="#55a868", s=22, alpha=0.65, label="baseline rigid(xy+yaw) mapped",
    )

    draw_principal_axes(
        ax, c_orig, a_orig, l_orig, axis_scale, min_axis_len,
        linestyle="-", linewidth=2.0, alpha=1.0,
    )
    draw_principal_axes(
        ax, c_def, a_def, l_def, axis_scale, min_axis_len,
        linestyle="--", linewidth=1.8, alpha=0.95,
    )
    draw_principal_axes(
        ax, c_rigid, a_rigid, l_rigid, axis_scale, min_axis_len,
        linestyle="-.", linewidth=1.8, alpha=0.95,
    )

    vec_def = c_def - c_orig
    vec_rigid = c_rigid - c_orig

    all_pts = np.concatenate([p_orig_local, p_def_local, p_rigid_local], axis=0)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) * 0.5
    span = max(float((maxs - mins).max()), 1e-3)
    radius = span * 0.6
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")
    ax.view_init(elev=22.0, azim=42.0)
    ax.set_title(title)

    style_legend = [
        Line2D([0], [0], color="black", lw=2, linestyle="-", label="PCA axes (orig)"),
        Line2D([0], [0], color="black", lw=2, linestyle="--", label="PCA axes (deformed)"),
        Line2D([0], [0], color="black", lw=2, linestyle="-.", label="PCA axes (rigid xy+yaw)"),
    ]
    ax.legend(loc="upper left", fontsize=8)
    fig.legend(handles=style_legend, loc="lower center", ncol=3, frameon=False, fontsize=9)

    info_text = (
        f"major axis angle: orig->def={major_angle_orig_def:.2f} deg\n"
        f"major axis angle: orig->rigid={major_angle_orig_rigid:.2f} deg\n"
        f"major axis angle: def->rigid={major_angle_def_rigid:.2f} deg"
    )
    fig.text(0.02, 0.02, info_text, fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 1.0])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {
        "major_axis_angle_orig_to_def_deg": major_angle_orig_def,
        "major_axis_angle_orig_to_rigid_deg": major_angle_orig_rigid,
        "major_axis_angle_def_to_rigid_deg": major_angle_def_rigid,
        "centroid_shift_orig_to_def_norm": float(np.linalg.norm(vec_def)),
        "centroid_shift_orig_to_rigid_norm": float(np.linalg.norm(vec_rigid)),
        "centroid_shift_def_to_rigid_xy_norm": float(np.linalg.norm((c_def - c_rigid)[:2])),
        "centroid_shift_def_to_rigid_z_abs": float(abs(c_def[2] - c_rigid[2])),
    }


@hydra.main(version_base="1.2", config_path="../cfg", config_name="deformation_warping")
def main(cfg) -> None:
    """Visualize local rigid principal-axis change around grasp region."""
    OmegaConf.resolve(cfg)

    deformation_path = cfg.get("deformation_path", None)
    if deformation_path is None:
        raise ValueError("Missing deformation_path. Use deformation_path=... to set.")

    demo_dir = Path(cfg.gt_dir)
    episode_id = int(cfg.get("episode_id", 1))
    start_idx: Optional[int] = cfg.get("start_idx", None)
    local_k = int(cfg.get("local_rigid_k", 5))
    if local_k < 3:
        raise ValueError("local_rigid_k must be >= 3 for PCA.")
    axis_scale = float(cfg.get("principal_axes_axis_scale", 3.0))
    min_axis_len = float(cfg.get("principal_axes_min_len", 0.02))

    output_dir = Path(
        cfg.get(
            "principal_axes_vis_dir",
            "log/experiments/visualizations/local_rigid_principal_axes",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_replay_trajectory(demo_dir, episode_id)
    if start_idx is None:
        start_idx = find_grasp_frame(trajectory["eef_gripper"])
    grasp_xyz = trajectory["eef_xyz"][start_idx, 0]

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
    env.close()

    grasp_xyz = grasp_xyz.to(p_orig.device)
    dists = torch.norm(p_orig - grasp_xyz[None, :], dim=1)
    k_local = min(local_k, int(p_orig.shape[0]))
    knn_idx = torch.topk(dists, k_local, largest=False).indices
    p_orig_local = p_orig[knn_idx]
    p_def_local = p_def[knn_idx]

    r_local, t_local = compute_optimal_rigid_transform(p_orig_local, p_def_local)
    yaw = float(torch.atan2(r_local[1, 0], r_local[0, 0]).item())
    r_yaw = build_world_z_rotation(yaw, p_orig.device, p_orig.dtype)
    centroid_orig = p_orig_local.mean(dim=0)
    centroid_def = p_def_local.mean(dim=0)
    t_xy = (centroid_def - (centroid_orig @ r_yaw.T)).clone()
    t_xy[2] = 0.0
    p_rigid_local = p_orig_local @ r_yaw.T + t_xy

    p_orig_np = p_orig_local.detach().cpu().numpy()
    p_def_np = p_def_local.detach().cpu().numpy()
    p_rigid_np = p_rigid_local.detach().cpu().numpy()

    stem = Path(str(deformation_path)).stem
    out_png = output_dir / f"episode_{episode_id:04d}_{stem}_k{k_local}_principal_axes.png"
    metrics = save_visualization(
        p_orig_np,
        p_def_np,
        p_rigid_np,
        out_png,
        title=f"Local rigid principal axes (k={k_local}, yaw={np.rad2deg(yaw):.2f} deg)",
        axis_scale=axis_scale,
        min_axis_len=min_axis_len,
    )

    print(f"Grasp frame: {start_idx}")
    print(f"K local points: {k_local}")
    print(f"principal_axes_axis_scale: {axis_scale}")
    print(f"principal_axes_min_len: {min_axis_len}")
    print(f"Estimated local yaw (deg): {np.rad2deg(yaw):.4f}")
    print(f"t_local (world): {t_local.detach().cpu().numpy()}")
    print(f"t_xy (world): {t_xy.detach().cpu().numpy()}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
