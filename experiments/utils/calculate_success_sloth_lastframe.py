import argparse
import json
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import open3d as o3d


def find_episode_dirs(root: Path) -> List[Path]:
    """Return sorted episode directories under a replay/eval output root."""
    return sorted([p for p in root.glob("episode_*") if p.is_dir()])


def _to_numpy_xyz(points_world: Any) -> np.ndarray:
    """Convert renderer points to a numpy (N, 3) array."""
    if hasattr(points_world, "detach"):
        points_world = points_world.detach().cpu().numpy()
    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_world must have shape (N, 3), got {points.shape}")
    return points


def _build_reference_obb(state_init: Dict[str, Any], obb_scale: float) -> o3d.geometry.OrientedBoundingBox:
    """Build the minimal OBB from the undeformed reference mesh, then scale it."""
    meshes = state_init["physics"]["static_meshes"]
    if len(meshes) != 1:
        raise ValueError(f"Expected exactly 1 static mesh, got {len(meshes)}")

    vertices = np.asarray(meshes[0]["vertices"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Reference mesh vertices must be (N, 3), got {vertices.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    obb = pcd.get_minimal_oriented_bounding_box(robust=True)
    obb = obb.scale(float(obb_scale), obb.get_center())
    return obb


def evaluate_sloth_last_frame_packed(
    state: Dict[str, Any],
    state_init: Dict[str, Any],
    success_min_points: int = 3050,
    obb_scale: float = 1.05,
) -> Dict[str, Any]:
    """Evaluate sloth success using the replay-aligned last-frame packed rule."""
    points_world = _to_numpy_xyz(state["renderer"]["x"])
    obb = _build_reference_obb(state_init, obb_scale=obb_scale)
    idx = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points_world))
    points_in_obb = int(len(idx))
    success = bool(points_in_obb >= int(success_min_points))

    return {
        "mode": "last_frame_packed",
        "success": success,
        "points_in_obb": points_in_obb,
        "success_min_points": int(success_min_points),
        "obb_scale": float(obb_scale),
    }


def _load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pkl.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Replay/eval output directory containing episode_* subdirectories.",
    )
    parser.add_argument(
        "--success-min-points",
        type=int,
        default=3050,
        help="Success threshold for points inside packed OBB.",
    )
    parser.add_argument(
        "--obb-scale",
        type=float,
        default=1.05,
        help="Uniform scale applied to the reference minimal OBB.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    episode_dirs = find_episode_dirs(data_dir)
    if not episode_dirs:
        raise SystemExit(f"No episodes under: {data_dir}")

    success_rows: List[int] = []
    for episode_dir in episode_dirs:
        ep_id = int(episode_dir.name.split("_")[-1])
        state_dir = episode_dir / "state"
        state_files = sorted(state_dir.glob("*.pkl"))
        print(f"Episode: {episode_dir}, Number of state files: {len(state_files)}")

        if not state_files:
            result = {
                "episode_id": ep_id,
                "mode": "last_frame_packed",
                "success": False,
                "reason": "missing_state",
                "success_min_points": int(args.success_min_points),
                "obb_scale": float(args.obb_scale),
            }
        else:
            init_state_path = state_dir / "000000.pkl"
            if not init_state_path.exists():
                result = {
                    "episode_id": ep_id,
                    "mode": "last_frame_packed",
                    "success": False,
                    "reason": "missing_init_state",
                    "success_min_points": int(args.success_min_points),
                    "obb_scale": float(args.obb_scale),
                }
            else:
                state_init = _load_pickle(init_state_path)
                state_last = _load_pickle(state_files[-1])
                result = evaluate_sloth_last_frame_packed(
                    state=state_last,
                    state_init=state_init,
                    success_min_points=args.success_min_points,
                    obb_scale=args.obb_scale,
                )
                result["episode_id"] = ep_id
                result["last_state_file"] = state_files[-1].name

        with (episode_dir / f"sloth_success_ep{ep_id:04d}.json").open("w") as f:
            json.dump(result, f, indent=2)

        success_rows.append(int(bool(result.get("success", False))))

    success = np.zeros((len(success_rows) + 2,), dtype=int)
    success[:-2] = np.asarray(success_rows, dtype=int)
    success[-2] = int(np.sum(success[:-2]))
    success[-1] = int(np.round(float(np.mean(success[:-2])) * 100.0))
    np.savetxt(data_dir / "success.txt", success, fmt="%d")

    print("pack_sloth last-frame success list:", success_rows)
    print(f"pack_sloth last-frame success rate: {success[-2]} / {len(success_rows)} = {success[-1]:.1f}%")


if __name__ == "__main__":
    main()
