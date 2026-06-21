import argparse
from pathlib import Path
import pickle as pkl

import numpy as np

CENTER = np.array([0.62, 0.05, 0.0], dtype=float)
BBOX_MIN = CENTER.copy()
BBOX_MAX = CENTER.copy()
BBOX_MIN[0] -= 0.035 / 2
BBOX_MAX[0] += 0.035 / 2
BBOX_MIN[1] -= 0.035 / 2
BBOX_MAX[1] += 0.035 / 2
BBOX_MIN[2] -= 0.0
BBOX_MAX[2] += 0.03


def _segment_plane_intersections_xz(p0, p1, y_plane, x_min, x_max, z_min, z_max, eps=1e-12):
    y0 = p0[:, 1]
    y1 = p1[:, 1]
    dy = y1 - y0

    parallel = np.isclose(dy, 0.0, atol=eps)
    t = np.zeros_like(dy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t[~parallel] = (y_plane - y0[~parallel]) / dy[~parallel]
    on_segment = (~parallel) & (t >= -eps) & (t <= 1.0 + eps)

    xi = p0[:, 0] + t * (p1[:, 0] - p0[:, 0])
    zi = p0[:, 2] + t * (p1[:, 2] - p0[:, 2])
    inside_rect = (xi >= x_min - eps) & (xi <= x_max + eps) & (zi >= z_min - eps) & (zi <= z_max + eps)
    hits_crossing = on_segment & inside_rect

    coplanar = parallel & np.isclose(y0 - y_plane, 0.0, atol=eps)
    end0_in = (p0[:, 0] >= x_min - eps) & (p0[:, 0] <= x_max + eps) & (p0[:, 2] >= z_min - eps) & (p0[:, 2] <= z_max + eps)
    end1_in = (p1[:, 0] >= x_min - eps) & (p1[:, 0] <= x_max + eps) & (p1[:, 2] >= z_min - eps) & (p1[:, 2] <= z_max + eps)
    hits_coplanar = coplanar & (end0_in | end1_in)

    return hits_crossing | hits_coplanar


def _count_intersections(vertices, springs):
    x_min, y_min, z_min = BBOX_MIN.tolist()
    x_max, y_max, z_max = BBOX_MAX.tolist()

    p0 = vertices[springs[:, 0]]
    p1 = vertices[springs[:, 1]]

    hits_min = _segment_plane_intersections_xz(p0, p1, y_min, x_min, x_max, z_min, z_max)
    hits_max = _segment_plane_intersections_xz(p0, p1, y_max, x_min, x_max, z_min, z_max)

    return int(np.count_nonzero(hits_min)), int(np.count_nonzero(hits_max))


def is_rope_success_lastframe(state_last, state_init):
    springs = state_init["physics"]["init_springs"].cpu().numpy()
    vertices = state_last["renderer"]["x"].cpu().numpy()
    y_min_count, y_max_count = _count_intersections(vertices, springs)
    return (y_min_count >= 100) and (y_max_count >= 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    episode_dirs = sorted([p for p in data_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise SystemExit("No episodes under: {}".format(data_dir))

    success_list = []
    for episode_dir in episode_dirs:
        state_files = sorted(episode_dir.glob("state/*.pkl"))
        if not state_files:
            success_list.append(False)
            continue

        init_file = episode_dir / "state/000000.pkl"
        if not init_file.exists():
            init_file = state_files[0]

        with open(init_file, "rb") as f:
            state_init = pkl.load(f)
        with open(state_files[-1], "rb") as f:
            state_last = pkl.load(f)

        success_list.append(is_rope_success_lastframe(state_last, state_init))

    success = np.zeros((len(episode_dirs) + 2), dtype=int)
    success[:-2] = np.array(success_list, dtype=int)
    success[-2] = success[:-2].sum()
    success[-1] = int(round(success[:-2].mean() * 100)) if len(success[:-2]) else 0

    np.savetxt(data_dir / "success.txt", success, fmt="%d")
    print("insert_rope(lastframe) success rate: {} / {} = {}%".format(success[-2], len(episode_dirs), success[-1]))


if __name__ == "__main__":
    main()
