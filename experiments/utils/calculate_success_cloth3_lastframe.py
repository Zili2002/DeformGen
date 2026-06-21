import argparse
import json
import pickle as pkl
from pathlib import Path
import sys
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from experiments.utils.calculate_success_cloth3 import (
    evaluate_cloth3_last_frame_triangle,
    save_cloth3_triangle_debug_image,
)


def find_episode_dirs(root: Path) -> List[Path]:
    """Return sorted episode directories under a replay/eval output root."""
    return sorted([p for p in root.glob("episode_*") if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Replay/eval output directory containing episode_* subdirectories.",
    )
    parser.add_argument(
        "--save-debug-image",
        action="store_true",
        help="Write cloth3 triangle debug png next to each metric json.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    episode_dirs = find_episode_dirs(data_dir)
    if not episode_dirs:
        raise SystemExit(f"No episodes under: {data_dir}")

    success_rows: List[int] = []
    for episode_dir in episode_dirs:
        ep_id = int(episode_dir.name.split("_")[-1])
        state_files = sorted((episode_dir / "state").glob("*.pkl"))
        print(f"Episode: {episode_dir}, Number of state files: {len(state_files)}")

        if not state_files:
            result = {
                "episode_id": ep_id,
                "mode": "last_frame_triangle",
                "success": False,
                "reason": "missing_state",
            }
        else:
            with state_files[-1].open("rb") as f:
                state = pkl.load(f)

            points_world = state.get("renderer", {}).get("x", None)
            if points_world is None:
                result = {
                    "episode_id": ep_id,
                    "mode": "last_frame_triangle",
                    "success": False,
                    "reason": "missing_renderer_x",
                }
            else:
                result = evaluate_cloth3_last_frame_triangle(points_world, cfg=None)
                result["episode_id"] = ep_id
                result["last_state_file"] = state_files[-1].name
                if args.save_debug_image:
                    save_cloth3_triangle_debug_image(
                        points_world,
                        result,
                        episode_dir / f"cloth3_success_ep{ep_id:04d}.png",
                    )

        with (episode_dir / f"cloth3_success_ep{ep_id:04d}.json").open("w") as f:
            json.dump(result, f, indent=2)

        success_rows.append(int(bool(result.get("success", False))))

    success = np.zeros((len(success_rows) + 2,), dtype=int)
    success[:-2] = np.asarray(success_rows, dtype=int)
    success[-2] = int(np.sum(success[:-2]))
    success[-1] = int(np.round(float(np.mean(success[:-2])) * 100.0))
    np.savetxt(data_dir / "success.txt", success, fmt="%d")

    print("cloth3 last-frame success list:", success_rows)
    print(f"cloth3 last-frame success rate: {success[-2]} / {len(success_rows)} = {success[-1]:.1f}%")


if __name__ == "__main__":
    main()
