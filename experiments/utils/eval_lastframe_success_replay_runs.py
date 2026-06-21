#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle as pkl
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.utils.calculate_success_rope_lastframe import is_rope_success_lastframe
from experiments.utils.calculate_success_sloth_lastframe import evaluate_sloth_last_frame_packed
from experiments.utils.calculate_success_cloth3 import evaluate_cloth3_last_frame_triangle


def iter_runs(patterns: Iterable[str]) -> list[Path]:
    runs: list[Path] = []
    for pat in patterns:
        runs.extend(sorted(Path().glob(pat)))
    uniq: list[Path] = []
    seen = set()
    for p in runs:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def load_state(path: Path):
    with path.open("rb") as f:
        return pkl.load(f)


def eval_one(case: str, state: dict, sloth_min_points: int, sloth_obb_scale: float) -> dict:
    if case == "rope":
        ok = bool(is_rope_success_lastframe(state, state))
        return {
            "mode": "last_frame_rope_intersections",
            "success": ok,
            "plane_min_bottom_threshold": 100,
            "plane_min_top_threshold": 100,
        }

    if case == "sloth":
        return evaluate_sloth_last_frame_packed(
            state=state,
            state_init=state,
            success_min_points=sloth_min_points,
            obb_scale=sloth_obb_scale,
        )

    if case == "cloth3":
        points_world = state.get("renderer", {}).get("x", None)
        if points_world is None:
            return {"mode": "last_frame_triangle", "success": False, "reason": "missing_renderer_x"}
        return evaluate_cloth3_last_frame_triangle(points_world, cfg=None)

    raise ValueError(f"unsupported case: {case}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    ap.add_argument(
        "--runs-glob",
        action="append",
        required=True,
        help="Glob for replay run directories, e.g. log/experiments/output_replay/myrun_*",
    )
    ap.add_argument("--episode-id", type=int, default=0)
    ap.add_argument("--final-state-dir-name", type=str, default="final_state")
    ap.add_argument("--summary-csv", type=Path, required=True)
    ap.add_argument("--sloth-success-min-points", type=int, default=3050)
    ap.add_argument("--sloth-obb-scale", type=float, default=1.05)
    args = ap.parse_args()

    runs = iter_runs(args.runs_glob)
    rows: list[list[object]] = []
    succ = 0

    json_name = {
        "rope": f"rope_success_ep{args.episode_id:04d}.json",
        "sloth": f"sloth_success_ep{args.episode_id:04d}.json",
        "cloth3": f"cloth3_success_ep{args.episode_id:04d}.json",
    }[args.case]

    for run_dir in runs:
        ep_dir = run_dir / f"episode_{args.episode_id:04d}"
        st_path = ep_dir / args.final_state_dir_name / "state.pkl"
        key = run_dir.name

        if not st_path.exists():
            rows.append([key, str(run_dir), 0, "missing_state"])
            continue

        try:
            st = load_state(st_path)
            result = eval_one(args.case, st, args.sloth_success_min_points, args.sloth_obb_scale)
            ok = int(bool(result.get("success", False)))
            succ += ok
            result["episode_id"] = args.episode_id
            result["run_dir"] = str(run_dir)
            with (ep_dir / json_name).open("w") as f:
                json.dump(result, f, indent=2)
            rows.append([key, str(run_dir), ok, ""])
        except Exception as e:  # noqa: BLE001
            rows.append([key, str(run_dir), 0, f"error:{e}"])

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "run_dir", "success", "note"])
        w.writerows(rows)

    total = len(rows)
    rate = (succ / total * 100.0) if total else 0.0
    success_arr = np.zeros((total + 2,), dtype=int)
    success_arr[:-2] = np.asarray([int(r[2]) for r in rows], dtype=int)
    success_arr[-2] = int(success_arr[:-2].sum())
    success_arr[-1] = int(round(success_arr[:-2].mean() * 100.0)) if total else 0
    np.savetxt(args.summary_csv.with_suffix(".success.txt"), success_arr, fmt="%d")

    print(f"case={args.case} total={total} success={succ} rate={rate:.2f}%")
    print(f"summary_csv={args.summary_csv}")


if __name__ == "__main__":
    main()
