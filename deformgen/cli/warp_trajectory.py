from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Sequence

_CASE_DEFAULTS = {
    "rope": ("rope", "log/phystwin/rope", "rope_0001"),
    "sloth": ("sloth", "log/phystwin/sloth", "sloth_0001"),
    "cloth3": ("cloth3", "log/phystwin/cloth3", "cloth3_0001"),
}

@dataclass(frozen=True)
class WarpItem:
    index: int
    state_path: Path | None
    output_dir: Path


def _quote(value: str) -> str:
    return "'" + value.replace("'", "\\'") + "'"


def _read_list(path: Path) -> list[str]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


def _resolve(entry: str, list_path: Path) -> Path:
    p = Path(entry)
    return p if p.is_absolute() else list_path.parent / p


def build_items(args: argparse.Namespace) -> list[WarpItem]:
    if args.state_path is not None and args.state_list is not None:
        raise ValueError("Use only one of --state-path or --state-list.")
    if args.state_list is not None:
        list_path = Path(args.state_list)
        states = [_resolve(x, list_path) for x in _read_list(list_path)]
        items = []
        for i, state in enumerate(states):
            if not state.exists():
                raise FileNotFoundError(f"state list item {i} does not exist: {state}")
            items.append(WarpItem(i, state, Path(args.out) / f"item{i:06d}"))
        return items
    state = Path(args.state_path) if args.state_path is not None else None
    if state is not None and not state.exists():
        raise FileNotFoundError(f"state path does not exist: {state}")
    return [WarpItem(0, state, Path(args.out))]


def build_command(args: argparse.Namespace, item: WarpItem) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    backend = repo_root / "experiments" / "create_interpolated_json_trajectory.py"
    if not backend.exists():
        raise FileNotFoundError(f"Cannot find trajectory backend: {backend}")
    demo = Path(args.demo)
    if not demo.exists() or not demo.is_dir():
        raise FileNotFoundError(f"Demo trajectory directory does not exist: {demo}")
    gs, ckpt_path, case_name = _CASE_DEFAULTS[args.case]
    overrides = [
        f"gs={gs}",
        f"physics.ckpt_path={_quote(ckpt_path)}",
        f"physics.case_name={_quote(case_name)}",
        f"gt_dir={_quote(str(demo))}",
        f"episode_id={int(args.episode_id)}",
        f"+output_dir={_quote(str(item.output_dir))}",
        f"num_interp_steps={int(args.num_interp_steps)}",
        f"num_approach_steps={int(args.num_approach_steps)}",
        f"num_grasp_steps={int(args.num_grasp_steps)}",
        f"+num_rotate_steps={int(args.num_rotate_steps)}",
    ]
    if item.state_path is not None:
        overrides.append(f"+deformation_path={_quote(str(item.state_path))}")
    if args.mode == "yawonly":
        overrides += [
            "+use_deformation_warping=true",
            "+grasp_yaw_only=true",
            "+disable_z_warp=true",
            f"+k_neighbors={int(args.grasp_local_k)}",
            f"+manip_k_neighbors={int(args.manip_local_k)}",
            f"+warp_decay={args.manip_decay}",
            f"+use_grasp_local_warp={'true' if args.use_grasp_local_warp else 'false'}",
            f"+grasp_local_k={int(args.grasp_local_k)}",
        ]
    elif args.mode in {"txy", "rigidlocal_txyfit"}:
        overrides += [
            "+use_deformation_warping=true",
            "+use_local_rigid_baseline=true",
            "+grasp_yaw_only=true",
            "+disable_z_warp=true",
            f"+local_rigid_k={int(args.grasp_local_k)}",
            f"+local_rigid_decay_mode={args.manip_decay}",
            f"+k_neighbors={int(args.grasp_local_k)}",
            f"+manip_k_neighbors={int(args.manip_local_k)}",
        ]
    else:
        raise ValueError(f"Unsupported warp mode: {args.mode}")
    if args.start_idx is not None:
        overrides.append(f"start_idx={int(args.start_idx)}")
    overrides.extend(args.override)
    return [sys.executable, str(backend), *overrides]


def _write_manifest(path: Path, row: dict, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_one(args: argparse.Namespace, item: WarpItem, manifest: Path, lock: Lock) -> int:
    cmd = build_command(args, item)
    started = time.time()
    if args.dry_run:
        print("DRY-RUN", item.index, " ".join(cmd))
        rc = 0
    else:
        print(f"[warp {item.index}] start output={item.output_dir}", flush=True)
        rc = subprocess.call(cmd, cwd=Path(__file__).resolve().parents[2])
        print(f"[warp {item.index}] done rc={rc}", flush=True)
    ep_dir = item.output_dir / f"episode_{int(args.episode_id):04d}"
    row = {
        "index": item.index,
        "state_path": str(item.state_path) if item.state_path else None,
        "output_dir": str(item.output_dir),
        "trajectory_dir": str(ep_dir),
        "return_code": rc,
        "elapsed_s": time.time() - started,
        "command": cmd,
    }
    _write_manifest(manifest, row, lock)
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warp demonstration trajectories to perturbed states.")
    parser.add_argument("--case", choices=sorted(_CASE_DEFAULTS), required=True)
    parser.add_argument("--demo", required=True, help="Demo replay trajectory directory.")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--state-path", default=None, help="Single target state path.")
    parser.add_argument("--state-list", default=None, help="Text file containing target state paths.")
    parser.add_argument("--mode", default="yawonly", choices=["yawonly", "txy", "rigidlocal_txyfit"])
    parser.add_argument("--grasp-local-k", type=int, default=5)
    parser.add_argument("--manip-local-k", type=int, default=5)
    parser.add_argument("--manip-decay", default="none", choices=["none", "linear", "exponential"])
    parser.add_argument("--use-grasp-local-warp", action="store_true")
    parser.add_argument("--num-interp-steps", type=int, default=118)
    parser.add_argument("--num-approach-steps", type=int, default=30)
    parser.add_argument("--num-grasp-steps", type=int, default=100)
    parser.add_argument("--num-rotate-steps", type=int, default=30)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--override", action="append", default=[], help="Additional raw Hydra override. May be repeated.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        items = build_items(args)
        manifest = Path(args.manifest_out) if args.manifest_out else Path(args.out) / "warp_manifest.jsonl"
        if manifest.exists():
            manifest.unlink()
        lock = Lock()
        workers = max(1, int(args.num_workers))
        if workers == 1:
            failures = [(i.index, _run_one(args, i, manifest, lock)) for i in items]
        else:
            failures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_run_one, args, i, manifest, lock): i for i in items}
                for fut in concurrent.futures.as_completed(futs):
                    item = futs[fut]
                    failures.append((item.index, fut.result()))
        bad = [(idx, rc) for idx, rc in failures if rc != 0]
        if bad:
            print(f"Warp failures: {bad}", file=sys.stderr)
            return bad[0][1]
        traj_file = Path(args.out) / "trajectories.txt"
        traj_file.parent.mkdir(parents=True, exist_ok=True)
        traj_file.write_text("\n".join(str(i.output_dir) for i in items) + "\n", encoding="utf-8")
        print(f"trajectory_list={traj_file}")
        return 0
    except Exception as exc:
        print(f"deformgen-warp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
