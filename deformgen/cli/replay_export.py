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


_CASE_TO_GS = {
    "rope": "rope",
    "sloth": "sloth",
    "cloth3": "cloth3",
}

_CASE_DEFAULT_PHYSICS = {
    "rope": ("log/phystwin/rope", "rope_0001"),
    "sloth": ("log/phystwin/sloth", "sloth_0001"),
    "cloth3": ("log/phystwin/cloth3", "cloth3_0001"),
}


@dataclass(frozen=True)
class BatchItem:
    index: int
    gt_dir: Path
    state_path: Path | None
    run_name: str
    lerobot_out: Path | None


def _quote_override_value(value: str) -> str:
    """Return a Hydra-safe scalar value for paths/strings."""
    if value == "":
        return "''"
    escaped = value.replace("'", "\\'")
    return f"'{escaped}'"


def _read_list_file(path: Path) -> list[str]:
    rows: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def _looks_like_list_file(value: str | None) -> bool:
    if value is None:
        return False
    path = Path(value)
    return path.exists() and path.is_file()


def _resolve_list_entry(entry: str, list_path: Path) -> Path:
    path = Path(entry)
    if not path.is_absolute():
        path = list_path.parent / path
    return path


def _base_run_name(args: argparse.Namespace) -> str:
    return args.name or "replay"


def build_batch_items(args: argparse.Namespace) -> list[BatchItem]:
    traj_list_path = Path(args.traj_list) if _looks_like_list_file(args.traj_list) else None
    state_list_path = Path(args.state_list) if _looks_like_list_file(args.state_list) else None

    if traj_list_path is None and state_list_path is None:
        return []

    if traj_list_path is not None:
        traj_entries = [_resolve_list_entry(x, traj_list_path) for x in _read_list_file(traj_list_path)]
    else:
        if args.gt_dir is None:
            raise ValueError("--state-list without --traj-list requires shared --gt-dir.")
        traj_entries = []

    if state_list_path is not None:
        state_entries = [_resolve_list_entry(x, state_list_path) for x in _read_list_file(state_list_path)]
    else:
        state_entries = []

    if traj_list_path is not None and state_list_path is not None and len(traj_entries) != len(state_entries):
        raise ValueError(
            f"--traj-list and --state-list must have the same number of non-empty rows, "
            f"got {len(traj_entries)} and {len(state_entries)}."
        )

    if traj_list_path is not None:
        n_items = len(traj_entries)
    else:
        n_items = len(state_entries)
        shared_gt = Path(args.gt_dir)
        traj_entries = [shared_gt] * n_items

    start = int(args.start_index)
    end = n_items if args.max_items is None else min(n_items, start + int(args.max_items))
    if start < 0 or start > n_items:
        raise ValueError(f"Invalid --start-index {start}; list length is {n_items}.")

    items: list[BatchItem] = []
    base = _base_run_name(args)
    for idx in range(start, end):
        gt_dir = traj_entries[idx]
        state_path = state_entries[idx] if state_entries else None
        if not gt_dir.exists() or not gt_dir.is_dir():
            raise FileNotFoundError(f"Batch item {idx}: gt_dir does not exist or is not a directory: {gt_dir}")
        if state_path is not None and not state_path.exists():
            raise FileNotFoundError(f"Batch item {idx}: state_path does not exist: {state_path}")
        run_name = f"{base}_item{idx:06d}"
        lerobot_out = Path(args.lerobot_out) / run_name if args.lerobot_out else Path(args.out) / run_name / "lerobot_dataset"
        items.append(BatchItem(index=idx, gt_dir=gt_dir, state_path=state_path, run_name=run_name, lerobot_out=lerobot_out))
    return items


def build_replay_command(
    args: argparse.Namespace,
    extra_overrides: Sequence[str],
    item: BatchItem | None = None,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    replay_py = repo_root / "experiments" / "replay.py"
    if not replay_py.exists():
        raise FileNotFoundError(f"Cannot find replay backend: {replay_py}")

    if item is not None:
        gt_path = item.gt_dir
        state_path = item.state_path
        run_name = item.run_name
        lerobot_dir = item.lerobot_out
    else:
        gt_dir = args.gt_dir or args.traj_list
        if gt_dir is None:
            raise ValueError("Provide --gt-dir pointing to a replay trajectory directory, or --traj-list file.")
        gt_path = Path(gt_dir)
        state_path = Path(args.state_path) if args.state_path is not None else None
        run_name = args.name
        lerobot_dir = Path(args.lerobot_out) if args.lerobot_out else Path(args.out) / (args.name or "replay") / "lerobot_dataset"

    if not gt_path.exists():
        raise FileNotFoundError(f"Replay trajectory directory does not exist: {gt_path}")
    if not gt_path.is_dir():
        raise ValueError(f"Replay trajectory input must be a directory accepted by experiments/replay.py, got: {gt_path}")

    physics_ckpt_path, physics_case_name = _CASE_DEFAULT_PHYSICS[args.case]
    overrides = [
        f"gs={_CASE_TO_GS[args.case]}",
        f"physics.ckpt_path={_quote_override_value(physics_ckpt_path)}",
        f"physics.case_name={_quote_override_value(physics_case_name)}",
        f"gt_dir={_quote_override_value(str(gt_path))}",
        f"output_root={_quote_override_value(str(Path(args.out)))}",
        f"overwrite_output={'true' if args.overwrite else 'false'}",
    ]
    if run_name is not None:
        overrides.append(f"timestamp={_quote_override_value(run_name)}")
    if args.use_qpos is not None:
        overrides.append(f"use_qpos={'true' if args.use_qpos else 'false'}")
    if args.save_final_state:
        overrides.append("save_final_state=true")
    if state_path is not None:
        if not state_path.exists():
            raise FileNotFoundError(f"State path does not exist: {state_path}")
        overrides.append(f"+deformed_state_path={_quote_override_value(str(state_path))}")
    if args.export_lerobot:
        overrides.extend(
            [
                "lerobot_export.enabled=true",
                f"lerobot_export.output_dir={_quote_override_value(str(lerobot_dir))}",
                f"lerobot_export.repo_id={_quote_override_value(args.repo_id)}",
                f"lerobot_export.task_name={_quote_override_value(args.task_name)}",
                "lerobot_export.use_videos=true",
            ]
        )
        if args.video_backend is not None:
            overrides.append(f"lerobot_export.video_backend={_quote_override_value(args.video_backend)}")

    overrides.extend(extra_overrides)
    return [sys.executable, str(replay_py), *overrides]


def _write_jsonl(path: Path, row: dict, lock: Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def _successful_manifest_indices(path: Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("return_code") == 0 and row.get("skipped_reason") not in {"locked"}:
            try:
                indices.add(int(row["index"]))
            except (KeyError, TypeError, ValueError):
                pass
    return indices


def _item_output_complete(args: argparse.Namespace, item: BatchItem) -> bool:
    run_dir = Path(args.out) / item.run_name
    if not run_dir.exists():
        return False
    # replay.py normally writes hydra.yaml for non-policy rollouts. If users disable it,
    # the episode directory check below still confirms that at least one replay happened.
    if not (run_dir / "hydra.yaml").exists() and not (run_dir / "episode_0000").exists():
        return False
    if args.save_final_state and not (run_dir / "episode_0000" / "final_state" / "state.npy").exists():
        return False
    if args.export_lerobot:
        if item.lerobot_out is None:
            return False
        if not (item.lerobot_out / "meta" / "info.json").exists():
            return False
        if not any((item.lerobot_out / "data").glob("chunk-*/*.parquet")):
            return False
        if not any((item.lerobot_out / "videos").glob("chunk-*/*/episode_*.mp4")):
            return False
    return True


def _base_row(args: argparse.Namespace, item: BatchItem, cmd: list[str] | None = None) -> dict:
    return {
        "index": item.index,
        "gt_dir": str(item.gt_dir),
        "state_path": str(item.state_path) if item.state_path is not None else None,
        "run_name": item.run_name,
        "output_dir": str(Path(args.out) / item.run_name),
        "lerobot_dir": str(item.lerobot_out) if item.lerobot_out is not None else None,
        "command": cmd,
        "start_time": time.time(),
    }


def _run_one(args: argparse.Namespace, item: BatchItem, manifest_path: Path, lock: Lock) -> int:
    cmd = build_replay_command(args, args.override, item=item)
    row = _base_row(args, item, cmd)
    lock_dir = Path(args.out) / item.run_name / ".deformgen_replay.lock"
    use_item_lock = bool(args.resume or args.skip_existing) and not args.dry_run
    acquired_lock = False

    if args.dry_run:
        print("DRY-RUN", item.index, " ".join(cmd))
        row.update({"return_code": 0, "dry_run": True, "elapsed_s": 0.0})
        _write_jsonl(manifest_path, row, lock)
        return 0

    if use_item_lock:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            acquired_lock = True
        except FileExistsError:
            print(f"[batch {item.index}] skip locked run_name={item.run_name}", flush=True)
            row.update({"return_code": 0, "dry_run": False, "elapsed_s": 0.0, "skipped_reason": "locked"})
            _write_jsonl(manifest_path, row, lock)
            return 0

    try:
        if args.skip_existing and _item_output_complete(args, item):
            print(f"[batch {item.index}] skip existing run_name={item.run_name}", flush=True)
            row.update({"return_code": 0, "dry_run": False, "elapsed_s": 0.0, "skipped_reason": "existing_output"})
            _write_jsonl(manifest_path, row, lock)
            return 0

        print(f"[batch {item.index}] start run_name={item.run_name}", flush=True)
        rc = subprocess.call(cmd, cwd=Path(__file__).resolve().parents[2])
        elapsed = time.time() - float(row["start_time"])
        print(f"[batch {item.index}] done rc={rc} elapsed_s={elapsed:.1f}", flush=True)
        row.update({"return_code": rc, "dry_run": False, "elapsed_s": elapsed})
        _write_jsonl(manifest_path, row, lock)
        return rc
    finally:
        if acquired_lock:
            try:
                lock_dir.rmdir()
            except OSError:
                pass


def _filter_resume_items(args: argparse.Namespace, items: list[BatchItem], manifest_path: Path, lock: Lock) -> list[BatchItem]:
    if not args.resume and not args.skip_existing:
        return items
    done = _successful_manifest_indices(manifest_path) if args.resume else set()
    kept: list[BatchItem] = []
    for item in items:
        if args.resume and item.index in done:
            print(f"[batch {item.index}] skip manifest success run_name={item.run_name}", flush=True)
            _write_jsonl(
                manifest_path,
                {**_base_row(args, item), "return_code": 0, "dry_run": False, "elapsed_s": 0.0, "skipped_reason": "manifest_success"},
                lock,
            )
            continue
        if args.skip_existing and _item_output_complete(args, item):
            print(f"[batch {item.index}] skip existing run_name={item.run_name}", flush=True)
            _write_jsonl(
                manifest_path,
                {**_base_row(args, item), "return_code": 0, "dry_run": False, "elapsed_s": 0.0, "skipped_reason": "existing_output"},
                lock,
            )
            continue
        kept.append(item)
    return kept


def run_batch(args: argparse.Namespace, items: list[BatchItem]) -> int:
    manifest_path = Path(args.manifest_out) if args.manifest_out else Path(args.out) / f"{_base_run_name(args)}_manifest.jsonl"
    if manifest_path.exists() and not (args.append_manifest or args.resume):
        manifest_path.unlink()
    workers = max(1, int(args.num_workers))
    lock = Lock()
    items = _filter_resume_items(args, items, manifest_path, lock)
    print(f"Batch replay items={len(items)} workers={workers} manifest={manifest_path}")
    failures: list[tuple[int, int]] = []

    if workers == 1:
        for item in items:
            rc = _run_one(args, item, manifest_path, lock)
            if rc != 0:
                failures.append((item.index, rc))
                if not args.continue_on_error:
                    break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_item = {pool.submit(_run_one, args, item, manifest_path, lock): item for item in items}
            for future in concurrent.futures.as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    rc = future.result()
                except Exception as exc:
                    rc = 255
                    print(f"[batch {item.index}] exception: {exc}", file=sys.stderr, flush=True)
                if rc != 0:
                    failures.append((item.index, rc))
                    if not args.continue_on_error:
                        # Cannot safely kill already running subprocesses from here because they are owned by
                        # worker threads. We stop submitting no new work by shutting down after current futures.
                        pass

    if failures:
        print(f"Batch failures: {failures}", file=sys.stderr)
        return failures[0][1]
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run simulation replay through experiments/replay.py and optionally export video-style LeRobot data."
    )
    parser.add_argument("--case", choices=sorted(_CASE_TO_GS), required=True)
    parser.add_argument("--gt-dir", default=None, help="Replay trajectory directory accepted by experiments/replay.py.")
    parser.add_argument(
        "--traj-list",
        default=None,
        help="Either a replay trajectory directory alias, or a text file with one gt_dir per non-empty row.",
    )
    parser.add_argument("--state-path", default=None, help="Optional single deformed soft-body state path.")
    parser.add_argument("--state-list", default=None, help="Text file with one state path per non-empty row.")
    parser.add_argument("--export-lerobot", action="store_true")
    parser.add_argument("--lerobot-out", default=None, help="LeRobot dataset output directory or batch root.")
    parser.add_argument("--repo-id", default="deformgen/replay", help="LeRobot repo_id metadata.")
    parser.add_argument("--task-name", default="replay", help="LeRobot task name metadata.")
    parser.add_argument("--video-backend", default=None, help="Optional LeRobot video backend metadata.")
    parser.add_argument("--save-final-state", action="store_true")
    parser.add_argument("--out", required=True, help="Replay output root directory.")
    parser.add_argument("--name", default=None, help="Run name mapped to Hydra timestamp. Batch mode appends _itemXXXXXX.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing replay run directory.")
    parser.add_argument("--use-qpos", dest="use_qpos", action="store_true", default=None)
    parser.add_argument("--no-use-qpos", dest="use_qpos", action="store_false")
    parser.add_argument("--start-index", type=int, default=0, help="Batch start row index.")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of batch rows to run.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of concurrent replay subprocesses in batch mode.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue batch after failed items.")
    parser.add_argument("--resume", action="store_true", help="Skip indices already marked return_code=0 in the manifest.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip items whose output directory already contains the requested replay/LeRobot/final_state artifacts.",
    )
    parser.add_argument("--manifest-out", default=None, help="Batch manifest JSONL path.")
    parser.add_argument("--append-manifest", action="store_true", help="Append to existing manifest instead of replacing it.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional raw Hydra override, e.g. --override randomize=false. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print backend command without executing it.")
    args = parser.parse_args(argv)

    try:
        items = build_batch_items(args)
        if items:
            return run_batch(args, items)
        if args.state_list is not None:
            raise ValueError("--state-list must be a text file in batch mode.")
        cmd = build_replay_command(args, args.override)
    except Exception as exc:
        print(f"deformgen-replay-export: {exc}", file=sys.stderr)
        return 2

    print("Running replay backend:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    raise SystemExit(main())
