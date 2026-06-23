from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    backend = repo_root / "experiments" / "utils" / "eval_lastframe_success_replay_runs.py"
    if not backend.exists():
        raise FileNotFoundError(f"Cannot find success backend: {backend}")

    patterns = args.runs_glob or []
    if args.replay_root is not None:
        root = Path(args.replay_root)
        if not root.exists():
            raise FileNotFoundError(f"Replay root does not exist: {root}")
        if (root / "episode_0000").exists():
            patterns.append(str(root))
        else:
            patterns.append(str(root / "*"))
    if not patterns:
        raise ValueError("Provide --replay-root or at least one --runs-glob.")

    out = Path(args.out) if args.out is not None else Path(patterns[0]).parent / f"{args.case}_success_summary.csv"
    cmd = [
        sys.executable,
        "-m",
        "experiments.utils.eval_lastframe_success_replay_runs",
        "--case",
        args.case,
        "--summary-csv",
        str(out),
    ]
    for pat in patterns:
        cmd.extend(["--runs-glob", pat])
    cmd.extend([
        "--episode-id", str(args.episode_id),
        "--final-state-dir-name", args.final_state_dir_name,
        "--sloth-success-min-points", str(args.sloth_success_min_points),
        "--sloth-obb-scale", str(args.sloth_obb_scale),
    ])
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate last-frame task success for replay outputs.")
    parser.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    parser.add_argument("--replay-root", default=None, help="One replay run dir or a directory containing run dirs.")
    parser.add_argument("--runs-glob", action="append", default=[], help="Glob for replay run dirs. May be repeated.")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--final-state-dir-name", default="final_state")
    parser.add_argument("--sloth-success-min-points", type=int, default=3050)
    parser.add_argument("--sloth-obb-scale", type=float, default=1.05)
    parser.add_argument("--out", default=None, help="Summary CSV path.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        cmd = build_command(args)
    except Exception as exc:
        print(f"deformgen-eval-success: {exc}", file=sys.stderr)
        return 2

    print("Running success backend:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    raise SystemExit(main())
