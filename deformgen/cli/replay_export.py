from __future__ import annotations

import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Replay warped trajectories and export policy datasets.")
    parser.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    parser.add_argument("--traj-list", default=None)
    parser.add_argument("--state-list", default=None)
    parser.add_argument("--export-lerobot", action="store_true")
    parser.add_argument("--save-final-state", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("DeformGen replay_export wrapper")
    print(f"case={args.case} out={args.out} export_lerobot={args.export_lerobot}")
    print("For simulation-backed replay, see docs/replay_and_export.md and experiments/replay.py.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
