from __future__ import annotations

import argparse

def main() -> int:
    parser = argparse.ArgumentParser(description="Warp demonstration trajectories to perturbed states.")
    parser.add_argument("--case", choices=["rope", "sloth", "cloth3"], required=True)
    parser.add_argument("--demo", required=True, help="Demo trajectory JSON or episode directory.")
    parser.add_argument("--state-list", required=True, help="Text file containing target state paths.")
    parser.add_argument("--mode", default="yawonly", choices=["yawonly", "txy", "rigidlocal_txyfit"])
    parser.add_argument("--grasp-local-k", type=int, default=5)
    parser.add_argument("--manip-local-k", type=int, default=5)
    parser.add_argument("--manip-decay", default="none")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("DeformGen warp_trajectory wrapper")
    print(f"case={args.case} mode={args.mode} demo={args.demo} state_list={args.state_list} out={args.out}")
    print("For full warping commands, see docs/trajectory_synthesis.md and scripts/warp_replay/.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
