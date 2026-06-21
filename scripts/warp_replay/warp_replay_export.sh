#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

# Minimal public template for trajectory warping, replay, LeRobot export, and success evaluation.
"${PYTHON}" -m deformgen.cli.warp_trajectory   --case rope   --demo examples/data/rope_tiny/demo_trajectory.json   --state-list examples/data/rope_tiny/state_paths.txt   --out outputs/examples/rope_warped   --mode yawonly

"${PYTHON}" -m deformgen.cli.replay_export   --case rope   --traj-list outputs/examples/rope_warped/trajectories.txt   --out outputs/examples/rope_replay   --export-lerobot   --save-final-state

"${PYTHON}" -m deformgen.cli.eval_success   --case rope   --replay-root outputs/examples/rope_replay
