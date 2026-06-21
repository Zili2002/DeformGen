#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

"${PYTHON}" -m deformgen.cli.perturb_states   --case rope   --out outputs/smoke_states   --num-states 1

"${PYTHON}" -m deformgen.cli.warp_trajectory   --case rope   --demo examples/data/rope_tiny/demo_trajectory.json   --state-list examples/data/rope_tiny/state_paths.txt   --out outputs/smoke_warp

"${PYTHON}" -m deformgen.cli.replay_export   --case rope   --out outputs/smoke_replay   --export-lerobot

"${PYTHON}" -m deformgen.cli.eval_success   --case rope   --replay-root outputs/smoke_replay
