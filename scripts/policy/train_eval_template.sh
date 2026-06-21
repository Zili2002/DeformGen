#!/usr/bin/env bash
set -euo pipefail

# Policy training/evaluation lives in the optional `policy` submodule.
# Initialize it first:
#   git submodule update --init --recursive policy
# Then adapt the command to the model family and dataset path.

POLICY_DIR=${POLICY_DIR:-policy}
DATASET=${DATASET:-examples/data/rope_tiny}
TASK=${TASK:-insert_rope}

cd "${POLICY_DIR}"
echo "Train policy for task=${TASK} dataset=${DATASET}"
echo "See docs/policy_training.md for ACT/DP/SVLA/pi0 command templates."
