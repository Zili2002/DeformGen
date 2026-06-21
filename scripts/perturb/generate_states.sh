#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

# Minimal public template for synthesizing perturbed deformable-object states.
# Replace config overrides with your dataset/checkpoint paths.
"${PYTHON}" -m deformgen.cli.perturb_states   --case rope   --out outputs/examples/rope_states   --num-states 1   --mode grid-rigid
