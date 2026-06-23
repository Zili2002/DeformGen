#!/usr/bin/env bash
set -euo pipefail

# Minimal public template for deterministic uniform-grid state synthesis.
# DEFORMGEN_ASSET_ROOT must contain log/gs and log/phystwin assets.
PYTHON=${PYTHON:-python3}
ASSET_ROOT=${DEFORMGEN_ASSET_ROOT:?set DEFORMGEN_ASSET_ROOT to the directory containing log/ assets}
DEMO=${DEFORMGEN_DEMO:?set DEFORMGEN_DEMO to a replay trajectory directory}
OUT=${DEFORMGEN_OUT:-outputs/examples/rope_grid_rigid_states}

"${PYTHON}" -m deformgen.cli.perturb_states \
  --case rope \
  --asset-root "${ASSET_ROOT}" \
  --demo "${DEMO}" \
  --out "${OUT}" \
  --mode uniform-grid \
  --grid-nx 3 \
  --grid-ny 3 \
  --grid-ntheta 3 \
  --grid-x-range=-0.05,0.05 \
  --grid-y-range=-0.05,0.05 \
  --grid-theta-range=-10,10
