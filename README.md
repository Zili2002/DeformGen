# DeformGen

DeformGen is a deformable-object data generation framework built on top of the real2sim-eval / PhysTwin simulation stack. It focuses on two core capabilities:

1. **State perturbation**: synthesize diverse deformable-object states from simulated assets.
2. **Trajectory synthesis**: warp demonstration trajectories to perturbed states, replay them, export LeRobot datasets, and evaluate task success.

The repository keeps the real2sim-eval simulation dependencies for compatibility. The policy training code is optional and provided through the `policy/` git submodule.

## Features

- Deformable state perturbation for rope, sloth, and cloth-like objects.
- Grid / random / grid-rigid perturbation pipelines with stabilization.
- Deformation-aware trajectory warping with `txy`, `yawonly`, and local-rigid variants.
- Replay, rendering, final-state export, LeRobot export, and last-frame success metrics.
- Policy training/evaluation command templates for ACT, DP, SVLA, and pi0 through the optional `policy/` submodule.

## Installation

```bash
uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .

# Optional: install policy training/evaluation submodule
git submodule update --init --recursive policy
cd policy && uv pip install -r pyproject.toml && cd ..
```

For simulation and rendering, also install the bundled third-party extensions:

```bash
cd third-party/diff-gaussian-rasterization-w-depth
uv pip install --no-build-isolation -e .
cd ../urdfpy-0.0.22
uv pip install -e .
cd ../..
```

## Quick Start

```bash
# 1. Generate perturbed states
python -m deformgen.cli.perturb_states --case rope --out outputs/rope_states --num-states 3

# 2. Warp a demonstration trajectory
python -m deformgen.cli.warp_trajectory   --case rope   --demo examples/data/rope_tiny/demo_trajectory.json   --state-list examples/data/rope_tiny/state_paths.txt   --mode yawonly   --out outputs/rope_warped

# 3. Replay/export placeholder command template
python -m deformgen.cli.replay_export --case rope --help
```

The CLI wrappers are intentionally lightweight and point to the underlying real2sim-eval-compatible implementation. See `docs/` for full workflows.

## Example Data

Tiny example manifests live under:

```text
examples/data/rope_tiny
examples/data/sloth_tiny
examples/data/cloth3_tiny
```

Large assets, checkpoints, and generated datasets are intentionally not included in git.

## Documentation

- `docs/installation.md`
- `docs/state_perturbation.md`
- `docs/trajectory_synthesis.md`
- `docs/replay_and_export.md`
- `docs/policy_training.md`
- `docs/data_format.md`
- `docs/examples.md`
- `docs/attribution.md`

## License

This repository keeps the original license from real2sim-eval. See `LICENSE`.
