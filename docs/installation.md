# Installation

The examples use a Linux shell and a Python 3.11 environment. Run commands from the repository root unless a command explicitly changes directory. The policy submodule is optional for data generation; initialize it for policy training or evaluation.

For xArm7 and RealSense deployment, continue with the [real-robot guide](../real/README.md).

```bash
git clone https://github.com/Zili2002/DeformGen.git DeformGen
cd DeformGen

uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .

# Optional policy submodule for ACT / DP / SVLA / pi0 training and evaluation.
git submodule update --init --recursive policy
# If HTTPS submodule cloning is unreliable on your cluster, use SSH rewriting instead:
# git -c url.git@github.com:.insteadOf=https://github.com/ submodule update --init --recursive policy
cd policy
uv pip install -e .
cd ..

# CUDA / geometry extensions used by the simulator.
cd third-party/diff-gaussian-rasterization-w-depth
uv pip install --no-build-isolation -e .
cd ../urdfpy-0.0.22
uv pip install -e .
cd ../..
```

If your environment does not provide `uv`, use an equivalent Python 3.11 virtual environment and `pip install -e .`.

## Command Entry Points

After `uv pip install -e .`, these console commands are available:

```bash
deformgen-perturb --help
deformgen-warp --help
deformgen-replay-export --help
deformgen-eval-success --help
deformgen-fetch --help
```

Equivalent module form:

```bash
python -m deformgen.cli.perturb_states --help
python -m deformgen.cli.warp_trajectory --help
python -m deformgen.cli.replay_export --help
python -m deformgen.cli.eval_success --help
python -m deformgen.cli.fetch_assets --help
```
