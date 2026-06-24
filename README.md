# DeformGen

DeformGen is a deformable-object data generation framework built on the real2sim-eval / PhysTwin simulation stack. It provides two core capabilities:

1. **State perturbation**: synthesize diverse rope, sloth, and cloth3 deformable-object states.
2. **Trajectory synthesis**: warp demonstrations to perturbed states, replay them in simulation, export LeRobot datasets, and evaluate task success.

The repository keeps the real2sim-eval simulation layout for compatibility. Policy training and evaluation are provided through the optional `policy/` submodule.

## Method Names

The code keeps engineering names because they are used by configs, scripts, and experiment metadata:

| Code name | Paper name | Meaning |
| --- | --- | --- |
| `yawonly` | `DG` | yaw-only local trajectory adaptation |
| `txy` | `DG*` | local rigid XY translation + yaw adaptation |
| `gridrigid` | `SMG*` | grid-rigid state generation / evaluation setting |

Do not rename directories or checkpoint folders for paper notation. Use the mapping above in papers and reports.

## Repository Layout

```text
DeformGen/
├── cfg/                    # Hydra configs for simulation, assets, and augmentation defaults
├── deformgen/cli/          # Public command-line wrappers
├── experiments/            # real2sim-eval-compatible backends
├── policy/                 # Optional policy-training submodule
├── scripts/                # Example launch scripts and policy helpers
├── sim/                    # PhysTwin / simulator code
├── log/                    # Local assets or symlinks to released assets
└── outputs/                # Generated outputs, replay runs, and local checks
```

## Installation

```bash
git clone <your-deformgen-repo-url> DeformGen
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

## Download Released Data

The command examples below assume the released artifacts are downloaded under `./release`. Replace the repo names if you mirror the data elsewhere.

```bash
pip install -U huggingface_hub

mkdir -p release

huggingface-cli download Zili2002/DeformGen-Datasets \
  --repo-type dataset \
  --local-dir release/DeformGen-Datasets

huggingface-cli download Zili2002/DeformGen-Checkpoints \
  --repo-type model \
  --local-dir release/DeformGen-Checkpoints
```

Expected released dataset layout:

```text
release/DeformGen-Datasets/
├── rope/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
├── sloth/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
└── cloth3/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
```

Expected checkpoint layout:

```text
release/DeformGen-Checkpoints/<case>/<model>/<mode>/checkpoint
# case:  rope | sloth | cloth3
# model: act | dp | svla | pi0
# mode:  txy | yawonly | gridrigid | single
```

Set common paths:

```bash
export DEFORMGEN_ROOT=$PWD
export RELEASE_ROOT=$DEFORMGEN_ROOT/release
export DATA_ROOT=$RELEASE_ROOT/DeformGen-Datasets
export CKPT_ROOT=$RELEASE_ROOT/DeformGen-Checkpoints
```

Install the simulator assets after cloning the repository. This command downloads fixed upstream rope/sloth assets, DeformGen-specific Cloth3 assets, and the three formal demonstrations. It then creates only local symlinks under `log/` and writes `log/external_assets/resolved_manifest.json`.

```bash
deformgen-fetch sim-assets \
  --case all \
  --repo-root "$DEFORMGEN_ROOT"

# Optional mirror endpoint.
# deformgen-fetch sim-assets --case all --repo-root "$DEFORMGEN_ROOT" --endpoint https://hf-mirror.com
```

The immutable source revisions are recorded in [`assets/sources.yaml`](assets/sources.yaml). The installer refuses to run if a source revision is not pinned; it never falls back to a mutable `main` branch.

For local development, symlinks are acceptable and were used to validate the release layout:

```bash
mkdir -p outputs/downloaded/{datasets,checkpoints,testsets}
ln -sfn "$DATA_ROOT/rope/train1000_yawonly" outputs/downloaded/datasets/rope_train1000_yawonly
ln -sfn "$DATA_ROOT/rope/test200" outputs/downloaded/testsets/rope_test200
ln -sfn "$CKPT_ROOT/rope/act/yawonly" outputs/downloaded/checkpoints/rope_act_yawonly
```

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

## 1. State Perturbation

### Runtime-Random Perturbation

Runtime-random perturbation follows a demonstration, closes the gripper at the configured frame, applies runtime perturbations, optionally releases the gripper, stabilizes the state, and writes `episode_0000/final_state/state.npy`.

Formal defaults are stored in `cfg/augmentation.yaml` under `case_defaults`. They align with the original three-case perturbation plan:

| Case | Default demo | Episode | Seed behavior | Replay randomization |
| --- | --- | ---: | --- | --- |
| `rope` | `log/policy_rollouts/rope_act_7000` | `1` | base `1600` | enabled |
| `sloth` | `log/dataset/teleop_replay_005_sloth_grasp_shift_s5a3f2_segmentxy_keepoffset` | `0` | base `42` | disabled |
| `cloth3` | `log/dataset/teleop_run_007_clean_noleading_interp80_trim_norelease` | `0` | base `44` | disabled |

Run one state per case:

```bash
# Rope
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode runtime-random \
  --num-states 1 \
  --out outputs/perturb/rope_runtime_random \
  --overwrite

# Sloth
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case sloth \
  --mode runtime-random \
  --num-states 1 \
  --out outputs/perturb/sloth_runtime_random \
  --overwrite

# Cloth3
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case cloth3 \
  --mode runtime-random \
  --num-states 1 \
  --out outputs/perturb/cloth3_runtime_random \
  --overwrite
```

Important outputs:

```text
outputs/perturb/<run>/demo_episode_XXXX/sample_0000/episode_0000/final_state/state.npy
outputs/perturb/<run>/demo_episode_XXXX/sample_0000/metadata.json
outputs/perturb/<run>/demo_episode_XXXX/summary.json
```

Useful overrides:

```bash
# Change batch seed while keeping the case default perturbation profile.
--seed 42

# Override friction when needed, e.g. sloth experiments.
--collide-fric-override 0.3

# Disable videos for large-scale generation.
--no-make-videos
```

### Default / Grid / Grid-Rigid State Generation

Use these modes when generating benchmark initial-state sets without runtime random actions.

```bash
# Deterministic default state.
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode default-state \
  --out outputs/states/rope_default \
  --overwrite

# Use the case YAML grid_randomization entries.
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode configured-grid \
  --num-states 27 \
  --out outputs/states/rope_configured_grid27 \
  --overwrite

# Explicit uniform rigid grid.
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode uniform-grid \
  --grid-x-range=-0.20,0.20 \
  --grid-y-range=-0.20,0.20 \
  --grid-theta-range=-60,60 \
  --grid-nx 10 \
  --grid-ny 10 \
  --grid-ntheta 10 \
  --num-states 1000 \
  --out outputs/states/rope_gridrigid_1000 \
  --overwrite
```

Each sample writes:

```text
outputs/states/<run>/demo_episode_XXXX/sample_0000/episode_0000/final_state/state.npy
outputs/states/<run>/demo_episode_XXXX/sample_0000/metadata.json
```

## 2. Trajectory Warping

`deformgen-warp` calls `experiments/create_interpolated_json_trajectory.py` and writes a replay-ready robot trajectory.

Formal yawonly / DG parameters used by the released three-case data:

```bash
--mode yawonly
--grasp-local-k 5
--manip-local-k 99999
--manip-decay none
--num-approach-steps 120
--num-grasp-steps 30
--num-rotate-steps 30
--num-interp-steps 300
--override grasp_warp_mode=knn
--override adapt_orientation=true
```

### Single-State Warp

```bash
CASE=rope
STATE=outputs/perturb/rope_runtime_random/demo_episode_0001/sample_0000/episode_0000/final_state/state.npy
DEMO=log/policy_rollouts/rope_act_7000
EPISODE_ID=1
OUT=outputs/warp/rope_yawonly_item000000

CUDA_VISIBLE_DEVICES=0 deformgen-warp \
  --case "$CASE" \
  --demo "$DEMO" \
  --episode-id "$EPISODE_ID" \
  --state-path "$STATE" \
  --mode yawonly \
  --grasp-local-k 5 \
  --manip-local-k 99999 \
  --manip-decay none \
  --num-approach-steps 120 \
  --num-grasp-steps 30 \
  --num-rotate-steps 30 \
  --num-interp-steps 300 \
  --override grasp_warp_mode=knn \
  --override adapt_orientation=true \
  --out "$OUT"
```

The trajectory directory is:

```text
outputs/warp/rope_yawonly_item000000/episode_0001
```

### Batch Warp

Prepare a state list:

```bash
find outputs/states/rope_gridrigid_1000 -path '*/episode_0000/final_state/state.npy' | sort > outputs/states/rope_gridrigid_1000_states.txt
```

Run parallel warp:

```bash
CUDA_VISIBLE_DEVICES=0 deformgen-warp \
  --case rope \
  --demo log/policy_rollouts/rope_act_7000 \
  --episode-id 1 \
  --state-list outputs/states/rope_gridrigid_1000_states.txt \
  --mode yawonly \
  --grasp-local-k 5 \
  --manip-local-k 99999 \
  --manip-decay none \
  --num-approach-steps 120 \
  --num-grasp-steps 30 \
  --num-rotate-steps 30 \
  --num-interp-steps 300 \
  --override grasp_warp_mode=knn \
  --override adapt_orientation=true \
  --out outputs/warp/rope_gridrigid_yawonly \
  --num-workers 4
```

Batch output includes:

```text
outputs/warp/rope_gridrigid_yawonly/trajectories.txt
outputs/warp/rope_gridrigid_yawonly/warp_manifest.jsonl
```

## 3. Replay, Rendering, LeRobot Export, and Success

`deformgen-replay-export` calls `experiments/replay.py`. For warped JSON trajectories, always use `--no-use-qpos`.

### Rope Replay + LeRobot + Online Rope Success

```bash
STATE=outputs/perturb/rope_runtime_random/demo_episode_0001/sample_0000/episode_0000/final_state/state.npy
GT=outputs/warp/rope_yawonly_item000000/episode_0001
ROOT=outputs/replay/rope_yawonly_item000000

CUDA_VISIBLE_DEVICES=0 deformgen-replay-export \
  --case rope \
  --gt-dir "$GT" \
  --state-path "$STATE" \
  --out "$ROOT" \
  --name rope_yawonly \
  --overwrite \
  --no-use-qpos \
  --save-final-state \
  --export-lerobot \
  --lerobot-out "$ROOT/lerobot_dataset" \
  --repo-id local/deformgen_rope_yawonly \
  --task-name replay_rope \
  --override make_videos=true \
  --override save_depth=false \
  --override clip_success_eval=true \
  --override clip_success_mode=rope_routed \
  --override clip_success_plane_min_bottom=100 \
  --override clip_success_plane_min_top=100 \
  --override clip_success_tail_steps=null \
  --override clip_success_tail_ratio=0.1111111111 \
  --override clip_success_required_routed_ratio=0.3
```

### Sloth Replay + LeRobot + Last-Frame Packed Success

```bash
STATE=outputs/perturb/sloth_runtime_random/demo_episode_0000/sample_0000/episode_0000/final_state/state.npy
GT=outputs/warp/sloth_yawonly_item000000/episode_0000
ROOT=outputs/replay/sloth_yawonly_item000000

CUDA_VISIBLE_DEVICES=0 deformgen-replay-export \
  --case sloth \
  --gt-dir "$GT" \
  --state-path "$STATE" \
  --out "$ROOT" \
  --name sloth_yawonly \
  --overwrite \
  --no-use-qpos \
  --save-final-state \
  --export-lerobot \
  --lerobot-out "$ROOT/lerobot_dataset" \
  --repo-id local/deformgen_sloth_yawonly \
  --task-name replay_sloth \
  --override physics.collide_fric_override=0.3 \
  --override final_state_stabilization_steps=0 \
  --override final_state_dir_name=final_state \
  --override make_videos=true \
  --override clip_success_eval=false \
  --override +sloth_success_eval=true \
  --override +sloth_success_mode=last_frame_packed \
  --override +sloth_success_min_points=3050 \
  --override +sloth_success_obb_scale=1.05
```

### Cloth3 Replay + LeRobot + Last-Frame Triangle Success

```bash
STATE=outputs/perturb/cloth3_runtime_random/demo_episode_0000/sample_0000/episode_0000/final_state/state.npy
GT=outputs/warp/cloth3_yawonly_item000000/episode_0000
ROOT=outputs/replay/cloth3_yawonly_item000000

CUDA_VISIBLE_DEVICES=0 deformgen-replay-export \
  --case cloth3 \
  --gt-dir "$GT" \
  --state-path "$STATE" \
  --out "$ROOT" \
  --name cloth3_yawonly \
  --overwrite \
  --no-use-qpos \
  --save-final-state \
  --export-lerobot \
  --lerobot-out "$ROOT/lerobot_dataset" \
  --repo-id local/deformgen_cloth3_yawonly \
  --task-name replay_cloth3 \
  --override final_state_stabilization_steps=0 \
  --override final_state_dir_name=final_state \
  --override make_videos=true \
  --override clip_success_eval=false \
  --override cloth3_success_eval=true \
  --override cloth3_success_mode=last_frame_triangle
```

### Offline Last-Frame Success

Use this when replay outputs already contain `episode_0000/final_state/state.npy`.

```bash
deformgen-eval-success \
  --case rope \
  --replay-root outputs/replay/rope_yawonly_item000000/rope_yawonly \
  --final-state-dir-name final_state \
  --out outputs/replay/rope_yawonly_item000000/success_summary.csv

deformgen-eval-success \
  --case sloth \
  --replay-root outputs/replay/sloth_yawonly_item000000/sloth_yawonly \
  --final-state-dir-name final_state \
  --sloth-success-min-points 3050 \
  --sloth-obb-scale 1.05 \
  --out outputs/replay/sloth_yawonly_item000000/success_summary.csv

deformgen-eval-success \
  --case cloth3 \
  --replay-root outputs/replay/cloth3_yawonly_item000000/cloth3_yawonly \
  --final-state-dir-name final_state \
  --out outputs/replay/cloth3_yawonly_item000000/success_summary.csv
```

### Batch Replay + LeRobot Export

`--traj-list` and `--state-list` must have the same number of rows. Each row is processed once. Use `--resume` and `--skip-existing` for long jobs.

```bash
CUDA_VISIBLE_DEVICES=0 deformgen-replay-export \
  --case rope \
  --traj-list outputs/warp/rope_gridrigid_yawonly/trajectories.txt \
  --state-list outputs/states/rope_gridrigid_1000_states.txt \
  --out outputs/replay/rope_gridrigid_yawonly \
  --name rope_gridrigid_yawonly \
  --overwrite \
  --no-use-qpos \
  --save-final-state \
  --export-lerobot \
  --lerobot-out outputs/policy_datasets/rope_gridrigid_yawonly_lerobot \
  --repo-id local/deformgen_rope_gridrigid_yawonly \
  --task-name replay_rope \
  --num-workers 4 \
  --resume \
  --skip-existing \
  --continue-on-error \
  --override make_videos=false \
  --override save_depth=false
```

## 4. Released Dataset Training Commands

The released LeRobot datasets can be used directly by ACT, DP, and SVLA through the `policy/third_party/lerobot` training entry point. The examples below train on `rope/train1000_yawonly`; replace `CASE`, `TASK`, and `TRAIN_ROOT` for other cases.

Task names:

| Case | Task name |
| --- | --- |
| `rope` | `insert_rope` |
| `sloth` | `pack_sloth` |
| `cloth3` | `fold_cloth` |

The current `policy/` submodule may ship only `insert_rope`, `pack_sloth`, and `pusht` templates. If `fold_cloth` configs are absent, create them once before training or evaluating cloth3:

```bash
cd "$DEFORMGEN_ROOT"
for model in act dp svla; do
  cp -n "policy/configs/training/${model}_pack_sloth.json" "policy/configs/training/${model}_fold_cloth.json"
done
cp -n policy/configs/inference/pack_sloth.json policy/configs/inference/fold_cloth.json
python - <<'PY'
import json
from pathlib import Path
inf = Path("policy/configs/inference/fold_cloth.json")
cfg = json.loads(inf.read_text())
cfg["task_name"] = "fold_cloth"
inf.write_text(json.dumps(cfg, indent=2))
PY
```

LeRobot refuses to start if `output_dir` already exists and `resume=false`. The commands below intentionally create only the config directory, not the training output directory.

### ACT Training

```bash
cd "$DEFORMGEN_ROOT"
CASE=rope
TASK=insert_rope
TRAIN_ROOT="$DATA_ROOT/$CASE/train1000_yawonly"
OUT="outputs/policy_train/$TASK/act_${CASE}_yawonly"
CFG="outputs/policy_train/configs/act_${CASE}_yawonly.json"
mkdir -p "$(dirname "$CFG")"

python - <<PY
import json
from pathlib import Path
case = "$CASE"
task = "$TASK"
train_root = "$TRAIN_ROOT"
out = "$OUT"
src = Path("policy/configs/training/act_" + task + ".json")
cfg = json.loads(src.read_text())
cfg["dataset"]["root"] = train_root
cfg["dataset"]["repo_id"] = "local/deformgen_" + case + "_yawonly"
cfg["dataset"]["episodes"] = None
# DeformGen LeRobot exports use front/wrist video keys. Some upstream policy
# templates use the older side/wrist image keys; map them here before training.
features = cfg["policy"].get("input_features", {})
if "observation.image.side" in features:
    features["observation.images.front"] = features.pop("observation.image.side")
if "observation.image.wrist" in features:
    features["observation.images.wrist"] = features.pop("observation.image.wrist")
cfg["output_dir"] = out
cfg["job_name"] = "act_" + case + "_yawonly"
cfg["wandb"]["enable"] = False
Path("$CFG").write_text(json.dumps(cfg, indent=2))
PY

cd policy
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/lerobot/scripts/train.py \
  --config_path "$DEFORMGEN_ROOT/$CFG" \
  --job_name "act_${CASE}_yawonly" \
  --output_dir "$DEFORMGEN_ROOT/$OUT"
```

### DP Training

```bash
cd "$DEFORMGEN_ROOT"
CASE=rope
TASK=insert_rope
TRAIN_ROOT="$DATA_ROOT/$CASE/train1000_yawonly"
OUT="outputs/policy_train/$TASK/dp_${CASE}_yawonly"
CFG="outputs/policy_train/configs/dp_${CASE}_yawonly.json"
mkdir -p "$(dirname "$CFG")"

python - <<PY
import json
from pathlib import Path
case = "$CASE"
task = "$TASK"
train_root = "$TRAIN_ROOT"
out = "$OUT"
src = Path("policy/configs/training/dp_" + task + ".json")
cfg = json.loads(src.read_text())
cfg["dataset"]["root"] = train_root
cfg["dataset"]["repo_id"] = "local/deformgen_" + case + "_yawonly"
cfg["dataset"]["episodes"] = None
# DeformGen LeRobot exports use front/wrist video keys. Some upstream policy
# templates use the older side/wrist image keys; map them here before training.
features = cfg["policy"].get("input_features", {})
if "observation.image.side" in features:
    features["observation.images.front"] = features.pop("observation.image.side")
if "observation.image.wrist" in features:
    features["observation.images.wrist"] = features.pop("observation.image.wrist")
cfg["output_dir"] = out
cfg["job_name"] = "dp_" + case + "_yawonly"
cfg["wandb"]["enable"] = False
Path("$CFG").write_text(json.dumps(cfg, indent=2))
PY

cd policy
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/lerobot/scripts/train.py \
  --config_path "$DEFORMGEN_ROOT/$CFG" \
  --job_name "dp_${CASE}_yawonly" \
  --output_dir "$DEFORMGEN_ROOT/$OUT"
```

### SVLA Training

SVLA uses relative actions. Keep the dataset-local `meta/relative_action_stats_Te50.pt` with the training dataset.

```bash
cd "$DEFORMGEN_ROOT"
CASE=rope
TASK=insert_rope
TRAIN_ROOT="$DATA_ROOT/$CASE/train1000_yawonly"
OUT="outputs/policy_train/$TASK/svla_${CASE}_yawonly"
CFG="outputs/policy_train/configs/svla_${CASE}_yawonly.json"
mkdir -p "$(dirname "$CFG")"

test -f "$TRAIN_ROOT/meta/relative_action_stats_Te50.pt"

python - <<PY
import json
from pathlib import Path
case = "$CASE"
task = "$TASK"
train_root = "$TRAIN_ROOT"
out = "$OUT"
src = Path("policy/configs/training/svla_" + task + ".json")
cfg = json.loads(src.read_text())
cfg["dataset"]["root"] = train_root
cfg["dataset"]["repo_id"] = "local/deformgen_" + case + "_yawonly"
cfg["dataset"]["episodes"] = None
# DeformGen LeRobot exports use front/wrist video keys. Some upstream policy
# templates use the older side/wrist image keys; map them here before training.
features = cfg["policy"].get("input_features", {})
if "observation.image.side" in features:
    features["observation.images.front"] = features.pop("observation.image.side")
if "observation.image.wrist" in features:
    features["observation.images.wrist"] = features.pop("observation.image.wrist")
cfg["output_dir"] = out
cfg["job_name"] = "svla_" + case + "_yawonly"
cfg["wandb"]["enable"] = False
Path("$CFG").write_text(json.dumps(cfg, indent=2))
PY

cd policy
CUDA_VISIBLE_DEVICES=0 python third_party/lerobot/lerobot/scripts/train.py \
  --config_path "$DEFORMGEN_ROOT/$CFG" \
  --job_name "svla_${CASE}_yawonly" \
  --output_dir "$DEFORMGEN_ROOT/$OUT"
```

### pi0 Training

pi0 uses OpenPI and requires the correct dataset-specific norm stats. Do not fall back to unrelated global norm stats.

The released datasets store pi0 norm stats with the standard name:

```text
$TRAIN_ROOT/meta/norm_stats.json
```

The file is computed with the max10000 setting. Older max10000/max200000-specific filenames are not part of the public training-set interface.

Example for rope yawonly:

```bash
cd "$DEFORMGEN_ROOT"
CASE=rope
TASK=insert_rope
TRAIN_ROOT="$DATA_ROOT/$CASE/train1000_yawonly"
OPENPI_REPO_ID="local/deformgen_${CASE}_yawonly"
NORM_SRC="$TRAIN_ROOT/meta/norm_stats.json"

# OpenPI resolves LeRobot datasets through the HF cache-style path.
mkdir -p "/root/.cache/huggingface/lerobot/local"
ln -sfn "$TRAIN_ROOT" "/root/.cache/huggingface/lerobot/local/deformgen_${CASE}_yawonly"

# Put norm stats where the OpenPI config expects them.
mkdir -p "outputs/policy_train/pi0_assets/pi0_lora_${TASK}/local/deformgen_${CASE}_yawonly"
cp "$NORM_SRC" "outputs/policy_train/pi0_assets/pi0_lora_${TASK}/local/deformgen_${CASE}_yawonly/norm_stats.json"

cd policy
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
python ../scripts/policy/run_deformgen_openpi.py \
  third_party/openpi/scripts/train.py \
  pi0_lora_${TASK}
```

The exact OpenPI config name depends on the `policy/` submodule configuration. The key requirement is that the config uses the same `repo_id` and norm stats copied from the matching DeformGen training dataset.

## 5. Policy Evaluation Commands

Closed-loop evaluation uses `experiments/eval_policy.py` with a checkpoint and a test-state list. For released `test200`, pass the mode-specific state list.

### ACT / DP / SVLA Evaluation

```bash
cd "$DEFORMGEN_ROOT"
CASE=rope
TASK=insert_rope
MODE=yawonly
MODEL=act
CKPT="$CKPT_ROOT/$CASE/$MODEL/$MODE/checkpoint"
STATE_LIST="$DATA_ROOT/$CASE/test200/${MODE}_state_paths.txt"
OUT="outputs/policy_eval/${CASE}_${MODEL}_${MODE}_test200"

CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
  gs=$CASE \
  env=xarm_gripper \
  physics.ckpt_path=log/phystwin/$CASE \
  physics.case_name=${CASE}_0001 \
  physics.duration=30 \
  env.sim.duration=30 \
  exp_root="$OUT" \
  timestamp=${CASE}_${MODEL}_${MODE}_test200 \
  policy.inference_cfg_path=policy/configs/inference/${TASK}.json \
  policy.checkpoint_path="$CKPT" \
  +deformed_states_file="$STATE_LIST" \
  +save_final_state=true \
  +final_state_dir_name=final_state
```

For `cloth3`, use `gs=cloth3`, `physics.ckpt_path=log/phystwin/cloth3`, `physics.case_name=cloth3_0001`, and `policy/configs/inference/fold_cloth.json` if present in your policy submodule.

### pi0 Evaluation

pi0 evaluation uses the same `experiments/eval_policy.py` entry point, but the inference config and checkpoint must point to the pi0 LoRA config/checkpoint from the `policy/` submodule.

```bash
cd "$DEFORMGEN_ROOT"
CASE=sloth
TASK=pack_sloth
MODE=yawonly
CKPT="$CKPT_ROOT/$CASE/pi0/$MODE/checkpoint"
STATE_LIST="$DATA_ROOT/$CASE/test200/${MODE}_state_paths.txt"
OUT="outputs/policy_eval/${CASE}_pi0_${MODE}_test200"

CUDA_VISIBLE_DEVICES=0 python experiments/eval_policy.py \
  gs=$CASE \
  env=xarm_gripper \
  physics.ckpt_path=log/phystwin/$CASE \
  physics.case_name=${CASE}_0001 \
  physics.duration=30 \
  env.sim.duration=30 \
  exp_root="$OUT" \
  timestamp=${CASE}_pi0_${MODE}_test200 \
  policy.inference_cfg_path=policy/configs/inference/${TASK}_pi0.json \
  policy.checkpoint_path="$CKPT" \
  +deformed_states_file="$STATE_LIST" \
  +save_final_state=true \
  +final_state_dir_name=final_state
```

If your policy submodule uses a different pi0 inference config path, keep the same checkpoint and state-list pattern but replace `policy.inference_cfg_path` accordingly.

### Evaluate Success for Policy Rollouts

```bash
deformgen-eval-success \
  --case rope \
  --runs-glob "outputs/policy_eval/rope_act_yawonly_test200/output_eval_policy/*" \
  --final-state-dir-name final_state \
  --out outputs/policy_eval/rope_act_yawonly_test200/success_summary.csv

deformgen-eval-success \
  --case sloth \
  --runs-glob "outputs/policy_eval/sloth_pi0_yawonly_test200/output_eval_policy/*" \
  --final-state-dir-name final_state \
  --sloth-success-min-points 3050 \
  --sloth-obb-scale 1.05 \
  --out outputs/policy_eval/sloth_pi0_yawonly_test200/success_summary.csv
```

## 6. End-to-End Minimal Example

This example runs the complete rope pipeline: perturb one state, warp one trajectory, replay/export LeRobot, and compute success.

```bash
cd "$DEFORMGEN_ROOT"

CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode runtime-random \
  --num-states 1 \
  --out outputs/e2e/rope_state \
  --overwrite

STATE=outputs/e2e/rope_state/demo_episode_0001/sample_0000/episode_0000/final_state/state.npy

CUDA_VISIBLE_DEVICES=0 deformgen-warp \
  --case rope \
  --demo log/policy_rollouts/rope_act_7000 \
  --episode-id 1 \
  --state-path "$STATE" \
  --mode yawonly \
  --grasp-local-k 5 \
  --manip-local-k 99999 \
  --manip-decay none \
  --num-approach-steps 120 \
  --num-grasp-steps 30 \
  --num-rotate-steps 30 \
  --num-interp-steps 300 \
  --override grasp_warp_mode=knn \
  --override adapt_orientation=true \
  --out outputs/e2e/rope_warp

CUDA_VISIBLE_DEVICES=0 deformgen-replay-export \
  --case rope \
  --gt-dir outputs/e2e/rope_warp/episode_0001 \
  --state-path "$STATE" \
  --out outputs/e2e/rope_replay \
  --name rope_yawonly \
  --overwrite \
  --no-use-qpos \
  --save-final-state \
  --export-lerobot \
  --lerobot-out outputs/e2e/rope_lerobot \
  --repo-id local/deformgen_rope_e2e \
  --task-name replay_rope \
  --override make_videos=true \
  --override save_depth=false \
  --override clip_success_eval=true \
  --override clip_success_mode=rope_routed \
  --override clip_success_plane_min_bottom=100 \
  --override clip_success_plane_min_top=100 \
  --override clip_success_tail_steps=null \
  --override clip_success_tail_ratio=0.1111111111 \
  --override clip_success_required_routed_ratio=0.3

deformgen-eval-success \
  --case rope \
  --replay-root outputs/e2e/rope_replay/rope_yawonly \
  --final-state-dir-name final_state \
  --out outputs/e2e/rope_success_summary.csv
```

## 7. Notes and Troubleshooting

- Keep the engineering directory names `yawonly`, `txy`, and `gridrigid`; map them to `DG`, `DG*`, and `SMG*` only in presentation text.
- For warped JSON trajectories, use `--no-use-qpos` in replay.
- For long batch replay jobs, use `--resume --skip-existing --continue-on-error`.
- For sloth evaluation, use the same object-ground friction used to generate the replay data, commonly `physics.collide_fric_override=0.3`.
- For pi0, always use the norm stats from the exact training dataset. A wrong global `norm_stats.json` can produce invalid evaluation results.
- Large generated outputs should stay outside git. Store them under `outputs/`, `release/`, or a project-specific data root.

## Documentation

- `docs/installation.md`
- `docs/assets.md`
- `docs/state_perturbation.md`
- `docs/trajectory_synthesis.md`
- `docs/replay_and_export.md`
- `docs/policy_training.md`
- `docs/data_format.md`
- `docs/examples.md`
- `docs/attribution.md`

## License and Attribution

This repository keeps the original license from real2sim-eval. See `LICENSE` and `docs/attribution.md`.
