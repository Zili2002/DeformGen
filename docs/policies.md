# Policy Training and Simulation Evaluation

Initialize and install the policy submodule using the [installation guide](installation.md). Download the [released datasets and checkpoints](assets.md#released-datasets-and-checkpoints), and set `DEFORMGEN_ROOT`, `DATA_ROOT`, and `CKPT_ROOT` as described under [local paths](assets.md#local-paths). Simulation evaluation also requires the [simulation assets](assets.md#simulation-assets).

For hardware deployment instead, see the [real-robot guide](../real/README.md).

## Training on Released Datasets

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

## Simulation Evaluation

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

deformgen-eval-success \
  --case cloth3 \
  --runs-glob "outputs/policy_eval/cloth3_svla_yawonly_test200/output_eval_policy/*" \
  --final-state-dir-name final_state \
  --out outputs/policy_eval/cloth3_svla_yawonly_test200/success_summary.csv
```
