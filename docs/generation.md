# Data Generation and Replay

Follow the [installation guide](installation.md) and install the [simulation assets](assets.md#simulation-assets) first. All paths and commands below assume the repository root as the working directory. For one complete rope run, see the [quick start](../README.md#quick-start).

## State Perturbation

### Runtime-Random Perturbation

Runtime-random perturbation follows a demonstration, closes the gripper at the configured frame, applies runtime perturbations, optionally releases the gripper, and saves its raw result. It then runs an independent 30-step stabilization replay from that raw state, holding the final robot command, and writes `final_state_stable/state.npy`. Use this stable artifact for trajectory warping.

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
  --post-stabilization-steps 30 \
  --out outputs/perturb/rope_runtime_random \
  --overwrite

# Sloth
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case sloth \
  --mode runtime-random \
  --num-states 1 \
  --post-stabilization-steps 30 \
  --out outputs/perturb/sloth_runtime_random \
  --overwrite

# Cloth3
CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case cloth3 \
  --mode runtime-random \
  --num-states 1 \
  --post-stabilization-steps 30 \
  --out outputs/perturb/cloth3_runtime_random \
  --overwrite
```

Important outputs:

```text
outputs/perturb/<run>/demo_episode_XXXX/sample_0000/soft_body_state.npy
outputs/perturb/<run>/demo_episode_XXXX/sample_0000/final_state_stable/state.npy
outputs/perturb/<run>/demo_episode_XXXX/sample_0000/metadata.json
outputs/perturb/<run>/demo_episode_XXXX/summary.json
```

Useful overrides:

```bash
# Change batch seed while keeping the case default perturbation profile.
--seed 42

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

## Trajectory Warping

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
STATE=outputs/perturb/rope_runtime_random/demo_episode_0001/sample_0000/final_state_stable/state.npy
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

## Replay, Rendering, LeRobot Export, and Success

`deformgen-replay-export` calls `experiments/replay.py`. For warped JSON trajectories, always use `--no-use-qpos`.

### Rope Replay + LeRobot + Online Rope Success

```bash
STATE=outputs/perturb/rope_runtime_random/demo_episode_0001/sample_0000/final_state_stable/state.npy
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
STATE=outputs/perturb/sloth_runtime_random/demo_episode_0000/sample_0000/final_state_stable/state.npy
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
STATE=outputs/perturb/cloth3_runtime_random/demo_episode_0000/sample_0000/final_state_stable/state.npy
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
