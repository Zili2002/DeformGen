# DeformGen

[Paper](https://arxiv.org/abs/2606.25939) | [Project Page](https://zili2002.github.io/DeformGen/) | [Hugging Face](https://huggingface.co/collections/Zili2002/deformgen-6a3c941e433459b0bfab97c3)

DeformGen is a dynamics-based topology augmentation framework for deformable manipulation policy learning. Starting from sparse demonstrations, it synthesizes physically plausible, topology-diverse object states and compatible robot trajectories for rope, sloth, and cloth manipulation.

1. **Dynamic topological transformation** generates and stabilizes new object states through localized physical disturbances.
2. **Deformation-field trajectory warping** adapts grasp poses and manipulation trajectories to the deformed geometry.

Generated demonstrations can be exported as LeRobot datasets for ACT, DP, SVLA, and pi0 training.

## Installation

The examples use a Linux shell and a CUDA-enabled environment.

```bash
git clone https://github.com/Zili2002/DeformGen.git DeformGen
cd DeformGen
uv venv --python=3.11
source .venv/bin/activate
uv pip install -e .

# Simulator geometry extensions.
uv pip install --no-build-isolation -e third-party/diff-gaussian-rasterization-w-depth
uv pip install -e third-party/urdfpy-0.0.22

export DEFORMGEN_ROOT=$PWD
```

For the optional policy submodule, alternative installation commands, and CLI entry points, see the [installation guide](docs/installation.md).

## Released Data and Checkpoints

Download the [LeRobot datasets](https://huggingface.co/datasets/Zili2002/DeformGen-Datasets) and [policy checkpoints](https://huggingface.co/Zili2002/DeformGen-Checkpoints) to train or evaluate policies without regenerating demonstrations:

```bash
huggingface-cli download Zili2002/DeformGen-Datasets \
  --repo-type dataset --local-dir release/DeformGen-Datasets
huggingface-cli download Zili2002/DeformGen-Checkpoints \
  --repo-type model --local-dir release/DeformGen-Checkpoints
```

Cases are `rope`, `sloth`, and `cloth3`. Directory names retain the code notation: `yawonly` = DG, `txy` = DG*, and `gridrigid` = SMG*. See [assets and released resources](docs/assets.md) for layouts, local paths, and simulation asset sources.

## Quick Start

This example generates one rope state, warps its demonstration, replays and exports a LeRobot episode, and computes success. Install the simulation assets first; downloading the released datasets above is not required for this workflow.

```bash
cd "$DEFORMGEN_ROOT"
deformgen-fetch sim-assets --case rope --repo-root "$DEFORMGEN_ROOT"

CUDA_VISIBLE_DEVICES=0 deformgen-perturb \
  --case rope \
  --mode runtime-random \
  --num-states 1 \
  --post-stabilization-steps 30 \
  --out outputs/e2e/rope_state \
  --overwrite

STATE=outputs/e2e/rope_state/demo_episode_0001/sample_0000/final_state_stable/state.npy

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

The exported dataset is under `outputs/e2e/rope_lerobot`. For other tasks, generation modes, batch jobs, and success metrics, see [data generation and replay](docs/generation.md). Training and simulation evaluation are covered in the [policy guide](docs/policies.md).

## Real-Robot Experiments

We thank [Chenhao Hong](https://github.com/Soohti) for contributing the real-robot evaluation code and providing deployment support.

We provide pi0 evaluation tools for an xArm7 with a UFACTORY gripper and front/wrist RealSense cameras, covering rope insertion, sloth packing, and cloth folding. The tools support camera alignment, no-motion inference checks, interactive rollouts, and video and command logging.

See the [real-robot guide](real/README.md) for hardware setup, calibration, and evaluation protocols.

## Documentation

| Guide | Contents |
| --- | --- |
| [Installation](docs/installation.md) | Dependencies, policy submodule, and CLI entry points |
| [Assets and released resources](docs/assets.md) | Downloads, directory layouts, local paths, and method names |
| [Data generation and replay](docs/generation.md) | State generation, trajectory warping, replay, export, and success metrics |
| [Policy training and simulation evaluation](docs/policies.md) | ACT, DP, SVLA, and pi0 workflows |
| [Real-robot evaluation](real/README.md) | xArm7 and RealSense setup, calibration, and rollout protocols |
| [Object appearance and physical parameters](docs/appearance-and-physics.md) | Color transforms, friction, and per-run overrides |

## Citation

If you find this repository helpful for your research, please cite our paper:

```bibtex
@misc{lin2026deformgen,
      title={DeformGen: Dynamics-Based Topology Augmentation for Deformable Manipulation Policy Learning},
      author={Zili Lin and Wenyao Zhang and Yuyang Zhang and Zekun Qi and Junyan Lin and Hanxin Zhu and Jiaolong Yang and Zhibo Chen and Yao Mu and Xiaokang Yang and Xin Jin and Wenjun Zeng},
      year={2026},
      eprint={2606.25939},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2606.25939},
}
```

## Acknowledgements

DeformGen builds on [real2sim-eval](https://github.com/kywind/real2sim-eval). We thank its authors for releasing their simulation, assets, and policy-evaluation infrastructure. Policy training and inference use the external `policy/` submodule.

## License

See [LICENSE](LICENSE). Third-party components and external assets have their own licenses; see the [asset guide](docs/assets.md) and linked upstream repositories.
