#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding
import importlib.util


def _load_train_script_module():
    script_path = Path(__file__).resolve().parents[2] / "policy/third_party/openpi/scripts/train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load train script from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_script = _load_train_script_module()


def _prepare_lerobot_repo_link(hf_lerobot_home: Path, repo_id: str, dataset_root: Path) -> Path:
    link_path = hf_lerobot_home / repo_id
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current = link_path.resolve()
            if current != dataset_root.resolve():
                link_path.unlink()
            else:
                return link_path
        else:
            raise RuntimeError(f"Refuse to overwrite non-symlink path: {link_path}")

    os.symlink(dataset_root.resolve(), link_path)
    return link_path


def _build_config(args: argparse.Namespace) -> _config.TrainConfig:
    cfg = _config.get_config(args.config_name)

    assets_dir = (Path(args.checkpoint_dir) / str(args.checkpoint_step) / "assets").resolve()
    if not assets_dir.exists():
        raise FileNotFoundError(f"assets dir not found: {assets_dir}")

    data_factory = dataclasses.replace(
        cfg.data,
        assets=dataclasses.replace(cfg.data.assets, assets_dir=str(assets_dir)),
    )

    cfg = dataclasses.replace(
        cfg,
        data=data_factory,
        batch_size=args.batch_size or cfg.batch_size,
        num_workers=args.num_workers,
        wandb_enabled=False,
        resume=True,
        overwrite=False,
        # Keep these local and inert for this offline script.
        exp_name=f"offline_loss_{int(time.time())}",
        checkpoint_base_dir=str((Path(args.checkpoint_dir).resolve().parent.parent)),
        assets_base_dir=str((Path(args.checkpoint_dir).resolve().parent.parent / "assets_stub")),
    )

    return cfg


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute Pi0 checkpoint loss on a LeRobot dataset.")
    parser.add_argument("--config-name", default="pi0_lora_pack_sloth")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=29999)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", default="shashuo0104/xarm7_pack_sloth")
    parser.add_argument("--hf-lerobot-home", default="/tmp/root/real2sim_cache/lerobot")
    parser.add_argument("--batch-size", type=int, default=0, help="0 means use config default")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full one pass")
    parser.add_argument("--train-mode", action="store_true", help="Use train=True when computing loss")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")

    hf_lerobot_home = Path(args.hf_lerobot_home).resolve()
    link_path = _prepare_lerobot_repo_link(hf_lerobot_home, args.repo_id, dataset_root)
    os.environ["HF_LEROBOT_HOME"] = str(hf_lerobot_home)

    cfg = _build_config(args)

    if cfg.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"batch_size={cfg.batch_size} not divisible by jax.device_count={jax.device_count()}"
        )

    # Build a concrete data config first to estimate one full pass batch count.
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    torch_dataset = _data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)
    full_num_batches = len(torch_dataset) // cfg.batch_size
    if full_num_batches <= 0:
        raise ValueError(
            f"dataset too small ({len(torch_dataset)}) for batch_size={cfg.batch_size}"
        )

    num_batches = full_num_batches
    if args.max_batches > 0:
        num_batches = min(num_batches, args.max_batches)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    loader = _data_loader.create_data_loader(
        cfg,
        sharding=data_sharding,
        shuffle=False,
        num_batches=num_batches,
    )

    init_rng = jax.random.key(args.seed)
    train_state_shape, train_state_sharding = train_script.init_train_state(cfg, init_rng, mesh, resume=True)

    ckpt_mgr, _ = _checkpoints.initialize_checkpoint_dir(
        args.checkpoint_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
    )
    state = _checkpoints.restore_state(
        ckpt_mgr,
        train_state_shape,
        loader,
        step=args.checkpoint_step,
    )

    def loss_step(rng, train_state, batch):
        model = nnx.merge(train_state.model_def, train_state.params)
        if args.train_mode:
            model.train()
        else:
            model.eval()
        observation, actions = batch
        chunked_loss = model.compute_loss(rng, observation, actions, train=args.train_mode)
        return jnp.mean(chunked_loss)

    ploss_step = jax.jit(
        loss_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    losses = []
    data_iter = iter(loader)
    base_rng = jax.random.key(args.seed + 1)
    t0 = time.time()

    for i in range(num_batches):
        batch = next(data_iter)
        step_rng = jax.random.fold_in(base_rng, i)
        with sharding.set_mesh(mesh):
            loss = ploss_step(step_rng, state, batch)
        losses.append(float(jax.device_get(loss)))
        if (i + 1) % 50 == 0 or (i + 1) == num_batches:
            print(f"[{i + 1}/{num_batches}] mean_so_far={np.mean(losses):.6f} last={losses[-1]:.6f}")

    elapsed = time.time() - t0
    stats = {
        "config_name": args.config_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "checkpoint_step": args.checkpoint_step,
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "hf_lerobot_home": str(hf_lerobot_home),
        "repo_symlink": str(link_path),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "num_samples": len(torch_dataset),
        "num_batches_full": full_num_batches,
        "num_batches_used": num_batches,
        "train_mode": bool(args.train_mode),
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "elapsed_sec": elapsed,
    }

    print("===OFFLINE_LOSS_STATS===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    out_json = args.output_json
    if not out_json:
        out_json = (
            Path("log/experiments/policy_eval/reports")
            / f"offline_loss_{args.config_name}_step{args.checkpoint_step}_{int(time.time())}.json"
        )
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"SAVED_JSON {out_json}")


if __name__ == "__main__":
    main()
