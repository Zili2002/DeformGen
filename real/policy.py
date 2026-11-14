"""Stateless, full-horizon pi0 inference using the released OpenPI transforms."""

import hashlib
import time
from typing import Any

import cv2
import numpy as np

from .config import resolve_path


def select_policy_variant(
    policy_config: dict[str, Any], model_variant: str
) -> tuple[str, dict]:
    if model_variant == "yawonly":
        selected = policy_config
    elif model_variant == "single":
        selected = policy_config["single_model"]
    elif model_variant == "base":
        selected = policy_config["base_model"]
    else:
        raise ValueError(f"Unknown model: {model_variant}")
    return model_variant, selected


def build_train_config(task: str, *, base: bool = False):
    from ._policy_paths import enable_policy_imports

    enable_policy_imports()
    from openpi.models import pi0
    from openpi.training import config

    model = (
        pi0.Pi0Config()
        if base
        else pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        )
    )
    return config.TrainConfig(
        name=f"pi0_{'base' if base else 'lora'}_{task}",
        model=model,
        data=config.LeRobotXarmDataConfig(
            repo_id=f"shashuo0104/xarm7_{task}",
            base_config=config.DataConfig(
                prompt_from_task=True, action_sequence_keys=("action",)
            ),
            use_delta_joint_actions=True,
        ),
    )


class PI0Runtime:
    def __init__(self, policy_config: dict[str, Any], model_variant: str = "yawonly"):
        from ._policy_paths import enable_policy_imports

        enable_policy_imports()
        import jax
        from inference.utils import InferenceTransform
        from openpi.policies import policy_config as policy_factory
        from openpi.training import checkpoints

        canonical, selected = select_policy_variant(policy_config, model_variant)
        self.model_variant = canonical
        self.augmentation_mode = None if canonical == "base" else canonical
        self.checkpoint_step = selected.get("checkpoint_step")
        self.checkpoint = resolve_path(selected["local_dir"])
        self.norm_stats_checkpoint = resolve_path(
            selected.get("norm_stats_dir", selected["local_dir"])
        )
        self.task = str(policy_config["task"])
        self.action_dim = int(policy_config["action_dim"])
        self.action_horizon = int(policy_config["action_horizon"])
        self.seed = int(policy_config.get("inference_seed", 0))
        config = build_train_config(self.task, base=canonical == "base")
        self.training_config = config.name
        if config.model.action_horizon != self.action_horizon:
            raise ValueError(
                "The pi0 checkpoint requires the full 50-step action horizon"
            )
        asset_id = f"shashuo0104/xarm7_{self.task}"
        stats_path = (
            self.norm_stats_checkpoint / "assets" / asset_id / "norm_stats.json"
        )
        self.norm_stats_sha256 = hashlib.sha256(stats_path.read_bytes()).hexdigest()
        self.image_transform = InferenceTransform(
            policy_config["image_preprocessing"], policy_type="openpi", device="cpu"
        )
        started = time.perf_counter()
        stats = checkpoints.load_norm_stats(
            self.norm_stats_checkpoint / "assets", asset_id
        )
        self.policy = policy_factory.create_trained_policy(
            config, self.checkpoint, norm_stats=stats
        )
        self.policy._rng = jax.random.key(self.seed)
        self.load_seconds = time.perf_counter() - started

    def reset(self) -> None:
        import jax

        self.policy._rng = jax.random.key(self.seed)

    def infer(
        self, frames: dict[str, np.ndarray], state: np.ndarray
    ) -> tuple[np.ndarray, float]:
        import torch

        observation = {
            "observation.state": np.asarray(state, dtype=np.float32).copy(),
            "task": self.task,
        }
        for role, camera_type in (("front", "fixed"), ("wrist", "wrist")):
            rgb = cv2.cvtColor(frames[role], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            transformed = self.image_transform(tensor, camera_type=camera_type)
            observation[f"observation.images.{role}"] = (
                transformed[0].permute(1, 2, 0).numpy()
            )
        started = time.perf_counter()
        actions = np.asarray(self.policy.infer(observation)["action"], dtype=np.float64)
        elapsed = time.perf_counter() - started
        if actions.shape != (self.action_horizon, self.action_dim) or not np.all(
            np.isfinite(actions)
        ):
            raise ValueError(
                f"Expected finite ({self.action_horizon}, {self.action_dim}) actions; got {actions.shape}"
            )
        return actions, elapsed
