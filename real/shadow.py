"""Capture current observations and predict one action chunk without motion."""

import argparse
from pathlib import Path

from .cameras import RealSensePair
from .config import ROOT, TASK_CASES, load_config, validate_hardware
from .controller import XArmPolicyController
from .logging import RunLogger
from .policy import PI0Runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_CASES), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "real/robot.local.yaml")
    parser.add_argument(
        "--model", choices=("yawonly", "single", "base"), default="yawonly"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs/real/shadow"
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config, args.task)
    validate_hardware(config)
    cameras = RealSensePair(config["cameras"])
    controller = XArmPolicyController(config["robot"])
    try:
        cameras.start(args.warmup_frames)
        controller.connect()
        controller.verify_feedback_available()
        policy = PI0Runtime(config["policy"], args.model)
        state, details = controller.read_policy_state()
        frames = cameras.capture()
        actions, seconds = policy.infer(frames, state)
        logger = RunLogger(
            args.output_root,
            "shadow",
            {
                "model_variant": args.model,
                "task": config["policy"]["task"],
                "settings": config,
                "norm_stats_sha256": policy.norm_stats_sha256,
            },
        )
        logger.images(0, frames)
        logger.event(
            {
                "type": "inference",
                "state": state.tolist(),
                "state_details": details,
                "actions": actions.tolist(),
                "inference_seconds": seconds,
                "sent_to_robot": False,
            }
        )
        print(f"Shadow output: {logger.directory}")
        print(f"First action: {actions[0].tolist()}")
    finally:
        controller.disconnect()
        cameras.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
