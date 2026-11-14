"""Download only inference parameters and assets at the task's pinned revision."""

import argparse
import shutil

from huggingface_hub import snapshot_download

from .config import ROOT, TASK_CASES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_CASES), required=True)
    parser.add_argument(
        "--models",
        choices=("yawonly", "single"),
        nargs="+",
        default=["yawonly", "single"],
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="also download the untouched official pi0 base",
    )
    parser.add_argument(
        "--revision", help="override the task's pinned checkpoint revision"
    )
    args = parser.parse_args()
    import yaml

    policy = yaml.safe_load((ROOT / "real/tasks" / f"{args.task}.yaml").read_text())[
        "policy"
    ]
    case = policy["case"]
    patterns = []
    for model in args.models:
        prefix = f"{case}/pi0/{model}"
        patterns.extend(
            [
                f"{prefix}/checkpoint/params/*",
                f"{prefix}/checkpoint/assets/*",
                f"{prefix}/metadata.json",
            ]
        )
    snapshot_download(
        policy["repo_id"],
        revision=args.revision or policy["revision"],
        allow_patterns=patterns,
        local_dir=ROOT / "release/DeformGen-Checkpoints",
    )
    if args.base:
        from ._policy_paths import enable_policy_imports

        enable_policy_imports()
        from openpi.shared.download import maybe_download

        source = maybe_download("gs://openpi-assets/checkpoints/pi0_base")
        destination = ROOT / policy["base_model"]["local_dir"]
        shutil.copytree(source, destination, dirs_exist_ok=True)
        print(f"Base checkpoint: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
