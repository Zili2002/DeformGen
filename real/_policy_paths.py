"""Locate the released policy submodule without modifying its tracked files."""

import sys

from .config import ROOT


def enable_policy_imports() -> None:
    paths = (
        ROOT / "policy",
        ROOT / "policy/third_party/openpi/src",
        ROOT / "policy/third_party/openpi/packages/openpi-client/src",
        ROOT / "policy/third_party/lerobot",
    )
    if not (paths[1] / "openpi").is_dir():
        raise RuntimeError(
            "Initialize the policy submodule: git submodule update --init --recursive policy"
        )
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
