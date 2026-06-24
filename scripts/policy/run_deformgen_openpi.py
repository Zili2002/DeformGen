#!/usr/bin/env python3
"""Run an OpenPI script with DeformGen's policy submodule first on sys.path."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _is_other_openpi_path(path: str, own_openpi_root: Path) -> bool:
    """Return whether a path points to a different policy OpenPI source tree."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    return (
        resolved.name == "src"
        and resolved.parent.name == "openpi"
        and resolved.parent.parent.name == "third_party"
        and resolved.parent.parent.parent.name == "policy"
        and resolved != own_openpi_root.resolve()
    )


def main() -> None:
    """Execute the requested OpenPI script against the local DeformGen submodule."""
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_deformgen_openpi.py <openpi_script.py> [script_args...]")

    repo_root = Path(__file__).resolve().parents[2]
    policy_root = repo_root / "policy"
    own_openpi_root = policy_root / "third_party" / "openpi" / "src"
    lerobot_root = policy_root / "third_party" / "lerobot"
    required = [own_openpi_root, lerobot_root]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing policy submodule paths: {missing}")
    preferred = [own_openpi_root, lerobot_root]
    optional_policy_src = policy_root / "src"
    if optional_policy_src.exists():
        preferred.insert(1, optional_policy_src)

    # Shared environments can inject another checkout's OpenPI package. Remove only
    # those foreign policy OpenPI roots; never depend on a machine-specific path.
    sys.path[:] = [path for path in sys.path if not _is_other_openpi_path(path, own_openpi_root)]
    for path in reversed([str(path) for path in preferred]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    script = Path(sys.argv[1]).resolve()
    if not script.is_file():
        raise FileNotFoundError(f"OpenPI script does not exist: {script}")
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
