"""Task presets and user-owned hardware configuration."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_CASES = {"cloth": "cloth3", "rope": "rope", "sloth": "sloth"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Path, task: str = "cloth") -> dict[str, Any]:
    preset = yaml.safe_load((ROOT / "real" / "tasks" / f"{task}.yaml").read_text())
    hardware = yaml.safe_load(path.expanduser().resolve().read_text())
    config = _merge(preset, hardware)
    if config["policy"]["case"] != TASK_CASES[task]:
        raise ValueError(f"The hardware override must retain the {task!r} task")
    return config


def validate_hardware(config: dict, *, motion: bool = False) -> None:
    for role in ("front", "wrist"):
        serial = str(config["cameras"][role]["serial"])
        if not serial.isdigit():
            raise ValueError(f"Set cameras.{role}.serial in your local hardware config")
    ip = str(config["robot"]["ip"])
    if not ip or ip == "ROBOT_IP":
        raise ValueError("Set robot.ip in your local hardware config")
    if motion and config["robot"].get("calibration_confirmed") is not True:
        raise ValueError(
            "Verify the robot calibration and set robot.calibration_confirmed: true"
        )
