"""Session progress, resume validation, and recoverable interrupted-attempt archival."""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import TASK_CASES

PROTOCOL_ORDERS = {
    "paired10": ("yawonly", "base"),
    "triple10": ("yawonly", "single", "base"),
    "single10": ("single",),
}


def _read_protocol_events(events_path: Path) -> tuple[list[dict[str, Any]], int]:
    raw = events_path.read_bytes()
    normalized = raw.rstrip(b"\x00")
    trailing_null_bytes = len(raw) - len(normalized)
    if b"\x00" in normalized:
        raise ValueError(f"Protocol log contains non-trailing NUL bytes: {events_path}")
    try:
        text = normalized.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Protocol log is not valid UTF-8: {events_path}: {error}"
        ) from error
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {events_path}:{line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise ValueError(f"Protocol event is not an object at line {line_number}")
        events.append(event)
    return events, trailing_null_bytes


def _archive_interrupted_rollouts(session_dir: Path) -> list[dict[str, Any]]:
    """Move power-loss artifacts aside and remove unmatched start events."""
    events_path = session_dir / "protocol_events.jsonl"
    events, trailing_null_bytes = _read_protocol_events(events_path)
    unmatched: dict[int, dict[str, Any]] = {}
    for event in events:
        rollout_index = event.get("rollout_index")
        if event.get("type") == "rollout_started" and isinstance(rollout_index, int):
            unmatched[rollout_index] = event
        elif event.get("type") == "rollout_finished" and isinstance(rollout_index, int):
            unmatched.pop(rollout_index, None)

    if not unmatched and trailing_null_bytes == 0:
        return []

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive_root = (
        session_dir.parent / ".interrupted_rollouts" / session_dir.name / stamp
    )
    moved: list[tuple[Path, Path]] = []
    archived: list[dict[str, Any]] = []
    temp_log = events_path.with_suffix(".jsonl.recovery.tmp")
    try:
        archive_root.mkdir(parents=True, exist_ok=False)
        for rollout_index, event in sorted(unmatched.items()):
            position = event.get("position_index")
            model = event.get("model_variant")
            if not isinstance(position, int) or not isinstance(model, str):
                raise ValueError(
                    f"Interrupted rollout {rollout_index} has an invalid protocol slot"
                )
            stem = f"rollout_{rollout_index:03d}_position_{position:02d}_{model}"
            rollout_archive = archive_root / stem
            rollout_archive.mkdir()
            targets = [
                session_dir / f"{stem}.mp4",
                session_dir / stem,
                session_dir / f"position_{position:02d}_{model}_initial_front.png",
                session_dir / f"position_{position:02d}_{model}_initial_wrist.png",
            ]
            rollout_moves: list[str] = []
            for source in targets:
                if not source.exists():
                    continue
                destination = rollout_archive / source.name
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
                rollout_moves.append(str(destination))
            archived.append(
                {
                    "rollout_index": rollout_index,
                    "position_index": position,
                    "model_variant": model,
                    "archive_directory": str(rollout_archive),
                    "moved_paths": rollout_moves,
                }
            )

        unmatched_event_ids = {id(event) for event in unmatched.values()}
        retained = [event for event in events if id(event) not in unmatched_event_ids]
        retained.append(
            {
                "wall_time": time.time(),
                "type": "interrupted_rollouts_archived",
                "trailing_null_bytes_removed": trailing_null_bytes,
                "rollouts": archived,
            }
        )
        temp_log.write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in retained)
        )
        os.replace(temp_log, events_path)
    except Exception:
        temp_log.unlink(missing_ok=True)
        for source, destination in reversed(moved):
            if destination.exists():
                shutil.move(str(destination), str(source))
        raise
    return archived


def _resume_progress(session_dir: Path, protocol: str) -> dict[str, Any]:
    """Reconstruct the next protocol slot without trusting filenames alone."""
    if protocol not in PROTOCOL_ORDERS:
        raise ValueError("--resume-session requires a 10-position protocol")
    events_path = session_dir / "protocol_events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing resume event log: {events_path}")

    events, _ = _read_protocol_events(events_path)
    max_rollout_index = 0
    attempted_slots: list[tuple[int, str]] = []
    finished_indices: set[int] = set()
    for event in events:
        rollout_index = event.get("rollout_index")
        if event.get("type") in {"rollout_started", "rollout_finished"} and isinstance(
            rollout_index, int
        ):
            max_rollout_index = max(max_rollout_index, rollout_index)
        if event.get("type") != "rollout_finished":
            continue
        if not isinstance(rollout_index, int):
            raise ValueError("rollout_finished lacks integer rollout_index")
        if rollout_index in finished_indices:
            raise ValueError(f"Duplicate rollout_finished for rollout {rollout_index}")
        finished_indices.add(rollout_index)
        if event.get("attempted") is True:
            position = event.get("position_index")
            model = event.get("model_variant")
            if not isinstance(position, int) or not isinstance(model, str):
                raise ValueError("Attempted rollout has invalid protocol slot")
            attempted_slots.append((position, model))

    order = PROTOCOL_ORDERS[protocol]
    expected = [(position, model) for position in range(1, 11) for model in order]
    if attempted_slots != expected[: len(attempted_slots)]:
        raise ValueError(
            "Resume event log is not a valid protocol prefix: "
            f"observed={attempted_slots}, expected_prefix={expected[: len(attempted_slots)]}"
        )
    if len(attempted_slots) > len(expected):
        raise ValueError(
            "Resume event log contains more attempted rollouts than planned"
        )
    if len(attempted_slots) == len(expected):
        position, model = expected[-1]
        complete = True
    else:
        position, model = expected[len(attempted_slots)]
        complete = False
    return {
        "rollout_index": max_rollout_index,
        "position_index": position,
        "model_variant": model,
        "attempted_count": len(attempted_slots),
        "complete": complete,
    }


def _validate_resume_session(
    session_dir: Path,
    *,
    task: str,
    protocol: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    if protocol not in PROTOCOL_ORDERS:
        raise ValueError("--resume-session requires a 10-position protocol")
    metadata_path = session_dir / "session.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing resume metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "protocol": protocol,
        "task_key": TASK_CASES[task],
        "task_prompt": config["policy"]["task"],
        "robot_ip": str(config["robot"]["ip"]),
        "video_codec": "h264",
    }
    for key, value in expected.items():
        if str(metadata.get(key)) != str(value):
            raise ValueError(
                f"Resume session {key} mismatch: saved={metadata.get(key)!r}, "
                f"requested={value!r}"
            )
    camera_serials = {
        role: str(camera["serial"]) for role, camera in config["cameras"].items()
    }
    saved_serials = {
        role: str(serial) for role, serial in metadata.get("camera_serials", {}).items()
    }
    if saved_serials != camera_serials:
        raise ValueError(
            f"Resume camera mismatch: saved={saved_serials}, requested={camera_serials}"
        )
    if metadata.get("protocol_order") != list(PROTOCOL_ORDERS[protocol]):
        raise ValueError(
            "Resume protocol_order does not match the current implementation"
        )
    return metadata
