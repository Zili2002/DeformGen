"""Per-rollout metadata, raw observations, and JSONL command logs."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class RunLogger:
    def __init__(
        self,
        output_root: Path,
        mode: str,
        metadata: dict[str, Any],
        *,
        directory: Path | None = None,
    ):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = (
            output_root.resolve() / f"{stamp}_{mode}"
            if directory is None
            else directory.resolve()
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "images").mkdir()
        (self.directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
        )
        self.events_path = self.directory / "events.jsonl"

    def event(self, payload: dict[str, Any]) -> None:
        record = {"wall_time": time.time(), **payload}
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def images(self, inference_index: int, frames: dict[str, np.ndarray]) -> None:
        for role, frame in frames.items():
            path = (
                self.directory
                / "images"
                / f"inference_{inference_index:04d}_{role}.jpg"
            )
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Could not save {path}")
