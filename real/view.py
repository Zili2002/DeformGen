"""Show the configured front and wrist RealSense streams side by side.

This script only talks to the cameras. It does not import the xArm SDK and
cannot enable or move the robot.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from .camera_controls import apply_color_controls
from .config import ROOT, load_config

DEFAULT_CONFIG = ROOT / "real/robot.local.yaml"
DEFAULT_SNAPSHOT_DIR = ROOT / "outputs/hardware_diagnostics/camera_alignment"
WINDOW_NAME = "DeformGen camera alignment | front D455 + wrist D405"


def _load_cameras(config_path: Path, task: str = "cloth") -> dict[str, dict[str, Any]]:
    config = load_config(config_path, task)
    cameras = config.get("cameras", {})
    if set(cameras) != {"front", "wrist"}:
        raise ValueError("Config must contain exactly cameras.front and cameras.wrist")
    return cameras


def _connected_devices() -> dict[str, str]:
    devices: dict[str, str] = {}
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices[serial] = name
    return devices


def _start_camera(camera: dict[str, Any], role: str) -> rs.pipeline:
    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_device(str(camera["serial"]))
    stream_config.enable_stream(
        rs.stream.color,
        int(camera["width"]),
        int(camera["height"]),
        rs.format.bgr8,
        int(camera["fps"]),
    )
    try:
        profile = pipeline.start(stream_config)
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot start {role} camera {camera['model']} SN {camera['serial']}. "
            "Close any RealSense Viewer, rollout, recording, or calibration process that is "
            f"using this camera. Original error: {error}"
        ) from error
    try:
        apply_color_controls(profile, camera, role)
    except Exception:
        pipeline.stop()
        raise
    return pipeline


def _read_color(pipeline: rs.pipeline, role: str) -> np.ndarray:
    try:
        frames = pipeline.wait_for_frames(3000)
    except RuntimeError as error:
        raise RuntimeError(
            f"Timed out waiting for the {role} camera. If the USB cable is secure, "
            "power-cycle that camera or reset it with librealsense before retrying. "
            f"Original error: {error}"
        ) from error
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError(f"No color frame received from {role}")
    return np.asanyarray(color.get_data()).copy()


def _draw_alignment_overlay(
    frame: np.ndarray,
    role: str,
    camera: dict[str, Any],
    fps: float,
    show_grid: bool,
) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]

    if show_grid:
        color = (0, 220, 255)
        for fraction in (1 / 3, 1 / 2, 2 / 3):
            x = round(width * fraction)
            y = round(height * fraction)
            thickness = 2 if fraction == 1 / 2 else 1
            cv2.line(display, (x, 0), (x, height - 1), color, thickness, cv2.LINE_AA)
            cv2.line(display, (0, y), (width - 1, y), color, thickness, cv2.LINE_AA)
        cv2.circle(display, (width // 2, height // 2), 10, color, 2, cv2.LINE_AA)

    label = (
        f"{role.upper()} | {camera['model']} | SN {camera['serial']} | "
        f"{width}x{height} | {fps:.1f} FPS"
    )
    cv2.rectangle(display, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        display,
        label,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def _fit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    size = (max_width, round(image.shape[0] * scale))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _save_snapshot(
    frames: dict[str, np.ndarray],
    cameras: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths: list[Path] = []
    for role, frame in frames.items():
        path = output_dir / f"{stamp}_{role}_{cameras[role]['serial']}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to save {path}")
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", choices=("cloth", "rope", "sloth"), default="cloth")
    parser.add_argument(
        "--list", action="store_true", help="list connected camera SDK serials and exit"
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--max-display-width",
        type=int,
        default=1600,
        help="resize only the combined display; saved frames remain native resolution",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="discard initial auto-exposure frames",
    )
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="capture N frames without opening a window, then exit (diagnostics)",
    )
    args = parser.parse_args()

    if args.list:
        for serial, model in _connected_devices().items():
            print(f"{model}: {serial}")
        return 0

    cameras = _load_cameras(args.config.resolve(), args.task)
    connected = _connected_devices()
    missing = [
        f"{role}={camera['serial']}"
        for role, camera in cameras.items()
        if str(camera["serial"]) not in connected
    ]
    if missing:
        available = ", ".join(
            f"{serial} ({name})" for serial, name in connected.items()
        )
        raise RuntimeError(
            f"Missing configured cameras: {', '.join(missing)}; connected: {available}"
        )

    pipelines: dict[str, rs.pipeline] = {}
    try:
        for role, camera in cameras.items():
            pipelines[role] = _start_camera(camera, role)

        frames: dict[str, np.ndarray] = {}
        for _ in range(max(args.warmup_frames, 0)):
            for role, pipeline in pipelines.items():
                frames[role] = _read_color(pipeline, role)

        if args.headless_frames > 0:
            started = time.perf_counter()
            for _ in range(args.headless_frames):
                for role, pipeline in pipelines.items():
                    frames[role] = _read_color(pipeline, role)
            elapsed = time.perf_counter() - started
            print(
                f"Captured {args.headless_frames} paired frames from "
                f"front={cameras['front']['serial']} and wrist={cameras['wrist']['serial']} "
                f"in {elapsed:.2f}s; no window opened."
            )
            return 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        show_grid = True
        fullscreen = False
        fps = 0.0
        frame_count = 0
        fps_started = time.perf_counter()

        print(
            "Controls: q/Esc quit | s save native frames | g toggle grid | f fullscreen"
        )
        while True:
            for role, pipeline in pipelines.items():
                frames[role] = _read_color(pipeline, role)

            frame_count += 1
            now = time.perf_counter()
            if now - fps_started >= 1.0:
                fps = frame_count / (now - fps_started)
                frame_count = 0
                fps_started = now

            panels = [
                _draw_alignment_overlay(
                    frames[role], role, cameras[role], fps, show_grid
                )
                for role in ("front", "wrist")
            ]
            combined = _fit_width(np.hstack(panels), args.max_display_width)
            cv2.imshow(WINDOW_NAME, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                saved = _save_snapshot(frames, cameras, args.snapshot_dir.resolve())
                print("Saved: " + ", ".join(str(path) for path in saved))
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
    finally:
        for pipeline in pipelines.values():
            pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
