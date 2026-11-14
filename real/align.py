"""Overlay a public DeformGen task frame on a live or saved D455 image.

This tool only opens the configured front camera. It never connects to xArm and
cannot unlock, enable, or move the robot.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs

from .camera_controls import apply_color_controls
from .config import ROOT, load_config

TASKS = {
    "cloth": {
        "case": "cloth3",
        "config": ROOT / "real/robot.local.yaml",
    },
    "rope": {
        "case": "rope",
        "config": ROOT / "real/robot.local.yaml",
    },
    "sloth": {
        "case": "sloth",
        "config": ROOT / "real/robot.local.yaml",
    },
}
DATASET_BASE_URL = (
    "https://huggingface.co/datasets/Zili2002/DeformGen-Datasets/resolve/main"
)


def _load_front_camera(config_path: Path, task: str = "cloth") -> dict[str, Any]:
    config = load_config(config_path, task)
    try:
        camera = config["cameras"]["front"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing cameras.front in {config_path}") from error
    for key in ("serial", "model", "width", "height", "fps"):
        if key not in camera:
            raise ValueError(f"Missing cameras.front.{key} in {config_path}")
    return camera


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached public episode: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url, headers={"User-Agent": "deformgen-camera-alignment/1.0"}
    )
    print(f"Downloading public DeformGen episode to {destination}")
    try:
        with (
            urllib.request.urlopen(request, timeout=30) as response,
            partial.open("wb") as output,
        ):
            expected = int(response.headers.get("Content-Length", 0))
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                received += len(chunk)
                if expected:
                    print(
                        f"  {100.0 * received / expected:5.1f}%", end="\r", flush=True
                    )
        if expected and received != expected:
            raise RuntimeError(
                f"Incomplete download: expected {expected} bytes, received {received}"
            )
        os.replace(partial, destination)
        print(f"Downloaded {received / (1024 * 1024):.1f} MiB{' ' * 12}")
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _extract_video_frame(video_path: Path, frame_index: int, output_path: Path) -> None:
    if frame_index < 0:
        raise ValueError("--reference-frame must be non-negative")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required because the public episode uses AV1 video. "
            "Install ffmpeg, then run this command again."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
    ]
    if frame_index:
        command += ["-vf", f"select=eq(n\\,{frame_index})", "-vsync", "0"]
    command += ["-frames:v", "1", str(output_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.is_file():
        detail = result.stderr.strip() or "ffmpeg did not create an image"
        raise RuntimeError(
            f"Could not extract frame {frame_index} from {video_path}: {detail}"
        )


def _load_reference(
    args: argparse.Namespace,
    case: str,
) -> tuple[np.ndarray, Path]:
    if args.reference_image:
        reference_path = args.reference_image.expanduser().resolve()
        if not reference_path.is_file():
            raise FileNotFoundError(f"Reference image does not exist: {reference_path}")
    else:
        video_path = args.reference_video.expanduser().resolve()
        if not video_path.is_file():
            _download(args.reference_url, video_path)
        reference_path = args.cache_dir.resolve() / (
            f"{case}_yawonly_episode_{args.episode:06d}_front_"
            f"frame_{args.reference_frame:06d}.png"
        )
        if (
            not reference_path.is_file()
            or reference_path.stat().st_mtime < video_path.stat().st_mtime
        ):
            print(
                f"Extracting public episode frame {args.reference_frame}: {reference_path}"
            )
            _extract_video_frame(video_path, args.reference_frame, reference_path)

    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise RuntimeError(f"OpenCV could not read reference image: {reference_path}")
    return reference, reference_path


def _connected_devices() -> dict[str, str]:
    return {
        device.get_info(rs.camera_info.serial_number): device.get_info(
            rs.camera_info.name
        )
        for device in rs.context().query_devices()
    }


def _start_front_camera(camera: dict[str, Any]) -> rs.pipeline:
    serial = str(camera["serial"])
    connected = _connected_devices()
    if serial not in connected:
        available = ", ".join(f"{value} (SN {key})" for key, value in connected.items())
        raise RuntimeError(
            f"Configured front D455 SN {serial} is missing; connected: {available}"
        )

    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_device(serial)
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
            f"Cannot start front D455 SN {serial}. Close RealSense Viewer and any other camera "
            f"script, then retry. Original error: {error}"
        ) from error
    try:
        apply_color_controls(profile, camera, "front")
    except Exception:
        pipeline.stop()
        raise
    return pipeline


def _read_color(pipeline: rs.pipeline) -> np.ndarray:
    try:
        frames = pipeline.wait_for_frames(3000)
    except RuntimeError as error:
        raise RuntimeError(
            "Timed out waiting for the front D455. Close other camera processes or power-cycle "
            f"the camera. Original error: {error}"
        ) from error
    color = frames.get_color_frame()
    if not color:
        raise RuntimeError("The front D455 returned no color frame")
    return np.asanyarray(color.get_data()).copy()


def _grid(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    color = (0, 220, 255)
    for fraction in (1 / 3, 1 / 2, 2 / 3):
        x = round(width * fraction)
        y = round(height * fraction)
        thickness = 2 if fraction == 1 / 2 else 1
        cv2.line(image, (x, 0), (x, height - 1), color, thickness, cv2.LINE_AA)
        cv2.line(image, (0, y), (width - 1, y), color, thickness, cv2.LINE_AA)


def _label(image: np.ndarray, text: str) -> None:
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _fit_width(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or image.shape[1] <= max_width:
        return image
    scale = max_width / image.shape[1]
    return cv2.resize(
        image,
        (max_width, round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _compose(
    live: np.ndarray,
    reference: np.ndarray,
    live_alpha: float,
    mirror_reference: bool,
    show_grid: bool,
    triptych: bool,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    fitted_reference = cv2.resize(
        reference, (live.shape[1], live.shape[0]), interpolation=cv2.INTER_AREA
    )
    if mirror_reference:
        fitted_reference = cv2.flip(fitted_reference, 1)
    overlay = cv2.addWeighted(live, live_alpha, fitted_reference, 1.0 - live_alpha, 0.0)

    if show_grid:
        _grid(overlay)
    _label(
        overlay,
        f"OVERLAY | live {live_alpha:.0%} + reference {1.0 - live_alpha:.0%} | {fps:.1f} FPS",
    )
    if not triptych:
        return overlay, fitted_reference

    reference_panel = fitted_reference.copy()
    live_panel = live.copy()
    if show_grid:
        _grid(reference_panel)
        _grid(live_panel)
    _label(reference_panel, "PUBLIC EPISODE | FRONT | REFERENCE FRAME")
    _label(live_panel, "LIVE | CONFIGURED FRONT D455")
    return np.hstack((reference_panel, overlay, live_panel)), fitted_reference


def _save_snapshot(
    output_dir: Path,
    live: np.ndarray,
    reference: np.ndarray,
    overlay: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = {
        "live": output_dir / f"{stamp}_live_d455.png",
        "reference": output_dir / f"{stamp}_reference.png",
        "overlay": output_dir / f"{stamp}_overlay.png",
    }
    for name, path in paths.items():
        image = {"live": live, "reference": reference, "overlay": overlay}[name]
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to save {path}")
    print("Saved: " + ", ".join(str(path) for path in paths.values()))


def _save_offline_comparison(
    output_prefix: Path,
    live: np.ndarray,
    reference: np.ndarray,
    live_alpha: float,
) -> list[Path]:
    fitted_reference = cv2.resize(
        reference,
        (live.shape[1], live.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    clean_overlay = cv2.addWeighted(
        live,
        live_alpha,
        fitted_reference,
        1.0 - live_alpha,
        0.0,
    )
    triptych, _ = _compose(
        live,
        reference,
        live_alpha,
        mirror_reference=False,
        show_grid=True,
        triptych=True,
        fps=0.0,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths_and_images = [
        (
            output_prefix.with_name(output_prefix.name + "_reference.png"),
            fitted_reference,
        ),
        (output_prefix.with_name(output_prefix.name + "_overlay.png"), clean_overlay),
        (output_prefix.with_name(output_prefix.name + "_triptych.png"), triptych),
    ]
    for path, image in paths_and_images:
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Failed to save {path}")
    paths = [path for path, _ in paths_and_images]
    print("Saved: " + ", ".join(str(path) for path in paths))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-url")
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument(
        "--reference-image", type=Path, help="use a local image instead of the video"
    )
    parser.add_argument("--reference-frame", type=int, default=0)
    parser.add_argument(
        "--live-image",
        type=Path,
        help="overlay a saved D455 image and exit instead of opening the live camera",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="offline output prefix (default: <snapshot-dir>/<timestamp>)",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--live-alpha", type=float, default=0.5)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-display-width", type=int, default=1800)
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="capture N live frames without opening a window, then exit (diagnostics)",
    )
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("--episode must be non-negative")
    if not 0.0 <= args.live_alpha <= 1.0:
        parser.error("--live-alpha must be between 0 and 1")

    task = TASKS[args.task]
    case = str(task["case"])
    config_path = (
        (Path(task["config"]) if args.config is None else args.config)
        .expanduser()
        .resolve()
    )
    output_dir = (
        ROOT / "outputs/hardware_diagnostics" / f"{args.task}_reference_alignment"
    )
    prepared_video = (
        ROOT
        / "outputs/reference"
        / f"deformgen_{case}_initial_states_yawonly"
        / "videos"
        / f"episode_{args.episode:06d}.mp4"
    )
    args.reference_video = (
        prepared_video
        if args.reference_video is None and prepared_video.is_file()
        else (
            output_dir / f"{case}_yawonly_episode_{args.episode:06d}_front.mp4"
            if args.reference_video is None
            else args.reference_video
        )
    )
    args.reference_url = args.reference_url or (
        f"{DATASET_BASE_URL}/{case}/train1000_yawonly/videos/chunk-000/"
        f"observation.images.front/episode_{args.episode:06d}.mp4"
    )
    args.cache_dir = output_dir if args.cache_dir is None else args.cache_dir
    args.snapshot_dir = (
        output_dir / "snapshots" if args.snapshot_dir is None else args.snapshot_dir
    )

    reference, reference_path = _load_reference(args, case)
    print(
        f"Reference image: {reference_path} ({reference.shape[1]}x{reference.shape[0]})"
    )

    if args.live_image:
        live_path = args.live_image.expanduser().resolve()
        if not live_path.is_file():
            raise FileNotFoundError(f"Live D455 image does not exist: {live_path}")
        live = cv2.imread(str(live_path), cv2.IMREAD_COLOR)
        if live is None:
            raise RuntimeError(f"OpenCV could not read live D455 image: {live_path}")
        output_prefix = args.output_prefix
        if output_prefix is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_prefix = args.snapshot_dir / stamp
        _save_offline_comparison(
            output_prefix.expanduser().resolve(),
            live,
            reference,
            args.live_alpha,
        )
        return 0

    camera = _load_front_camera(config_path, args.task)
    print(
        f"Opening configured front camera: {camera['model']} SN {camera['serial']} "
        f"at {camera['width']}x{camera['height']}@{camera['fps']}"
    )

    pipeline = _start_front_camera(camera)
    try:
        live = np.empty((0, 0, 3), dtype=np.uint8)
        for _ in range(max(args.warmup_frames, 0)):
            live = _read_color(pipeline)

        if args.headless_frames > 0:
            started = time.perf_counter()
            for _ in range(args.headless_frames):
                live = _read_color(pipeline)
            elapsed = time.perf_counter() - started
            print(
                f"Captured {args.headless_frames} live frames from D455 SN {camera['serial']} "
                f"in {elapsed:.2f}s; reference decode and camera test passed."
            )
            return 0

        window_name = (
            f"DeformGen {args.task} reference | episode {args.episode:06d} "
            f"frame {args.reference_frame} + live D455"
        )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        live_alpha = args.live_alpha
        mirror_reference = False
        show_grid = True
        triptych = False
        fullscreen = False
        fps = 0.0
        frame_count = 0
        fps_started = time.perf_counter()

        print(
            "Controls: q/Esc quit | [/ ] reference/live opacity | v overlay/triptych | "
            "m mirror reference | g grid | s save | f fullscreen"
        )
        while True:
            live = _read_color(pipeline)
            frame_count += 1
            now = time.perf_counter()
            if now - fps_started >= 1.0:
                fps = frame_count / (now - fps_started)
                frame_count = 0
                fps_started = now

            display, fitted_reference = _compose(
                live,
                reference,
                live_alpha,
                mirror_reference,
                show_grid,
                triptych,
                fps,
            )
            cv2.imshow(window_name, _fit_width(display, args.max_display_width))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("["):
                live_alpha = max(0.0, round(live_alpha - 0.05, 2))
            elif key == ord("]"):
                live_alpha = min(1.0, round(live_alpha + 0.05, 2))
            elif key == ord("v"):
                triptych = not triptych
            elif key == ord("m"):
                mirror_reference = not mirror_reference
            elif key == ord("g"):
                show_grid = not show_grid
            elif key == ord("s"):
                clean_overlay = cv2.addWeighted(
                    live, live_alpha, fitted_reference, 1.0 - live_alpha, 0.0
                )
                _save_snapshot(
                    args.snapshot_dir.resolve(), live, fitted_reference, clean_overlay
                )
            elif key == ord("f"):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
