"""RealSense capture, recovery, latest-frame handoff, and H.264 recording."""

import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

from .camera_controls import apply_color_controls


class RealSensePair:
    def __init__(self, camera_config: dict[str, Any]):
        if set(camera_config) != {"front", "wrist"}:
            raise ValueError("Camera config must contain exactly front and wrist")
        self.config = camera_config
        self.pipelines: dict[str, rs.pipeline] = {}
        self._record_thread: threading.Thread | None = None
        self._record_stop = threading.Event()
        self._record_lock = threading.Lock()
        self._record_latest: np.ndarray | None = None
        self._record_latest_monotonic: float | None = None
        self._record_error: Exception | None = None
        self._record_path: Path | None = None
        self._record_frame_count = 0

    def _start_pipeline(self, role: str) -> rs.pipeline:
        camera = self.config[role]
        serial = str(camera["serial"])
        # Do not enumerate and interrogate every RealSense before opening the
        # requested serial. One unrelated D400 with a broken UVC extension-unit
        # state can make get_info() raise and previously prevented a healthy
        # configured camera from even reaching pipeline.start().
        pipeline = rs.pipeline()
        stream = rs.config()
        stream.enable_device(serial)
        stream.enable_stream(
            rs.stream.color,
            int(camera["width"]),
            int(camera["height"]),
            rs.format.bgr8,
            int(camera["fps"]),
        )
        try:
            profile = pipeline.start(stream)
        except RuntimeError as error:
            self._stop_pipeline(pipeline)
            raise RuntimeError(
                f"Cannot start {role} camera SN {serial}; "
                f"close camera viewers or reset that USB device: {error}"
            ) from error

        try:
            apply_color_controls(profile, camera, role)
        except Exception:
            self._stop_pipeline(pipeline)
            raise
        return pipeline

    @staticmethod
    def _stop_pipeline(pipeline: rs.pipeline) -> None:
        try:
            pipeline.stop()
        except RuntimeError:
            pass

    def _restart_pipeline(self, role: str, warmup_frames: int = 5) -> None:
        if role == "front" and self._record_thread is not None:
            raise RuntimeError("Cannot restart front camera while recording")
        previous = self.pipelines.pop(role, None)
        if previous is not None:
            self._stop_pipeline(previous)
        time.sleep(0.1)
        pipeline = self._start_pipeline(role)
        self.pipelines[role] = pipeline
        for _ in range(max(0, warmup_frames)):
            frame_set = pipeline.wait_for_frames(1000)
            if not frame_set.get_color_frame():
                raise RuntimeError(
                    f"No color frame while warming restarted {role} camera"
                )

    def start(self, warmup_frames: int) -> None:
        if self.pipelines:
            raise RuntimeError("Cameras are already started")
        for role in ("front", "wrist"):
            errors: list[str] = []
            for attempt in range(1, 4):
                try:
                    self.pipelines[role] = self._start_pipeline(role)
                    break
                except RuntimeError as error:
                    errors.append(str(error))
                    if attempt < 3:
                        time.sleep(0.5)
            else:
                self.stop()
                raise RuntimeError(
                    f"Could not start {role} camera after 3 attempts: "
                    + " | ".join(errors)
                )
        for _ in range(max(0, warmup_frames)):
            self.capture()

    def recover(self, warmup_frames: int = 5) -> None:
        if self._record_thread is not None:
            raise RuntimeError("Cannot recover camera pipelines while recording")
        for pipeline in self.pipelines.values():
            self._stop_pipeline(pipeline)
        self.pipelines.clear()
        time.sleep(0.2)
        self.start(warmup_frames)

    def capture(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for role in ("front", "wrist"):
            if role == "front" and self._record_thread is not None:
                deadline = time.monotonic() + 1.5
                while True:
                    with self._record_lock:
                        error = self._record_error
                        latest = self._record_latest
                        latest_monotonic = self._record_latest_monotonic
                    if error is not None:
                        raise RuntimeError(
                            f"Front-camera recording failed: {error}"
                        ) from error
                    if (
                        latest is not None
                        and latest_monotonic is not None
                        and time.monotonic() - latest_monotonic <= 1.0
                    ):
                        frames[role] = latest.copy()
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Front-camera recorder has not produced a fresh frame "
                            "for more than one second"
                        )
                    time.sleep(0.01)
                continue
            try:
                frame_set = self.pipelines[role].wait_for_frames(1000)
            except RuntimeError as first_error:
                try:
                    self._restart_pipeline(role)
                    frame_set = self.pipelines[role].wait_for_frames(1000)
                except RuntimeError as recovery_error:
                    raise RuntimeError(
                        f"Timed out waiting for {role} camera and pipeline "
                        f"restart failed: initial={first_error}; "
                        f"recovery={recovery_error}"
                    ) from recovery_error
            color = frame_set.get_color_frame()
            if not color:
                raise RuntimeError(f"No color frame from {role} camera")
            frames[role] = np.asanyarray(color.get_data()).copy()
        return frames

    def start_front_recording(self, output_path: Path) -> None:
        if self._record_thread is not None:
            raise RuntimeError("Front-camera recording is already running")
        if "front" not in self.pipelines:
            raise RuntimeError("Start cameras before starting front-camera recording")
        camera = self.config["front"]
        width, height, fps = (
            int(camera["width"]),
            int(camera["height"]),
            int(camera["fps"]),
        )
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if ffmpeg.stdin is None:
            ffmpeg.kill()
            raise RuntimeError("Could not open ffmpeg stdin for H.264 recording")

        self._record_stop.clear()
        self._record_latest = None
        self._record_latest_monotonic = None
        self._record_error = None
        self._record_path = output_path
        self._record_frame_count = 0

        def record() -> None:
            consecutive_timeouts = 0
            try:
                while not self._record_stop.is_set():
                    try:
                        frame_set = self.pipelines["front"].wait_for_frames(1000)
                    except RuntimeError as error:
                        if self._record_stop.is_set():
                            break
                        consecutive_timeouts += 1
                        if consecutive_timeouts < 3:
                            continue
                        raise RuntimeError(
                            "Front D455 timed out for three consecutive seconds"
                        ) from error
                    consecutive_timeouts = 0
                    color = frame_set.get_color_frame()
                    if not color:
                        raise RuntimeError(
                            "Front D455 returned no color frame while recording"
                        )
                    frame = np.asanyarray(color.get_data()).copy()
                    if frame.shape[:2] != (height, width):
                        raise RuntimeError(
                            f"Unexpected front frame shape {frame.shape}; expected {height}x{width}"
                        )
                    ffmpeg.stdin.write(frame.tobytes())
                    with self._record_lock:
                        self._record_latest = frame
                        self._record_latest_monotonic = time.monotonic()
                        self._record_frame_count += 1
            except Exception as error:
                with self._record_lock:
                    self._record_error = error
            finally:
                try:
                    ffmpeg.stdin.close()
                except Exception:
                    pass
                try:
                    return_code = ffmpeg.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    ffmpeg.kill()
                    return_code = ffmpeg.wait()
                if return_code != 0:
                    stderr = (
                        ffmpeg.stderr.read().decode("utf-8", errors="replace")
                        if ffmpeg.stderr is not None
                        else ""
                    )
                    with self._record_lock:
                        if self._record_error is None:
                            self._record_error = RuntimeError(
                                f"ffmpeg H.264 encoder exited {return_code}: {stderr}"
                            )

        self._record_thread = threading.Thread(
            target=record,
            name="front-d455-recorder",
            daemon=True,
        )
        self._record_thread.start()

    def stop_front_recording(self) -> dict[str, Any] | None:
        thread = self._record_thread
        if thread is None:
            return None
        self._record_stop.set()
        thread.join(timeout=12.0)
        if thread.is_alive():
            raise RuntimeError("Front-camera recorder did not stop within 12 seconds")
        with self._record_lock:
            error = self._record_error
            frame_count = self._record_frame_count
        path = self._record_path
        self._record_thread = None
        self._record_path = None
        if error is not None:
            raise RuntimeError(f"Front-camera recording failed: {error}") from error
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        codec = probe.stdout.strip()
        if codec != "h264":
            raise RuntimeError(
                f"Front video codec verification failed: expected h264, got {codec!r}"
            )
        return {
            "path": None if path is None else str(path),
            "frame_count": frame_count,
            "fps": int(self.config["front"]["fps"]),
            "codec": codec,
        }

    def stop(self) -> None:
        recording_error: Exception | None = None
        try:
            self.stop_front_recording()
        except Exception as error:
            recording_error = error
        for pipeline in self.pipelines.values():
            self._stop_pipeline(pipeline)
        self.pipelines.clear()
        if recording_error is not None:
            raise recording_error


class FrameStore:
    """Thread-safe latest-frame handoff; only the GUI thread reads RealSense."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray] | None = None

    def update(self, frames: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._frames = {role: frame.copy() for role, frame in frames.items()}

    def get(self) -> dict[str, np.ndarray]:
        with self._lock:
            if self._frames is None:
                raise RuntimeError("No paired camera frame is available yet")
            return {role: frame.copy() for role, frame in self._frames.items()}
