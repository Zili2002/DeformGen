"""Interactive dual-camera console for DeformGen pi0 real-robot rollouts.

Keys:
  c  start one 900-step rollout from the verified reset pose
  s  stop an active rollout and leave xArm enabled/holding
  r  release, lift, and return xArm to the common DeformGen reset pose
  Delete (or d)  remove the previous rollout and repeat its protocol slot
  q  stop if needed, then close the window

With triple10, each of ten task initial states is evaluated in the fixed order
``yawonly -> single -> base`` (30 commanded rollouts total).  The two exact
confirmation tokens authorize this interactive process. Motion
still requires an explicit ``c`` or ``r`` key press, clean xArm feedback, and
the existing reset-pose check.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cameras import FrameStore, RealSensePair
from .config import ROOT, TASK_CASES, load_config, validate_hardware
from .controller import (
    MOTION_CONFIRMATION,
    PI0_CONTINUOUS_CONFIRMATION,
    SafetyViolation,
    XArmPolicyController,
    require_motion_authorization,
)
from .logging import RunLogger
from .policy import PI0Runtime
from .protocol import (
    PROTOCOL_ORDERS,
    _archive_interrupted_rollouts,
    _resume_progress,
    _validate_resume_session,
)
from .reset import reset_robot
from .view import _draw_alignment_overlay, _fit_width

WINDOW_NAME = "DeformGen interactive | D455 front + D405 wrist"
DELETE_KEY_CODES = {8, 127, 0xFFFF, 0xFF9F, 0x2E0000}


class InteractiveSession:
    def __init__(
        self,
        config_path: Path,
        config: dict[str, Any],
        policies: dict[str, PI0Runtime],
        cameras: RealSensePair,
        frames: FrameStore,
        session_dir: Path,
        max_steps: int,
        protocol: str,
        reference_session: Path | None = None,
        resume_progress: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = config
        self.robot_config = config["robot"]
        self.policy_config = config["policy"]
        self.policies = policies
        self.cameras = cameras
        self.frames = frames
        self.session_dir = session_dir
        self.max_steps = max_steps
        self.protocol = protocol
        self.reference_session = (
            reference_session.resolve() if reference_session is not None else None
        )
        self._protocol_order = PROTOCOL_ORDERS.get(protocol, tuple(policies))
        self._position_protocol = protocol in PROTOCOL_ORDERS
        self._rollout_index = 0
        self._pair_index = 1
        self._pair_stage = self._protocol_order[0]
        self._pair_count = 10
        self._protocol_complete = False
        self._reference_frames: dict[str, np.ndarray] | None = None
        self._active_model_variant = ""
        self._active_pair_index: int | None = None
        self._active_attempted = False
        self._active_total_steps = 0
        self._protocol_events_path = session_dir / "protocol_events.jsonl"

        if resume_progress is not None:
            self._rollout_index = int(resume_progress["rollout_index"])
            self._pair_index = int(resume_progress["position_index"])
            self._pair_stage = str(resume_progress["model_variant"])
            self._protocol_complete = bool(resume_progress["complete"])

        self._lock = threading.Lock()
        self._activity = "idle"
        if self._protocol_complete:
            commanded = self._pair_count * len(self._protocol_order)
            self._status = f"{self.protocol.upper()} COMPLETE | {commanded} commanded attempts | q exits"
        elif self._position_protocol:
            self._status = (
                f"POSITION {self._pair_index}/10 | r reset, then place "
                f"{self.policy_config['case']}, "
                f"c runs {self._pair_stage}"
            )
        else:
            only_policy = next(iter(policies.values()))
            self._status = f"IDLE | {only_policy.model_variant} | r reset, c rollout"
        self._status_kind = "complete" if self._protocol_complete else "idle"
        self._worker: threading.Thread | None = None
        self._worker_done = False
        self._worker_outcome = ""
        self._worker_kind = ""
        self._stop = threading.Event()
        self._logger: RunLogger | None = None
        self._recording = False
        self._external_reference_frames: dict[int, dict[str, np.ndarray]] | None = None
        if self.reference_session is not None and not self._protocol_complete:
            if not self._position_protocol:
                raise ValueError("--reference-session requires a 10-position protocol")
            self._external_reference_frames = self._load_external_references()
            self._activate_external_reference(self._pair_index)
            self._status = (
                f"POSITION {self._pair_index}/10 | r reset, align "
                f"{self.policy_config['case']} "
                f"to saved overlay, c runs {self._pair_stage}"
            )
        elif (
            resume_progress is not None
            and not self._protocol_complete
            and self._position_protocol
            and self._pair_stage != self._protocol_order[0]
        ):
            self._load_current_session_reference(self._pair_index)
            self._status = (
                f"RESUMED POSITION {self._pair_index}/10 | restore SAME "
                f"{self.policy_config['case']} with overlay, r then c runs "
                f"{self._pair_stage}"
            )
        elif resume_progress is not None and not self._protocol_complete:
            self._status = (
                f"RESUMED POSITION {self._pair_index}/10 | r reset, then place NEW "
                f"{self.policy_config['case']} state, c runs {self._pair_stage}"
            )
        if resume_progress is not None:
            self._protocol_event(
                {
                    "type": "session_resumed",
                    **resume_progress,
                }
            )

    def _load_current_session_reference(self, position: int) -> None:
        reference: dict[str, np.ndarray] = {}
        for role in ("front", "wrist"):
            path = self.session_dir / f"position_{position:02d}_reference_{role}.png"
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(
                    f"Cannot resume mid-position without saved reference: {path}"
                )
            reference[role] = frame
        with self._lock:
            self._reference_frames = reference

    def _load_external_references(self) -> dict[int, dict[str, np.ndarray]]:
        assert self.reference_session is not None
        references: dict[int, dict[str, np.ndarray]] = {}
        for position in range(1, self._pair_count + 1):
            references[position] = {}
            for role in ("front", "wrist"):
                path = (
                    self.reference_session
                    / f"position_{position:02d}_reference_{role}.png"
                )
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise FileNotFoundError(
                        f"Missing or unreadable reference image: {path}"
                    )
                references[position][role] = frame
        return references

    def _activate_external_reference(self, position: int) -> None:
        assert self._external_reference_frames is not None
        reference = self._external_reference_frames[position]
        with self._lock:
            self._reference_frames = {
                role: frame.copy() for role, frame in reference.items()
            }
        for role, frame in reference.items():
            destination = (
                self.session_dir / f"position_{position:02d}_reference_{role}.png"
            )
            if not cv2.imwrite(str(destination), frame):
                raise RuntimeError(f"Could not save {destination}")
        self._protocol_event(
            {
                "type": "external_reference_activated",
                "position_index": position,
                "source_session": str(self.reference_session),
            }
        )

    def _protocol_event(self, payload: dict[str, Any]) -> None:
        record = {"wall_time": time.time(), **payload}
        with self._protocol_events_path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _next_model_variant(self) -> str:
        if self._position_protocol:
            return self._pair_stage
        return next(iter(self.policies))

    def reference_frames_for_display(self) -> dict[str, np.ndarray] | None:
        with self._lock:
            if (
                not self._position_protocol
                or self._activity == "rollout"
                or self._reference_frames is None
            ):
                return None
            return {
                role: frame.copy() for role, frame in self._reference_frames.items()
            }

    def display_state(self) -> tuple[str, str, str]:
        with self._lock:
            return self._activity, self._status, self._status_kind

    def _set_status(self, text: str, kind: str | None = None) -> None:
        with self._lock:
            self._status = text
            if kind is not None:
                self._status_kind = kind
        print(text, flush=True)

    def _claim_activity(self, kind: str, status: str) -> bool:
        with self._lock:
            if self._activity != "idle":
                return False
            self._activity = kind
            self._status = status
            self._status_kind = kind
            self._worker_done = False
            self._worker_outcome = ""
            self._worker_kind = kind
            self._active_attempted = False
            self._active_total_steps = 0
            self._stop.clear()
            return True

    def _finish_worker(
        self,
        outcome: str,
        status_kind: str,
        *,
        attempted: bool = False,
        total_steps: int = 0,
    ) -> None:
        with self._lock:
            self._worker_outcome = outcome
            self._status_kind = status_kind
            self._active_attempted = attempted
            self._active_total_steps = total_steps
            self._worker_done = True

    def start_rollout(self) -> None:
        if self._protocol_complete:
            self._set_status(f"{self.protocol.upper()} COMPLETE | q exits", "complete")
            return
        model_variant = self._next_model_variant()
        policy = self.policies[model_variant]
        if not self._claim_activity(
            "rollout",
            f"ROLLOUT | {model_variant} | preparing log and H.264 video",
        ):
            self._set_status("IGNORED c | another operation is active", "warning")
            return
        self._rollout_index += 1
        pair_index = self._pair_index if self._position_protocol else None
        self._active_model_variant = model_variant
        self._active_pair_index = pair_index
        rollout_name = f"rollout_{self._rollout_index:03d}"
        if pair_index is not None:
            rollout_name += f"_position_{pair_index:02d}_{model_variant}"
        metadata = {
            "policy": "pi0",
            "model_variant": model_variant,
            "protocol": self.protocol,
            "protocol_order": (
                list(self._protocol_order) if self._position_protocol else None
            ),
            "position_index": pair_index,
            "mode": "interactive_rollout",
            "config": str(self.config_path),
            "robot_ip": self.robot_config["ip"],
            "ik_solver": self.robot_config.get("ik_solver", "firmware"),
            "firmware_ik_equivalent_euler_retry": self.robot_config.get(
                "firmware_ik_equivalent_euler_retry", False
            ),
            "camera_serials": {
                role: camera["serial"]
                for role, camera in self.config["cameras"].items()
            },
            "checkpoint": str(policy.checkpoint),
            "checkpoint_step": policy.checkpoint_step,
            "augmentation_mode": policy.augmentation_mode,
            "training_config": policy.training_config,
            "norm_stats_checkpoint": str(policy.norm_stats_checkpoint),
            "norm_stats_sha256": policy.norm_stats_sha256,
            "base_comparison_semantics": (
                f"untouched_pi0_base_weights_with_deformgen_"
                f"{self.policy_config['case']}_xarm_norm_stats"
                if model_variant == "base"
                else None
            ),
            "task": self.policy_config["task"],
            "action_horizon": self.policy_config["action_horizon"],
            "inference_seed": self.policy_config["inference_seed"],
            "max_steps": self.max_steps,
            "session_directory": str(self.session_dir),
            "rollout_index": self._rollout_index,
            "queue_semantics": "full_action_horizon_then_replan",
            "image_preprocessing": self.policy_config["image_preprocessing"],
            "gripper_geometry_clearance_mm": self.robot_config["safety"][
                "gripper_geometry_clearance_mm"
            ],
            "interactive_keys": {
                "start": "c",
                "stop": "s",
                "reset": "r",
                "delete_previous": "Delete or d",
                "quit": "q",
            },
        }
        try:
            initial_frames = self.frames.get()
            if self._position_protocol:
                assert pair_index is not None
                if (
                    model_variant == self._protocol_order[0]
                    and self.reference_session is None
                ):
                    with self._lock:
                        self._reference_frames = {
                            role: frame.copy() for role, frame in initial_frames.items()
                        }
                    for role, frame in initial_frames.items():
                        reference_path = (
                            self.session_dir
                            / f"position_{pair_index:02d}_reference_{role}.png"
                        )
                        if not cv2.imwrite(str(reference_path), frame):
                            raise RuntimeError(f"Could not save {reference_path}")
                for role, frame in initial_frames.items():
                    initial_path = (
                        self.session_dir
                        / f"position_{pair_index:02d}_{model_variant}_initial_{role}.png"
                    )
                    if not cv2.imwrite(str(initial_path), frame):
                        raise RuntimeError(f"Could not save {initial_path}")
            self._logger = RunLogger(
                self.session_dir,
                "interactive_rollout",
                metadata,
                directory=self.session_dir / rollout_name,
            )
            video_path = self.session_dir / f"{rollout_name}.mp4"
            self.cameras.start_front_recording(video_path)
            self._recording = True
            self._logger.event({"type": "front_video_started", "path": str(video_path)})
            self._protocol_event(
                {
                    "type": "rollout_started",
                    "rollout_index": self._rollout_index,
                    "position_index": pair_index,
                    "model_variant": model_variant,
                    "video": str(video_path),
                }
            )
        except Exception as error:
            if self._recording:
                try:
                    self.cameras.stop_front_recording()
                except Exception:
                    pass
                self._recording = False
            with self._lock:
                self._activity = "idle"
                self._status = f"FAULT | could not start rollout: {error}"
                self._status_kind = "fault"
            return
        self._worker = threading.Thread(
            target=self._rollout_worker,
            args=(policy,),
            name="interactive-pi0-rollout",
            daemon=True,
        )
        self._worker.start()

    def _rollout_worker(self, policy: PI0Runtime) -> None:
        controller = XArmPolicyController(self.robot_config)
        motion_started = False
        keep_enabled_on_exit = False
        total_steps = 0
        inference_index = 0
        logger = self._logger
        assert logger is not None
        try:
            policy.reset()
            self._set_status("ROLLOUT | connecting and checking reset pose", "rollout")
            controller.connect()
            readiness = controller.verify_ready_for_motion()
            logger.event({"type": "ready", "details": readiness})
            if self._stop.is_set():
                raise InterruptedError("operator stop before inference")

            chunk_steps = int(self.policy_config["action_horizon"])
            while total_steps < self.max_steps:
                state, state_details = controller.read_policy_state()
                observation = self.frames.get()
                logger.images(inference_index, observation)
                self._set_status(
                    f"ROLLOUT | inference {inference_index} | step {total_steps}/{self.max_steps}",
                    "rollout",
                )
                actions, inference_seconds = policy.infer(observation, state)
                logger.event(
                    {
                        "type": "inference",
                        "inference_index": inference_index,
                        "state": state.tolist(),
                        "state_details": state_details,
                        "inference_seconds": inference_seconds,
                        "action_min": actions.min(axis=0).tolist(),
                        "action_max": actions.max(axis=0).tolist(),
                        "first_action": actions[0].tolist(),
                        "possible_policy_action_modification": (
                            "gripper_geometry_z_floor_only"
                        ),
                        "gripper_geometry_clearance_mm": (
                            controller.limits.gripper_geometry_clearance_mm
                        ),
                    }
                )
                if self._stop.is_set():
                    raise InterruptedError("operator stop after inference")
                if not motion_started:
                    controller.start_motion()
                    motion_started = True
                controller.begin_action_chunk()
                deadline = time.monotonic()
                steps_this_chunk = min(chunk_steps, self.max_steps - total_steps)
                for chunk_step in range(steps_this_chunk):
                    if self._stop.is_set():
                        raise InterruptedError("operator stop")
                    command = controller.execute_policy_action(actions[chunk_step])
                    logger.event(
                        {
                            "type": "command",
                            "global_step": total_steps,
                            "chunk_step": chunk_step,
                            **command,
                        }
                    )
                    total_steps += 1
                    if total_steps % 10 == 0:
                        self._set_status(
                            f"ROLLOUT | step {total_steps}/{self.max_steps} | s stops",
                            "rollout",
                        )
                    deadline += 1.0 / controller.limits.control_hz
                    remaining = deadline - time.monotonic()
                    if remaining > 0 and self._stop.wait(remaining):
                        raise InterruptedError("operator stop")
                    if remaining <= -controller.limits.command_timeout_s:
                        raise SafetyViolation(
                            f"Control loop missed deadline by {-remaining:.3f}s"
                        )
                inference_index += 1

            logger.event({"type": "complete", "total_steps": total_steps})
            keep_enabled_on_exit = True
            self._finish_worker(
                f"COMPLETE | {total_steps}/{self.max_steps} steps | xArm enabled/holding",
                "complete",
                attempted=True,
                total_steps=total_steps,
            )
        except InterruptedError as error:
            keep_enabled_on_exit = True
            logger.event(
                {
                    "type": "operator_stop",
                    "message": str(error),
                    "total_steps": total_steps,
                }
            )
            self._finish_worker(
                f"STOPPED | {total_steps}/{self.max_steps} steps | xArm enabled/holding",
                "warning",
                attempted=motion_started or total_steps > 0,
                total_steps=total_steps,
            )
        except Exception as error:
            logger.event(
                {
                    "type": "fault",
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "total_steps": total_steps,
                }
            )
            self._finish_worker(
                f"FAULT | {type(error).__name__}: {error}",
                "fault",
                attempted=motion_started or total_steps > 0,
                total_steps=total_steps,
            )
        finally:
            # Always zero velocity and leave velocity mode. Normal completion
            # and operator stop retain servo hold for a fast reset/restart;
            # faults still disable the motors and engage the brakes.
            controller.stop_motion(disable=not keep_enabled_on_exit)
            controller.disconnect()

    def start_reset(self) -> None:
        if not self._claim_activity("reset", "RESET | release, vertical lift, reset"):
            self._set_status("IGNORED r | press s and wait before reset", "warning")
            return
        self._worker = threading.Thread(
            target=self._reset_worker,
            name="interactive-xarm-reset",
            daemon=True,
        )
        self._worker.start()

    def _reset_worker(self) -> None:
        try:
            metrics = reset_robot(
                self.robot_config,
                recover_collision_31=True,
                release_before_lift=True,
                disable_after=False,
            )
            self._finish_worker(
                "RESET COMPLETE | error "
                f"{metrics['translation_error_mm']:.2f} mm | xArm enabled/holding",
                "complete",
            )
        except Exception as error:
            self._finish_worker(
                f"RESET FAULT | {type(error).__name__}: {error}", "fault"
            )

    def request_stop(self) -> None:
        activity, _, _ = self.display_state()
        if activity == "rollout":
            self._stop.set()
            self._set_status(
                "STOPPING | waiting for current command/inference boundary", "warning"
            )
        elif activity == "reset":
            self._set_status(
                "RESET ACTIVE | reset completes before another command", "warning"
            )
        else:
            self._set_status("IDLE | nothing to stop", "idle")

    def report_camera_fault(self, error: Exception) -> None:
        self._protocol_event(
            {
                "type": "camera_fault",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        self.request_stop()

    def report_camera_recovered(self) -> None:
        self._protocol_event({"type": "camera_recovered"})
        if self.is_idle():
            if self._position_protocol:
                status = (
                    f"CAMERAS RECOVERED | POSITION {self._pair_index}/10 | "
                    f"c runs {self._pair_stage}"
                )
            else:
                status = f"CAMERAS RECOVERED | c runs {self._next_model_variant()}"
            self._set_status(status, "warning")

    def delete_previous_rollout(self) -> None:
        """Remove the latest finished rollout and restore its protocol slot."""
        with self._lock:
            if self._activity != "idle":
                self._status = "IGNORED Delete | stop and wait until idle"
                self._status_kind = "warning"
                return
        if not self._position_protocol:
            self._set_status(
                "IGNORED Delete | available only in 10-position protocols",
                "warning",
            )
            return

        parsed: list[dict[str, Any]] = []
        last_finished: dict[str, Any] | None = None
        for line_number, line in enumerate(
            self._protocol_events_path.read_text().splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                self._set_status(
                    f"DELETE FAULT | invalid protocol log line {line_number}: {error}",
                    "fault",
                )
                return
            parsed.append(event)
            if event.get("type") == "rollout_finished":
                last_finished = event
        if last_finished is None:
            self._set_status("IGNORED Delete | no finished rollout exists", "warning")
            return

        rollout_index = last_finished.get("rollout_index")
        position = last_finished.get("position_index")
        model = last_finished.get("model_variant")
        if (
            not isinstance(rollout_index, int)
            or not isinstance(position, int)
            or not isinstance(model, str)
            or model not in self._protocol_order
        ):
            self._set_status("DELETE FAULT | malformed final rollout event", "fault")
            return

        stem = f"rollout_{rollout_index:03d}_position_{position:02d}_{model}"
        targets = [
            self.session_dir / f"{stem}.mp4",
            self.session_dir / stem,
            self.session_dir / f"position_{position:02d}_{model}_initial_front.png",
            self.session_dir / f"position_{position:02d}_{model}_initial_wrist.png",
        ]
        existing_targets = [path for path in targets if path.exists()]
        trash_dir = (
            self.session_dir.parent
            / ".deleted_rollouts"
            / self.session_dir.name
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{stem}"
        )
        moved: list[tuple[Path, Path]] = []
        temp_log = self._protocol_events_path.with_suffix(".jsonl.tmp")
        try:
            trash_dir.mkdir(parents=True, exist_ok=False)
            for source in existing_targets:
                destination = trash_dir / source.name
                shutil.move(str(source), str(destination))
                moved.append((source, destination))

            retained = [
                event
                for event in parsed
                if not (
                    event.get("type") in {"rollout_started", "rollout_finished"}
                    and event.get("rollout_index") == rollout_index
                )
            ]
            retained.append(
                {
                    "wall_time": time.time(),
                    "type": "rollout_deleted",
                    "deleted_rollout_index": rollout_index,
                    "position_index": position,
                    "model_variant": model,
                    "attempted": last_finished.get("attempted"),
                    "trash_directory": str(trash_dir),
                    "moved_paths": [str(destination) for _, destination in moved],
                }
            )
            temp_log.write_text(
                "".join(
                    json.dumps(event, ensure_ascii=False) + "\n" for event in retained
                )
            )
            os.replace(temp_log, self._protocol_events_path)
        except Exception as error:
            try:
                temp_log.unlink(missing_ok=True)
                for source, destination in reversed(moved):
                    if destination.exists():
                        shutil.move(str(destination), str(source))
                if trash_dir.exists() and not any(trash_dir.iterdir()):
                    trash_dir.rmdir()
            except Exception as rollback_error:
                self._set_status(
                    f"DELETE FAULT | {error}; rollback failed: {rollback_error}",
                    "fault",
                )
                return
            self._set_status(f"DELETE FAULT | {error}", "fault")
            return

        progress = _resume_progress(self.session_dir, self.protocol)
        self._rollout_index = int(progress["rollout_index"])
        self._pair_index = int(progress["position_index"])
        self._pair_stage = str(progress["model_variant"])
        self._protocol_complete = bool(progress["complete"])
        with self._lock:
            self._reference_frames = None
        if self.reference_session is not None:
            self._activate_external_reference(self._pair_index)
        else:
            reference_paths = [
                self.session_dir
                / f"position_{self._pair_index:02d}_reference_{role}.png"
                for role in ("front", "wrist")
            ]
            if all(path.is_file() for path in reference_paths):
                self._load_current_session_reference(self._pair_index)
        self._set_status(
            f"DELETED {stem} | POSITION {self._pair_index}/10 | "
            f"r reset, restore SAME state with overlay, c reruns {self._pair_stage}",
            "warning",
        )

    def finalize_worker_if_done(self) -> None:
        with self._lock:
            done = self._worker_done
            outcome = self._worker_outcome
            worker_kind = self._worker_kind
            logger = self._logger
            attempted = self._active_attempted
            total_steps = self._active_total_steps
            model_variant = self._active_model_variant
            pair_index = self._active_pair_index
            outcome_kind = self._status_kind
        if not done:
            return
        worker = self._worker
        if worker is not None:
            worker.join()
        if worker_kind == "rollout" and self._recording:
            try:
                recording_info = self.cameras.stop_front_recording()
                if recording_info is not None and logger is not None:
                    logger.event({"type": "front_video_complete", **recording_info})
                    outcome += f" | H.264: {recording_info['path']}"
            except Exception as error:
                outcome += f" | VIDEO FAULT: {error}"
            finally:
                self._recording = False
        final_status = outcome
        final_kind = outcome_kind
        if worker_kind == "rollout":
            self._protocol_event(
                {
                    "type": "rollout_finished",
                    "rollout_index": self._rollout_index,
                    "position_index": pair_index,
                    "model_variant": model_variant,
                    "attempted": attempted,
                    "total_steps": total_steps,
                    "outcome_kind": outcome_kind,
                    "outcome": outcome,
                }
            )
            if self._position_protocol:
                if not attempted:
                    final_status = (
                        f"POSITION {self._pair_index}/10 | PRECHECK NOT COUNTED | "
                        f"fix/reset, c retries {self._pair_stage}"
                    )
                    final_kind = "warning"
                elif model_variant != self._protocol_order[-1]:
                    stage_index = self._protocol_order.index(model_variant)
                    self._pair_stage = self._protocol_order[stage_index + 1]
                    final_status = (
                        f"POSITION {self._pair_index}/10 | restore SAME "
                        f"{self.policy_config['case']} (50% overlay), r then c runs "
                        f"{self._pair_stage}"
                    )
                    final_kind = "warning"
                elif self._pair_index >= self._pair_count:
                    self._protocol_complete = True
                    commanded = self._pair_count * len(self._protocol_order)
                    final_status = (
                        f"{self.protocol.upper()} COMPLETE | {commanded} commanded "
                        "attempts | q exits"
                    )
                    final_kind = "complete"
                else:
                    self._pair_index += 1
                    self._pair_stage = self._protocol_order[0]
                    if self.reference_session is not None:
                        self._activate_external_reference(self._pair_index)
                        final_status = (
                            f"POSITION {self._pair_index}/10 | r reset, align "
                            f"{self.policy_config['case']} to saved overlay, c runs "
                            f"{self._pair_stage}"
                        )
                    else:
                        with self._lock:
                            self._reference_frames = None
                        final_status = (
                            f"POSITION {self._pair_index}/10 | r reset, then place NEW "
                            f"{self.policy_config['case']} state, c runs "
                            f"{self._pair_stage}"
                        )
                    final_kind = "idle"
        elif worker_kind == "reset" and self._position_protocol:
            if (
                outcome_kind == "complete"
                and self._pair_stage != self._protocol_order[0]
            ):
                final_status = (
                    f"POSITION {self._pair_index}/10 | RESET COMPLETE | restore SAME "
                    f"{self.policy_config['case']} with overlay, c runs "
                    f"{self._pair_stage}"
                )
                final_kind = "warning"
            elif outcome_kind == "complete":
                if self.reference_session is not None:
                    final_status = (
                        f"POSITION {self._pair_index}/10 | RESET COMPLETE | align "
                        f"{self.policy_config['case']} to saved overlay, c runs "
                        f"{self._pair_stage}"
                    )
                elif self._reference_frames is not None:
                    final_status = (
                        f"POSITION {self._pair_index}/10 | RESET COMPLETE | restore SAME "
                        f"{self.policy_config['case']} with overlay, c runs "
                        f"{self._pair_stage}"
                    )
                else:
                    final_status = (
                        f"POSITION {self._pair_index}/10 | RESET COMPLETE | place "
                        f"{self.policy_config['case']}, c runs {self._pair_stage}"
                    )
                final_kind = "idle"
        with self._lock:
            self._activity = "idle"
            self._status = final_status
            self._status_kind = final_kind
            self._worker = None
            self._worker_done = False
            self._worker_kind = ""
            self._logger = None
        print(outcome, flush=True)
        if final_status != outcome:
            print(final_status, flush=True)

    def is_idle(self) -> bool:
        with self._lock:
            return self._activity == "idle"

    def shutdown(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None:
            # Do not tear down the process while a reset or a final stop is
            # still executing.  The GUI loop normally reaches shutdown only
            # after idle; this also makes Ctrl-C follow the same safe behavior.
            while worker.is_alive():
                worker.join(timeout=1.0)
        self.finalize_worker_if_done()


def _draw_console(
    frames: dict[str, np.ndarray],
    reference_frames: dict[str, np.ndarray] | None,
    cameras: dict[str, dict[str, Any]],
    fps: float,
    status: str,
    status_kind: str,
    max_width: int,
) -> np.ndarray:
    panels = []
    for role in ("front", "wrist"):
        frame = frames[role]
        reference_active = (
            reference_frames is not None
            and role in reference_frames
            and reference_frames[role].shape == frame.shape
        )
        if reference_active:
            frame = cv2.addWeighted(frame, 0.5, reference_frames[role], 0.5, 0.0)
        panel = _draw_alignment_overlay(frame, role, cameras[role], fps, False)
        if reference_active:
            cv2.putText(
                panel,
                "50% SAVED INITIAL + 50% LIVE",
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        panels.append(panel)
    canvas = np.hstack(panels)
    colors = {
        "idle": (220, 220, 220),
        "rollout": (0, 220, 255),
        "reset": (255, 180, 0),
        "complete": (80, 230, 80),
        "warning": (0, 180, 255),
        "fault": (60, 60, 255),
    }
    color = colors.get(status_kind, (255, 255, 255))
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, height - 68), (width, height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status[:150],
        (12, height - 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "c: rollout   s: stop   r: reset   Delete/d: redo previous   q: quit",
        (12, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return _fit_width(canvas, max_width)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=tuple(TASK_CASES),
        required=True,
        help="real-robot task; selects its checkpoint, prompt, and image preprocessing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "real/robot.local.yaml",
        help="local hardware config; task policy fields may be overridden",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--resume-session",
        type=Path,
        default=None,
        help="append to an interrupted 10-position session and restore its next slot",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-display-width", type=int, default=1600)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--protocol",
        choices=("single", "paired10", "triple10", "single10"),
        default="single",
        help="triple10 runs yawonly, single, then base at each of 10 states",
    )
    parser.add_argument(
        "--model",
        choices=("yawonly", "single", "base"),
        default="yawonly",
        help="used only with --protocol single",
    )
    parser.add_argument(
        "--reference-session",
        type=Path,
        default=None,
        help=(
            "source session containing position_NN_reference_{front,wrist}.png; "
            "required by --protocol single10"
        ),
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirm", help=f"must equal {MOTION_CONFIRMATION}")
    parser.add_argument(
        "--confirm-continuous",
        help=f"must equal {PI0_CONTINUOUS_CONFIRMATION}",
    )
    return parser


def _create_session_directory(
    output_root: Path,
    *,
    model_variant: str,
    protocol: str,
    config_path: Path,
    config: dict[str, Any],
    reference_session: Path | None = None,
) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = output_root / f"{stamp}_interactive_session"
    session_dir.mkdir(exist_ok=False)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "model_variant": model_variant,
                "protocol": protocol,
                "task_key": config["policy"]["case"],
                "task_prompt": config["policy"]["task"],
                "protocol_order": (
                    list(PROTOCOL_ORDERS[protocol])
                    if protocol in PROTOCOL_ORDERS
                    else None
                ),
                "position_count": 10 if protocol in PROTOCOL_ORDERS else None,
                "planned_commanded_rollouts": (
                    10 * len(PROTOCOL_ORDERS[protocol])
                    if protocol in PROTOCOL_ORDERS
                    else None
                ),
                "config": str(config_path),
                "settings": config,
                "robot_ip": config["robot"]["ip"],
                "camera_serials": {
                    role: camera["serial"] for role, camera in config["cameras"].items()
                },
                "video_naming": (
                    "rollout_NNN_position_NN_MODEL.mp4"
                    if protocol in PROTOCOL_ORDERS
                    else "rollout_NNN.mp4"
                ),
                "video_codec": "h264",
                "reference_session": (
                    str(reference_session.resolve())
                    if reference_session is not None
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    return session_dir


def main() -> int:
    args = _parser().parse_args()
    if args.resume_session is not None and args.output_root is not None:
        raise ValueError(
            "--resume-session reuses its directory; do not pass --output-root"
        )
    require_motion_authorization(
        args.allow_motion,
        args.confirm,
        continuous=True,
        continuous_confirmation=args.confirm_continuous,
    )

    config_path = args.config.resolve()
    config = load_config(config_path, args.task)
    validate_hardware(config, motion=True)
    expected_case = TASK_CASES[args.task]
    if config["policy"]["case"] != expected_case:
        raise ValueError(
            f"--task {args.task!r} requires policy.case={expected_case!r}; "
            f"{config_path} contains {config['policy']['case']!r}"
        )
    resume_progress = None
    if args.resume_session is not None:
        session_dir = args.resume_session.resolve()
        metadata = _validate_resume_session(
            session_dir,
            task=args.task,
            protocol=args.protocol,
            config=config,
        )
        saved_reference = metadata.get("reference_session")
        if saved_reference is not None:
            saved_reference = Path(saved_reference).resolve()
            if args.reference_session is None:
                args.reference_session = saved_reference
            elif args.reference_session.resolve() != saved_reference:
                raise ValueError(
                    "--reference-session differs from the resumed session metadata"
                )
        elif args.reference_session is not None:
            raise ValueError(
                "Cannot add --reference-session when resuming a session without one"
            )
        archived = _archive_interrupted_rollouts(session_dir)
        for interrupted in archived:
            print(
                "Archived interrupted rollout "
                f"{interrupted['rollout_index']:03d} "
                f"(position {interrupted['position_index']} / "
                f"{interrupted['model_variant']}) to "
                f"{interrupted['archive_directory']}",
                flush=True,
            )
        resume_progress = _resume_progress(session_dir, args.protocol)
    if args.protocol == "single10" and args.reference_session is None:
        raise ValueError("--protocol single10 requires --reference-session")
    if args.protocol != "single10" and args.reference_session is not None:
        raise ValueError(
            "--reference-session is currently only valid with --protocol single10"
        )
    limits = XArmPolicyController(config["robot"]).limits
    max_steps = limits.max_episode_steps if args.max_steps is None else args.max_steps
    if not 1 <= max_steps <= limits.max_episode_steps:
        raise ValueError(f"--max-steps must be in [1, {limits.max_episode_steps}]")

    if args.resume_session is None:
        output_root = (
            ROOT / f"outputs/real/{args.task}"
            if args.output_root is None
            else args.output_root
        )
        session_dir = _create_session_directory(
            output_root,
            model_variant=(args.model if args.protocol == "single" else args.protocol),
            protocol=args.protocol,
            config_path=config_path,
            config=config,
            reference_session=args.reference_session,
        )
    else:
        print(
            "Resuming after "
            f"{resume_progress['attempted_count']} attempted rollouts; next is "
            f"position {resume_progress['position_index']} / "
            f"{resume_progress['model_variant']}",
            flush=True,
        )
    print(f"Session output: {session_dir}", flush=True)

    model_variants = (
        (args.model,) if args.protocol == "single" else PROTOCOL_ORDERS[args.protocol]
    )
    policies: dict[str, PI0Runtime] = {}
    for model_variant in model_variants:
        print(f"Loading {model_variant} checkpoint...", flush=True)
        policies[model_variant] = PI0Runtime(
            config["policy"], model_variant=model_variant
        )
        print(
            f"{model_variant} loaded in {policies[model_variant].load_seconds:.2f}s",
            flush=True,
        )
    cameras = RealSensePair(config["cameras"])
    frame_store = FrameStore()
    session = InteractiveSession(
        config_path,
        config,
        policies,
        cameras,
        frame_store,
        session_dir,
        max_steps,
        args.protocol,
        args.reference_session,
        resume_progress,
    )

    exit_requested = False
    fps = 0.0
    fps_frames = 0
    fps_started = time.perf_counter()
    camera_fault: Exception | None = None
    next_recovery_attempt = 0.0
    try:
        print("Starting configured cameras...", flush=True)
        cameras.start(args.warmup_frames)
        print("Cameras ready.", flush=True)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print(
            "Controls: c rollout | s stop | r reset | "
            "Delete/d remove previous | q quit",
            flush=True,
        )
        while True:
            session.finalize_worker_if_done()
            if exit_requested and session.is_idle():
                break
            if camera_fault is not None:
                if session.is_idle() and time.monotonic() >= next_recovery_attempt:
                    try:
                        cameras.recover(max(5, args.warmup_frames))
                    except RuntimeError as recovery_error:
                        camera_fault = recovery_error
                        next_recovery_attempt = time.monotonic() + 1.0
                        print(f"CAMERA RECOVERY RETRY: {recovery_error}", flush=True)
                    else:
                        camera_fault = None
                        fps_frames = 0
                        fps_started = time.perf_counter()
                        session.report_camera_recovered()
                if camera_fault is not None:
                    key = cv2.waitKeyEx(20)
                    window_closed = (
                        cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
                    )
                    if key in (27, ord("q")) or window_closed:
                        exit_requested = True
                        session.request_stop()
                    continue
            try:
                paired = cameras.capture()
            except RuntimeError as error:
                camera_fault = error
                next_recovery_attempt = time.monotonic()
                print(f"CAMERA FAULT: {error}", flush=True)
                session.report_camera_fault(error)
                continue
            frame_store.update(paired)
            fps_frames += 1
            now = time.perf_counter()
            if now - fps_started >= 1.0:
                fps = fps_frames / (now - fps_started)
                fps_frames = 0
                fps_started = now
            _, status, status_kind = session.display_state()
            reference_frames = session.reference_frames_for_display()
            cv2.imshow(
                WINDOW_NAME,
                _draw_console(
                    paired,
                    reference_frames,
                    config["cameras"],
                    fps,
                    status,
                    status_kind,
                    args.max_display_width,
                ),
            )
            key = cv2.waitKeyEx(1)
            window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key == ord("c"):
                session.start_rollout()
            elif key == ord("s"):
                session.request_stop()
            elif key == ord("r"):
                session.start_reset()
            elif key in DELETE_KEY_CODES or key == ord("d"):
                session.delete_previous_rollout()
            elif key in (27, ord("q")) or window_closed:
                exit_requested = True
                session.request_stop()
    except KeyboardInterrupt:
        session.request_stop()
    finally:
        session.shutdown()
        cameras.stop()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
