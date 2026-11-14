"""Adapter from DeformGen 8D absolute actions to xArm7.

Importing this module does not connect to hardware. Motion callers authorize
execution explicitly; configured recovery and enable behavior runs only then.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from .config import ROOT

MOTION_CONFIRMATION = "MOVE_XARM"
PI0_CONTINUOUS_CONFIRMATION = "RUN_CONTINUOUS_PI0"
EXPECTED_GRIPPER_BAUD = 2_000_000


class SafetyViolation(RuntimeError):
    """Raised before or during execution when a safety invariant is false."""


def _vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise SafetyViolation(
            f"{name} must contain {length} finite values; got {values!r}"
        )
    return vector


def normalize_quaternion_wxyz(values: Sequence[float]) -> np.ndarray:
    # Never mutate a view into the policy action while converting its
    # quaternion to a mathematically valid Rotation.
    quaternion = _vector(values, 4, "quaternion_wxyz").copy()
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise SafetyViolation("Policy quaternion norm is nearly zero")
    quaternion /= norm
    return quaternion


def rotation_from_wxyz(values: Sequence[float]) -> Rotation:
    qw, qx, qy, qz = normalize_quaternion_wxyz(values)
    return Rotation.from_quat([qx, qy, qz, qw])


def wxyz_from_rotation(rotation: Rotation) -> np.ndarray:
    qx, qy, qz, qw = rotation.as_quat()
    result = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    # q and -q are geometrically equivalent but not numerically equivalent to
    # a neural policy.  Kornia's DeformGen export keeps the largest-magnitude
    # quaternion component positive (qx around the fold-cloth workspace).
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0:
        result *= -1
    return result


def pose_matrix_mm_degrees(pose: Sequence[float]) -> np.ndarray:
    values = _vector(pose, 6, "xArm pose")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
    transform[:3, 3] = values[:3] / 1000.0
    return transform


def policy_pose_matrix(policy_state_or_action: Sequence[float]) -> np.ndarray:
    values = _vector(policy_state_or_action, 8, "policy state/action")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_from_wxyz(values[3:7]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def eef_policy_pose_from_xarm(
    tcp_pose_mm_degrees: Sequence[float],
    tcp_offset_mm_degrees: Sequence[float],
    flange_to_policy_eef_mm_degrees: Sequence[float],
) -> tuple[np.ndarray, Rotation]:
    """Map the xArm configured-TCP report into DeformGen's nominal link7 frame.

    The controller pose is ``base->configured_tcp``.  Its TCP offset is
    ``flange->configured_tcp``.  DeformGen uses the nominal URDF link7 pose,
    whose fixed calibration from this physical controller's flange is supplied
    explicitly in the robot config.
    """
    base_to_tcp = pose_matrix_mm_degrees(tcp_pose_mm_degrees)
    flange_to_tcp = pose_matrix_mm_degrees(tcp_offset_mm_degrees)
    flange_to_policy_eef = pose_matrix_mm_degrees(flange_to_policy_eef_mm_degrees)
    base_to_flange = base_to_tcp @ np.linalg.inv(flange_to_tcp)
    base_to_policy_eef = base_to_flange @ flange_to_policy_eef
    return base_to_policy_eef[:3, 3], Rotation.from_matrix(base_to_policy_eef[:3, :3])


def xarm_tcp_pose_from_policy_action(
    action: Sequence[float],
    tcp_offset_mm_degrees: Sequence[float],
    flange_to_policy_eef_mm_degrees: Sequence[float],
) -> np.ndarray:
    """Invert :func:`eef_policy_pose_from_xarm` for a Cartesian command."""
    base_to_policy_eef = policy_pose_matrix(action)
    flange_to_policy_eef = pose_matrix_mm_degrees(flange_to_policy_eef_mm_degrees)
    flange_to_tcp = pose_matrix_mm_degrees(tcp_offset_mm_degrees)
    base_to_flange = base_to_policy_eef @ np.linalg.inv(flange_to_policy_eef)
    base_to_tcp = base_to_flange @ flange_to_tcp
    xyz_mm = base_to_tcp[:3, 3] * 1000.0
    rpy_degrees = Rotation.from_matrix(base_to_tcp[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    return np.concatenate([xyz_mm, rpy_degrees])


def rotation_distance_degrees(first: Rotation, second: Rotation) -> float:
    return math.degrees(float((first.inv() * second).magnitude()))


FIRMWARE_IK_EQUIVALENT_EULER_RETRIES = (
    ("roll+360", np.asarray([360.0, 0.0, 0.0])),
    ("roll-360", np.asarray([-360.0, 0.0, 0.0])),
    ("pitch+360", np.asarray([0.0, 360.0, 0.0])),
    ("pitch-360", np.asarray([0.0, -360.0, 0.0])),
    ("yaw+360", np.asarray([0.0, 0.0, 360.0])),
    ("yaw-360", np.asarray([0.0, 0.0, -360.0])),
)


def solve_xarm_firmware_ik(
    arm: Any,
    tcp_target_mm_degrees: Sequence[float],
    current_joints_rad: Sequence[float],
    *,
    expected_axis_count: int = 7,
    retry_equivalent_euler: bool = True,
    retry_position_tolerance_mm: float = 0.1,
    retry_orientation_tolerance_deg: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run xArm IK, recovering legacy-firmware Euler branch failures.

    xArm firmware v2.5.1 can return API code 10 for a TCP pose and then solve
    the exact same SE(3) target when one Euler component is shifted by 360
    degrees.  A shift by a full turn changes only the numeric representation,
    never the requested physical pose.  Retry solutions are checked with the
    controller's own FK and the joint-space-nearest valid branch is selected.
    """
    target = _vector(tcp_target_mm_degrees, 6, "tcp_target_mm_degrees")
    current = _vector(current_joints_rad, expected_axis_count, "current_joints_rad")
    code, joints = arm.get_inverse_kinematics(
        target.tolist(),
        input_is_radian=False,
        return_is_radian=True,
        limited=True,
        ref_angles=np.rad2deg(current).tolist(),
    )
    original_code = int(code)
    if original_code == 0 and joints is not None:
        return _vector(
            joints[:expected_axis_count], expected_axis_count, "target_joints"
        ), {
            "solver": "xarm_firmware",
            "sdk_code": 0,
            "original_sdk_code": 0,
            "equivalent_euler_retry_used": False,
        }
    if original_code != 10 or not retry_equivalent_euler:
        raise SafetyViolation(
            f"get_inverse_kinematics failed with SDK code {original_code}"
        )

    target_matrix = pose_matrix_mm_degrees(target)
    attempts: list[dict[str, Any]] = []
    solutions: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for name, euler_delta in FIRMWARE_IK_EQUIVALENT_EULER_RETRIES:
        equivalent_target = target.copy()
        equivalent_target[3:] += euler_delta
        retry_code, retry_joints = arm.get_inverse_kinematics(
            equivalent_target.tolist(),
            input_is_radian=False,
            return_is_radian=True,
            limited=True,
            ref_angles=np.rad2deg(current).tolist(),
        )
        attempt: dict[str, Any] = {
            "representation": name,
            "sdk_code": int(retry_code),
        }
        attempts.append(attempt)
        if retry_code != 0 or retry_joints is None:
            continue
        candidate = _vector(
            retry_joints[:expected_axis_count],
            expected_axis_count,
            f"target_joints_{name}",
        )
        fk_code, achieved_pose = arm.get_forward_kinematics(
            candidate.tolist(),
            input_is_radian=True,
            return_is_radian=False,
        )
        attempt["fk_sdk_code"] = int(fk_code)
        if fk_code != 0 or achieved_pose is None:
            continue
        achieved_matrix = pose_matrix_mm_degrees(achieved_pose)
        position_error_mm = float(
            np.linalg.norm(achieved_matrix[:3, 3] - target_matrix[:3, 3]) * 1000.0
        )
        orientation_error_deg = rotation_distance_degrees(
            Rotation.from_matrix(target_matrix[:3, :3]),
            Rotation.from_matrix(achieved_matrix[:3, :3]),
        )
        attempt["fk_position_error_mm"] = position_error_mm
        attempt["fk_orientation_error_deg"] = orientation_error_deg
        if position_error_mm > retry_position_tolerance_mm or (
            orientation_error_deg > retry_orientation_tolerance_deg
        ):
            continue
        joint_delta = np.arctan2(
            np.sin(candidate - current), np.cos(candidate - current)
        )
        joint_distance = float(np.linalg.norm(joint_delta))
        attempt["joint_distance_from_current_rad"] = joint_distance
        solutions.append((joint_distance, candidate, attempt))

    if not solutions:
        raise SafetyViolation(
            "get_inverse_kinematics failed with SDK code 10 and all "
            f"equivalent-Euler retries failed: {attempts}"
        )
    _, selected, selected_attempt = min(solutions, key=lambda item: item[0])
    status_code, err_warn = arm.get_err_warn_code()
    if status_code != 0 or err_warn is None or len(err_warn) < 2:
        raise SafetyViolation(
            "Could not verify controller status after recovering firmware IK: "
            f"code={status_code}, err_warn={err_warn}"
        )
    controller_error = int(err_warn[0])
    controller_warning = int(err_warn[1])
    if controller_error != 0 or controller_warning not in {0, 14}:
        raise SafetyViolation(
            "Unexpected controller error/warning after recovering firmware IK: "
            f"error={controller_error}, warning={controller_warning}"
        )
    warning_14_cleared = False
    if controller_warning == 14:
        clean_code = int(arm.clean_warn())
        verify_code, verified_err_warn = arm.get_err_warn_code()
        if (
            clean_code != 0
            or verify_code != 0
            or verified_err_warn is None
            or (list(map(int, verified_err_warn[:2])) != [0, 0])
        ):
            raise SafetyViolation(
                "Failed to clear the expected firmware IK warning 14: "
                f"clean_code={clean_code}, verify_code={verify_code}, "
                f"err_warn={verified_err_warn}"
            )
        warning_14_cleared = True
    return selected, {
        "solver": "xarm_firmware",
        "sdk_code": 0,
        "original_sdk_code": original_code,
        "equivalent_euler_retry_used": True,
        "selected_representation": selected_attempt["representation"],
        "valid_retry_solution_count": len(solutions),
        "selected_fk_position_error_mm": selected_attempt["fk_position_error_mm"],
        "selected_fk_orientation_error_deg": selected_attempt[
            "fk_orientation_error_deg"
        ],
        "warning_14_cleared": warning_14_cleared,
        "attempts": attempts,
    }


@dataclass(frozen=True)
class SafetyLimits:
    control_hz: float
    reset_position_tolerance_m: float
    reset_orientation_tolerance_deg: float
    reset_gripper_must_be_open: bool
    command_timeout_s: float
    max_chunk_steps: int
    max_episode_steps: int
    gripper_open_threshold: float
    gripper_open_pulse: int
    gripper_closed_pulse: int
    gripper_speed: int
    gripper_max_normalized_step: float
    max_joint_delta_norm_rad: float
    velocity_gain: float
    command_check_interval_s: float
    velocity_command_duration_s: float
    gripper_geometry_clearance_mm: float

    @classmethod
    def from_config(cls, values: dict[str, Any]) -> "SafetyLimits":
        limits = cls(
            control_hz=float(values["control_hz"]),
            reset_position_tolerance_m=float(values["reset_position_tolerance_m"]),
            reset_orientation_tolerance_deg=float(
                values["reset_orientation_tolerance_deg"]
            ),
            reset_gripper_must_be_open=bool(values["reset_gripper_must_be_open"]),
            command_timeout_s=float(values["command_timeout_s"]),
            max_chunk_steps=int(values["max_chunk_steps"]),
            max_episode_steps=int(values["max_episode_steps"]),
            gripper_open_threshold=float(values["gripper_open_threshold"]),
            gripper_open_pulse=int(values["gripper_open_pulse"]),
            gripper_closed_pulse=int(values["gripper_closed_pulse"]),
            gripper_speed=int(values["gripper_speed"]),
            gripper_max_normalized_step=float(
                values.get("gripper_max_normalized_step", 2.0 / 30.0)
            ),
            max_joint_delta_norm_rad=float(
                values.get("max_joint_delta_norm_rad", 0.10)
            ),
            velocity_gain=float(values.get("velocity_gain", 0.15)),
            command_check_interval_s=float(
                values.get("command_check_interval_s", 0.02)
            ),
            velocity_command_duration_s=float(
                values.get("velocity_command_duration_s", 0.05)
            ),
            gripper_geometry_clearance_mm=float(
                values["gripper_geometry_clearance_mm"]
            ),
        )
        if limits.control_hz <= 0 or limits.control_hz > 50:
            raise ValueError("control_hz must be in (0, 50]")
        for name in (
            "reset_position_tolerance_m",
            "reset_orientation_tolerance_deg",
            "command_timeout_s",
        ):
            if getattr(limits, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= limits.gripper_open_threshold <= 1:
            raise ValueError("gripper_open_threshold must be in [0, 1]")
        if not 0 < limits.gripper_max_normalized_step <= 1:
            raise ValueError("gripper_max_normalized_step must be in (0, 1]")
        if limits.max_joint_delta_norm_rad <= 0:
            raise ValueError("max_joint_delta_norm_rad must be positive")
        if limits.velocity_gain <= 0:
            raise ValueError("velocity_gain must be positive")
        if limits.command_check_interval_s <= 0:
            raise ValueError("command_check_interval_s must be positive")
        if limits.velocity_command_duration_s <= 0:
            raise ValueError("velocity_command_duration_s must be positive")
        if limits.gripper_geometry_clearance_mm < 0:
            raise ValueError("gripper_geometry_clearance_mm must be nonnegative")
        if not 1 <= limits.max_chunk_steps <= 64:
            raise ValueError("max_chunk_steps must be in [1, 64]")
        if limits.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        return limits


def require_motion_authorization(
    allow_motion_cli: bool,
    confirmation: str | None,
    continuous: bool = False,
    continuous_confirmation: str | None = None,
) -> None:
    failures: list[str] = []
    if not allow_motion_cli:
        failures.append("--allow-motion was not supplied")
    if confirmation != MOTION_CONFIRMATION:
        failures.append(f"--confirm must equal {MOTION_CONFIRMATION!r}")
    if continuous and continuous_confirmation != PI0_CONTINUOUS_CONFIRMATION:
        failures.append(
            "continuous mode requires --confirm-continuous "
            f"{PI0_CONTINUOUS_CONFIRMATION!r}"
        )
    if failures:
        raise SafetyViolation("Motion authorization refused: " + "; ".join(failures))


def load_gripper_collision_envelope(
    urdf_path: Path,
    opening_samples: int,
) -> np.ndarray:
    """Return a conservative convex envelope in configured-TCP coordinates."""
    if opening_samples < 2:
        raise ValueError("gripper_geometry_opening_samples must be at least 2")
    try:
        from urdfpy import URDF
    except ImportError as error:
        raise RuntimeError(
            "urdfpy is required for the configured gripper geometry guard"
        ) from error
    robot = URDF.load(str(urdf_path))
    tcp_link = next((link for link in robot.links if link.name == "link_tcp"), None)
    if tcp_link is None:
        raise ValueError(f"{urdf_path} has no link_tcp")
    guarded_links = {
        link.name
        for link in robot.links
        if link.name == "xarm_gripper_base_link"
        or link.name.startswith("left_")
        or link.name.startswith("right_")
    }
    all_points: list[np.ndarray] = []
    for drive_joint in np.linspace(0.0, 0.8, opening_samples):
        configuration = {
            joint.name: 0.0
            for joint in robot.actuated_joints
            if joint.name != "drive_joint"
        }
        configuration["drive_joint"] = float(drive_joint)
        link_fk = robot.link_fk(cfg=configuration)
        tcp_inverse = np.linalg.inv(link_fk[tcp_link])
        for link in robot.links:
            if link.name not in guarded_links:
                continue
            for collision in link.collisions:
                if collision.geometry.mesh is None:
                    continue
                transform = tcp_inverse @ link_fk[link] @ collision.origin
                for mesh in collision.geometry.mesh.meshes:
                    vertices = np.asarray(mesh.vertices, dtype=np.float64)
                    homogeneous = np.concatenate(
                        [vertices, np.ones((len(vertices), 1))], axis=1
                    )
                    all_points.append((transform @ homogeneous.T).T[:, :3])
    if not all_points:
        raise ValueError(f"No gripper collision vertices found in {urdf_path}")
    points = np.concatenate(all_points, axis=0)
    # A support-point query only needs the convex hull vertices.  The hull of
    # every sampled opening is conservative for intermediate finger positions.
    hull = ConvexHull(points)
    return points[hull.vertices].astype(np.float64)


def enforce_gripper_geometry_clearance(
    tcp_target_mm_degrees: Sequence[float],
    collision_envelope_tcp_m: np.ndarray,
    clearance_mm: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Raise only TCP z so the gripper envelope stays above the table plane."""
    target = _vector(tcp_target_mm_degrees, 6, "tcp_target_mm_degrees").copy()
    envelope = np.asarray(collision_envelope_tcp_m, dtype=np.float64)
    if (
        envelope.ndim != 2
        or envelope.shape[1] != 3
        or not np.all(np.isfinite(envelope))
    ):
        raise SafetyViolation(
            "Gripper collision envelope must have finite shape (N, 3)"
        )
    rotation = Rotation.from_euler("xyz", target[3:], degrees=True)
    world_offsets_mm = (rotation.as_matrix() @ envelope.T).T[:, 2] * 1000.0
    minimum_offset_mm = float(np.min(world_offsets_mm))
    required_tcp_z_mm = float(clearance_mm) - minimum_offset_mm
    raw_geometry_minimum_z_mm = float(target[2]) + minimum_offset_mm
    z_raise_mm = max(0.0, required_tcp_z_mm - float(target[2]))
    target[2] += z_raise_mm
    return (
        target,
        z_raise_mm,
        {
            "gripper_geometry_minimum_offset_mm": minimum_offset_mm,
            "gripper_geometry_required_tcp_z_mm": required_tcp_z_mm,
            "gripper_geometry_clearance_mm": float(clearance_mm),
            "raw_gripper_geometry_minimum_z_mm": raw_geometry_minimum_z_mm,
            "commanded_gripper_geometry_minimum_z_mm": (
                raw_geometry_minimum_z_mm + z_raise_mm
            ),
        },
    )


class SapienXArmIK:
    """Solve policy-space link7 targets with DeformGen's xArm7 model.

    DeformGen generates and executes its simulated demonstrations with SAPIEN's
    Pinocchio model.  Using the same model here avoids depending on the legacy
    controller IK implementation shipped by older xArm firmware.  This class
    is deliberately independent of the live robot: the caller supplies the
    measured joints as the IK seed and remains responsible for commanding the
    resulting joint target.
    """

    def __init__(
        self,
        urdf_path: Path,
        *,
        eef_link_name: str = "link7",
        position_tolerance_m: float = 0.001,
        rotation_matrix_tolerance: float = 0.01,
    ):
        if position_tolerance_m <= 0:
            raise ValueError("sapien_ik_position_tolerance_m must be positive")
        if rotation_matrix_tolerance <= 0:
            raise ValueError("sapien_ik_rotation_matrix_tolerance must be positive")
        if not urdf_path.is_file():
            raise FileNotFoundError(f"SAPIEN IK URDF does not exist: {urdf_path}")

        # Keep SAPIEN optional unless this solver is explicitly selected.
        try:
            import sapien.core as sapien
        except ImportError as error:
            raise RuntimeError(
                "SAPIEN is required when robot.ik_solver is 'sapien'"
            ) from error

        self._sapien = sapien
        self._engine = sapien.Engine()
        self._scene = self._engine.create_scene()
        loader = self._scene.create_urdf_loader()
        self._robot = loader.load(str(urdf_path))
        if self._robot is None:
            raise RuntimeError(f"SAPIEN could not load IK URDF: {urdf_path}")
        self._model = self._robot.create_pinocchio_model()
        self._eef_link_index = next(
            (
                index
                for index, link in enumerate(self._robot.get_links())
                if link.name == eef_link_name
            ),
            -1,
        )
        if self._eef_link_index < 0:
            raise ValueError(
                f"SAPIEN IK link {eef_link_name!r} is missing from {urdf_path}"
            )
        active_joints = self._robot.get_active_joints()
        if len(active_joints) != 7:
            raise ValueError(
                f"SAPIEN IK requires 7 active joints, found {len(active_joints)}"
            )
        limits = np.asarray(
            [joint.get_limits()[0] for joint in active_joints], dtype=np.float64
        )
        if limits.shape != (7, 2) or not np.all(np.isfinite(limits)):
            raise ValueError(f"Invalid SAPIEN joint limits: {limits!r}")
        self._joint_lower = limits[:, 0]
        self._joint_upper = limits[:, 1]
        self._active_qmask = np.ones(7, dtype=bool)
        self.position_tolerance_m = float(position_tolerance_m)
        self.rotation_matrix_tolerance = float(rotation_matrix_tolerance)
        self.urdf_path = urdf_path
        self.eef_link_name = eef_link_name

    def solve(
        self,
        policy_action: Sequence[float],
        current_joints_rad: Sequence[float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target = policy_pose_matrix(policy_action)
        current = _vector(current_joints_rad, 7, "current_joints_rad")
        result = self._model.compute_inverse_kinematics(
            link_index=self._eef_link_index,
            pose=self._sapien.Pose.from_transformation_matrix(target),
            initial_qpos=current,
            active_qmask=self._active_qmask,
        )
        if len(result) < 2 or not bool(result[1]):
            residual = None if len(result) < 3 else np.asarray(result[2]).tolist()
            raise SafetyViolation(
                "SAPIEN inverse kinematics failed"
                + ("" if residual is None else f"; residual={residual}")
            )
        joints = _vector(result[0], 7, "sapien_target_joints")
        joint_tolerance = 1e-6
        if np.any(joints < self._joint_lower - joint_tolerance) or np.any(
            joints > self._joint_upper + joint_tolerance
        ):
            raise SafetyViolation(
                "SAPIEN IK returned joints outside the xArm7 URDF limits: "
                f"joints={joints.tolist()}"
            )

        self._model.compute_forward_kinematics(joints)
        achieved = self._model.get_link_pose(
            self._eef_link_index
        ).to_transformation_matrix()
        position_error_m = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
        rotation_matrix_error = float(np.linalg.norm(achieved[:3, :3] - target[:3, :3]))
        if position_error_m > self.position_tolerance_m:
            raise SafetyViolation(
                "SAPIEN IK FK position residual "
                f"{1000.0 * position_error_m:.3f} mm exceeds "
                f"{1000.0 * self.position_tolerance_m:.3f} mm"
            )
        if rotation_matrix_error > self.rotation_matrix_tolerance:
            raise SafetyViolation(
                "SAPIEN IK FK rotation residual "
                f"{rotation_matrix_error:.6f} exceeds "
                f"{self.rotation_matrix_tolerance:.6f}"
            )
        joint_limit_margin_rad = float(
            np.min(
                np.minimum(
                    joints - self._joint_lower,
                    self._joint_upper - joints,
                )
            )
        )
        return joints, {
            "solver": "sapien_pinocchio",
            "urdf": str(self.urdf_path),
            "eef_link": self.eef_link_name,
            "position_error_m": position_error_m,
            "rotation_matrix_error": rotation_matrix_error,
            "joint_limit_margin_rad": joint_limit_margin_rad,
        }


class XArmPolicyController:
    """Configured xArm feedback, IK, gripper limits, and velocity execution."""

    def __init__(self, robot_config: dict[str, Any]):
        self.config = robot_config
        self.ip = str(robot_config["ip"])
        self.expected_axis_count = int(robot_config.get("expected_axis_count", 7))
        self.auto_enable_for_authorized_motion = bool(
            robot_config.get("auto_enable_for_authorized_motion", False)
        )
        self.gripper_scale = float(robot_config.get("gripper_scale", 800.0))
        self.use_controller_tcp_offset = bool(
            robot_config.get("use_controller_tcp_offset", True)
        )
        self.expected_controller_tcp_offset = _vector(
            robot_config.get(
                "expected_controller_tcp_offset_mm_degrees",
                [0, 0, 0, 0, 0, 0],
            ),
            6,
            "expected_controller_tcp_offset_mm_degrees",
        )
        self.reset_state = _vector(
            robot_config["reset_policy_state"], 8, "reset_policy_state"
        )
        self.flange_to_policy_eef = _vector(
            robot_config["flange_to_policy_eef_mm_degrees"],
            6,
            "flange_to_policy_eef_mm_degrees",
        )
        self.limits = SafetyLimits.from_config(robot_config["safety"])
        self.ik_solver_name = str(robot_config.get("ik_solver", "firmware"))
        if self.ik_solver_name not in {"firmware", "sapien"}:
            raise ValueError("robot.ik_solver must be either 'firmware' or 'sapien'")
        self.sapien_ik: SapienXArmIK | None = None
        self.firmware_ik_equivalent_euler_retry = bool(
            robot_config.get("firmware_ik_equivalent_euler_retry", False)
        )
        self.firmware_ik_retry_position_tolerance_mm = float(
            robot_config.get("firmware_ik_retry_position_tolerance_mm", 0.1)
        )
        self.firmware_ik_retry_orientation_tolerance_deg = float(
            robot_config.get("firmware_ik_retry_orientation_tolerance_deg", 0.01)
        )
        if self.ik_solver_name == "sapien":
            sapien_urdf = Path(
                robot_config.get("sapien_ik_urdf", "assets/robots/xarm/xarm7.urdf")
            ).expanduser()
            if not sapien_urdf.is_absolute():
                sapien_urdf = ROOT / sapien_urdf
            self.sapien_ik = SapienXArmIK(
                sapien_urdf.resolve(),
                eef_link_name=str(robot_config.get("sapien_ik_eef_link", "link7")),
                position_tolerance_m=float(
                    robot_config.get("sapien_ik_position_tolerance_m", 0.001)
                ),
                rotation_matrix_tolerance=float(
                    robot_config.get("sapien_ik_rotation_matrix_tolerance", 0.01)
                ),
            )
        geometry_urdf = Path(robot_config["gripper_collision_urdf"]).expanduser()
        if not geometry_urdf.is_absolute():
            geometry_urdf = ROOT / geometry_urdf
        self.gripper_collision_envelope = load_gripper_collision_envelope(
            geometry_urdf.resolve(),
            int(robot_config.get("gripper_geometry_opening_samples", 9)),
        )
        self.arm: XArmAPI | None = None
        self.controller_tcp_offset = np.zeros(6, dtype=np.float64)
        self.tcp_offset = np.zeros(6, dtype=np.float64)
        self.motion_started = False
        self.previous_command: np.ndarray | None = None
        self.previous_raw_action: np.ndarray | None = None
        self.last_gripper_discrete: str | None = None
        self.last_gripper_command_pulse: float | None = None

    def connect(self, *, configure_tcp_offset: bool = False) -> None:
        if self.arm is not None:
            raise RuntimeError("Controller is already connected")
        arm = XArmAPI(self.ip, is_radian=False, do_not_open=True)
        try:
            arm.connect()
        except Exception as error:
            arm.disconnect()
            raise SafetyViolation(
                f"Could not connect to xArm at {self.ip}: {error}"
            ) from error
        if not arm.connected:
            arm.disconnect()
            raise SafetyViolation(f"Could not connect to xArm at {self.ip}")
        if int(arm.axis) != self.expected_axis_count:
            arm.disconnect()
            raise SafetyViolation(
                f"Expected xArm{self.expected_axis_count}, reported axis={arm.axis}"
            )
        self.arm = arm
        # tcp_offset arrives on the asynchronous report channel and is commonly
        # still all-zero immediately after connect(). Wait for the configured
        # value instead of silently treating a stale report as calibration.
        deadline = time.monotonic() + 3.0
        configuration_attempted = False
        while True:
            self.controller_tcp_offset = _vector(
                arm.tcp_offset, 6, "controller_tcp_offset"
            )
            if np.allclose(
                self.controller_tcp_offset,
                self.expected_controller_tcp_offset,
                atol=np.asarray([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]),
                rtol=0.0,
            ):
                break
            if configure_tcp_offset and not configuration_attempted:
                # This changes only the volatile tool-coordinate setting; it
                # does not move the arm and deliberately does not call
                # save_conf(), so a controller reboot restores its saved value.
                set_code = arm.set_tcp_offset(
                    self.expected_controller_tcp_offset.tolist(),
                    is_radian=False,
                    wait=True,
                )
                configuration_attempted = True
                if set_code != 0:
                    arm.disconnect()
                    self.arm = None
                    raise SafetyViolation(
                        "Could not configure the required volatile xArm TCP "
                        f"offset: SDK code={set_code}"
                    )
                deadline = time.monotonic() + 3.0
            if time.monotonic() >= deadline:
                arm.disconnect()
                self.arm = None
                raise SafetyViolation(
                    "Controller TCP offset does not match the required xArm "
                    f"gripper TCP: expected {self.expected_controller_tcp_offset.tolist()}, "
                    f"reported {self.controller_tcp_offset.tolist()}. Configure it in "
                    "xArm Studio or explicitly request volatile configuration."
                )
            # Refresh report-backed properties without changing robot state.
            arm.get_position(is_radian=False)
            time.sleep(0.05)
        self.tcp_offset = (
            self.controller_tcp_offset.copy()
            if self.use_controller_tcp_offset
            else np.zeros(6, dtype=np.float64)
        )

    def disconnect(self) -> None:
        if self.arm is not None:
            self.arm.disconnect()
            self.arm = None

    def _connected_arm(self) -> XArmAPI:
        if self.arm is None or not self.arm.connected:
            raise SafetyViolation("xArm is not connected")
        return self.arm

    def status(self) -> dict[str, Any]:
        arm = self._connected_arm()
        state_code, state = arm.get_state()
        err_code, err_warn = arm.get_err_warn_code()
        return {
            "connected": bool(arm.connected),
            "mode": int(arm.mode),
            "state_sdk_code": int(state_code),
            "state": int(state),
            "err_warn_sdk_code": int(err_code),
            "controller_error": int(err_warn[0]),
            "controller_warning": int(err_warn[1]),
            "motor_enable_states": list(
                arm.motor_enable_states[: self.expected_axis_count]
            ),
            "motor_brake_states": list(
                arm.motor_brake_states[: self.expected_axis_count]
            ),
        }

    def verify_feedback_available(self) -> dict[str, Any]:
        """Require coherent live feedback without changing controller state."""
        arm = self._connected_arm()
        status = self.status()
        if status["state_sdk_code"] != 0 or status["err_warn_sdk_code"] != 0:
            raise SafetyViolation(f"xArm status API failed: {status}")
        if status["controller_error"] != 0 or status["controller_warning"] != 0:
            raise SafetyViolation(
                "xArm feedback is not trustworthy while the controller has an error/warning: "
                f"{status}. Release E-stop and clear the error in UFactory Studio, then rerun shadow."
            )
        if status["state"] == 4:
            raise SafetyViolation(
                "xArm feedback is not accepted while state=4 (stopped/E-stopped). "
                "Restore a clean controller state in UFactory Studio, then rerun shadow."
            )
        angle_code, angles = arm.get_servo_angle(is_radian=False)
        if angle_code != 0 or angles is None or not np.all(np.isfinite(angles)):
            raise SafetyViolation(
                f"xArm joint feedback is unavailable: SDK code={angle_code}, angles={angles}"
            )
        # The SDK also refreshes motor/brake properties while fetching joint
        # feedback, so sample status again instead of returning connect-time
        # cache values.
        status = self.status()
        if status["controller_error"] != 0 or status["controller_warning"] != 0:
            raise SafetyViolation(
                f"xArm developed an error/warning while refreshing feedback: {status}"
            )
        if status["state"] == 4:
            raise SafetyViolation("xArm entered state=4 while refreshing feedback")
        return {**status, "joint_angles_degrees": list(map(float, angles))}

    def read_policy_state(
        self, gripper_pulse_override: float | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        arm = self._connected_arm()
        pose_code, tcp_pose = arm.get_position(is_radian=False)
        if pose_code != 0 or tcp_pose is None:
            raise SafetyViolation(f"get_position failed with SDK code {pose_code}")
        baud_code, baudrate = arm.get_rs485_baudrate()
        if gripper_pulse_override is None:
            if baud_code != 0 or baudrate != EXPECTED_GRIPPER_BAUD:
                raise SafetyViolation(
                    f"Tool RS485 must be {EXPECTED_GRIPPER_BAUD}; "
                    f"got code={baud_code}, value={baudrate}"
                )
            gripper_code, pulse = arm.get_gripper_position(check_baud=False)
            if gripper_code != 0 or pulse is None:
                raise SafetyViolation(
                    f"get_gripper_position failed with SDK code {gripper_code}"
                )
            gripper_source = "verified_xarm_feedback"
        else:
            pulse = float(gripper_pulse_override)
            if not 0 <= pulse <= 850:
                raise SafetyViolation("gripper_pulse_override must be in [0, 850]")
            gripper_code = None
            gripper_source = "explicit_shadow_override"

        xyz, rotation = eef_policy_pose_from_xarm(
            tcp_pose, self.tcp_offset, self.flange_to_policy_eef
        )
        gripper = float(np.clip(1.0 - float(pulse) / self.gripper_scale, 0.0, 1.0))
        state = np.concatenate([xyz, wxyz_from_rotation(rotation), [gripper]])
        details = {
            "tcp_pose_mm_degrees": list(map(float, tcp_pose)),
            "tcp_offset_mm_degrees": self.tcp_offset.tolist(),
            "controller_tcp_offset_mm_degrees": self.controller_tcp_offset.tolist(),
            "use_controller_tcp_offset": self.use_controller_tcp_offset,
            "flange_to_policy_eef_mm_degrees": self.flange_to_policy_eef.tolist(),
            "gripper_pulse": float(pulse),
            "gripper_feedback_sdk_code": gripper_code,
            "gripper_source": gripper_source,
            "rs485_baudrate_sdk_code": int(baud_code),
            "rs485_baudrate": int(baudrate),
        }
        return state.astype(np.float64), details

    def verify_reset_pose(self, state: Sequence[float]) -> dict[str, float]:
        measured = _vector(state, 8, "measured state")
        position_error = float(np.linalg.norm(measured[:3] - self.reset_state[:3]))
        rotation_error = rotation_distance_degrees(
            rotation_from_wxyz(measured[3:7]), rotation_from_wxyz(self.reset_state[3:7])
        )
        if position_error > self.limits.reset_position_tolerance_m:
            raise SafetyViolation(
                f"Robot is {position_error:.4f} m from reset pose; limit is "
                f"{self.limits.reset_position_tolerance_m:.4f} m"
            )
        if rotation_error > self.limits.reset_orientation_tolerance_deg:
            raise SafetyViolation(
                f"Robot orientation is {rotation_error:.2f} deg from reset; limit is "
                f"{self.limits.reset_orientation_tolerance_deg:.2f} deg"
            )
        if (
            self.limits.reset_gripper_must_be_open
            and measured[7] > self.limits.gripper_open_threshold
        ):
            raise SafetyViolation(
                f"Gripper must be open at reset; policy-space feedback is {measured[7]:.3f}"
            )
        return {
            "reset_position_error_m": position_error,
            "reset_rotation_error_deg": rotation_error,
        }

    def verify_ready_for_motion(self) -> dict[str, Any]:
        # Reading servo angles first also refreshes Cartesian feedback on this
        # controller/SDK combination; a fresh connection may otherwise expose
        # a stale cached pose even though get_position() reports SDK code 0.
        initial_status = self.status()
        if (
            initial_status["state_sdk_code"] != 0
            or initial_status["err_warn_sdk_code"] != 0
            or initial_status["controller_error"] != 0
            or initial_status["controller_warning"] != 0
        ):
            raise SafetyViolation(
                "Automatic enable is forbidden while controller status is not clean: "
                f"{initial_status}"
            )
        auto_enable_details: dict[str, Any] = {
            "auto_enable_requested": False,
            "auto_enable_performed": False,
        }
        if not all(value == 1 for value in initial_status["motor_enable_states"]):
            if not self.auto_enable_for_authorized_motion:
                raise SafetyViolation(
                    "Robot motors are not enabled. Enable them in UFactory Studio or "
                    "explicitly configure auto_enable_for_authorized_motion."
                )
            auto_enable_details["auto_enable_requested"] = True
            arm = self._connected_arm()
            enable_code = arm.motion_enable(True)
            mode_code = arm.set_mode(0) if enable_code == 0 else None
            state_code = arm.set_state(0) if mode_code == 0 else None
            auto_enable_details.update(
                {
                    "motion_enable_sdk_code": enable_code,
                    "normal_mode_sdk_code": mode_code,
                    "ready_state_sdk_code": state_code,
                }
            )
            if enable_code != 0 or mode_code != 0 or state_code != 0:
                arm.motion_enable(False)
                raise SafetyViolation(
                    "xArm refused authorized automatic enable: "
                    f"motion_enable={enable_code}, set_mode={mode_code}, "
                    f"set_state={state_code}"
                )
            deadline = time.monotonic() + 2.0
            while True:
                # Refresh report-backed motor and state fields.
                arm.get_servo_angle(is_radian=False)
                enabled_status = self.status()
                if (
                    enabled_status["controller_error"] == 0
                    and enabled_status["controller_warning"] == 0
                    and enabled_status["state"] != 4
                    and all(
                        value == 1 for value in enabled_status["motor_enable_states"]
                    )
                ):
                    auto_enable_details["auto_enable_performed"] = True
                    break
                if time.monotonic() >= deadline:
                    arm.motion_enable(False)
                    raise SafetyViolation(
                        "xArm did not report enabled motors within 2 s: "
                        f"{enabled_status}"
                    )
                time.sleep(0.02)

        try:
            status = self.verify_feedback_available()
            if not all(value == 1 for value in status["motor_enable_states"]):
                raise SafetyViolation(
                    f"Robot motors are not enabled after readiness preparation: {status}"
                )
            state, details = self.read_policy_state()
            reset_metrics = self.verify_reset_pose(state)
        except Exception:
            if auto_enable_details["auto_enable_performed"]:
                arm = self._connected_arm()
                for operation in (
                    lambda: arm.set_mode(0),
                    lambda: arm.set_state(4),
                    lambda: arm.motion_enable(False),
                ):
                    try:
                        operation()
                    except Exception:
                        pass
            raise
        self.last_gripper_discrete = "closed" if state[7] >= 0.5 else "open"
        self.last_gripper_command_pulse = float(details["gripper_pulse"])
        return {
            **status,
            **auto_enable_details,
            **details,
            **reset_metrics,
            "policy_state": state.tolist(),
        }

    def start_motion(self) -> None:
        arm = self._connected_arm()
        self.verify_ready_for_motion()
        # Real2Sim-Eval's xArm-equivalent controller converts each absolute
        # Cartesian target to IK, clips the joint error, then commands joint
        # velocity. xArm mode 4 is the SDK's joint-velocity mode.
        mode_code = arm.set_mode(4)
        state_code = arm.set_state(0) if mode_code == 0 else None
        if mode_code != 0 or state_code != 0:
            self.stop_motion()
            raise SafetyViolation(
                f"Could not enter servo mode: set_mode={mode_code}, set_state={state_code}"
            )
        # This firmware does not publish mode changes back into the SDK's
        # ``arm.mode`` cache.  Confirm readiness with the live controller state
        # plus a zero-velocity command, whose response comes from the motion
        # controller and cannot move the robot.
        report_deadline = time.monotonic() + 1.0
        while True:
            state_report_code, reported_state = arm.get_state()
            if state_report_code == 0 and int(reported_state) in {0, 1, 2}:
                break
            if time.monotonic() >= report_deadline:
                self.stop_motion()
                raise SafetyViolation(
                    "xArm did not enter a velocity-command-ready state: "
                    f"state_code={state_report_code}, state={reported_state}"
                )
            time.sleep(0.01)
        zero_velocity_code = arm.vc_set_joint_velocity(
            [0.0] * self.expected_axis_count,
            is_radian=True,
            is_sync=True,
            duration=self.limits.velocity_command_duration_s,
        )
        if zero_velocity_code != 0:
            self.stop_motion()
            raise SafetyViolation(
                "xArm did not accept the joint-velocity mode probe: "
                f"SDK code={zero_velocity_code}"
            )
        self.motion_started = True
        self.previous_command = None
        self.previous_raw_action = None
        if self.last_gripper_command_pulse is None:
            _, details = self.read_policy_state()
            self.last_gripper_command_pulse = float(details["gripper_pulse"])

    def stop_motion(self, *, disable: bool = True) -> None:
        arm = self.arm
        self.motion_started = False
        self.previous_command = None
        self.previous_raw_action = None
        self.last_gripper_command_pulse = None
        if arm is None or not arm.connected:
            return
        # First command zero velocity, then leave velocity mode. Interactive
        # operator stops can retain servo hold for a fast reset/restart; faults
        # and non-interactive callers keep the conservative disable behavior.
        operations = [
            lambda: arm.vc_set_joint_velocity(
                [0.0] * self.expected_axis_count,
                is_radian=True,
                duration=self.limits.velocity_command_duration_s,
            ),
            lambda: arm.set_mode(0),
        ]
        if disable:
            operations.extend(
                [
                    lambda: arm.set_state(4),
                    lambda: arm.motion_enable(False),
                ]
            )
        else:
            operations.append(lambda: arm.set_state(0))
        for operation in operations:
            try:
                operation()
            except Exception:
                pass

    def begin_action_chunk(self) -> None:
        """Confirm that execution is active before consuming a new plan."""
        if not self.motion_started:
            raise SafetyViolation("Cannot begin an action chunk before motion starts")

    def _assert_runtime_status(self) -> dict[str, Any]:
        status = self.status()
        if (
            status["state_sdk_code"] != 0
            or status["err_warn_sdk_code"] != 0
            or status["controller_error"] != 0
            or status["controller_warning"] != 0
            or status["state"] == 4
        ):
            raise SafetyViolation(f"Runtime xArm status is unsafe: {status}")
        return status

    def execute_policy_action(self, raw_action: Sequence[float]) -> dict[str, Any]:
        if not self.motion_started:
            raise SafetyViolation("Motion was not started")
        arm = self._connected_arm()
        started = time.perf_counter()
        status = self._assert_runtime_status()
        status_seconds = time.perf_counter() - started
        current, feedback = self.read_policy_state()
        feedback_seconds = time.perf_counter() - started - status_seconds
        policy_action = _vector(raw_action, 8, "raw_action").copy()
        # The only Cartesian action modification is a z raise derived from the
        # full gripper collision envelope and calibrated table plane.
        raw_tcp_target = xarm_tcp_pose_from_policy_action(
            policy_action, self.tcp_offset, self.flange_to_policy_eef
        )
        tcp_target, geometry_z_raise_mm, geometry = enforce_gripper_geometry_clearance(
            raw_tcp_target,
            self.gripper_collision_envelope,
            self.limits.gripper_geometry_clearance_mm,
        )
        commanded_policy_action = policy_action.copy()
        commanded_policy_action[2] += geometry_z_raise_mm / 1000.0

        joint_code, current_joints = arm.get_servo_angle(is_radian=True)
        if joint_code != 0 or current_joints is None:
            raise SafetyViolation(
                f"get_servo_angle failed before velocity command: code={joint_code}"
            )
        current_joints_array = _vector(
            current_joints[: self.expected_axis_count],
            self.expected_axis_count,
            "current_joints",
        )
        ik_sdk_code: int | None = None
        if self.ik_solver_name == "sapien":
            if self.sapien_ik is None:
                raise SafetyViolation("SAPIEN IK solver was not initialized")
            target_joints_array, ik_details = self.sapien_ik.solve(
                commanded_policy_action,
                current_joints_array,
            )
        else:
            target_joints_array, ik_details = solve_xarm_firmware_ik(
                arm,
                tcp_target,
                current_joints_array,
                expected_axis_count=self.expected_axis_count,
                retry_equivalent_euler=(self.firmware_ik_equivalent_euler_retry),
                retry_position_tolerance_mm=(
                    self.firmware_ik_retry_position_tolerance_mm
                ),
                retry_orientation_tolerance_deg=(
                    self.firmware_ik_retry_orientation_tolerance_deg
                ),
            )
            ik_sdk_code = int(ik_details["sdk_code"])
        ik_complete_seconds = time.perf_counter() - started
        joint_delta = target_joints_array - current_joints_array
        joint_delta = np.arctan2(np.sin(joint_delta), np.cos(joint_delta))
        joint_delta_norm = float(np.linalg.norm(joint_delta))
        clipped_joint_delta = joint_delta.copy()
        if joint_delta_norm > self.limits.max_joint_delta_norm_rad:
            clipped_joint_delta *= (
                self.limits.max_joint_delta_norm_rad / joint_delta_norm
            )
        joint_velocity = (
            clipped_joint_delta
            / self.limits.command_check_interval_s
            * self.limits.velocity_gain
        )
        arm_code = arm.vc_set_joint_velocity(
            joint_velocity.tolist(),
            is_radian=True,
            is_sync=True,
            duration=self.limits.velocity_command_duration_s,
        )
        if arm_code != 0:
            raise SafetyViolation(
                f"vc_set_joint_velocity failed with SDK code {arm_code}"
            )
        velocity_command_complete_seconds = time.perf_counter() - started

        gripper_code: int | None = None
        desired_pulse = self.limits.gripper_open_pulse + float(policy_action[7]) * (
            self.limits.gripper_closed_pulse - self.limits.gripper_open_pulse
        )
        previous_pulse = (
            float(feedback["gripper_pulse"])
            if self.last_gripper_command_pulse is None
            else self.last_gripper_command_pulse
        )
        max_pulse_step = (
            abs(self.limits.gripper_open_pulse - self.limits.gripper_closed_pulse)
            * self.limits.gripper_max_normalized_step
        )
        pulse_delta = float(
            np.clip(desired_pulse - previous_pulse, -max_pulse_step, max_pulse_step)
        )
        commanded_pulse = float(
            np.clip(
                previous_pulse + pulse_delta,
                min(
                    self.limits.gripper_open_pulse,
                    self.limits.gripper_closed_pulse,
                ),
                max(
                    self.limits.gripper_open_pulse,
                    self.limits.gripper_closed_pulse,
                ),
            )
        )
        if abs(commanded_pulse - previous_pulse) >= 0.5:
            gripper_code = arm.set_gripper_position(
                int(round(commanded_pulse)),
                wait=False,
                speed=self.limits.gripper_speed,
                auto_enable=True,
                check_baud=False,
                # Joint velocity mode is intentionally active here.  The SDK's
                # default wait_move synchronization adds avoidable latency and
                # serializes an otherwise non-blocking gripper command.
                wait_motion=False,
            )
            if gripper_code != 0:
                raise SafetyViolation(
                    f"set_gripper_position failed with SDK code {gripper_code}"
                )
            self.last_gripper_command_pulse = commanded_pulse

        elapsed = time.perf_counter() - started
        if elapsed > self.limits.command_timeout_s:
            raise SafetyViolation(
                f"xArm command cycle took {elapsed:.3f}s; limit is {self.limits.command_timeout_s:.3f}s"
            )
        self.previous_command = commanded_policy_action.copy()
        self.previous_raw_action = _vector(raw_action, 8, "raw_action").copy()
        return {
            "status": status,
            "feedback": feedback,
            "current_policy_state": current.tolist(),
            "raw_action": _vector(raw_action, 8, "raw_action").tolist(),
            "commanded_policy_action": commanded_policy_action.tolist(),
            "policy_action_modified": geometry_z_raise_mm > 0.0,
            "policy_action_modification": (
                "gripper_geometry_z_floor" if geometry_z_raise_mm > 0.0 else None
            ),
            "raw_tcp_target_mm_degrees": raw_tcp_target.tolist(),
            "tcp_target_mm_degrees": tcp_target.tolist(),
            "geometry_z_raise_mm": geometry_z_raise_mm,
            "gripper_geometry": geometry,
            "arm_sdk_code": int(arm_code),
            "gripper_sdk_code": None if gripper_code is None else int(gripper_code),
            "gripper_desired_pulse": desired_pulse,
            "gripper_commanded_pulse": commanded_pulse,
            "joint_sdk_code": int(joint_code),
            "ik_solver": self.ik_solver_name,
            "ik_sdk_code": ik_sdk_code,
            "ik_details": ik_details,
            "current_joints_rad": current_joints_array.tolist(),
            "target_joints_rad": target_joints_array.tolist(),
            "joint_delta_norm_rad": joint_delta_norm,
            "clipped_joint_delta_rad": clipped_joint_delta.tolist(),
            "joint_velocity_rad_s": joint_velocity.tolist(),
            "command_seconds": elapsed,
            "timing_seconds": {
                "status": status_seconds,
                "feedback": feedback_seconds,
                "ik_and_joint_feedback": (
                    ik_complete_seconds - status_seconds - feedback_seconds
                ),
                "velocity_command": (
                    velocity_command_complete_seconds - ik_complete_seconds
                ),
                "gripper_and_checks": elapsed - velocity_command_complete_seconds,
            },
        }
