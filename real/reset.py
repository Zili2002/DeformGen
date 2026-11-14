"""Return xArm to the verified common DeformGen reset TCP pose."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from xarm.wrapper import XArmAPI

from .config import ROOT, load_config, validate_hardware
from .controller import (
    XArmPolicyController,
    require_motion_authorization,
    xarm_tcp_pose_from_policy_action,
)


def reset_robot(
    robot: dict,
    *,
    recover_collision_31: bool = False,
    release_before_lift: bool = True,
    disable_after: bool = False,
) -> dict[str, float]:
    """Run the verified release/lift/reset sequence after caller authorization."""
    adapter = XArmPolicyController(robot)
    reset_speed = float(robot["safety"].get("reset_cartesian_speed_mm_s", 30.0))
    reset_acceleration = float(
        robot["safety"].get("reset_cartesian_acceleration_mm_s2", 100.0)
    )
    if not 1.0 <= reset_speed <= 500.0:
        raise ValueError("reset_cartesian_speed_mm_s must be in [1, 500]")
    if not 1.0 <= reset_acceleration <= 5000.0:
        raise ValueError("reset_cartesian_acceleration_mm_s2 must be in [1, 5000]")
    arm = XArmAPI(str(robot["ip"]), is_radian=False, do_not_open=True)
    motion_enabled = False
    succeeded = False
    try:
        arm.connect()
        if not arm.connected or int(arm.axis) != int(
            robot.get("expected_axis_count", 7)
        ):
            raise RuntimeError("Could not verify the expected xArm7")
        state_code, state = arm.get_state()
        err_code, err_warn = arm.get_err_warn_code()
        if (
            recover_collision_31
            and state_code == 0
            and err_code == 0
            and err_warn == [31, 0]
        ):
            clean_code = arm.clean_error()
            if clean_code != 0:
                raise RuntimeError(
                    f"Could not clear the explicitly authorized C31: {clean_code}"
                )
            arm.set_mode(0)
            deadline = time.monotonic() + 2.0
            while True:
                err_code, err_warn = arm.get_err_warn_code()
                if err_code == 0 and err_warn == [0, 0]:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"C31 did not clear within 2 s: {err_warn}")
                time.sleep(0.02)
            state_code, state = arm.get_state()
        angle_code, angles = arm.get_servo_angle(is_radian=False)
        pose_code, current = arm.get_position(is_radian=False)
        if state_code != 0 or err_code != 0 or err_warn != [0, 0]:
            raise RuntimeError(
                f"Controller must be error-free: state={state}, err_warn={err_warn}, "
                f"sdk_codes={[state_code, err_code]}"
            )
        if angle_code != 0 or pose_code != 0 or angles is None or current is None:
            raise RuntimeError(
                f"Feedback unavailable: angles={angle_code}, position={pose_code}"
            )
        current = np.asarray(current, dtype=np.float64)
        guard_min = np.asarray(
            robot["safety"].get("reset_guard_tcp_min_mm", [150.0, -300.0, 0.0]),
            dtype=np.float64,
        )
        guard_max = np.asarray(
            robot["safety"].get("reset_guard_tcp_max_mm", [700.0, 300.0, 450.0]),
            dtype=np.float64,
        )
        if (
            guard_min.shape != (3,)
            or guard_max.shape != (3,)
            or np.any(guard_min >= guard_max)
        ):
            raise ValueError(
                "reset_guard_tcp_min_mm/max_mm must be ordered xyz vectors"
            )
        if (
            not np.all(np.isfinite(current))
            or np.any(current[:3] < guard_min)
            or np.any(current[:3] > guard_max)
        ):
            raise RuntimeError(
                "Current TCP is outside reset-script guard bounds: "
                f"xyz={current[:3].tolist()}, min={guard_min.tolist()}, "
                f"max={guard_max.tolist()}"
            )

        command_tcp_offset = (
            adapter.expected_controller_tcp_offset
            if adapter.use_controller_tcp_offset
            else np.zeros(6, dtype=np.float64)
        )
        reset_tcp = xarm_tcp_pose_from_policy_action(
            adapter.reset_state,
            command_tcp_offset,
            adapter.flange_to_policy_eef,
        )
        lift = current.copy()
        lift[2] = max(current[2], min(float(reset_tcp[2]), current[2] + 100.0))

        if arm.motion_enable(True) != 0:
            raise RuntimeError("motion_enable(True) failed")
        motion_enabled = True
        if arm.set_mode(0) != 0 or arm.set_state(0) != 0:
            raise RuntimeError("Could not enter position mode/state")
        if release_before_lift:
            release_code = arm.set_gripper_position(
                800,
                wait=True,
                speed=1000,
                auto_enable=True,
                check_baud=False,
            )
            if release_code != 0:
                raise RuntimeError(
                    f"Could not release gripper before lift: SDK code {release_code}"
                )
        for name, target in (("lift", lift), ("reset", reset_tcp)):
            code = arm.set_position(
                *target.tolist(),
                speed=reset_speed,
                mvacc=reset_acceleration,
                wait=True,
                timeout=30,
                is_radian=False,
            )
            if code != 0:
                raise RuntimeError(f"{name} move failed with SDK code {code}")
            print(f"{name} target reached: {target.tolist()}")
        gripper_code = arm.set_gripper_position(
            800,
            wait=True,
            speed=1000,
            auto_enable=True,
            check_baud=False,
        )
        if gripper_code != 0:
            raise RuntimeError(f"Could not open gripper: SDK code {gripper_code}")
        _, final = arm.get_position(is_radian=False)
        error_mm = float(np.linalg.norm(np.asarray(final[:3]) - reset_tcp[:3]))
        if error_mm > 3.0:
            raise RuntimeError(f"Reset TCP error {error_mm:.2f} mm exceeds 3 mm")
        print(
            "DeformGen reset complete; "
            f"speed={reset_speed:.0f} mm/s, acceleration={reset_acceleration:.0f} mm/s^2, "
            f"TCP translation error={error_mm:.2f} mm"
        )
        succeeded = True
        return {
            "translation_error_mm": error_mm,
            "reset_tcp_x_mm": float(reset_tcp[0]),
            "reset_tcp_y_mm": float(reset_tcp[1]),
            "reset_tcp_z_mm": float(reset_tcp[2]),
            "reset_cartesian_speed_mm_s": reset_speed,
            "reset_cartesian_acceleration_mm_s2": reset_acceleration,
        }
    finally:
        if motion_enabled and (disable_after or not succeeded):
            for operation in (
                lambda: arm.set_mode(0),
                lambda: arm.set_state(4),
                lambda: arm.motion_enable(False),
            ):
                try:
                    operation()
                except Exception:
                    pass
        arm.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "real/robot.local.yaml")
    parser.add_argument("--task", choices=("cloth", "rope", "sloth"), default="cloth")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument("--confirm")
    parser.add_argument(
        "--recover-collision-31",
        action="store_true",
        help="clear only xArm collision error C31, then lift vertically before reset",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.task)
    validate_hardware(config, motion=True)
    robot = config["robot"]
    require_motion_authorization(args.allow_motion, args.confirm)

    reset_robot(
        robot,
        recover_collision_31=args.recover_collision_31,
        release_before_lift=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
