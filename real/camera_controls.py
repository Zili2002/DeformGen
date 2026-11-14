"""Apply the same RGB sensor controls in camera previews and policy runners."""

from typing import Any

import pyrealsense2 as rs


def apply_color_controls(
    profile: rs.pipeline_profile, camera: dict[str, Any], role: str
) -> None:
    options = {
        "auto_exposure": rs.option.enable_auto_exposure,
        "exposure": rs.option.exposure,
        "gain": rs.option.gain,
        "auto_white_balance": rs.option.enable_auto_white_balance,
        "white_balance": rs.option.white_balance,
    }
    requested = {key: option for key, option in options.items() if key in camera}
    if not requested:
        return
    if "exposure" in camera and camera.get("auto_exposure", True):
        raise ValueError(f"{role}: exposure requires auto_exposure: false")
    if "white_balance" in camera and camera.get("auto_white_balance", True):
        raise ValueError(f"{role}: white_balance requires auto_white_balance: false")
    # D455's depth sensor also supports exposure: choose by the COLOR stream.
    # D405 delivers its color stream through the Stereo Module.
    sensor = next(
        (
            candidate
            for candidate in profile.get_device().query_sensors()
            if any(
                p.stream_type() == rs.stream.color
                for p in candidate.get_stream_profiles()
            )
            and all(candidate.supports(option) for option in requested.values())
        ),
        None,
    )
    if sensor is None:
        raise RuntimeError(f"{role}: no color sensor supports {list(requested)}")
    for key, option in requested.items():
        sensor.set_option(option, float(camera[key]))
    actual = {key: sensor.get_option(option) for key, option in requested.items()}
    print(f"{role} SN {camera['serial']} RGB controls: {actual}", flush=True)
