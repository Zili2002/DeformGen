from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf


_CASE_DEFAULTS = {
    "rope": ("rope", "log/phystwin/rope", "rope_0001"),
    "sloth": ("sloth", "log/phystwin/sloth", "sloth_0001"),
    "cloth3": ("cloth3", "log/phystwin/cloth3", "cloth3_0001"),
}

_RUNTIME_MODE_TO_BACKEND = {
    "runtime-random": "teleop_random",
    "runtime-fixed": "fixed",
}

_STATE_MODES = {"default-state", "configured-grid", "uniform-grid"}

# Legacy aliases remain accepted so existing batch commands do not silently change behavior.
_MODE_ALIASES = {
    "runtime-random": "runtime-random",
    "random": "runtime-random",
    "teleop-random": "runtime-random",
    "teleop_random": "runtime-random",
    "runtime-fixed": "runtime-fixed",
    "fixed": "runtime-fixed",
    "default-state": "default-state",
    "default": "default-state",
    "configured-grid": "configured-grid",
    "grid": "configured-grid",
    "uniform-grid": "uniform-grid",
    "grid-rigid": "uniform-grid",
}


def _csv_floats(value: str, expected: int, name: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated values, got {value!r}")
    return vals


def _csv_float_list(value: str, name: str) -> list[float]:
    vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} expects at least one comma-separated value, got {value!r}")
    return vals


def _linspace(start: float, end: float, count: int, name: str) -> list[float]:
    if count < 1:
        raise ValueError(f"{name} must be positive, got {count}.")
    if count == 1:
        return [(start + end) / 2.0]
    return [start + idx * (end - start) / float(count - 1) for idx in range(count)]


def _format_hydra_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def _resolve_asset_root(args: argparse.Namespace, repo_root: Path) -> Path:
    """Return the internal DeformGen asset root.

    External asset roots are deprecated because runtime code and relative assets
    must resolve from the same repository root.
    """
    if args.asset_root is not None:
        print(
            "deformgen-perturb: --asset-root is deprecated and ignored; "
            "runtime assets are loaded from DeformGen/log.",
            file=sys.stderr,
        )
    if not (repo_root / "log").is_dir():
        raise FileNotFoundError(f"Missing internal runtime assets: {repo_root / 'log'}")
    return repo_root


def _load_case_profile(repo_root: Path, case: str) -> dict[str, Any]:
    """Load the documented case-specific runtime perturbation defaults."""
    config_path = repo_root / "cfg" / "augmentation.yaml"
    cfg = OmegaConf.load(config_path)
    profiles = cfg.get("case_defaults", None)
    if profiles is None or case not in profiles:
        raise ValueError(f"No case_defaults profile for case={case!r} in {config_path}.")
    profile = OmegaConf.to_container(profiles[case], resolve=True)
    if not isinstance(profile, dict):
        raise TypeError(f"Invalid case_defaults profile for {case!r}.")
    return profile


def _resolve_demo_path(
    args: argparse.Namespace,
    asset_root: Path,
    profile: dict[str, Any],
) -> Path:
    """Resolve an explicit demo path or the documented case-default demo."""
    raw_demo = args.demo if args.demo is not None else profile["demo"]["gt_dir"]
    demo = Path(raw_demo).expanduser()
    if not demo.is_absolute():
        demo = asset_root / demo
    if not demo.is_dir():
        raise FileNotFoundError(f"Demo trajectory directory does not exist: {demo}")
    return demo


def _runtime_profile_values(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Resolve case defaults, then apply typed CLI overrides."""
    cfg = OmegaConf.load(repo_root / "cfg" / "augmentation.yaml")
    profile = _load_case_profile(repo_root, args.case)
    perturbation = profile["perturbation"]
    release = profile["release"]
    replay_init = profile["replay_init"]
    output = profile["output"]
    seed = profile["seed"]
    batch_seed = (
        int(args.seed)
        if args.seed is not None
        else int(seed["base"]) + int(args.seed_shard_index) * int(seed["stride"])
    )
    return {
        "batch_seed": batch_seed,
        "reset_seed": int(args.reset_seed) if args.reset_seed is not None else seed["reset"],
        "random_steps": int(args.random_steps or perturbation["random_steps"]),
        "translation_step_sizes": (
            _csv_float_list(args.translation_step_sizes, "--translation-step-sizes")
            if args.translation_step_sizes is not None
            else [float(value) for value in perturbation["translation_step_sizes"]]
        ),
        "rotation_step_deg": float(
            args.rotation_step_deg
            if args.rotation_step_deg is not None
            else perturbation["rotation_step_deg"]
        ),
        "rotation_prob": float(
            args.rotation_prob if args.rotation_prob is not None else perturbation["rotation_prob"]
        ),
        "gripper_state": float(
            args.gripper_state if args.gripper_state is not None else 1.0
        ),
        "release_enabled": bool(release["enabled"]) if args.release is None else bool(args.release),
        "release_waypoints": int(
            args.release_waypoints
            if args.release_waypoints is not None
            else release["num_waypoints"]
        ),
        "release_gripper_open": float(
            args.release_gripper_open
            if args.release_gripper_open is not None
            else release["gripper_open"]
        ),
        "replay_init_index": int(
            args.replay_init_index
            if args.replay_init_index is not None
            else replay_init["index"]
        ),
        "init_stabilization_steps": int(
            args.init_stabilization_steps
            if args.init_stabilization_steps is not None
            else replay_init["stabilization_steps"]
        ),
        "run_demo_tail": bool(replay_init["run_demo_tail"])
        if args.run_demo_tail is None
        else bool(args.run_demo_tail),
        "final_stabilization_steps": int(
            args.stabilization_steps
            if args.stabilization_steps is not None
            else output["final_state_stabilization_steps"]
        ),
        "replay_randomize": (
            bool(profile.get("replay_randomize", cfg.replay_randomize))
            if args.randomize is None
            else bool(args.randomize)
        ),
        "make_videos": True if args.make_videos is None else bool(args.make_videos),
    }


def _state_stabilization_steps(args: argparse.Namespace, repo_root: Path) -> int:
    """Use the grid/default-state stabilization default unless explicitly overridden."""
    if args.stabilization_steps is not None:
        return int(args.stabilization_steps)
    cfg = OmegaConf.load(repo_root / "cfg" / "augmentation.yaml")
    return int(cfg.state_synthesis.final_state_stabilization_steps)


def _resolve_episode_dir(demo: Path, episode_id: int) -> Path:
    episode_dir = demo / f"episode_{episode_id:04d}"
    if (episode_dir / "robot").is_dir():
        return episode_dir
    if (demo / "robot").is_dir():
        return demo
    raise FileNotFoundError(
        "Expected either <demo>/episode_XXXX/robot or <demo>/robot, "
        f"but neither exists under {demo}."
    )


def _resolve_initial_robot_json(demo: Path, episode_id: int) -> Path:
    episode_dir = _resolve_episode_dir(demo, episode_id)
    robot_paths = sorted((episode_dir / "robot").glob("*.json"))
    if not robot_paths:
        raise FileNotFoundError(f"No robot JSON files in {episode_dir / 'robot'}.")
    return robot_paths[0]


def _load_case_grid(repo_root: Path, case: str) -> list[dict[str, Any]]:
    config_path = repo_root / "cfg" / "gs" / f"{case}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Grid configuration not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    grid_cfg = cfg.object.get("grid_randomization", None)
    if grid_cfg is None:
        raise ValueError(f"{config_path} has no object.grid_randomization section.")

    xy_values = [[float(pair[0]), float(pair[1])] for pair in grid_cfg.xy]
    theta_values = [float(theta) for theta in grid_cfg.theta]
    one_to_one = bool(grid_cfg.get("one_to_one", False))
    if not xy_values or not theta_values:
        raise ValueError(f"{config_path} has an empty grid_randomization definition.")
    if one_to_one:
        if len(xy_values) != len(theta_values):
            raise ValueError(
                f"{config_path} uses one_to_one=true but has {len(xy_values)} XY values "
                f"and {len(theta_values)} theta values."
            )
        return [
            {"grid_index": idx, "xy": xy, "theta_deg": theta, "grid_source": "case_config"}
            for idx, (xy, theta) in enumerate(zip(xy_values, theta_values))
        ]
    return [
        {"grid_index": idx, "xy": xy, "theta_deg": theta, "grid_source": "case_config"}
        for idx, (xy, theta) in enumerate(product(xy_values, theta_values))
    ]


def _build_rigid_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    x_min, x_max = _csv_floats(args.grid_x_range, 2, "--grid-x-range")
    y_min, y_max = _csv_floats(args.grid_y_range, 2, "--grid-y-range")
    theta_min, theta_max = _csv_floats(args.grid_theta_range, 2, "--grid-theta-range")
    xs = _linspace(x_min, x_max, args.grid_nx, "--grid-nx")
    ys = _linspace(y_min, y_max, args.grid_ny, "--grid-ny")
    thetas = _linspace(theta_min, theta_max, args.grid_ntheta, "--grid-ntheta")
    return [
        {
            "grid_index": idx,
            "xy": [round(x, 8), round(y, 8)],
            "theta_deg": round(theta, 8),
            "grid_source": "cli_rigid_grid",
        }
        for idx, (x, y, theta) in enumerate(product(xs, ys, thetas))
    ]


def _select_grid_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if args.grid_start < 0:
        raise ValueError("--grid-start must be non-negative.")
    selected = samples[args.grid_start:]
    if args.num_states is not None:
        if args.num_states < 1:
            raise ValueError("--num-states must be positive when provided.")
        selected = selected[:args.num_states]
    if not selected:
        raise ValueError("Grid selection is empty; check --grid-start and --num-states.")
    return selected


def build_runtime_command(args: argparse.Namespace) -> list[str]:
    """Build a case-profiled runtime-perturbation backend command."""
    repo_root = Path(__file__).resolve().parents[2]
    backend = repo_root / "experiments" / "generate_augmented_dataset.py"
    if not backend.exists():
        raise FileNotFoundError(f"Cannot find perturbation backend: {backend}")
    asset_root = _resolve_asset_root(args, repo_root)
    profile = _load_case_profile(repo_root, args.case)
    demo = _resolve_demo_path(args, asset_root, profile)
    runtime = _runtime_profile_values(args, repo_root)

    gs, ckpt_path, case_name = _CASE_DEFAULTS[args.case]
    trans = (
        _csv_floats(args.translation_range_xy, 2, "--translation-range-xy")
        if args.translation_range_xy is not None
        else None
    )
    rot = (
        _csv_floats(args.rotation_range_z, 2, "--rotation-range-z")
        if args.rotation_range_z is not None
        else None
    )
    num_states = 1 if args.num_states is None else args.num_states
    overrides = [
        f"gs={gs}",
        f"gs_config={gs}",
        f"physics.ckpt_path={ckpt_path}",
        f"physics.case_name={case_name}",
        f"demo.gt_dir={demo}",
        f"demo.episode_id={int(args.episode_id)}",
        f"output.root_dir={Path(args.out)}",
        f"output.overwrite={'true' if args.overwrite else 'false'}",
        f"batch.num_samples={int(num_states)}",
        f"batch.seed={runtime['batch_seed']}",
        f"perturbation.mode={_RUNTIME_MODE_TO_BACKEND[args.mode]}",
        f"perturbation.random_steps={runtime['random_steps']}",
        f"perturbation.translation_step_sizes={_format_hydra_value(runtime['translation_step_sizes'])}",
        f"perturbation.rotation_step_deg={runtime['rotation_step_deg']}",
        f"perturbation.rotation_prob={runtime['rotation_prob']}",
        f"perturbation.gripper_state={runtime['gripper_state']}",
        f"replay_randomize={'true' if runtime['replay_randomize'] else 'false'}",
        f"release.enabled={'true' if runtime['release_enabled'] else 'false'}",
        f"release.num_waypoints={runtime['release_waypoints']}",
        f"release.gripper_open={runtime['release_gripper_open']}",
        "replay_init.enabled=true",
        "replay_init.frame=index",
        f"replay_init.index={runtime['replay_init_index']}",
        f"replay_init.stabilization_steps={runtime['init_stabilization_steps']}",
        f"replay_init.run_demo_tail={'true' if runtime['run_demo_tail'] else 'false'}",
        f"output.final_state_stabilization_steps={runtime['final_stabilization_steps']}",
        f"make_videos={'true' if runtime['make_videos'] else 'false'}",
    ]
    if trans is not None:
        overrides.append(f"perturbation.translation_range_xy={_format_hydra_value(trans)}")
    if rot is not None:
        overrides.append(f"perturbation.rotation_range_z={_format_hydra_value(rot)}")
    if runtime["reset_seed"] is None:
        overrides.append("reset_seed=null")
    else:
        overrides.append(f"reset_seed={int(runtime['reset_seed'])}")
    if args.collide_fric_override is not None:
        overrides.append(f"physics.collide_fric_override={float(args.collide_fric_override)}")
    overrides.extend(args.override)
    return [sys.executable, str(backend), *overrides]


def _build_state_replay_command(
    args: argparse.Namespace,
    input_root: Path,
    replay_root: Path,
    run_name: str,
    grid_sample: dict[str, Any] | None,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    replay = repo_root / "experiments" / "replay.py"
    gs, ckpt_path, case_name = _CASE_DEFAULTS[args.case]
    use_grid = grid_sample is not None
    stabilization_steps = _state_stabilization_steps(args, repo_root)
    command = [
        sys.executable,
        str(replay),
        f"gs={gs}",
        "env=xarm_gripper",
        f"physics.ckpt_path={ckpt_path}",
        f"physics.case_name={case_name}",
        f"gt_dir={input_root}",
        "use_qpos=false",
        f"randomize={'true' if use_grid else 'false'}",
        "save_final_state=true",
        f"final_state_stabilization_steps={stabilization_steps}",
        "final_state_dir_name=final_state",
        f"output_root={replay_root}",
        "overwrite_output=true",
        "make_videos=false",
        "save_depth=false",
        f"timestamp={run_name}",
    ]
    if args.collide_fric_override is not None:
        command.append(f"physics.collide_fric_override={float(args.collide_fric_override)}")
    if use_grid:
        xy = grid_sample["xy"]
        theta = grid_sample["theta_deg"]
        command.extend([
            "gs.use_grid_randomization=true",
            f"++gs.object.grid_randomization.xy={_format_hydra_value([xy])}",
            f"++gs.object.grid_randomization.theta={_format_hydra_value([theta])}",
            "++gs.object.grid_randomization.one_to_one=false",
            "+reset_seed=0",
        ])
        # Sloth's box is intentionally fixed while its soft object is moved.
        if args.case == "sloth":
            command.append("gs.meshes.0.grid_randomization=null")
    command.extend(args.override)
    return command


def _prepare_state_output_root(args: argparse.Namespace) -> Path:
    output_root = Path(args.out) / f"demo_episode_{int(args.episode_id):04d}"
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    return output_root


def _write_initial_replay_input(
    source_robot_json: Path,
    sample_input_root: Path,
    hold_frames: int,
) -> None:
    """Create a static initial-pose trajectory long enough for one simulation second."""
    if hold_frames < 2:
        raise ValueError("--state-hold-frames must be at least 2.")
    robot_dir = sample_input_root / "episode_0000" / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(hold_frames):
        shutil.copy2(source_robot_json, robot_dir / f"{frame_index:06d}.json")


def _run_state_synthesis(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    asset_root = _resolve_asset_root(args, repo_root)
    profile = _load_case_profile(repo_root, args.case)
    demo = _resolve_demo_path(args, asset_root, profile)
    source_robot_json = _resolve_initial_robot_json(demo, args.episode_id)
    stabilization_steps = _state_stabilization_steps(args, repo_root)

    if args.mode == "default-state":
        if args.num_states not in (None, 1):
            raise ValueError("default mode has exactly one deterministic simulator state; use --num-states 1.")
        selected: list[dict[str, Any] | None] = [None]
    elif args.mode == "configured-grid":
        selected = _select_grid_samples(_load_case_grid(repo_root, args.case), args)
    elif args.mode == "uniform-grid":
        selected = _select_grid_samples(_build_rigid_grid(args), args)
    else:
        raise ValueError(f"Unsupported state synthesis mode: {args.mode}")

    if args.dry_run:
        print(f"State synthesis mode={args.mode}, samples={len(selected)}")
        for local_index, grid_sample in enumerate(selected):
            print(
                " ".join(
                    _build_state_replay_command(
                        args,
                        Path("<input>"),
                        Path("<replay-output>"),
                        f"sample_{local_index:04d}",
                        grid_sample,
                    )
                )
            )
        return 0

    output_root = _prepare_state_output_root(args)
    replay_root = output_root / "replay_runs"
    input_root = output_root / "replay_inputs"
    summary: dict[str, Any] = {
        "case": args.case,
        "demo": str(demo),
        "demo_episode": int(args.episode_id),
        "mode": args.mode,
        "source_robot_json": str(source_robot_json),
        "asset_root": str(asset_root),
        "stabilization_steps": stabilization_steps,
        "samples": [],
    }

    for local_index, grid_sample in enumerate(selected):
        sample_name = f"sample_{local_index:04d}"
        sample_dir = output_root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=False)
        sample_input_root = input_root / sample_name
        _write_initial_replay_input(source_robot_json, sample_input_root, args.state_hold_frames)
        command = _build_state_replay_command(
            args,
            sample_input_root,
            replay_root,
            sample_name,
            grid_sample,
        )
        print("Running state synthesis:")
        print(" ".join(command))
        subprocess.run(command, cwd=asset_root, check=True)

        replay_dir = replay_root / sample_name / "episode_0000"
        state_path = replay_dir / "final_state" / "state.npy"
        if not state_path.exists():
            raise FileNotFoundError(f"Expected final state was not produced: {state_path}")
        exported_state = sample_dir / "soft_body_state.npy"
        shutil.copy2(state_path, exported_state)
        metadata = {
            "case": args.case,
            "mode": args.mode,
            "sample_index": local_index,
            "source_robot_json": str(source_robot_json),
            "asset_root": str(asset_root),
            "replay_dir": str(replay_dir),
            "soft_body_state": str(exported_state),
            "stabilization_steps": stabilization_steps,
            "state_hold_frames": int(args.state_hold_frames),
            "grid": grid_sample,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        summary["samples"].append(metadata)
        print(f"Completed {sample_name} ({local_index + 1}/{len(selected)})")

    with open(output_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate perturbed deformable-object states through simulation.")
    parser.add_argument("--case", choices=sorted(_CASE_DEFAULTS), required=True)
    parser.add_argument(
        "--asset-root",
        default=None,
        help="Deprecated compatibility option; runtime assets are loaded from this repository's log/ directory.",
    )
    parser.add_argument("--demo", default=None, help="Source demonstration; defaults to the case profile under --asset-root.")
    parser.add_argument("--episode-id", type=int, default=None, help="Episode id; defaults to the case profile.")
    parser.add_argument("--out", required=True, help="Output directory for generated states/manifests.")
    parser.add_argument("--num-states", type=int, default=None, help="Runtime sample count or selected grid-state count.")
    parser.add_argument(
        "--mode",
        default="runtime-random",
        choices=sorted(_MODE_ALIASES),
        help=(
            "Canonical modes: runtime-random, runtime-fixed, default-state, "
            "configured-grid, uniform-grid. Legacy aliases remain accepted."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the case profile batch seed.")
    parser.add_argument("--seed-shard-index", type=int, default=0, help="Profile seed-stride index; cloth3 uses a stride of 1000.")
    parser.add_argument("--reset-seed", type=int, default=None, help="Override the case profile simulator reset seed.")
    parser.add_argument(
        "--translation-range-xy",
        default=None,
        help="Override continuous runtime-fixed min,max XY range in meters; defaults to cfg/augmentation.yaml.",
    )
    parser.add_argument(
        "--rotation-range-z",
        default=None,
        help="Override continuous runtime-fixed min,max yaw range in degrees; defaults to cfg/augmentation.yaml.",
    )
    parser.add_argument("--grid-start", type=int, default=0, help="First deterministic grid index to synthesize.")
    parser.add_argument("--grid-nx", type=int, default=3, help="uniform-grid X samples.")
    parser.add_argument("--grid-ny", type=int, default=3, help="uniform-grid Y samples.")
    parser.add_argument("--grid-ntheta", type=int, default=3, help="uniform-grid yaw samples.")
    parser.add_argument("--grid-x-range", default="-0.05,0.05", help="uniform-grid X min,max in meters.")
    parser.add_argument("--grid-y-range", default="-0.05,0.05", help="uniform-grid Y min,max in meters.")
    parser.add_argument("--grid-theta-range", default="-10,10", help="uniform-grid yaw min,max in degrees.")
    parser.add_argument("--random-steps", type=int, default=None, help="Override the case profile runtime-random step count.")
    parser.add_argument("--translation-step-sizes", default=None, help="Override runtime-random XY step sizes in meters.")
    parser.add_argument("--rotation-step-deg", type=float, default=None, help="Override runtime-random yaw step size in degrees.")
    parser.add_argument("--rotation-prob", type=float, default=None, help="Override runtime-random yaw-step probability.")
    parser.add_argument("--gripper-state", type=float, default=None, help="Override the closed-gripper command during runtime perturbation.")
    parser.add_argument("--replay-init-index", type=int, default=None, help="Override the case profile initialization frame.")
    parser.add_argument("--init-stabilization-steps", type=int, default=None, help="Override pre-replay stabilization steps.")
    parser.add_argument("--run-demo-tail", action=argparse.BooleanOptionalAction, default=None, help="Override the case profile demo-tail behavior.")
    parser.add_argument("--stabilization-steps", type=int, default=None, help="Override final stabilization steps.")
    parser.add_argument(
        "--state-hold-frames",
        type=int,
        default=30,
        help="Static source-pose frames replayed before saving a default-state/grid state.",
    )
    parser.add_argument("--collide-fric-override", type=float, default=None)
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override non-grid reset randomization for runtime perturbation replay; defaults to cfg/augmentation.yaml.",
    )
    parser.add_argument("--release", action=argparse.BooleanOptionalAction, default=None, help="Override whether to open the gripper after runtime perturbation.")
    parser.add_argument("--release-waypoints", type=int, default=None, help="Override release interpolation frames.")
    parser.add_argument("--release-gripper-open", type=float, default=None, help="Override the release gripper command.")
    parser.add_argument("--make-videos", action=argparse.BooleanOptionalAction, default=None, help="Override replay video export; defaults to enabled.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", default=None, help="Deprecated compatibility option; use --override for Hydra values.")
    parser.add_argument("--override", action="append", default=[], help="Additional raw Hydra override. May be repeated.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    requested_mode = args.mode
    args.mode = _MODE_ALIASES[requested_mode]
    if args.episode_id is None:
        profile = _load_case_profile(Path(__file__).resolve().parents[2], args.case)
        args.episode_id = int(profile["demo"]["episode_id"])
    if requested_mode != args.mode:
        print(
            f"deformgen-perturb: --mode {requested_mode!r} is a legacy alias; "
            f"using canonical mode {args.mode!r}.",
            file=sys.stderr,
        )

    if args.config is not None:
        print("deformgen-perturb: --config is not loaded; use --override for explicit Hydra values.", file=sys.stderr)

    try:
        if args.mode in _RUNTIME_MODE_TO_BACKEND:
            command = build_runtime_command(args)
            print("Running runtime perturbation backend:")
            print(" ".join(command))
            if args.dry_run:
                return 0
            return subprocess.call(
                command,
                cwd=Path(__file__).resolve().parents[2],
            )
        return _run_state_synthesis(args)
    except Exception as exc:
        print(f"deformgen-perturb: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
