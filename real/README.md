# Real-robot evaluation

xArm7, UFACTORY gripper, front D455, wrist D405; pi0 checkpoints for `cloth`, `rope`, and `sloth`.
Run commands from the repository root. Install the repository and `policy/` environment using the [installation guide](../docs/installation.md), then:

```bash
pip install pyrealsense2 xarm-python-sdk PyYAML
pip install -e third-party/urdfpy-0.0.22
# Install ffmpeg and ffprobe using your system package manager.
cp real/robot.example.yaml real/robot.local.yaml
python -m real.view --list
```

Edit `real/robot.local.yaml`: set the robot IP and camera SDK serials, verify the TCP offset,
flange-to-policy transform, reset pose, and table plane; then set `robot.calibration_confirmed: true`.
The local file is gitignored. Relative asset/checkpoint paths resolve from the repository root.
Camera settings may include `auto_exposure: false`, `exposure`, `gain`,
`auto_white_balance: false`, and `white_balance` (D455 exposure units: 100 us; D405: us).

```bash
python -m real.download --task sloth
python -m real.view --task sloth
python -m real.align --task sloth
python -m real.shadow --task sloth --model yawonly
```

Alignment keys: `v` toggles the three-panel view, `[`/`]` change opacity, `s` saves images, `q` quits.
Downloads pin the corrected checkpoint stats. `--revision` overrides that pin.
Each finetuned model uses its own checkpoint statistics.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m real.interactive \
  --task sloth --protocol single --model yawonly \
  --allow-motion --confirm MOVE_XARM --confirm-continuous RUN_CONTINUOUS_PI0
```

Keys: `r` releases/lifts/resets, `c` runs, `s` stops, `q` exits.
Motion requires these CLI flags and the verified local calibration.
Each replan uses one current front image, one wrist image, current 8D state, and the task prompt;
all 50 predicted actions run at 30 Hz before replanning.
Outputs go to `outputs/real/<task>/`, with H.264 front videos, observations, metadata, and command logs.
The controller retains the original real-run gripper rate limit (`2/30` per step relative to the last command),
joint-velocity limits, and logged table-plane z correction.

For three models at each of ten initial states:

```bash
python -m real.download --task sloth --base
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m real.interactive \
  --task sloth --protocol triple10 \
  --allow-motion --confirm MOVE_XARM --confirm-continuous RUN_CONTINUOUS_PI0
```

Order: `yawonly -> single -> base`; reset and restore the saved initial-state overlay between models.
`base` is untouched official pi0 with task yawonly stats.
Resume with the same command plus `--resume-session outputs/real/sloth/SESSION`.
While idle, `d`/Delete moves the previous attempt to recoverable trash and rewinds its slot.

Optional: `python -m real.reference_videos --help` prepares 16 public reference episodes per task.
