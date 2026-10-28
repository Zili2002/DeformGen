# Object Appearance and Physical Parameters

The existing YAML configs provide these controls; no simulator code changes are needed. Edit the configs before starting a new run. Color changes affect rendering only, while friction changes affect subsequent physics rollouts, not already saved states or images.

All paths and commands below assume the repository root as the working directory.

## Object Appearance (Color)

Edit `object.color_A` and `object.color_b` in the selected GS config: [`cfg/gs/rope.yaml`](../cfg/gs/rope.yaml), [`cfg/gs/sloth.yaml`](../cfg/gs/sloth.yaml), or [`cfg/gs/cloth3.yaml`](../cfg/gs/cloth3.yaml). Their full Hydra keys are `gs.object.color_A` and `gs.object.color_b`.

These are the transformation coefficients estimated by upstream real2sim-eval's [5.3 Color alignment](https://github.com/kywind/real2sim-eval#53-color-alignment), not a new material or lighting parameter. The [GS renderer](../sim/renderer/gs_renderer.py) applies them to the object's Gaussian colors:

- A 3-by-3 `color_A` (9 row-major values) and a 3-vector `color_b` define `rgb_out = A @ [r, g, b] + b`, with RGB values on the 0-to-1 scale.
- A 3-by-6 `color_A` (18 row-major values) defines the quadratic mapping `rgb_out = A @ [r*r, g*g, b*b, r, g, b] + b`. This form requires `gs.use_shs=false`, as in the released case configs.

For example, replace only the color fields under the existing `object` block with the following to reduce green/blue intensity relative to red. Keep the other object fields, such as `path` and `pose`, unchanged:

```yaml
object:
  color_A: [
    1.0, 0.0, 0.0,
    0.0, 0.7, 0.0,
    0.0, 0.0, 0.7
  ]
  color_b: [0.0, 0.0, 0.0]
```

This replaces the existing color transform rather than composing with it. An identity 3-by-3 matrix and zero bias retain the original PLY colors. Editing `object` affects only the deformable object; `scene.color_A` / `scene.color_b` control the scene/robot scans, and individual `meshes` entries have their own color fields.

To estimate a transform from paired source/reference images instead of setting it manually, update `src_img_paths`, `ref_img_paths`, and the image dimensions in [`assets/scans/color_alignment.py`](../assets/scans/color_alignment.py), then run:

```bash
python assets/scans/color_alignment.py
```

The script writes `color_A` and `color_b` to `log/color_alignment/result/color_mapping.yaml`. Copy those fields into the desired `object` (or `scene` / mesh) block; the generated file is not loaded automatically.

## Physical Parameters (Friction)

For object contact with the ground/table or static meshes, set `collide_fric_override` in [`cfg/physics/default.yaml`](../cfg/physics/default.yaml), for example:

```yaml
collide_fric_override: 0.5
```

The full Hydra key is `physics.collide_fric_override`. **Use this override rather than changing `physics.collide_fric` directly:** [`sim/physics/phystwin.py`](../sim/physics/phystwin.py) loads the learned `collide_fric` from the PhysTwin checkpoint and applies the explicit override afterward.

The effective priority is `collide_fric_override` (if not `null`), then a matching `collide_fric_override_by_case` entry, then the checkpoint value. The case-map key is the prefix of `physics.case_name` before the first `_` (for example, `sloth_0001` uses `sloth`). To restore the checkpoint value, set the global override and any matching case entry to `null`. The [collision kernels](../sim/physics/spring_mass_warp.py) clamp friction coefficients to `[0, 2]`.

This override does not change all contact types: gripper contact uses `physics.collide_eef_fric` (the pusher path forces it to `0.2`), while self-contact friction is loaded from the checkpoint's `collide_object_fric`, not from a direct edit of `physics.collide_self_fric`.

## Per-Run Overrides

Instead of editing YAML for a single replay, append these flags to the `deformgen-replay-export` commands in the [README replay section](../README.md#3-replay-rendering-lerobot-export-and-success):

```bash
--override 'gs.object.color_A=[1.0,0.0,0.0,0.0,0.7,0.0,0.0,0.0,0.7]' \
--override 'gs.object.color_b=[0.0,0.0,0.0]' \
--override physics.collide_fric_override=0.5
```

For state generation, `deformgen-perturb --collide-fric-override 0.5` sets the same friction override. For color changes in `runtime-random` / `runtime-fixed` generation, edit the case GS YAML: that launcher reloads the named GS config in child replays and does not forward arbitrary `gs.object.*` overrides. Use the same intended friction setting during generation and replay when evaluating a material variant.
