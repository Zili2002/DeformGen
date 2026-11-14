# Assets and Released Resources

## Released Datasets and Checkpoints

The command examples below assume the released artifacts are downloaded under `./release`. Replace the repo names if you mirror the data elsewhere.

```bash
pip install -U huggingface_hub

mkdir -p release

huggingface-cli download Zili2002/DeformGen-Datasets \
  --repo-type dataset \
  --local-dir release/DeformGen-Datasets

huggingface-cli download Zili2002/DeformGen-Checkpoints \
  --repo-type model \
  --local-dir release/DeformGen-Checkpoints
```

Expected released dataset layout:

```text
release/DeformGen-Datasets/
+-- rope/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
+-- sloth/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
`-- cloth3/{train1000_txy,train1000_yawonly,train1000_gridrigid,test200}
```

Expected checkpoint layout:

```text
release/DeformGen-Checkpoints/<case>/<model>/<mode>/checkpoint
# case:  rope | sloth | cloth3
# model: act | dp | svla | pi0
# mode:  txy | yawonly | gridrigid | single
```

### Local Paths

Set these variables from the repository root before using the training and evaluation examples:

```bash
export DEFORMGEN_ROOT=$PWD
export RELEASE_ROOT=$DEFORMGEN_ROOT/release
export DATA_ROOT=$RELEASE_ROOT/DeformGen-Datasets
export CKPT_ROOT=$RELEASE_ROOT/DeformGen-Checkpoints
```

## Simulation Assets

DeformGen separates code from large simulation assets. Install the released assets after cloning the repository:

```bash
deformgen-fetch sim-assets \
  --case all \
  --repo-root "$DEFORMGEN_ROOT"

# Optional mirror endpoint.
# deformgen-fetch sim-assets --case all --repo-root "$DEFORMGEN_ROOT" --endpoint https://hf-mirror.com
```

The command downloads the upstream default branch unless a source declares an explicit `revision` in [`assets/sources.yaml`](../assets/sources.yaml), caches the downloaded files under `~/.cache/deformgen`, and creates local links under `log/`. It writes the requested source revision (`default` when unpinned) and link targets to `log/external_assets/resolved_manifest.json`.

The source registry uses:

- `shashuo0104/gs-scans` for upstream rope/sloth Gaussian scans and shared rigid scene scans.
- `shashuo0104/phystwin-rope` for rope PhysTwin assets.
- `shashuo0104/phystwin-toy` for sloth PhysTwin assets.
- `Zili2002/DeformGen-SimAssets` for Cloth3 assets and the formal rope/sloth/cloth3 demonstration trajectories.

Use a single case when disk capacity is limited:

```bash
deformgen-fetch sim-assets --case rope --repo-root /path/to/DeformGen
```

The command refuses to overwrite an existing nonmatching `log/` asset target. Review the target or pass `--force` intentionally. The `--dry-run` mode resolves the download plan without changing repository links.

```bash
deformgen-fetch sim-assets --case cloth3 --dry-run
```

### Local Symlinks

For local development, symlinks are acceptable and were used to validate the release layout:

```bash
mkdir -p outputs/downloaded/{datasets,checkpoints,testsets}
ln -sfn "$DATA_ROOT/rope/train1000_yawonly" outputs/downloaded/datasets/rope_train1000_yawonly
ln -sfn "$DATA_ROOT/rope/test200" outputs/downloaded/testsets/rope_test200
ln -sfn "$CKPT_ROOT/rope/act/yawonly" outputs/downloaded/checkpoints/rope_act_yawonly
```

## Method Names

The code keeps engineering names because they are used by configs, scripts, and experiment metadata:

| Code name | Paper name | Meaning |
| --- | --- | --- |
| `yawonly` | `DG` | yaw-only local trajectory adaptation |
| `txy` | `DG*` | local rigid XY translation + yaw adaptation |
| `gridrigid` | `SMG*` | grid-rigid state generation / evaluation setting |

Do not rename directories or checkpoint folders for paper notation. Use the mapping above in papers and reports.
