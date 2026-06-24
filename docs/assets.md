# Simulation Assets

DeformGen separates code from large simulation assets. Install the released assets after cloning the repository:

```bash
deformgen-fetch sim-assets --case all --repo-root /path/to/DeformGen
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
