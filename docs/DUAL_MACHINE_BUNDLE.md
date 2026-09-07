# Dual-machine reconstruction bundle

This workflow runs reconstruction on a high-memory H20 host and transfers one
self-contained `.tar.gz` to a 4090 host for OmniGibson import and policy work.
The archive contains reconstructed object assets, settled poses, and optionally
the Gaussian-splat background. It never contains a robot or a shared
BEHAVIOR-1K asset library.

## H20: reconstruct and export

Install all reconstruction environments while omitting Isaac Sim,
OmniGibson, BEHAVIOR-1K, robot assets, and LeRobot from the main environment:

```bash
bash scripts/installation/install_everything.sh --reconstruction-only
```

An interrupted installer can be rerun. Package caches and complete dependency
checkouts are reused, but installation is not a stage checkpoint system; repair
or remove a partial Git checkout if an installer reports that it is invalid.

Before the first reconstruction, complete the service login and checkpoint
steps when credentials are available:

```bash
bash scripts/installation/login_services.sh
bash scripts/installation/download_checkpoints.sh
```

These can be deferred during environment installation, but the model-backed
reconstruction stages cannot run until their checkpoints and service
credentials are available.

Run the canonical pipeline only through stage 12:

```bash
bash scripts/pipeline/A_reconstruction/run.sh \
  --scene-name my_scene \
  --video-fpath /absolute/path/input.mp4 \
  --include 1b,2,3,4,5,6,7,8,10,11,12 \
  --no-stream -- \
  s1_video.splat_prep=true \
  s1_video.n_subsampled_frames=400 \
  s1_video.target_w=672 \
  s1_video.target_h=384 \
  s5_scene.pda_geometric_backend=depth_pro \
  s7_mesh.low_vram=false
```

Authentication-backed reconstruction stages still require their normal
credentials when they execute. Installation itself does not perform the login.

Generate and convert the automatic background without stage 14:

```bash
bash scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh \
  my_scene /absolute/path/input.mp4 --mode export
```

Create the single transfer file:

```bash
mamba run -n simfoundry python \
  scripts/pipeline/A_reconstruction/export_scene_bundle.py \
  --scene-name my_scene \
  --output /transfer/my_scene.tar.gz
```

Use `--no-background` only when no background was generated. The exporter
validates the required stage 11 and 12 outputs and records a SHA-256 for every
file. Paths inside the archive are relative.

## 4090: import and build the scene

Install only the simulation and rollout profile:

```bash
bash scripts/installation/install_everything.sh --simulation-only --skip-robot-assets
```

This creates only the `simfoundry` environment. It installs OmniGibson and its
runtime dependencies but does not create DA3, VOID, Hunyuan, Nerfstudio, or
3DGRUT environments, and does not download official Franka/YAM robot assets.
The reconstruction bundle is robot-free; attach your own robot at rollout.

Copy `my_scene.tar.gz` to this host, then validate, restore, and build:

```bash
mamba run -n simfoundry python \
  scripts/pipeline/A_reconstruction/import_scene_bundle.py \
  /transfer/my_scene.tar.gz
```

The importer rejects absolute paths, parent-directory traversal, links, unknown
archive members, size mismatches, SHA-256 mismatches, and archives above the
default 250 GiB unpacked limit. It restores
`Data/my_scene`, runs stages 13 and 14 headlessly with
`s14_og.include_robot=false`, then assembles a prebuilt background without
calling 3DGRUT.

With a background, the final scene is:

```text
assets/scenes/my_scene/my_scene_scene_state_auto_bg.json
```

Without a background, it is:

```text
Data/my_scene/s14_og/reconstructed_og_scene.json
```

Pass `--extract-only` to validate and unpack without starting Isaac Sim. Pass
`--force` only when replacing an existing `Data/my_scene` directory. Override
the size guard with `--max-unpacked-gb` only for a known larger scene.

## Add a robot at rollout time

The saved scene is intentionally robot-free. Select the robot and its assets in
the teleoperation, rollout, or evaluation configuration rather than modifying
the archive. This keeps the same reconstructed scene reusable across robots and
policies.
