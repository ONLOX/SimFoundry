# Qwen + manual reconstruction

This profile runs reconstruction without a Google Gemini or Vertex AI account:

- object names are supplied manually;
- Qwen Image edits the scene and improves object crops;
- rigid-object mass and friction are supplied manually;
- Gemini-based validation and front selection are disabled.

SAM3 and the local geometry models are still required.

## Configure Alibaba Model Studio

Create an API key in Alibaba Cloud Model Studio (百炼), then export it in the shell that
starts the pipeline:

```bash
export DASHSCOPE_API_KEY='sk-your-key'
```

The client defaults to the China-region DashScope endpoint. A compatible endpoint can be
selected with `DASHSCOPE_IMAGE_ENDPOINT`. API responses are downloaded immediately because
their image URLs are temporary.

The API key and endpoint must belong to the same region. For example:

```bash
# International/Singapore key
export DASHSCOPE_IMAGE_ENDPOINT='https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'

# Workspace-specific Beijing key (replace the placeholder)
export DASHSCOPE_IMAGE_ENDPOINT='https://YOUR_WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation'
```

## Fruits example

From the repository root:

```bash
mamba run -n simfoundry python \
  scripts/pipeline/A_reconstruction/run_reconstruction.py \
  --exclude 13,14 \
  --skip-successful \
  'scene_name=fruits_example' \
  "s1_video.video_fpath=$PWD/docs/assets/example_videos/Fruits.mp4" \
  's1_video.splat_prep=true' \
  's1_video.n_subsampled_frames=400' \
  's1_video.target_w=672' \
  's1_video.target_h=384' \
  's3_ground.frame_selection.mode=heuristic' \
  's3_ground.allow_vlm_fallback=false' \
  's5_scene.force_categories=["green plate","orange fruit","red apple","orange plate","banana","pear"]' \
  's5_scene.removal_model=qwen-image-3.0' \
  's5_scene.use_upsampled_source_image=false' \
  's6_upsample.model=qwen-image-3.0' \
  's6_upsample.check_valid=false' \
  's8_pose.canonicalize_front=false' \
  's11_sim.physics_mode=manual' \
  '+s11_sim.manual_physics.overrides={green_plate:{mass:0.10,friction:0.5},orange_fruit:{mass:0.15,friction:0.5},red_apple:{mass:0.18,friction:0.5},orange_plate:{mass:0.10,friction:0.5},banana:{mass:0.12,friction:0.5},pear:{mass:0.18,friction:0.5}}'
```

`force_categories` should contain visible, individually removable objects. Keep the order
stable when resuming a partially completed stage 5.

The example treats every object as rigid, so it does not enable optional stage 9. For a
scene with articulated objects, add `--detect-articulation` and explicitly configure:

```text
s9_articulate_objects.classification_mode=manual
s9_articulate_objects.force_articulated=["cabinet"]
```

Manual physics values use kilograms for `mass`; `friction` is the unitless Coulomb
coefficient. Overrides may be keyed by raw object name, sanitized category, `iter_N`, or
numeric index. An object-specific override is merged over:

```yaml
s11_sim:
  manual_physics:
    default: {mass: 0.2, friction: 0.5}
```

Inspect and correct these initial estimates in the scene editor before policy training.

## Background reconstruction without Gemini

The same manual object list can drive the VOID quadmask:

```bash
bash scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/run_auto_align.sh \
  fruits_example \
  "$PWD/docs/assets/example_videos/Fruits.mp4" \
  --mode export \
  --manual-objects '["green plate","orange fruit","red apple","orange plate","banana","pear"]'
```

Qwen requests support the repository's existing remote-call cache. Set `CACHE_MODE=1` to
record successful responses and `TEST_MODE=1` to replay them without network access.
