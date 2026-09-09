# Qwen + manual reconstruction

Normal reconstruction uses Qwen only for stage 5 object removal and stage 6 crop
upsample (`qwen-image-3.0`). Object names are supplied with
`s5_scene.force_categories`. Pin the reconstruction frame with
`s3_ground.img_idx=<N>`, or leave it `auto` to take the highest heuristic score.
Front-pick, validity, and physics stay off the Gemini VLM.

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
  'scene_name=bottle_box' \
  "s1_video.video_fpath=$PWD/docs/assets/example_videos/bottle-box.mp4" \
  's1_video.splat_prep=true' \
  's1_video.n_subsampled_frames=400' \
  's1_video.target_w=672' \
  's1_video.target_h=384' \
  's3_ground.img_idx=0' \
  's5_scene.force_categories=["clear water bottle","open cardboard box"]' \
  '+s11_sim.manual_physics.overrides={clear_water_bottle:{mass:0.10,friction:0.8},open_cardboard_box:{mass:0.5,friction:0.6}}'
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
