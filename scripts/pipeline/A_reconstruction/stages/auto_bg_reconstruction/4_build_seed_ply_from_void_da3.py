# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Build a clean-tabletop seed PLY for splatfacto from VOID-frame DA3.

Training uses void RGB, void poses and void depth, so the seed stays in
void-DA3. Mapping it into orig-DA3 (Umeyama on camera centres only) used to
leave an orientation residual; splatfacto then painted a second desk.

Umeyama against orig-DA3 is still logged as a diagnostic — it is not baked
into the PLY. The step-6 bridge maps void-DA3 → OG through the same
canonical-frame cam2world the object meshes use.

Reads config from scripts/cfg/auto_bg.yaml (Hydra), section `s4_seed_ply`.
Run from simfoundry env (Hydra override syntax):
  mamba run -n simfoundry python \
      scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/4_build_seed_ply_from_void_da3.py \
      scene_name=<scene>
Per-stage values can be overridden directly, e.g.
  `s4_seed_ply.orig_da3_npz=...`.
"""
import logging
from pathlib import Path

import hydra
import numpy as np
from plyfile import PlyData, PlyElement

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

bootstrap_hydra_workdir(__file__)

from simfoundry import CFG_DIR  # noqa: E402
from simfoundry.utils.transform_utils import camera_centers_from_world2cam, umeyama_alignment


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_seed_ply_from_void_da3")


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s4_seed_ply

    for desc, p in (("orig DA3 npz (run stage 2b: 2b_run_da.py at num_frames frames)", sec.orig_da3_npz),
                    ("void DA3 npz (run stage 2b on the void cleaned_frames)", sec.void_da3_npz)):
        if not Path(p).exists():
            raise SystemExit(f"missing {desc}: {p}")

    orig = np.load(sec.orig_da3_npz)
    void = np.load(sec.void_da3_npz)

    ext_orig = orig["extrinsics"]  # (N,3,4) world2cam
    ext_void = void["extrinsics"]
    if ext_orig.shape != ext_void.shape:
        raise SystemExit(f"Frame counts differ: orig {ext_orig.shape} vs void {ext_void.shape}")
    logger.info("Both NPZs report %d frames", ext_orig.shape[0])

    centers_orig = camera_centers_from_world2cam(ext_orig)
    centers_void = camera_centers_from_world2cam(ext_void)
    T_v2o, _ = umeyama_alignment(centers_void, centers_orig, with_scale=False)
    logger.info("Umeyama rigid void→orig (diagnostic only, not applied to the seed):")
    logger.info("  R det = %+.6f (must be +1)", float(np.linalg.det(T_v2o[:3, :3])))
    logger.info("  t = [% .4f % .4f % .4f]", *T_v2o[:3, 3])
    aligned = (centers_void @ T_v2o[:3, :3].T) + T_v2o[:3, 3]
    res = np.linalg.norm(aligned - centers_orig, axis=1)
    logger.info("  cam-center residual: mean=%.4fm median=%.4fm max=%.4fm",
                res.mean(), np.median(res), res.max())
    if float(res.max()) > 0.05:
        logger.warning(
            "void/orig camera centres differ by up to %.3fm after a rigid fit. "
            "Training stays in void-DA3 so this residual does not paint a second desk.",
            float(res.max()),
        )

    intr_void = void["intrinsics"]   # (N,3,3)
    depth_void = void["depth"]       # (N,H,W)
    img_void = void["image"]         # (N,H,W,3) uint8
    conf_void = void["conf"]         # (N,H,W)
    N, H, W = depth_void.shape

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    chunks_xyz, chunks_rgb = [], []
    for i in range(N):
        K = intr_void[i]
        # Void-world cam2world
        T_w2c = np.eye(4, dtype=np.float64)
        T_w2c[:3, :4] = ext_void[i]
        T_c2w_void = np.linalg.inv(T_w2c)
        keep = (depth_void[i] > 0) & (conf_void[i] >= sec.conf_min)
        if not keep.any():
            continue
        z = depth_void[i][keep].astype(np.float64)
        uu = u[keep].astype(np.float64)
        vv = v[keep].astype(np.float64)
        x_cam = (uu - K[0, 2]) * z / K[0, 0]
        y_cam = (vv - K[1, 2]) * z / K[1, 1]
        cam_pts = np.stack([x_cam, y_cam, z], axis=1)
        world_pts_void = cam_pts @ T_c2w_void[:3, :3].T + T_c2w_void[:3, 3]
        chunks_xyz.append(world_pts_void.astype(np.float32))
        chunks_rgb.append(img_void[i][keep])

    xyz = np.concatenate(chunks_xyz, axis=0)
    rgb = np.concatenate(chunks_rgb, axis=0)
    logger.info("Backprojected %d void-frame depth points (conf>=%.1f)", len(xyz), sec.conf_min)

    if len(xyz) > sec.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=sec.max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        logger.info("Subsampled to %d points", sec.max_points)

    arr = np.empty(len(xyz), dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["red"] = rgb[:, 0]
    arr["green"] = rgb[:, 1]
    arr["blue"] = rgb[:, 2]
    out_ply = Path(sec.out_ply)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(out_ply))
    logger.info("Wrote seed PLY -> %s", out_ply)


if __name__ == "__main__":
    main()
