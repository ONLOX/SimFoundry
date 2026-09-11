# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy.spatial.transform import Rotation

from simfoundry.reconstruction.bg_bridge import same_scene_src_to_og, se3_src_to_og
from simfoundry.utils.transform_utils import camera_centers_from_world2cam


def _lookat_w2c(center: np.ndarray) -> np.ndarray:
    """OpenCV w2c for a camera at ``center`` looking along +Z, +Y down."""
    rot = np.eye(3)
    t = -rot @ np.asarray(center, dtype=np.float64)
    ext = np.zeros((3, 4), dtype=np.float64)
    ext[:3, :3] = rot
    ext[:3, 3] = t
    return ext


def test_se3_is_anchor_only():
    ext = _lookat_w2c([0.2, 0.0, 1.0])
    image_c2w = np.eye(4)
    image_c2w[:3, 3] = [1.0, 2.0, 3.0]
    matrix = se3_src_to_og(ext_src_anchor=ext, image_c2w=image_c2w)
    # ext @ p_world = p_cam; image_c2w maps that camera into OG.
    p_void = np.array([0.2, 0.0, 1.0, 1.0])
    # Camera centre in void maps to image_c2w origin.
    np.testing.assert_allclose((matrix @ p_void)[:3], image_c2w[:3, 3], atol=1e-9)


def test_same_scene_recovers_known_scale():
    rng = np.random.default_rng(0)
    n = 8
    centers_orig = rng.normal(scale=0.4, size=(n, 3)) + np.array([0.0, 0.0, 0.8])
    scale = 0.97
    rot = Rotation.from_euler("z", 3, degrees=True).as_matrix()
    t = np.array([0.01, -0.02, 0.03])
    # p_orig = s * R @ p_void + t  =>  p_void = R.T @ (p_orig - t) / s
    centers_void = ((centers_orig - t) @ rot) / scale
    ext_orig = np.stack([_lookat_w2c(c) for c in centers_orig])
    ext_void = np.stack([_lookat_w2c(c) for c in centers_void])
    image_c2w = np.eye(4)
    image_c2w[:3, 3] = [0.05, 0.0, 0.1]

    matrix, recovered, info = same_scene_src_to_og(
        ext_void=ext_void,
        ext_orig=ext_orig,
        image_c2w=image_c2w,
        anchor_row=2,
    )
    assert info["alignment"] == "sim3"
    np.testing.assert_allclose(recovered, scale, atol=1e-6)
    # A void-world point that is a camera centre lands on the matching orig
    # centre, then on the s4 camera origin after image_c2w @ ext_orig[2].
    p_void = np.append(centers_void[2], 1.0)
    p_og = matrix @ p_void
    np.testing.assert_allclose(p_og[:3], image_c2w[:3, 3], atol=1e-6)


def test_same_scene_accepts_eighteen_percent_when_residual_is_small():
    rng = np.random.default_rng(1)
    n = 8
    centers_orig = rng.normal(scale=0.3, size=(n, 3)) + np.array([0.0, 0.0, 0.7])
    scale = 1.178
    centers_void = centers_orig / scale
    ext_orig = np.stack([_lookat_w2c(c) for c in centers_orig])
    ext_void = np.stack([_lookat_w2c(c) for c in centers_void])
    _, recovered, info = same_scene_src_to_og(
        ext_void=ext_void,
        ext_orig=ext_orig,
        image_c2w=np.eye(4),
        anchor_row=0,
    )
    assert info["alignment"] == "sim3"
    np.testing.assert_allclose(recovered, scale, atol=1e-6)
    assert "scale_warning" in info


def test_same_scene_falls_back_on_crazy_scale():
    n = 5
    ext_orig = np.stack([_lookat_w2c([float(i), 0.0, 0.5]) for i in range(n)])
    ext_void = np.stack([_lookat_w2c([0.2 * float(i), 0.0, 0.5]) for i in range(n)])
    matrix, scale, info = same_scene_src_to_og(
        ext_void=ext_void,
        ext_orig=ext_orig,
        image_c2w=np.eye(4),
        anchor_row=0,
        scale_min=0.85,
        scale_max=1.15,
    )
    assert info["alignment"] == "se3_fallback"
    assert scale == 1.0
    expected = se3_src_to_og(ext_src_anchor=ext_void[0], image_c2w=np.eye(4))
    np.testing.assert_allclose(matrix, expected)


def test_same_scene_falls_back_on_frame_count_mismatch():
    ext_void = np.stack([_lookat_w2c([0.0, 0.0, 1.0])] * 4)
    ext_orig = np.stack([_lookat_w2c([0.0, 0.0, 1.0])] * 3)
    matrix, scale, info = same_scene_src_to_og(
        ext_void=ext_void,
        ext_orig=ext_orig,
        image_c2w=np.eye(4),
        anchor_row=0,
    )
    assert info["alignment"] == "se3_fallback"
    assert scale == 1.0
    assert "shape" in info["reason"]


def test_camera_centers_match_lookat_helper():
    ext = np.stack([_lookat_w2c([0.3, -0.1, 0.9])])
    np.testing.assert_allclose(camera_centers_from_world2cam(ext)[0], [0.3, -0.1, 0.9])
