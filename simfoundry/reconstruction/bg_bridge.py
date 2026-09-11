# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Void-DA3 splat → OG-world prim pose.

Objects live in orig-DA3 / s4 metres. The background splat is trained in
void-DA3, whose depth scale can differ by a few percent. Same-scene bridging
therefore fits a Sim(3) on camera centres (``with_scale=True``) and applies it
as one prim transform — the PLY is not rewritten.

The seed PLY stays in void-DA3. Baking this Sim(3) into the seed is what used
to paint a second desk.
"""

from __future__ import annotations

import numpy as np

from simfoundry.utils.transform_utils import camera_centers_from_world2cam, umeyama_alignment

# Residual is the quality gate. The soft band is only logged; void vs orig DA3
# has already shown ~18% scale on a real desk with a 3 cm residual.
DEFAULT_SCALE_MIN = 0.5
DEFAULT_SCALE_MAX = 2.0
DEFAULT_WARN_SCALE_MIN = 0.85
DEFAULT_WARN_SCALE_MAX = 1.15
DEFAULT_MAX_RESIDUAL_M = 0.08


def _as_w2c_4x4(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = np.asarray(extrinsic, dtype=np.float64)[:3, :4]
    return matrix


def se3_src_to_og(
    *,
    ext_src_anchor: np.ndarray,
    image_c2w: np.ndarray,
    step4_to_og: np.ndarray | None = None,
) -> np.ndarray:
    """Anchor-camera SE(3): void-DA3 → OG with scale fixed at 1."""
    if step4_to_og is None:
        step4_to_og = np.eye(4, dtype=np.float64)
    return (
        np.asarray(step4_to_og, dtype=np.float64)
        @ np.asarray(image_c2w, dtype=np.float64)
        @ _as_w2c_4x4(ext_src_anchor)
    )


def same_scene_src_to_og(
    *,
    ext_void: np.ndarray,
    ext_orig: np.ndarray,
    image_c2w: np.ndarray,
    step4_to_og: np.ndarray | None = None,
    anchor_row: int,
    scale_min: float = DEFAULT_SCALE_MIN,
    scale_max: float = DEFAULT_SCALE_MAX,
    max_residual_m: float = DEFAULT_MAX_RESIDUAL_M,
) -> tuple[np.ndarray, float, dict]:
    """Same-scene void-DA3 → OG.

    Preferred: Sim(3) ``T`` mapping void camera centres onto orig, then

        M = step4_to_og @ image_c2w @ ext_orig[anchor] @ T

    Falls back to the old single-anchor SE(3) when the two NPZs cannot be
    paired or the fit looks unstable.
    """
    if step4_to_og is None:
        step4_to_og = np.eye(4, dtype=np.float64)
    ext_void = np.asarray(ext_void, dtype=np.float64)
    ext_orig = np.asarray(ext_orig, dtype=np.float64)
    fallback = se3_src_to_og(
        ext_src_anchor=ext_void[anchor_row],
        image_c2w=image_c2w,
        step4_to_og=step4_to_og,
    )
    info = {"alignment": "se3_fallback", "reason": None, "scale": 1.0}

    if ext_void.shape != ext_orig.shape:
        info["reason"] = (
            f"void/orig extrinsics shape {tuple(ext_void.shape)} vs {tuple(ext_orig.shape)}"
        )
        return fallback, 1.0, info
    if ext_void.shape[0] < 3:
        info["reason"] = f"need >=3 frames for Sim(3), got {ext_void.shape[0]}"
        return fallback, 1.0, info
    if not (0 <= anchor_row < ext_void.shape[0]):
        info["reason"] = f"anchor_row={anchor_row} out of range for {ext_void.shape[0]} frames"
        return fallback, 1.0, info

    centers_void = camera_centers_from_world2cam(ext_void)
    centers_orig = camera_centers_from_world2cam(ext_orig)
    transform, scale = umeyama_alignment(centers_void, centers_orig, with_scale=True)
    predicted = (centers_void @ transform[:3, :3].T) + transform[:3, 3]
    residual = np.linalg.norm(predicted - centers_orig, axis=1)
    info.update(
        {
            "scale": float(scale),
            "residual_mean_m": float(residual.mean()),
            "residual_max_m": float(residual.max()),
        }
    )
    if float(residual.max()) > max_residual_m:
        info["reason"] = (
            f"Sim(3) max residual {residual.max():.4f} m > {max_residual_m} m"
        )
        return fallback, 1.0, info
    if not (scale_min <= scale <= scale_max):
        info["reason"] = (
            f"Sim(3) scale {scale:.4f} outside [{scale_min}, {scale_max}]"
        )
        return fallback, 1.0, info
    if not (DEFAULT_WARN_SCALE_MIN <= scale <= DEFAULT_WARN_SCALE_MAX):
        info["scale_warning"] = (
            f"Sim(3) scale {scale:.4f} is outside "
            f"[{DEFAULT_WARN_SCALE_MIN}, {DEFAULT_WARN_SCALE_MAX}] "
            "but residual is acceptable; applying it"
        )

    matrix = (
        np.asarray(step4_to_og, dtype=np.float64)
        @ np.asarray(image_c2w, dtype=np.float64)
        @ _as_w2c_4x4(ext_orig[anchor_row])
        @ transform
    )
    info["alignment"] = "sim3"
    info["reason"] = None
    return matrix, float(scale), info
