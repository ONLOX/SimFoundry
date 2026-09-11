# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchor-frame camera for training / Isaac spawn.

Stage 4 already has the reconstruction camera: OpenCV ``K`` at the DA3 (or
FoundationStereo) pixel size, and ``image_<N>_cam2world.npy`` taking that
camera into the Z-up scene used by object poses. This module writes those
numbers as JSON so a training importer can spawn ``/World/camera`` without
re-reading ``results.npz``.

The pose is already in the object / GS world. Isaac ``add_camera`` must use
``world_to_scene = I``. Applying ``CV_TO_ZUP`` a second time is wrong — the
same rule as ``gs_auto.pose.json``.

USD cameras look down ``-Z`` with ``+Y`` up; this file stores OpenCV
(``+Z`` forward, ``+Y`` down). Convert with ``R @ diag(1, -1, -1)`` at spawn,
which is what ``real2sim-for-manipulation`` ``add_camera`` already does after
the world map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "simfoundry.anchor_camera.v1"
ANCHOR_CAMERA_RELPATH = "s4_frame/anchor_camera.json"


def _as_4x4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = matrix
        return out
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 3x4 or 4x4 matrix, got {matrix.shape}")
    return matrix.copy()


def _as_k(matrix: np.ndarray) -> np.ndarray:
    k = np.asarray(matrix, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got {k.shape}")
    return k


def build_anchor_camera(
    *,
    frame_index: int,
    K: np.ndarray,
    width: int,
    height: int,
    cam2world: np.ndarray,
    world_from_s4: np.ndarray | None = None,
) -> dict:
    """Build the JSON payload. ``cam2world`` maps OpenCV camera points to s4.

    ``world_from_s4`` is ``step4_to_og`` when that file exists; otherwise the
    s4 frame *is* the OG / object world.
    """
    k = _as_k(K)
    c2w_s4 = _as_4x4(cam2world)
    world_from_s4 = np.eye(4, dtype=np.float64) if world_from_s4 is None else _as_4x4(world_from_s4)
    c2w = world_from_s4 @ c2w_s4
    w2c = np.linalg.inv(c2w)
    rot = c2w[:3, :3]
    # s4's floor alignment is a proper rotation; reject a leftover scale so a
    # training camera cannot inherit a sheared USD basis.
    scales = np.linalg.norm(rot, axis=0)
    if np.any(scales < 1e-8):
        raise ValueError("cam2world rotation has a zero column")
    rot_n = rot / scales
    if float(np.linalg.det(rot_n)) < 0:
        raise ValueError("cam2world rotation is improper; cannot spawn a camera")
    u, _, vt = np.linalg.svd(rot_n)
    rot_n = u @ vt
    if np.linalg.det(rot_n) < 0:
        u[:, -1] *= -1
        rot_n = u @ vt
    pos = c2w[:3, 3]
    ori_xyzw = Rotation.from_matrix(rot_n).as_quat()
    return {
        "schema": SCHEMA,
        "frame_index": int(frame_index),
        "world": "og",
        "camera_convention": "opencv",
        "units": "m",
        "width": int(width),
        "height": int(height),
        "K": k.tolist(),
        "cam2world": c2w.tolist(),
        "world2cam": w2c.tolist(),
        "pos": pos.tolist(),
        "ori_xyzw": ori_xyzw.tolist(),
        "isaac_world_to_scene": "identity",
    }


def write_anchor_camera(out_dir: str | Path, payload: dict) -> Path:
    """Write ``anchor_camera.json`` and ``image_<N>_camera.json`` under s4."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    stable = out_dir / "anchor_camera.json"
    named = out_dir / f"image_{payload['frame_index']}_camera.json"
    stable.write_text(text)
    named.write_text(text)
    return stable


def load_anchor_camera(path: str | Path) -> dict:
    """Read and lightly validate an anchor-camera JSON."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported anchor camera schema: {payload.get('schema')!r}")
    for key in ("frame_index", "K", "cam2world", "width", "height"):
        if key not in payload:
            raise ValueError(f"anchor camera JSON missing {key}")
    return payload


def _load_step4_to_og(s4_dir: Path) -> np.ndarray:
    path = s4_dir / "step4_to_og_tf.npy"
    if path.is_file():
        return _as_4x4(np.load(path))
    return np.eye(4, dtype=np.float64)


def _find_s4_cam2world(s4_dir: Path) -> tuple[int, np.ndarray]:
    files = sorted(s4_dir.glob("image_*_cam2world.npy"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one image_*_cam2world.npy in {s4_dir}, found {len(files)}"
        )
    frame_index = int(files[0].stem.split("_")[1])
    return frame_index, _as_4x4(np.load(files[0]))


def _load_k_and_size(scene_dir: Path, frame_index: int) -> tuple[np.ndarray, int, int]:
    da_npz = scene_dir / "s2_da" / "da" / "exports" / "npz" / "results.npz"
    if da_npz.is_file():
        results = np.load(da_npz)
        k = _as_k(results["intrinsics"][frame_index])
        image = results["image"][frame_index]
        height, width = int(image.shape[0]), int(image.shape[1])
        return k, width, height
    fs_dir = scene_dir / "s2_fs"
    k_path = fs_dir / f"image_{frame_index}_K.npy"
    rgb_path = fs_dir / f"image_{frame_index}_rgb.npy"
    if k_path.is_file() and rgb_path.is_file():
        rgb = np.load(rgb_path)
        return _as_k(np.load(k_path)), int(rgb.shape[1]), int(rgb.shape[0])
    raise FileNotFoundError(
        f"No intrinsics for frame {frame_index}: missing {da_npz} and {k_path}"
    )


def ensure_anchor_camera(scene_dir: str | Path) -> Path:
    """Write ``s4_frame/anchor_camera.json`` from stage-4 / stage-2 outputs."""
    scene_dir = Path(scene_dir)
    s4_dir = scene_dir / "s4_frame"
    frame_index, cam2world = _find_s4_cam2world(s4_dir)
    k, width, height = _load_k_and_size(scene_dir, frame_index)
    payload = build_anchor_camera(
        frame_index=frame_index,
        K=k,
        width=width,
        height=height,
        cam2world=cam2world,
        world_from_s4=_load_step4_to_og(s4_dir),
    )
    return write_anchor_camera(s4_dir, payload)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = _parse_args(argv)
    path = ensure_anchor_camera(args.scene_dir)
    print(path)
    return path


if __name__ == "__main__":
    main()
