# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from simfoundry.reconstruction.anchor_camera import (
    SCHEMA,
    build_anchor_camera,
    ensure_anchor_camera,
    load_anchor_camera,
    write_anchor_camera,
)
from simfoundry.reconstruction.bundle import export_bundle, import_bundle


def _identity_payload():
    k = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = [0.1, 0.2, 0.8]
    return build_anchor_camera(
        frame_index=80,
        K=k,
        width=640,
        height=360,
        cam2world=c2w,
    )


def test_build_anchor_camera_round_trip():
    payload = _identity_payload()
    assert payload["schema"] == SCHEMA
    assert payload["world"] == "og"
    assert payload["camera_convention"] == "opencv"
    assert payload["isaac_world_to_scene"] == "identity"
    assert payload["frame_index"] == 80
    assert payload["width"] == 640
    assert payload["height"] == 360
    np.testing.assert_allclose(payload["pos"], [0.1, 0.2, 0.8])
    c2w = np.asarray(payload["cam2world"])
    w2c = np.asarray(payload["world2cam"])
    np.testing.assert_allclose(c2w @ w2c, np.eye(4), atol=1e-9)
    rot = Rotation.from_quat(payload["ori_xyzw"]).as_matrix()
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-9)


def test_world_from_s4_is_composed():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 0.0, 0.0]
    world_from_s4 = np.eye(4)
    world_from_s4[:3, 3] = [0.0, 0.0, 0.5]
    payload = build_anchor_camera(
        frame_index=0,
        K=np.eye(3),
        width=2,
        height=2,
        cam2world=c2w,
        world_from_s4=world_from_s4,
    )
    np.testing.assert_allclose(payload["pos"], [1.0, 0.0, 0.5])


def test_write_and_load(tmp_path):
    payload = _identity_payload()
    path = write_anchor_camera(tmp_path, payload)
    assert path.name == "anchor_camera.json"
    assert (tmp_path / "image_80_camera.json").is_file()
    loaded = load_anchor_camera(path)
    assert loaded["frame_index"] == 80


def test_ensure_from_stage_outputs(tmp_path):
    scene = tmp_path / "bottle_box"
    s4 = scene / "s4_frame"
    s4.mkdir(parents=True)
    c2w = np.eye(4)
    c2w[:3, 3] = [0.0, -0.4, 0.9]
    np.save(s4 / "image_80_cam2world.npy", c2w)
    k = np.array([[350.0, 0.0, 336.0], [0.0, 350.0, 192.0], [0.0, 0.0, 1.0]])
    image = np.zeros((384, 672, 3), dtype=np.uint8)
    da_dir = scene / "s2_da/da/exports/npz"
    da_dir.mkdir(parents=True)
    np.savez(da_dir / "results.npz", intrinsics=np.stack([np.eye(3)] * 80 + [k]), image=np.stack([image] * 81))

    path = ensure_anchor_camera(scene)
    payload = json.loads(path.read_text())
    assert payload["frame_index"] == 80
    assert payload["width"] == 672
    assert payload["height"] == 384
    np.testing.assert_allclose(payload["K"], k)
    np.testing.assert_allclose(payload["pos"], [0.0, -0.4, 0.9])


def _minimal_scene(root: Path) -> Path:
    scene = root / "test_scene"
    (scene / "s11_sim/objects/cup/model/urdf").mkdir(parents=True)
    (scene / "s11_sim/objects/cup/model/mesh").mkdir()
    (scene / "s11_sim/scene_objects_info.json").write_text('{"0": {"name": "cup"}}')
    (scene / "s11_sim/objects/cup/model/urdf/model.urdf").write_text("<robot/>")
    (scene / "s11_sim/objects/cup/model/mesh/model.obj").write_text("v 0 0 0")
    (scene / "s12_physics").mkdir()
    (scene / "s12_physics/pb_scene_poses.json").write_text('{"cup": [0, 0, 0]}')
    return scene


def test_bundle_includes_anchor_camera(tmp_path):
    source = _minimal_scene(tmp_path / "source")
    write_anchor_camera(source / "s4_frame", _identity_payload())
    archive = export_bundle(source, tmp_path / "scene.tar.gz", include_background=False)
    restored, _ = import_bundle(archive, tmp_path / "Data")
    camera = json.loads((restored / "s4_frame/anchor_camera.json").read_text())
    assert camera["schema"] == SCHEMA
    assert camera["frame_index"] == 80
