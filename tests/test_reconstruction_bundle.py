# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from pathlib import Path

import pytest

from simfoundry.pipeline.background_bundle import materialize_bg_usdz
from simfoundry.reconstruction.bundle import (
    BundleError,
    build_simulation_scene,
    export_bundle,
    import_bundle,
)


def _scene(root: Path, *, background: bool = True) -> Path:
    scene = root / "test_scene"
    (scene / "s11_sim/objects/cup/model/urdf").mkdir(parents=True)
    (scene / "s11_sim/objects/cup/model/mesh").mkdir()
    (scene / "s11_sim/scene_objects_info.json").write_text('{"0": {"name": "cup"}}')
    (scene / "s11_sim/objects/cup/model/urdf/model.urdf").write_text("<robot/>")
    (scene / "s11_sim/objects/cup/model/mesh/model.obj").write_text("v 0 0 0")
    (scene / "s12_physics").mkdir()
    (scene / "s12_physics/pb_scene_poses.json").write_text('{"cup": [0, 0, 0]}')
    if background:
        (scene / "auto_bg/export").mkdir(parents=True)
        (scene / "auto_bg/export/gs_auto.usdz").write_bytes(b"usdz")
        (scene / "auto_bg/export/gs_auto.pose.json").write_text(
            '{"pos": [0, 0, 0], "ori_xyzw": [0, 0, 0, 1], "scale": 1}'
        )
    return scene


def _rewrite_archive(source: Path, destination: Path, replacements: dict[str, bytes]) -> None:
    with tarfile.open(source, "r:gz") as original, tarfile.open(destination, "w:gz") as changed:
        for member in original.getmembers():
            stream = original.extractfile(member) if member.isfile() else None
            data = stream.read() if stream is not None else None
            if member.name in replacements:
                data = replacements[member.name]
                member.size = len(data)
            changed.addfile(member, io.BytesIO(data) if data is not None else None)


def test_bundle_round_trip_with_background(tmp_path):
    source = _scene(tmp_path / "source")
    archive = export_bundle(source, tmp_path / "scene.tar.gz")
    restored, manifest = import_bundle(archive, tmp_path / "Data")

    assert manifest["schema"] == "simfoundry.reconstruction.bundle.v1"
    assert manifest["background"] is True
    assert (restored / "s11_sim/objects/cup/model/mesh/model.obj").read_text() == "v 0 0 0"
    assert (restored / "auto_bg/export/gs_auto.usdz").read_bytes() == b"usdz"
    assert all(not Path(record["path"]).is_absolute() for record in manifest["files"])


def test_bundle_requires_stage_outputs(tmp_path):
    scene = _scene(tmp_path, background=False)
    (scene / "s12_physics/pb_scene_poses.json").unlink()
    with pytest.raises(BundleError, match="Required reconstruction output"):
        export_bundle(scene, tmp_path / "scene.tar.gz", include_background=False)


def test_bundle_rejects_hash_tampering(tmp_path):
    source = _scene(tmp_path / "source")
    archive = export_bundle(source, tmp_path / "scene.tar.gz")
    tampered = tmp_path / "tampered.tar.gz"
    _rewrite_archive(archive, tampered, {"scene/s11_sim/scene_objects_info.json": b"tampered"})

    with pytest.raises(BundleError, match="mismatch"):
        import_bundle(tampered, tmp_path / "Data")


def test_bundle_rejects_path_traversal(tmp_path):
    archive = tmp_path / "traversal.tar.gz"
    manifest = {
        "schema": "simfoundry.reconstruction.bundle.v1",
        "scene_name": "safe",
        "files": [{"path": "../outside", "size": 1, "sha256": "0" * 64}],
    }
    with tarfile.open(archive, "w:gz") as output:
        payload = json.dumps(manifest).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        output.addfile(info, io.BytesIO(payload))
        evil = tarfile.TarInfo("../outside")
        evil.size = 1
        output.addfile(evil, io.BytesIO(b"x"))

    with pytest.raises(BundleError, match="Unsafe archive member"):
        import_bundle(archive, tmp_path / "Data")
    assert not (tmp_path / "outside").exists()


def test_bundle_without_background(tmp_path):
    source = _scene(tmp_path / "source", background=False)
    archive = export_bundle(source, tmp_path / "scene.tar.gz", include_background=False)
    restored, manifest = import_bundle(archive, tmp_path / "Data")
    assert manifest["background"] is False
    assert not (restored / "auto_bg").exists()


def test_bundle_enforces_unpacked_size_limit(tmp_path):
    source = _scene(tmp_path / "source", background=False)
    archive = export_bundle(source, tmp_path / "scene.tar.gz", include_background=False)
    with pytest.raises(BundleError, match="above limit"):
        import_bundle(archive, tmp_path / "Data", max_unpacked_bytes=1)


def test_bundle_enforces_manifest_size_limit(tmp_path, monkeypatch):
    source = _scene(tmp_path / "source", background=False)
    archive = export_bundle(source, tmp_path / "scene.tar.gz", include_background=False)
    monkeypatch.setattr("simfoundry.reconstruction.bundle.MAX_MANIFEST_BYTES", 1)
    with pytest.raises(BundleError, match="manifest.json.*above limit"):
        import_bundle(archive, tmp_path / "Data")


def test_bundle_rejects_asset_symlink_outside_scene(tmp_path):
    source = _scene(tmp_path / "source", background=False)
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "host.txt").write_text("do not archive")
    (source / "s11_sim/objects/external").symlink_to(secret, target_is_directory=True)
    with pytest.raises(BundleError, match="escapes the scene"):
        export_bundle(source, tmp_path / "scene.tar.gz", include_background=False)


def test_prebuilt_usdz_is_copied_without_converter(tmp_path):
    source = tmp_path / "source.usdz"
    destination = tmp_path / "scene/objects/gs_background/gs_auto.usdz"
    source.write_bytes(b"prebuilt")
    assert materialize_bg_usdz(destination, prebuilt_usdz=source) == destination.resolve()
    assert destination.read_bytes() == b"prebuilt"


def test_build_simulation_scene_is_robot_free_and_assembles_background(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    scene = tmp_path / "Data/test_scene"
    (scene / "auto_bg/export").mkdir(parents=True)
    (scene / "auto_bg/export/gs_auto.usdz").write_bytes(b"usdz")
    (scene / "auto_bg/export/gs_auto.pose.json").write_text("{}")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "run.sh" in str(command[1]):
            (scene / "s14_og").mkdir()
            (scene / "s14_og/reconstructed_og_scene.json").write_text("{}")
        else:
            assembled = repo / "assets/scenes/test_scene"
            assembled.mkdir(parents=True)
            (assembled / "test_scene_scene_state_auto_bg.json").write_text("{}")

    monkeypatch.setattr("simfoundry.reconstruction.bundle.subprocess.run", fake_run)
    result = build_simulation_scene(scene, repo_root=repo, python_bin="python")

    assert result == repo / "assets/scenes/test_scene/test_scene_scene_state_auto_bg.json"
    assert "--include" in calls[0]
    assert calls[0][calls[0].index("--include") + 1] == "13,14"
    assert "s14_og.include_robot=false" in calls[0]
    assert any(str(token).endswith("7_build_og_scene_assets.py") for token in calls[1])
