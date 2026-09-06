# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create, validate, and import portable reconstruction bundles."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from simfoundry import REPO_DIR


SCHEMA = "simfoundry.reconstruction.bundle.v1"
MANIFEST_NAME = "manifest.json"
DEFAULT_MAX_UNPACKED_BYTES = 250 * 1024**3
MAX_MANIFEST_BYTES = 16 * 1024**2
_SCENE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class BundleError(ValueError):
    """Raised when a reconstruction bundle violates its contract."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scene_name(name: str) -> str:
    if not _SCENE_NAME.fullmatch(name) or name in {".", ".."}:
        raise BundleError(f"Unsafe scene name: {name!r}")
    return name


def _bundle_sources(scene_dir: Path, include_background: bool) -> list[tuple[Path, str]]:
    required = (
        scene_dir / "s11_sim/scene_objects_info.json",
        scene_dir / "s12_physics/pb_scene_poses.json",
    )
    for path in required:
        if not path.is_file():
            raise BundleError(f"Required reconstruction output is missing: {path}")

    objects_dir = scene_dir / "s11_sim/objects"
    object_files = []
    visited_dirs = set()
    for root, dirnames, filenames in os.walk(objects_dir, followlinks=True):
        root_path = Path(root)
        try:
            root_path.resolve().relative_to(scene_dir)
        except ValueError as exc:
            raise BundleError(f"Object asset symlink escapes the scene: {root_path}") from exc
        stat = root_path.stat()
        directory_id = (stat.st_dev, stat.st_ino)
        if directory_id in visited_dirs:
            dirnames.clear()
            continue
        visited_dirs.add(directory_id)
        for filename in filenames:
            path = root_path / filename
            if path.is_file():
                try:
                    path.resolve().relative_to(scene_dir)
                except ValueError as exc:
                    raise BundleError(f"Object asset symlink escapes the scene: {path}") from exc
                object_files.append(path)
    object_files.sort()
    if not object_files:
        raise BundleError(f"No stage-11 object assets found under: {objects_dir}")

    paths = list(required) + object_files
    bg_usdz = scene_dir / "auto_bg/export/gs_auto.usdz"
    bg_pose = scene_dir / "auto_bg/export/gs_auto.pose.json"
    present = (bg_usdz.is_file(), bg_pose.is_file())
    if include_background and present != (True, True):
        raise BundleError(
            "Background export is incomplete; expected both "
            f"{bg_usdz} and {bg_pose}"
        )
    if include_background:
        paths.extend((bg_usdz, bg_pose))

    seen = set()
    sources = []
    for path in paths:
        try:
            path.resolve().relative_to(scene_dir)
        except ValueError as exc:
            raise BundleError(f"Bundle input symlink escapes the scene: {path}") from exc
        relative = path.relative_to(scene_dir).as_posix()
        if relative not in seen:
            seen.add(relative)
            sources.append((path, f"scene/{relative}"))
    return sorted(sources, key=lambda item: item[1])


def export_bundle(
    scene_dir: str | Path,
    output_path: str | Path,
    *,
    include_background: bool = True,
) -> Path:
    """Package stage 11/12 and optional auto-background outputs."""
    scene_dir = Path(scene_dir).resolve()
    scene_name = _validate_scene_name(scene_dir.name)
    output_path = Path(output_path).resolve()
    sources = _bundle_sources(scene_dir, include_background)
    files = [
        {
            "path": archive_path,
            "size": source.stat().st_size,
            "sha256": sha256_file(source),
        }
        for source, archive_path in sources
    ]
    manifest = {
        "schema": SCHEMA,
        "scene_name": scene_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "background": any(item["path"].endswith("gs_auto.usdz") for item in files),
        "files": files,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(temporary, "w:gz", dereference=True) as archive:
            info = tarfile.TarInfo(MANIFEST_NAME)
            info.size = len(manifest_bytes)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(manifest_bytes))
            for source, archive_path in sources:
                archive.add(source, arcname=archive_path, recursive=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise BundleError(f"Unsafe archive member path: {name!r}")
    return path


def _read_manifest(archive: tarfile.TarFile) -> dict:
    members = [member for member in archive.getmembers() if member.name == MANIFEST_NAME]
    if len(members) != 1 or not members[0].isfile():
        raise BundleError("Bundle must contain exactly one regular manifest.json")
    if members[0].size > MAX_MANIFEST_BYTES:
        raise BundleError(
            f"manifest.json is {members[0].size} bytes, above limit {MAX_MANIFEST_BYTES}"
        )
    stream = archive.extractfile(members[0])
    if stream is None:
        raise BundleError("Cannot read manifest.json")
    try:
        manifest = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"Invalid manifest.json: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise BundleError(f"Unsupported bundle schema: {manifest.get('schema')!r}")
    _validate_scene_name(str(manifest.get("scene_name", "")))
    return manifest


def import_bundle(
    archive_path: str | Path,
    data_root: str | Path,
    *,
    force: bool = False,
    max_unpacked_bytes: int = DEFAULT_MAX_UNPACKED_BYTES,
) -> tuple[Path, dict]:
    """Validate and restore a bundle under ``Data/<scene_name>``."""
    archive_path = Path(archive_path).resolve()
    data_root = Path(data_root).resolve()
    if not archive_path.is_file():
        raise BundleError(f"Bundle does not exist: {archive_path}")

    with tarfile.open(archive_path, "r:*") as archive:
        manifest = _read_manifest(archive)
        scene_name = manifest["scene_name"]
        records = manifest.get("files")
        if not isinstance(records, list):
            raise BundleError("Bundle manifest files must be a list")
        declared = {}
        declared_size = 0
        for record in records:
            if not isinstance(record, dict):
                raise BundleError("Every bundle file record must be an object")
            path = _safe_member_name(str(record.get("path", ""))).as_posix()
            if not path.startswith("scene/") or path in declared:
                raise BundleError(f"Invalid or duplicate manifest path: {path!r}")
            try:
                size = int(record.get("size", -1))
            except (TypeError, ValueError) as exc:
                raise BundleError(f"Invalid size for {path}") from exc
            if size < 0:
                raise BundleError(f"Invalid size for {path}")
            declared_size += size
            declared[path] = record
        if declared_size > max_unpacked_bytes:
            raise BundleError(
                f"Bundle expands to {declared_size} bytes, above limit {max_unpacked_bytes}"
            )
        if not declared:
            raise BundleError("Bundle manifest contains no files")
        required = {
            "scene/s11_sim/scene_objects_info.json",
            "scene/s12_physics/pb_scene_poses.json",
        }
        if not required.issubset(declared):
            raise BundleError(f"Bundle lacks required files: {sorted(required - set(declared))}")
        if not any(path.startswith("scene/s11_sim/objects/") for path in declared):
            raise BundleError("Bundle contains no stage-11 object assets")
        background_files = {
            "scene/auto_bg/export/gs_auto.usdz",
            "scene/auto_bg/export/gs_auto.pose.json",
        }
        has_background = background_files.issubset(declared)
        if background_files.intersection(declared) and not has_background:
            raise BundleError("Bundle contains an incomplete background export")
        if bool(manifest.get("background")) != has_background:
            raise BundleError("Bundle background flag does not match its files")

        actual = {}
        for member in archive.getmembers():
            path = _safe_member_name(member.name).as_posix()
            if path == MANIFEST_NAME:
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise BundleError(f"Links and special archive members are forbidden: {path}")
            if path in actual:
                raise BundleError(f"Duplicate archive member: {path}")
            actual[path] = member
        if set(actual) != set(declared):
            missing = sorted(set(declared) - set(actual))
            extra = sorted(set(actual) - set(declared))
            raise BundleError(f"Bundle file list mismatch; missing={missing}, extra={extra}")

        data_root.mkdir(parents=True, exist_ok=True)
        target = data_root / scene_name
        if target.exists() and not force:
            raise BundleError(f"Target scene already exists (use --force): {target}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{scene_name}.import-", dir=data_root))
        try:
            for path, member in actual.items():
                record = declared[path]
                if member.size != int(record.get("size", -1)):
                    raise BundleError(f"Size mismatch for {path}")
                destination = temporary / PurePosixPath(path).relative_to("scene")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise BundleError(f"Cannot read archive member: {path}")
                digest = hashlib.sha256()
                with destination.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        output.write(chunk)
                if digest.hexdigest() != record.get("sha256"):
                    raise BundleError(f"SHA-256 mismatch for {path}")
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return target, manifest


def build_simulation_scene(
    scene_dir: str | Path,
    *,
    repo_root: str | Path = REPO_DIR,
    python_bin: str = sys.executable,
) -> Path:
    """Run stages 13/14 without a robot, then assemble the optional background."""
    scene_dir = Path(scene_dir).resolve()
    repo_root = Path(repo_root).resolve()
    data_root = scene_dir.parent
    scene_name = scene_dir.name
    env = os.environ.copy()
    env["OMNIGIBSON_HEADLESS"] = "1"
    subprocess.run(
        [
            "bash", str(repo_root / "scripts/pipeline/A_reconstruction/run.sh"),
            "--scene-name", scene_name,
            "--root-dir", str(data_root),
            "--include", "13,14",
            "--no-stream",
            "--exec-mode", "direct",
            "--python-bin", python_bin,
            "--",
            "s14_og.include_robot=false",
            "s14_og.include_table=false",
            "s14_og.interactive=false",
            "s14_og.capture_image=false",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    scene_state = scene_dir / "s14_og/reconstructed_og_scene.json"
    if not scene_state.is_file():
        raise BundleError(f"Stages 13/14 did not produce: {scene_state}")
    bg_usdz = scene_dir / "auto_bg/export/gs_auto.usdz"
    bg_pose = scene_dir / "auto_bg/export/gs_auto.pose.json"
    if bg_usdz.is_file() or bg_pose.is_file():
        if not (bg_usdz.is_file() and bg_pose.is_file()):
            raise BundleError("Imported background is incomplete")
        subprocess.run(
            [
                python_bin,
                str(
                    repo_root
                    / "scripts/pipeline/A_reconstruction/stages/"
                    "auto_bg_reconstruction/7_build_og_scene_assets.py"
                ),
                f"scene_name={scene_name}",
                f"root_dir={data_root}",
                f"s7_build_assets.scene_state={scene_state}",
                f"s7_build_assets.bg_usdz={bg_usdz}",
                f"s7_build_assets.bg_pose_json={bg_pose}",
            ],
            cwd=repo_root,
            env=env,
            check=True,
        )
        assembled = (
            repo_root
            / "assets/scenes"
            / scene_name
            / f"{scene_name}_scene_state_auto_bg.json"
        )
        if not assembled.is_file():
            raise BundleError(f"Background assembly did not produce: {assembled}")
        return assembled
    return scene_state
