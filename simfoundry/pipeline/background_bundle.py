# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable background-splat conversion and pose helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from simfoundry import REPO_DIR


_BUILD_ENV_KEYS = (
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CFLAGS", "CXXFLAGS",
    "CPPFLAGS", "LDFLAGS", "CC", "CXX", "CC_FOR_BUILD", "CXX_FOR_BUILD",
    "GCC", "GCC_AR", "GCC_NM", "GCC_RANLIB", "CXXFILT", "CUDAARCHS",
    "CMAKE_ARGS", "CUDA_HOME",
)


def convert_ply_to_usdz(
    in_ply: str | Path,
    out_usdz: str | Path,
    *,
    env_name: str = "3dgrut",
    repo_root: str | Path = REPO_DIR,
) -> Path:
    """Convert a Gaussian-splat PLY into a portable NuRec USDZ."""
    in_ply = Path(in_ply).resolve()
    out_usdz = Path(out_usdz).resolve()
    if not in_ply.is_file():
        raise FileNotFoundError(f"Background splat PLY does not exist: {in_ply}")
    exporter = (
        Path(repo_root).resolve()
        / "deps/3dgrut/threedgrut/export/scripts/ply_to_usd.py"
    )
    if not exporter.is_file():
        raise FileNotFoundError(f"3DGRUT exporter does not exist: {exporter}")

    out_usdz.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = env.get(
        "TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;9.0;10.0;12.0+PTX"
    )
    for key in _BUILD_ENV_KEYS:
        env.pop(key, None)
    subprocess.run(
        [
            "mamba", "run", "-n", env_name, "python", str(exporter),
            str(in_ply), "--output_file", str(out_usdz),
        ],
        check=True,
        env=env,
    )
    if not out_usdz.is_file():
        raise RuntimeError(f"3DGRUT did not produce the expected USDZ: {out_usdz}")
    return out_usdz


def load_bg_pose_sidecar(path: str | Path) -> dict | None:
    """Load and validate a background pose sidecar."""
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if "pos" not in payload or "ori_xyzw" not in payload:
        raise ValueError(f"Background pose sidecar lacks pos/ori_xyzw: {path}")
    return {
        "pos": payload["pos"],
        "ori_xyzw": payload["ori_xyzw"],
        "scale": float(payload.get("scale", 1.0)),
        "sidecar_path": path,
    }


def pose_sidecar_for_ply(bg_ply: str | Path) -> Path:
    """Return the sidecar path written by the background bridge stage."""
    bg_ply = Path(bg_ply)
    return bg_ply.with_name(bg_ply.name + ".pose.json")


def materialize_bg_usdz(
    destination: str | Path,
    *,
    prebuilt_usdz: str | Path | None = None,
    source_ply: str | Path | None = None,
) -> Path:
    """Copy a prebuilt USDZ or convert a PLY into ``destination``."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if prebuilt_usdz is not None:
        source = Path(prebuilt_usdz).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Prebuilt background USDZ does not exist: {source}")
        if source != destination:
            shutil.copy2(source, destination)
    elif source_ply is not None:
        convert_ply_to_usdz(source_ply, destination)
    else:
        raise ValueError("Either prebuilt_usdz or source_ply is required")
    if not destination.is_file():
        raise RuntimeError(f"Background USDZ was not materialized: {destination}")
    return destination
