# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from simfoundry.pipeline.background_bundle import (
    _nvcc_host_compiler_exports,
    _ply_to_usd_command,
    _resolve_nvcc_host_cxx,
)


def test_resolve_nvcc_host_cxx_picks_first_executable(tmp_path):
    missing = tmp_path / "missing-g++"
    host = tmp_path / "g++"
    host.write_text("")
    host.chmod(0o755)
    assert _resolve_nvcc_host_cxx((str(missing), str(host))) == str(host)


def test_resolve_nvcc_host_cxx_errors_when_none_exist(tmp_path):
    with pytest.raises(RuntimeError, match="No C\\+\\+ host compiler"):
        _resolve_nvcc_host_cxx((str(tmp_path / "missing"), None))


def test_install_3dgrut_installs_gxx_for_cc1plus():
    script = Path(__file__).resolve().parents[1] / "scripts/installation/install_3dgrut.sh"
    text = script.read_text()
    assert "gxx_linux-64" in text
    assert "cc1plus" in text


def test_ply_to_usd_command_forces_host_cxx_after_activation():
    exporter = Path("/repo/deps/3dgrut/threedgrut/export/scripts/ply_to_usd.py")
    command = _ply_to_usd_command(
        exporter,
        Path("/in.ply"),
        Path("/out.usdz"),
        env_name="3dgrut",
        host_cxx="/usr/bin/g++",
    )
    assert command[:6] == ["mamba", "run", "-n", "3dgrut", "bash", "-c"]
    inner = command[6]
    exports = _nvcc_host_compiler_exports("/usr/bin/g++")
    assert inner.startswith(exports)
    assert "CC=/usr/bin/g++" in inner
    assert "CXX=/usr/bin/g++" in inner
    assert "CUDAHOSTCXX=/usr/bin/g++" in inner
    assert "NVCC_APPEND_FLAGS=" in inner
    assert "-ccbin /usr/bin/g++" in inner
    assert str(exporter) in inner
    assert "--output_file /out.usdz" in inner
