# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_DIR = REPO_ROOT / "scripts/installation"


def _env():
    env = os.environ.copy()
    mamba = shutil.which("mamba") or "/root/miniconda3/bin/mamba"
    env["PATH"] = f"{Path(mamba).parent}:{env.get('PATH', '')}"
    return env


def test_installer_help_lists_both_profiles():
    for script in ("install_simfoundry.sh", "install_everything.sh"):
        result = subprocess.run(
            ["bash", str(INSTALL_DIR / script), "--help"],
            env=_env(),
            text=True,
            capture_output=True,
            check=True,
        )
        assert "--reconstruction-only" in result.stdout
        assert "--simulation-only" in result.stdout
        assert "--skip-robot-assets" in result.stdout


def test_simfoundry_profiles_are_mutually_exclusive():
    result = subprocess.run(
        [
            "bash", str(INSTALL_DIR / "install_simfoundry.sh"),
            "--default", "--reconstruction-only", "--simulation-only",
        ],
        env=_env(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "mutually exclusive" in result.stderr


def test_simfoundry_installer_repairs_charset_normalizer_before_og_import():
    text = (INSTALL_DIR / "install_simfoundry.sh").read_text()
    assert "charset-normalizer==3.3.2" in text
    assert "--no-binary charset-normalizer" in text
    assert "download_omnigibson_robot_assets" in text
    assert "--skip-robot-assets" in text
    assert '"${SKIP_ROBOT_ASSETS}" == false' in text


def test_simulation_profile_rejects_reconstruction_envs():
    result = subprocess.run(
        [
            "bash", str(INSTALL_DIR / "install_everything.sh"),
            "--simulation-only", "--only", "da3",
        ],
        env=_env(),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "may only install" in result.stderr
