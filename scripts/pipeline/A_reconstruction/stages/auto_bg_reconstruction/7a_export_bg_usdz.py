# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export an aligned background splat as a simulator-portable USDZ."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from simfoundry import DATA_DIR
from simfoundry.pipeline.background_bundle import (
    convert_ply_to_usdz,
    pose_sidecar_for_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--root-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--env-3dgrut", default="3dgrut")
    args = parser.parse_args()

    scene_dir = args.root_dir.resolve() / args.scene_name
    bg_ply = scene_dir / "auto_bg/splat/export" / f"{args.scene_name}_bg.ply"
    source_pose = pose_sidecar_for_ply(bg_ply)
    if not source_pose.is_file():
        raise FileNotFoundError(
            f"Background pose sidecar is missing; run bridge step 6 first: {source_pose}"
        )

    export_dir = scene_dir / "auto_bg/export"
    out_usdz = export_dir / "gs_auto.usdz"
    out_pose = export_dir / "gs_auto.pose.json"
    convert_ply_to_usdz(bg_ply, out_usdz, env_name=args.env_3dgrut)
    shutil.copy2(source_pose, out_pose)
    print(f"background_usdz={out_usdz}")
    print(f"background_pose={out_pose}")


if __name__ == "__main__":
    main()
