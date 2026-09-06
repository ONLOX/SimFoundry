# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import a reconstruction bundle and build its robot-free OmniGibson scene."""

import argparse
from pathlib import Path

from simfoundry import DATA_DIR, REPO_DIR
from simfoundry.reconstruction.bundle import build_simulation_scene, import_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--root-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-unpacked-gb",
        type=float,
        default=250.0,
        help="Reject archives larger than this after decompression.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Validate and restore the bundle without running stages 13/14.",
    )
    args = parser.parse_args()

    scene_dir, manifest = import_bundle(
        args.bundle,
        args.root_dir,
        force=args.force,
        max_unpacked_bytes=int(args.max_unpacked_gb * 1024**3),
    )
    print(f"restored_scene={scene_dir}")
    print(f"background={manifest['background']}")
    if not args.extract_only:
        scene_json = build_simulation_scene(scene_dir, repo_root=REPO_DIR)
        print(f"scene_json={scene_json}")


if __name__ == "__main__":
    main()
