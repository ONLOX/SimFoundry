# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export H20 reconstruction outputs as one portable tar.gz."""

import argparse
from pathlib import Path

from simfoundry import DATA_DIR
from simfoundry.reconstruction.bundle import export_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--root-dir", type=Path, default=Path(DATA_DIR))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Export only stage 11/12 outputs.",
    )
    args = parser.parse_args()
    scene_dir = args.root_dir.resolve() / args.scene_name
    output = args.output or (args.root_dir.resolve() / f"{args.scene_name}.tar.gz")
    result = export_bundle(
        scene_dir,
        output,
        include_background=not args.no_background,
    )
    print(result)


if __name__ == "__main__":
    main()
