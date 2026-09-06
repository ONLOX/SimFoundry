# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a local SAM3 checkpoint before falling back to Hugging Face."""

import os
from pathlib import Path

from simfoundry import CHECKPOINT_DIR


def resolve_sam3_checkpoint() -> str | None:
    """Return an explicitly configured or repo-local SAM3 checkpoint."""
    explicit = os.environ.get("SAM3_CHECKPOINT_PATH")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM3_CHECKPOINT_PATH does not exist: {path}")
        return str(path)

    default = Path(CHECKPOINT_DIR) / "sam3.pt"
    return str(default) if default.is_file() else None
