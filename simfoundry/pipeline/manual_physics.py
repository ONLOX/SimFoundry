# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation and precedence for manually authored rigid-object physics."""

import math


def resolve_manual_physics(section, *, obj_phrase: str, obj_category: str, idx: int) -> dict:
    """Resolve mass and friction using object-specific values over defaults."""
    values = dict(section.get("default", {}))
    overrides = section.get("overrides", {})
    for key in (obj_phrase, obj_category, f"iter_{idx}", str(idx)):
        if key in overrides:
            values.update(dict(overrides[key]))
    missing = {"mass", "friction"} - set(values)
    if missing:
        raise ValueError(
            f"Manual physics for {obj_phrase!r} is missing {sorted(missing)}; "
            "set s11_sim.manual_physics.default or an object override"
        )
    mass = float(values["mass"])
    friction = float(values["friction"])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"Manual mass for {obj_phrase!r} must be positive, got {mass}")
    if not math.isfinite(friction) or friction < 0:
        raise ValueError(f"Manual friction for {obj_phrase!r} must be non-negative, got {friction}")
    return {"mass": mass, "friction": friction}
