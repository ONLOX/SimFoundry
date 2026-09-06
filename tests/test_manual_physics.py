# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from simfoundry.pipeline.manual_physics import resolve_manual_physics


def test_manual_physics_uses_defaults_and_object_override():
    section = {
        "default": {"mass": 0.2, "friction": 0.5},
        "overrides": {
            "banana": {"mass": 0.12},
            "iter_4": {"friction": 0.7},
        },
    }
    assert resolve_manual_physics(
        section, obj_phrase="banana", obj_category="banana", idx=4
    ) == {"mass": 0.12, "friction": 0.7}


@pytest.mark.parametrize(
    "values, message",
    [
        ({"mass": 0.0, "friction": 0.5}, "mass"),
        ({"mass": 0.2, "friction": -0.1}, "friction"),
    ],
)
def test_manual_physics_rejects_invalid_values(values, message):
    with pytest.raises(ValueError, match=message):
        resolve_manual_physics(
            {"default": values, "overrides": {}},
            obj_phrase="object",
            obj_category="object",
            idx=0,
        )
