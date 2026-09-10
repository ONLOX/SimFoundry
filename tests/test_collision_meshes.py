# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collision generation: visual keeps a cavity; convex fills it."""

import numpy as np
import pytest
import trimesh

from simfoundry.utils.asset_conversion_utils import (
    generate_collision_meshes,
    resolve_collision_method,
)


def _open_box_surface(size=(0.30, 0.20, 0.15)):
    """Five quads: bottom and four walls, no lid. Not a volume."""
    sx, sy, sz = size
    hx, hy = sx / 2, sy / 2
    verts = np.array([
        [-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0],
        [-hx, -hy, sz], [hx, -hy, sz], [hx, hy, sz], [-hx, hy, sz],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def test_visual_collision_keeps_open_box_cavity():
    visual = _open_box_surface()
    hulls = generate_collision_meshes(visual, method="visual", max_faces=50)
    assert len(hulls) == 1
    collision = hulls[0]
    solid = visual.convex_hull
    assert collision.faces.shape[0] >= 8
    if collision.is_volume:
        assert collision.volume < 0.4 * solid.volume
    np.testing.assert_allclose(collision.extents, visual.extents, rtol=0.05, atol=0.01)


def test_convex_collision_fills_open_box():
    visual = _open_box_surface()
    hulls = generate_collision_meshes(visual, method="convex")
    assert len(hulls) == 1
    assert hulls[0].is_volume
    assert hulls[0].volume == pytest.approx(visual.convex_hull.volume, rel=0.05)


def test_auto_collision_keeps_box_visual_and_bottle_convex():
    assert resolve_collision_method("auto", "open_cardboard_box") == "visual"
    assert resolve_collision_method("auto", "clear_water_bottle") == "convex"
    assert resolve_collision_method("visual", "clear_water_bottle") == "visual"
    assert resolve_collision_method(
        "auto", "clear_water_bottle", overrides={"clear_water_bottle": "visual"}
    ) == "visual"


def test_visual_method_rejects_unknown():
    visual = _open_box_surface()
    with pytest.raises(ValueError, match="Invalid collision"):
        generate_collision_meshes(visual, method="not-a-method")
