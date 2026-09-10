#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Set PhysX mesh-collision approximation on every collider in a USD.

Isaac's URDF importer authors ``convexHull`` on each collision mesh. That is
correct for CoACD pieces (already convex) and wrong for a visual-mesh collider:
the hull fills an open box. ``sdf`` follows the authored triangles, so a
hollow container stays a cavity and dynamic bodies still collide.

Usage:
    python set_usd_collision_approximation.py <usd_path> --approximation sdf
"""
import argparse
import os
import site
import sys
from pathlib import Path

os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

_PXR_BOOTSTRAP_ENV = "SIMFOUNDRY_PXR_BOOTSTRAPPED"

# Isaac / PhysX tokens on physics:approximation.
_APPROXIMATIONS = (
    "none",
    "convexHull",
    "convexDecomposition",
    "meshSimplification",
    "sdf",
    "boundingSphere",
    "boundingCube",
)


def _prepend_env_path(env: dict[str, str], key: str, value: str) -> None:
    current = env.get(key)
    paths = [] if not current else current.split(os.pathsep)
    if value not in paths:
        env[key] = value if not current else value + os.pathsep + current


def _find_usd_extension_root() -> Path:
    roots: list[Path] = []
    for site_root in site.getsitepackages():
        roots.extend(Path(site_root).glob("isaacsim/extscache/omni.usd.libs-*"))
    user_site = site.getusersitepackages()
    if user_site:
        roots.extend(Path(user_site).glob("isaacsim/extscache/omni.usd.libs-*"))

    valid_roots = sorted(root for root in roots if (root / "pxr").is_dir() and (root / "bin").is_dir())
    if not valid_roots:
        raise RuntimeError("Could not locate Isaac Sim omni.usd.libs extension containing pxr bindings.")
    return valid_roots[-1]


def _bootstrap_pxr():
    """Import pxr, re-execing once with the Isaac Sim USD libs on the path if needed."""
    try:
        from pxr import Usd, UsdPhysics

        return Usd, UsdPhysics
    except (ImportError, ModuleNotFoundError):
        if os.environ.get(_PXR_BOOTSTRAP_ENV) == "1":
            raise
        usd_root = _find_usd_extension_root()
        env = os.environ.copy()
        _prepend_env_path(env, "PYTHONPATH", str(usd_root))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(Path(sys.prefix) / "lib"))
        _prepend_env_path(env, "LD_LIBRARY_PATH", str(usd_root / "bin"))
        env[_PXR_BOOTSTRAP_ENV] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
        raise RuntimeError("Failed to re-exec with USD bindings enabled.")


def set_collision_approximation(usd_path: str, approximation: str) -> int:
    """Author ``physics:approximation`` on every mesh collider in ``usd_path``.

    Returns:
        int: Number of prims changed.
    """
    Usd, UsdPhysics = _bootstrap_pxr()

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Error: Could not open USD stage at {usd_path}")
        return 0

    changed = 0
    for prim in stage.Traverse():
        if not (
            prim.HasAPI(UsdPhysics.CollisionAPI)
            or prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            or "collision" in prim.GetPath().pathString.lower()
        ):
            continue
        if not prim.IsA("Mesh") and prim.GetTypeName() != "Mesh":
            # Apply the API only on mesh colliders; cubes/spheres keep their shape.
            if prim.GetTypeName() not in ("Mesh",):
                continue
        api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        attr = api.CreateApproximationAttr()
        current = attr.Get()
        if current is not None and str(current) == approximation:
            continue
        attr.Set(approximation)
        print(f"  {prim.GetPath()}: approximation {current} -> {approximation}")
        changed += 1

    if changed:
        stage.GetRootLayer().Save()
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd_path")
    parser.add_argument("--approximation", required=True, choices=_APPROXIMATIONS)
    args = parser.parse_args()

    if not os.path.isfile(args.usd_path):
        print(f"Error: USD file not found: {args.usd_path}")
        return 1

    changed = set_collision_approximation(args.usd_path, args.approximation)
    print(f"Set approximation={args.approximation} on {changed} collider(s) in {args.usd_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
