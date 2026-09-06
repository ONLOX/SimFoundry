# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Materialise an `assets/scenes/<scene>/` directory in the nv_desk style from the
outputs of stages 10-14 + the BG splat produced by the auto-bg pipeline.

Inputs:
  - Stage-14 scene state JSON:  Data/<scene>/s14_og/reconstructed_og_scene.json
  - Imported dataset USDs:      deps/BEHAVIOR-1K/datasets/<dataset_name>/objects/<cat>/<model>/
  - Trained BG splat PLY:       passed via --bg-splat-ply (default
                                Data/<scene>/auto_bg/splat/export/<scene>_bg.ply).
                                The PLY is in DA3 world; the bridge's pose
                                sidecar (`<bg_ply>.pose.json`) carries the
                                rigid transform to OG world.
  - Or a prebuilt BG USDZ and explicit pose sidecar, supplied with
    `s7_build_assets.bg_usdz` and `s7_build_assets.bg_pose_json`.

Outputs:
  - assets/scenes/<scene>/objects/<cat>/<model>/{usd,material,misc}/...
  - assets/scenes/<scene>/objects/gs_background/gs_auto.usdz
  - assets/scenes/<scene>/<scene>_scene_state_auto_bg.json

What this script does:
  1. Read stage-14 scene state.
  2. For each `DatasetObject` entry, copy (or symlink) the dataset object tree
     under assets/scenes/<scene>/objects/ and rewrite the init_info entry to
     `USDObject` with an explicit absolute `usd_path`.
  3. Install a prebuilt USDZ, or convert the BG PLY through the 3DGRUT env.
  4. Inject the `gs_background` USDObject (fixed_base, visual_only) and its
     `root_link` state. If `<bg_ply>.pose.json` exists next to the splat
     (written by bridge_bg_splat_to_og.py), apply that pose onto the prim
     — so OG renders the splat as a single rigid transform on unrotated
     gaussians. Falls back to identity pose when the sidecar is absent
     (pre-pose-sidecar PLYs / legacy flows).
  5. Inject viewer_camera_state / lighting_state / ground_plane_info defaults
     from the reference nv_desk scene state when missing in stage-14's output.
  6. Recompute `expected_file_hash` for the BG USDZ.

Run from simfoundry env (Hydra override syntax):
  python scripts/pipeline/A_reconstruction/stages/auto_bg_reconstruction/7_build_og_scene_assets.py \\
      scene_name=<scene> \\
      s7_build_assets.bg_splat_ply=Data/<scene>/auto_bg/splat/export/<scene>_bg.ply
"""
import copy
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import hydra

from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir
from simfoundry.pipeline.background_bundle import (
    load_bg_pose_sidecar,
    materialize_bg_usdz,
    pose_sidecar_for_ply,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_og_scene_assets")

REPO_ROOT = Path(__file__).resolve().parents[5]

bootstrap_hydra_workdir(__file__)
from simfoundry import CFG_DIR  # noqa: E402

REF_SCENE_STATE = REPO_ROOT / "assets" / "scenes" / "nv_desk" / "nv_desk_scene_state_auto_bg.json"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _copy_asset_tree(src_dir: Path, dst_dir: Path) -> None:
    """Copy a dataset object's tree (usd/, material/, misc/, ...) into dst_dir.

    Idempotent: removes dst_dir first if it exists, then copies fresh.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.is_dir():
            shutil.copytree(child, dst_dir / child.name)
        else:
            shutil.copy2(child, dst_dir / child.name)


def _resolve_dataset_obj_dir(dataset_name: str, category: str, model: str) -> Path:
    # Use exactly the dataset named by stage 14. Selecting a same-named model
    # from another dataset can silently assemble a different asset than the one
    # that was settled and serialized.
    datasets_root = REPO_ROOT / "deps" / "BEHAVIOR-1K" / "datasets"
    return datasets_root / dataset_name / "objects" / category / model


def _build_usdobject_entry(name: str, usd_path: Path, category: str, hash_hex: str) -> dict:
    """Convert a DatasetObject init_info entry into a USDObject one with an explicit usd_path.

    Mirrors the format used in assets/scenes/nv_desk/nv_desk_scene_state_auto_bg.json.
    """
    return {
        "class_module": "omnigibson.objects.usd_object",
        "class_name": "USDObject",
        "args": {
            "name": name,
            "usd_path": str(usd_path),
            "category": category,
            "expected_file_hash": hash_hex,
        },
    }


def _build_gs_background_entry(usdz_path: Path, scale: float = 1.0) -> dict:
    return {
        "class_module": "omnigibson.objects.usd_object",
        "class_name": "USDObject",
        "args": {
            "name": "gs_background",
            "usd_path": str(usdz_path),
            "scale": [float(scale), float(scale), float(scale)],
            "fixed_base": True,
            "visual_only": True,
            "expected_file_hash": _md5(usdz_path),
        },
    }


def _build_gs_background_state(pos: list[float] | None = None,
                               ori_xyzw: list[float] | None = None) -> dict:
    """Static root-link state for the visual-only fixed splat.

    If `pos` / `ori_xyzw` are provided (e.g. from the bridge sidecar), they place
    the splat prim at that pose in OG world; the gaussians inside the USDZ stay in
    their native DA3 trained frame so no per-gaussian rotation quality loss occurs.
    Defaults are identity (used when the PLY was pre-baked to OG world by an older
    bridge run).
    """
    if pos is None:
        pos = [0.0, 0.0, 0.0]
    if ori_xyzw is None:
        ori_xyzw = [0.0, 0.0, 0.0, 1.0]
    return {
        "is_asleep": True,
        "root_link": {
            "pos": [float(v) for v in pos],
            "ori": [float(v) for v in ori_xyzw],
            "lin_vel": [0.0, 0.0, 0.0],
            "ang_vel": [0.0, 0.0, 0.0],
        },
    }


def _load_bg_pose_sidecar(bg_ply: Path) -> dict | None:
    """Return {pos, ori_xyzw, scale} parsed from `<bg_ply>.pose.json` if present.

    The bridge writes this file when it leaves the PLY in its native trained frame
    and stores the OG-world transform as a prim pose instead of baking it in.
    """
    try:
        return load_bg_pose_sidecar(pose_sidecar_for_ply(bg_ply))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Pose sidecar for %s is invalid (%s); ignoring.", bg_ply, e)
        return None


@hydra.main(config_name="auto_bg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    sec = cfg.s7_build_assets

    if not cfg.scene_name:
        sys.exit("scene_name is required (set scene_name=<scene>)")

    scene = cfg.scene_name
    scene_state = Path(sec.scene_state).resolve() if sec.scene_state else REPO_ROOT / "Data" / scene / "s14_og" / "reconstructed_og_scene.json"
    bg_ply = Path(sec.bg_splat_ply).resolve() if sec.bg_splat_ply else \
        REPO_ROOT / "Data" / scene / "auto_bg" / "splat" / "export" / f"{scene}_bg.ply"
    prebuilt_usdz = Path(sec.bg_usdz).resolve() if sec.get("bg_usdz") else None
    explicit_pose = Path(sec.bg_pose_json).resolve() if sec.get("bg_pose_json") else None
    # Resolve out_scene_dir to absolute so all usd_path entries in the written
    # scene state JSON are absolute too (OG resolves usd_path from CWD; relative
    # paths break unless the loader happens to start in the repo root).
    out_scene_dir = Path(sec.out_scene_dir).resolve() if sec.out_scene_dir else REPO_ROOT / "assets" / "scenes" / scene
    out_state_name = sec.out_state_name or f"{scene}_scene_state_auto_bg.json"

    if not scene_state.exists():
        sys.exit(f"missing stage-14 scene state: {scene_state}")
    out_scene_dir.mkdir(parents=True, exist_ok=True)
    (out_scene_dir / "objects").mkdir(exist_ok=True)

    logger.info("Loading stage-14 scene state: %s", scene_state)
    state = json.loads(scene_state.read_text())
    obj_init = state.setdefault("objects_info", {}).setdefault("init_info", {})
    obj_reg = state.setdefault("state", {}).setdefault("registry", {}).setdefault("object_registry", {})

    # 1. Materialise each dataset object's asset tree, rewrite to USDObject with explicit usd_path
    rewritten = []
    for name, entry in list(obj_init.items()):
        args_d = entry.get("args", {})
        if entry.get("class_name") != "DatasetObject":
            logger.info("Keeping non-DatasetObject entry: %s (%s)", name, entry.get("class_name"))
            continue
        category = args_d["category"]
        model = args_d["model"]
        dataset_name = args_d.get("dataset_name", "real2sim-assets")
        src_dir = _resolve_dataset_obj_dir(dataset_name, category, model)
        if not src_dir.exists():
            sys.exit(f"source object dir missing for {name}: {src_dir}")
        dst_dir = out_scene_dir / "objects" / category / model

        if sec.symlink_assets:
            if dst_dir.exists() or dst_dir.is_symlink():
                if dst_dir.is_symlink():
                    dst_dir.unlink()
                else:
                    shutil.rmtree(dst_dir)
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            dst_dir.symlink_to(src_dir.resolve())
            logger.info("symlinked %s -> %s", dst_dir, src_dir)
        else:
            _copy_asset_tree(src_dir, dst_dir)
            logger.info("copied %s -> %s", src_dir, dst_dir)

        usd_path = dst_dir / "usd" / f"{model}.usd"
        if not usd_path.exists():
            sys.exit(f"expected USD missing after copy: {usd_path}")

        # Preserve hash if the dataset provided one (it's the same file).
        # Otherwise compute fresh.
        hash_hex = args_d.get("expected_file_hash") or _md5(usd_path)
        obj_init[name] = _build_usdobject_entry(name, usd_path, category, hash_hex)
        rewritten.append(name)
    logger.info("Rewrote %d DatasetObject entries -> USDObject: %s", len(rewritten), rewritten)

    # 2. BG splat: install a prebuilt USDZ, or convert the local PLY.
    if not sec.skip_bg_splat:
        gs_bg_dir = out_scene_dir / "objects" / "gs_background"
        gs_bg_dir.mkdir(parents=True, exist_ok=True)
        gs_usdz = gs_bg_dir / "gs_auto.usdz"
        if prebuilt_usdz is not None:
            try:
                materialize_bg_usdz(gs_usdz, prebuilt_usdz=prebuilt_usdz)
            except FileNotFoundError as e:
                sys.exit(str(e))
            logger.info("Installed prebuilt BG USDZ %s -> %s", prebuilt_usdz, gs_usdz)
        else:
            if not bg_ply.exists():
                sys.exit(f"missing BG splat PLY: {bg_ply}")
            logger.info("Converting BG splat %s -> %s", bg_ply, gs_usdz)
            materialize_bg_usdz(gs_usdz, source_ply=bg_ply)
        if not gs_usdz.exists():
            sys.exit(f"BG splat USDZ failed to write: {gs_usdz}")
        if explicit_pose is not None and not explicit_pose.is_file():
            sys.exit(f"missing explicit BG pose sidecar: {explicit_pose}")
        try:
            bg_pose = (
                load_bg_pose_sidecar(explicit_pose)
                if explicit_pose is not None
                else _load_bg_pose_sidecar(bg_ply)
            )
        except (json.JSONDecodeError, ValueError) as e:
            sys.exit(f"invalid BG pose sidecar: {e}")
        if bg_pose is None:
            logger.info("No BG pose sidecar; assuming the splat is pre-baked into OG world.")
            obj_init["gs_background"] = _build_gs_background_entry(gs_usdz, scale=1.0)
            obj_reg["gs_background"] = _build_gs_background_state()
        else:
            logger.info(
                "Applying pose sidecar %s: pos=%s ori_xyzw=%s scale=%.6f",
                bg_pose["sidecar_path"], bg_pose["pos"], bg_pose["ori_xyzw"], bg_pose["scale"],
            )
            obj_init["gs_background"] = _build_gs_background_entry(gs_usdz, scale=bg_pose["scale"])
            obj_reg["gs_background"] = _build_gs_background_state(
                pos=bg_pose["pos"], ori_xyzw=bg_pose["ori_xyzw"],
            )
        logger.info("Installed BG splat USDZ at %s (md5 %s)",
                    gs_usdz, obj_init["gs_background"]["args"]["expected_file_hash"])
    else:
        logger.info("--skip-bg-splat: gs_background not injected.")

    # 3. Defaults for top-level keys missing in the stage-14 state.
    if REF_SCENE_STATE.exists():
        ref = json.loads(REF_SCENE_STATE.read_text())
        for k in ("viewer_camera_state", "lighting_state", "ground_plane_info"):
            if k not in state:
                state[k] = copy.deepcopy(ref.get(k, {}))
                logger.info("Injected %s from reference nv_desk scene", k)
    else:
        logger.warning("No reference scene at %s; viewer/lighting/ground_plane left as-is", REF_SCENE_STATE)

    # Write
    out_path = out_scene_dir / out_state_name
    out_path.write_text(json.dumps(state, indent=2))
    logger.info("Wrote %s (%.1f KB, %d init entries)",
                out_path, out_path.stat().st_size / 1024.0, len(obj_init))


if __name__ == "__main__":
    main()
