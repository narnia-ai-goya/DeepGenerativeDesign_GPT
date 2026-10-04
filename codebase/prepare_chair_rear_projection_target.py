#!/usr/bin/env python3
"""Add a provisional mirrored-front rear silhouette to chair dense guidance."""
from __future__ import annotations

import json

import numpy as np
import trimesh
from PIL import Image

from make_chair_domain import ROOT
from prepare_chair_camera_projection_targets import camera_grid, camera_from_elev_azim


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
SOURCE = BASE / "open_arm/camera_projection_targets.npz"
OUT = BASE / "visual_hull_dense/rear_projection"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE)
    arrays = {key: source[key].copy() for key in source.files}
    arrays["target_rear"] = arrays["target_front"][:, ::-1].copy()
    arrays["weight_rear"] = arrays["weight_front"][:, ::-1].copy()
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    voxel = np.load(ROOT / "data_real/chair/voxel.npz")
    center = env.bounds.mean(axis=0)
    radius = np.linalg.norm(env.extents) * 1.5
    eye, up = camera_from_elev_azim(center, radius, 15, 180)
    arrays["camera_grid_rear"] = camera_grid(
        center, eye, up, env.extents.max() / 2 * 1.15,
        voxel["origin"], voxel["pitch_xyz"], np.linalg.norm(env.extents))
    path = OUT / "camera_projection_targets_with_rear.npz"
    np.savez_compressed(path, **arrays)
    Image.fromarray(((1 - arrays["target_rear"]) * 255).astype(np.uint8)).resize(
        (512, 512), Image.Resampling.NEAREST).save(OUT / "rear_target_provisional.png")
    (OUT / "target_metadata.json").write_text(json.dumps({
        "source": str(SOURCE), "output": str(path),
        "rear_target_source": "horizontally mirrored registered front silhouette",
        "rear_camera": {"elev_deg": 15, "azim_deg": 180},
        "assumption": "approximate bilateral chair symmetry; not an observed rear image"
    }, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
