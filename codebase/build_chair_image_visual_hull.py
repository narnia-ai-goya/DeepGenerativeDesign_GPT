#!/usr/bin/env python3
"""Build a BC-clipped 64^3 multi-view occupancy proxy for chair dense guidance."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from scipy.ndimage import binary_closing, label
from skimage.measure import marching_cubes

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim  # noqa: E402


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
OUT = BASE / "visual_hull_dense"
SOURCE = BASE / "open_arm/camera_projection_targets.npz"
VIEWS = (("front", 15, 0), ("right", 15, 90), ("top", 85, 0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "data_real/chair/voxel.npz")
    t = np.load(SOURCE)
    envelope = d["bracket"].astype(bool)
    bc = d["bc"].astype(bool)
    origin = d["origin"].astype(float)
    pitch = d["pitch_xyz"].astype(float)
    env_mesh = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env_mesh.bounds.mean(axis=0)
    radius = np.linalg.norm(env_mesh.extents) * 1.5
    half_extent = env_mesh.extents.max() / 2 * 1.15
    ijk = np.indices((64, 64, 64)).reshape(3, -1).T
    world = origin + (ijk + .5) * pitch
    hull = np.ones(len(world), dtype=bool)
    per_view = {}
    for view, elev, azim in VIEWS:
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        forward = (center - eye) / np.linalg.norm(center - eye)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        true_up /= np.linalg.norm(true_up)
        relative = world - center
        x = np.floor((relative @ right / (2 * half_extent) + .5) * 64).astype(int)
        y = np.floor((.5 - relative @ true_up / (2 * half_extent)) * 64).astype(int)
        valid = (x >= 0) & (x < 64) & (y >= 0) & (y < 64)
        selected = np.zeros(len(world), dtype=bool)
        mask = t[f"target_{view}"] > .35
        selected[valid] = mask[y[valid], x[valid]]
        hull &= selected
        per_view[view] = int(selected.reshape((64, 64, 64)).__and__(envelope).sum())
    hull = hull.reshape((64, 64, 64)) & envelope
    # Fill single-voxel gaps from sampled, antialiased silhouettes; do not change BC.
    hull = binary_closing(hull, iterations=1) & envelope
    hull |= bc
    labels, n = label(hull)
    components = np.bincount(labels[labels > 0].ravel()).tolist()
    np.savez_compressed(OUT / "visual_hull_prototype.npz",
                        prototypes=hull[None].astype(np.float32))
    verts, faces, _, _ = marching_cubes(np.pad(hull, 1), level=.5)
    verts = origin + (verts - .5) * pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(OUT / "visual_hull_proxy.obj")
    metrics = {"source": str(SOURCE), "envelope_voxels": int(envelope.sum()),
               "hull_voxels": int(hull.sum()), "bc_voxels": int(bc.sum()),
               "components": n, "component_sizes": sorted(components, reverse=True),
               "per_view_envelope_voxels": per_view,
               "prototype": str(OUT / "visual_hull_prototype.npz"),
               "proxy_mesh": str(OUT / "visual_hull_proxy.obj")}
    (OUT / "hull_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
