#!/usr/bin/env python3
"""Make a chair design-space variant with a functional underseat clearance."""
from __future__ import annotations

import json
import argparse

import numpy as np
import trimesh
from scipy.ndimage import label
from skimage.measure import marching_cubes

from make_chair_domain import ROOT

BASE = ROOT / "experiments/chair/sofa_style_2026-09-28/visual_hull_dense"
DEFAULT_OUT = BASE / "underseat_clearance"


def main(variant: str = "broad") -> None:
    out = DEFAULT_OUT if variant == "broad" else BASE / "rear_midspan_clearance"
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(ROOT / "data_real/chair/voxel.npz")
    arrays = {key: data[key].copy() for key in data.files}
    origin = arrays["origin"]
    pitch = arrays["pitch_xyz"]
    coord = [origin[i] + (np.arange(64) + .5) * pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*coord, indexing="ij")
    # Functional free space below the seat, between the four fixed foot columns.
    if variant == "broad":
        limits = ((-.15, .15), (-.235, .235), (.08, .425))
    else:
        limits = ((-.10, .10), (.12, .235), (.16, .425))
    clearance = ((x > limits[0][0]) & (x < limits[0][1])
                 & (y > limits[1][0]) & (y < limits[1][1])
                 & (z > limits[2][0]) & (z < limits[2][1]))
    old_envelope = arrays["bracket"].astype(bool)
    bc = arrays["bc"].astype(bool)
    if np.any(clearance & bc):
        raise ValueError("clearance intersects mandatory BC")
    new_envelope = (old_envelope & ~clearance) | bc
    arrays["bracket"] = new_envelope
    arrays["design"] = new_envelope & ~bc
    if np.any(bc & ~new_envelope):
        raise ValueError("BC outside new envelope")
    np.savez_compressed(out / "voxel.npz", **arrays)
    source = np.load(BASE / "visual_hull_prototype.npz")["prototypes"][0] > .5
    prototype = (source & new_envelope) | bc
    np.savez_compressed(out / "prototype.npz",
                        prototypes=prototype[None].astype(np.float32))
    verts, faces, _, _ = marching_cubes(np.pad(new_envelope, 1), .5)
    verts = origin + (verts - .5) * pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Marching-cubes normals point into the occupied voxels here. Flip the
    # entire shell so the exterior is outward and the cavity is inward.
    mesh.invert()
    mesh.export(out / "envelope.stl")
    labels, n_components = label(new_envelope)
    metrics = {
        "clearance_x_m": limits[0], "clearance_y_m": limits[1],
        "clearance_z_m": limits[2],
        "original_envelope_voxels": int(old_envelope.sum()),
        "new_envelope_voxels": int(new_envelope.sum()),
        "removed_voxels": int((old_envelope & ~new_envelope).sum()),
        "bc_overlap_voxels": int((clearance & bc).sum()),
        "new_envelope_components": int(n_components),
        "prototype_voxels": int(prototype.sum()),
        "paths": {"voxel": str(out / "voxel.npz"),
                  "prototype": str(out / "prototype.npz"),
                  "envelope": str(out / "envelope.stl")}}
    (out / "clearance_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("broad", "rear_midspan"), default="broad")
    main(parser.parse_args().variant)
