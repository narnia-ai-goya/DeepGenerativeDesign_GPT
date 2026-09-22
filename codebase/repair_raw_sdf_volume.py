#!/usr/bin/env python3
"""Extract a volume-calibrated mesh from a saved final-refiner SDF.

The descriptor reference contains fixed CAD sample points inside the design
domain.  Interpolating the SDF at those points lets us choose an iso-level that
targets the same material-volume definition used by the RA-QD archive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from skimage.measure import marching_cubes


def sample_reference(sdf, xyz, origin, pitch):
    """Trilinearly sample an (x,y,z)-indexed SDF at world-space points."""
    coords = ((xyz - origin[None, :]) / pitch[None, :] - .5).T
    return map_coordinates(sdf, coords, order=1, mode="constant", cval=np.inf)


def calibrated_level(values, target):
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No descriptor-reference point lies inside the saved SDF grid")
    if not 0 < target < 1:
        raise ValueError("target must lie strictly between zero and one")
    # Inside is sdf <= level in the captured Direct3D-S2 refiner convention.
    return float(np.quantile(finite, target, method="linear"))


def extract_mesh(sdf, level, origin, pitch, keep_largest=True):
    vertices, faces, _, _ = marching_cubes(sdf, level=level, spacing=tuple(pitch))
    vertices += origin[None, :] + .5 * pitch[None, :]
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    if keep_largest:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            mesh = max(components, key=lambda part: abs(part.volume))
            mesh.remove_unreferenced_vertices()
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-sdf", type=Path, required=True)
    ap.add_argument("--descriptor-reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="raw repaired mesh; run the normal post stage afterward")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--target-final", type=float, default=.5)
    ap.add_argument("--transfer-slope", type=float, default=.9929862555158612)
    ap.add_argument("--transfer-intercept", type=float, default=.016006001427529146)
    args = ap.parse_args()

    raw_path = args.raw_sdf.resolve()
    ref_path = args.descriptor_reference.resolve()
    out = args.out.resolve()
    report_path = (args.report.resolve() if args.report else out.with_suffix(".repair.json"))
    with np.load(raw_path) as raw:
        sdf = np.asarray(raw["sdf"], dtype=np.float32)
        origin = np.asarray(raw["env_origin"], dtype=np.float64)
        pitch = np.asarray(raw["env_pitch"], dtype=np.float64)
        original_level = float(raw["mc_threshold"])
    with np.load(ref_path) as reference:
        xyz = np.asarray(reference["xyz"], dtype=np.float64)

    target_raw = (args.target_final - args.transfer_intercept) / args.transfer_slope
    values = sample_reference(sdf, xyz, origin, pitch)
    level = calibrated_level(values, target_raw)
    predicted_raw = float(np.mean(values <= level))
    unfiltered = extract_mesh(sdf, level, origin, pitch, keep_largest=False)
    raw_components = unfiltered.split(only_watertight=False)
    raw_volume = sum(abs(part.volume) for part in raw_components)
    mesh = (max(raw_components, key=lambda part: abs(part.volume)).copy()
            if len(raw_components) > 1 else unfiltered)
    mesh.remove_unreferenced_vertices()
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out)
    components = mesh.split(only_watertight=False)
    report = {
        "raw_sdf": str(raw_path),
        "descriptor_reference": str(ref_path),
        "output_mesh": str(out),
        "target_final_volume_fraction": args.target_final,
        "target_raw_volume_fraction": target_raw,
        "predicted_raw_volume_fraction": predicted_raw,
        "original_iso_level": original_level,
        "repaired_iso_level": level,
        "iso_level_delta": level - original_level,
        "reference_points": len(xyz),
        "finite_reference_samples": int(np.isfinite(values).sum()),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight_before_post": bool(mesh.is_watertight),
        "raw_components": len(raw_components),
        "components_before_post": len(components),
        "discarded_component_volume_fraction": float(
            1 - abs(mesh.volume) / raw_volume) if raw_volume else 0.0,
        "mesh_volume_mm3_before_post": float(abs(mesh.volume) * 1e9),
        "transfer_model": {
            "final_equals_slope_times_raw_plus_intercept": [
                args.transfer_slope, args.transfer_intercept
            ],
            "source": "/home/goya/SDL/3d_qd/experiments/bracket/raqd_online_2026-09-13/methodology_diagnostics.json"
        },
        "next_step": "Run post_hybrid_union_clip.py and surface_remesh_pre.py, then remeasure descriptors and FEA."
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(out)
    print(report_path)


if __name__ == "__main__":
    main()
