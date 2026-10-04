#!/usr/bin/env python3
"""Repair sparse-stage pits on the *outside wall* near bracket fix bosses.

This is an isolated, opt-in post experiment. It leaves the source OBJ and the
standard generator/post recipe unchanged. Four outer-side strips are filled
from the original allowed design domain, then the seam is locally faired.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from post_hybrid_union_clip import _clean, _mf
from repair_fix_interface_local import repair

ROOT = Path(__file__).resolve().parents[1]


def outer_wall_patch(source: Path, envelope: Path, fix_path: Path, load_path: Path,
                     strip_depth_mm: float, strip_width_mm: float,
                     strip_height_mm: float, offset_mm: float) -> tuple[trimesh.Trimesh, list[dict]]:
    body = _clean(trimesh.load_mesh(source, force="mesh", process=False))
    env = _clean(trimesh.load_mesh(envelope, force="mesh", process=False))
    fix = _clean(trimesh.load_mesh(fix_path, force="mesh", process=False))
    load = _clean(trimesh.load_mesh(load_path, force="mesh", process=False))
    allowed = _mf(env) + _mf(fix) + _mf(load)
    result = _mf(body)
    components = [c for c in fix.split(only_watertight=False) if len(c.faces) >= 100]
    if len(components) != 4:
        raise ValueError(f"Expected four bracket fix bosses, found {len(components)}")
    patches = []
    for component in components:
        center = component.bounds.mean(axis=0)
        # In this bracket frame the two near bosses are at x≈0 and the two far
        # bosses at x≈38/52 mm. Repair only their outward-facing side wall.
        side = -1 if center[0] < .010 else 1
        tool = trimesh.creation.box(extents=np.array([
            strip_depth_mm, strip_width_mm, strip_height_mm]) / 1000.0)
        tool.apply_translation([center[0] + side * offset_mm / 1000.0,
                                center[1], strip_height_mm / 2000.0])
        patch = allowed ^ _mf(tool)
        result = result + patch
        patches.append({"boss_center_mm": (center[:2] * 1000).tolist(),
                        "outward_x_sign": side, "allowed_patch_mm3": patch.volume() * 1e9})
    result = result ^ allowed
    parts = sorted(result.decompose(), key=lambda p: abs(p.volume()), reverse=True)
    if not parts:
        raise RuntimeError("Local wall union produced no manifold")
    raw = parts[0].to_mesh()
    mesh = trimesh.Trimesh(raw.vert_properties.copy(), raw.tri_verts.copy(), process=True)
    mesh.merge_vertices()
    shells = mesh.split(only_watertight=False)
    if len(shells) > 1:
        mesh = max(shells, key=lambda c: abs(c.volume))
        mesh.remove_unreferenced_vertices()
    if mesh.volume < 0:
        mesh.invert()
    return mesh, patches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strip-depth-mm", type=float, default=12)
    ap.add_argument("--strip-width-mm", type=float, default=44)
    ap.add_argument("--strip-height-mm", type=float, default=23)
    ap.add_argument("--outward-offset-mm", type=float, default=10)
    ap.add_argument("--smooth-iterations", type=int, default=25)
    ap.add_argument("--smooth-strength", type=float, default=.35)
    ap.add_argument("--smooth-inner-mm", type=float, default=.5)
    ap.add_argument("--smooth-outer-mm", type=float, default=32)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    post = cfg["stages"]["post"]
    envelope, fix, load = [Path(post[k]).resolve() for k in ("bracket_stl", "fix", "load")]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hybrid, patches = outer_wall_patch(args.source.resolve(), envelope, fix, load,
                                       args.strip_depth_mm, args.strip_width_mm,
                                       args.strip_height_mm, args.outward_offset_mm)
    hybrid_path = out / "hybrid.obj"
    hybrid.export(hybrid_path)
    final_path = out / "final.obj"
    subprocess.run([sys.executable, str(ROOT / "codebase/code/surface_remesh_pre.py"),
                    "--config", str(args.config.resolve()), "--in", str(hybrid_path),
                    "--out", str(final_path)], cwd=ROOT, check=True)
    final = trimesh.load_mesh(final_path, force="mesh", process=False)
    fixture = trimesh.load_mesh(fix, force="mesh", process=True)
    fixture.merge_vertices()
    smooth, smooth_report = repair(final, fixture,
                                   iterations=args.smooth_iterations,
                                   strength=args.smooth_strength,
                                   inner_mm=args.smooth_inner_mm,
                                   outer_mm=args.smooth_outer_mm)
    result_path = out / "final_smoothed.obj"
    smooth.export(result_path)
    report = {"source": str(args.source.resolve()), "config": str(args.config.resolve()),
              "hybrid": str(hybrid_path), "post_final": str(final_path),
              "repaired_final": str(result_path), "patches": patches,
              "strip_mm": {"depth": args.strip_depth_mm, "width": args.strip_width_mm,
                           "height": args.strip_height_mm, "offset": args.outward_offset_mm},
              "smoothing": smooth_report}
    (out / "repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(result_path)


if __name__ == "__main__":
    main()
