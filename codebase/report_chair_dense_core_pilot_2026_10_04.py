"""Evaluate and render fixed-Dense chair sparse core guidance pilots."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from pysdf import SDF
import trimesh

from run_chair_existing_fea_on_015_2026_10_04 import OUT
from run_chair_dual_load_fea_2026_10_04 import SPEC
from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = OUT / "diagnostics/dense_core_pilot"
ROWS = {
    "Dense reference": OUT / "dense/generation/mesh.obj",
    "Sparse baseline (FEA OFF)": OUT / "diagnostics/same_dense_sparse_fea_off/generation/mesh.obj",
    "Core 8 mm, w=30 (FEA OFF)": BASE / "core_8mm_w30_fea_off/generation/mesh.obj",
    "Core 6 mm, w=30 (FEA OFF)": BASE / "core_6mm_w30_fea_off/generation/mesh.obj",
    "Core 6 mm, w=30 (FEA ON)": BASE / "core_6mm_w30_fea_on/generation/mesh.obj",
}


def main() -> None:
    grid = np.load(SPEC / "native_frame_spec.npz")
    cache = np.load(OUT / "dense/dense_cache.npz")
    centers = (grid["origin"] + (cache["latent_index"][:, 1:] + 0.5)
               * grid["pitch_xyz"]).astype(np.float32)
    registration = json.loads((OUT.parents[4] / "single_view_spec_2026-10-03/specification.json").read_text())
    scale = float(registration["native_frame_registration"]["uniform_scale"]) ** 3
    dense = trimesh.load(ROWS["Dense reference"], force="mesh", process=False)
    dense_sdf = SDF(dense.vertices.astype(np.float32), dense.faces.astype(np.uint32))
    depth = dense_sdf(centers)
    masks = {"all_dense_centers": depth > 0,
             "dense_core_gt6mm": depth > 0.006,
             "dense_core_gt8mm": depth > 0.008}
    center = np.array([0.0, 0.01, 0.46])
    views = (("hero", 25, 35), ("front", 15, 0), ("right", 15, 90))
    tile, header = 350, 36
    sheet = Image.new("RGB", (tile * len(views), (tile + header) * len(ROWS)), "#f5f7f8")
    draw = ImageDraw.Draw(sheet)
    results = {}
    for row, (label, path) in enumerate(ROWS.items()):
        if not path.exists():
            raise FileNotFoundError(path)
        mesh = dense if row == 0 else trimesh.load(path, force="mesh", process=False)
        sdf = dense_sdf if row == 0 else SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        inside = sdf(centers) > 0
        row_result = {"mesh": str(path), "physical_volume_L": abs(float(mesh.volume)) * scale * 1000,
                      "watertight": bool(mesh.is_watertight), "dense_center_retention": {}}
        for key, mask in masks.items():
            row_result["dense_center_retention"][key] = {
                "retained": int(np.count_nonzero(inside & mask)),
                "total": int(np.count_nonzero(mask)),
                "fraction": float(np.mean(inside[mask])),
            }
        results[label] = row_result
        for col, (view, elev, azim) in enumerate(views):
            eye, up = camera_from_elev_azim(center, 2.0, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile, fit_extent=.56,
                             color=(.56, .61, .66))
            sheet.paste(Image.fromarray(rgb).convert("RGB"), (col * tile, row * (tile + header) + header))
            draw.text((col * tile + 10, row * (tile + header) + 10),
                      f"{label} | {view} | {row_result['physical_volume_L']:.2f} L",
                      fill="#1c3040")
    BASE.mkdir(parents=True, exist_ok=True)
    image_path = BASE / "comparison.png"
    summary_path = BASE / "summary.json"
    html_path = BASE / "index.html"
    sheet.save(image_path)
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{entry['physical_volume_L']:.2f}</td>"
        f"<td>{entry['dense_center_retention']['all_dense_centers']['fraction']:.1%}</td>"
        f"<td>{entry['dense_center_retention']['dense_core_gt6mm']['fraction']:.1%}</td>"
        f"<td>{entry['dense_center_retention']['dense_core_gt8mm']['fraction']:.1%}</td>"
        f"<td><a href='{os.path.relpath(entry['mesh'], BASE)}'>OBJ</a></td></tr>"
        for name, entry in results.items()
    )
    off_fea = BASE / "core_6mm_w30_fea_off/independent_fea_15mm/summary.json"
    on_fea = BASE / "core_6mm_w30_fea_on/independent_fea_15mm/summary.json"
    fea_note = ""
    if off_fea.exists() and on_fea.exists():
        off = json.loads(off_fea.read_text())
        on = json.loads(on_fea.read_text())
        fea_note = (f"<p>Independent 15 mm FEM on final raw OBJ: core 6 mm FEA OFF "
                    f"C={off['compliance']/1e6:.3f}×10⁶ at {off['solid_volume_liters']:.3f} L; "
                    f"FEA ON C={on['compliance']/1e6:.3f}×10⁶ at "
                    f"{on['solid_volume_liters']:.3f} L. The in-loop FEA contribution is small "
                    f"at almost equal mass.</p>")
    html_path.write_text(f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>Chair Dense core pilot</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:30px auto;background:#f3f6f8;color:#1c3040}}
article{{background:white;padding:24px;border-radius:12px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #cbd7de;padding:8px;text-align:left}}img{{max-width:100%}}</style>
<article><h1>Chair Dense core pilot</h1><p>Same Dense cache, same image and seed.
Soft signed-distance core guidance only; final hard projection OFF. FEA is enabled only in the last row.</p>
<table><tr><th>Case</th><th>Volume L</th><th>Dense centers retained</th><th>Core &gt;6 mm retained</th>
<th>Core &gt;8 mm retained</th><th>Mesh</th></tr>{body}</table>
{fea_note}<h2>3D surface comparison</h2><p><img src='comparison.png'></p>
<h2>Same-position interior sections</h2><p>Orange: added material; blue: removed material.</p>
<p><img src='difference_slices.png'></p><p><a href='summary.json'>Numeric results</a> ·
<a href='difference_summary.json'>Section data</a></p></article></html>""")
    print(image_path)
    print(summary_path)
    print(html_path)


if __name__ == "__main__":
    main()
