"""Quantify and visualize fixed-Dense, fixed-core sparse loss weight search."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pysdf import SDF
import trimesh

from make_chair_domain import ROOT
from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = OUT / "diagnostics/dense_core_pilot/weight_search"
PILOT = OUT / "diagnostics/dense_core_pilot"
CASES = {
    0: OUT / "diagnostics/same_dense_sparse_fea_off/generation/mesh.obj",
    1: PILOT / "core_6mm_fea_off/generation/mesh.obj",
    5: BASE / "w005/generation/mesh.obj",
    10: BASE / "w010/generation/mesh.obj",
    15: BASE / "w015/generation/mesh.obj",
    30: PILOT / "core_6mm_w30_fea_off/generation/mesh.obj",
    60: BASE / "w060/generation/mesh.obj",
    120: BASE / "w120/generation/mesh.obj",
}


def main() -> None:
    grid = np.load(SPEC / "native_frame_spec.npz")
    cache = np.load(OUT / "dense/dense_cache.npz")
    centers = (grid["origin"] + (cache["latent_index"][:, 1:] + .5)
               * grid["pitch_xyz"]).astype(np.float32)
    dense = trimesh.load(OUT / "dense/generation/mesh.obj", force="mesh", process=False)
    dense_sdf = SDF(dense.vertices.astype(np.float32), dense.faces.astype(np.uint32))
    depth = dense_sdf(centers)
    masks = {"all": depth > 0, "core6": depth > .006, "core8": depth > .008}
    registration = json.loads((OUT.parents[4] / "single_view_spec_2026-10-03/specification.json").read_text())
    physical_scale = float(registration["native_frame_registration"]["uniform_scale"]) ** 3
    center = np.array([0.0, .01, .46])
    views = (("hero", 25, 35), ("right", 15, 90))
    tile, header = 370, 35
    sheet = Image.new("RGB", (tile * len(views), (tile + header) * len(CASES)), "#f5f7f8")
    draw = ImageDraw.Draw(sheet)
    records = []
    for row, (weight, path) in enumerate(CASES.items()):
        mesh = trimesh.load(path, force="mesh", process=False)
        sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        inside = sdf(centers) > 0
        run_dir = BASE / f"w{weight:03d}"
        log = run_dir / "generation.log"
        trace = log.read_text(errors="replace") if log.exists() else ""
        gradients = [float(v) for v in re.findall(r"\[sparse grad diagnostic\].*?grad_norm=([0-9.eE+-]+)", trace)]
        updates = [float(v) for v in re.findall(r"\[sparse grad diagnostic\].*?param_delta_norm=([0-9.eE+-]+)", trace)]
        record = {
            "effective_weight": weight, "mesh": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "physical_volume_L": abs(float(mesh.volume)) * physical_scale * 1000,
            "surface_area_m2": float(mesh.area) * physical_scale ** (2/3),
            "watertight": bool(mesh.is_watertight),
            "retention": {name: float(np.mean(inside[mask])) for name, mask in masks.items()},
            "grad_norms": gradients, "param_update_norms": updates,
        }
        if weight in (0, 1):
            fea_path = BASE / "w000/independent_fea_15mm/summary.json"
        elif weight == 30:
            fea_path = PILOT / "core_6mm_w30_fea_off/independent_fea_15mm/summary.json"
        else:
            fea_path = run_dir / "independent_fea_15mm/summary.json"
        if fea_path.exists():
            fea = json.loads(fea_path.read_text())
            record["independent_fea"] = {"compliance": fea["compliance"],
                                         "solid_volume_liters": fea["solid_volume_liters"],
                                         "summary": str(fea_path)}
        records.append(record)
        for col, (name, elev, azim) in enumerate(views):
            eye, up = camera_from_elev_azim(center, 2.0, elev, azim)
            image = render_lit(mesh, eye, center, up, size=tile, fit_extent=.56,
                               color=(.56, .61, .66))
            sheet.paste(Image.fromarray(image).convert("RGB"), (col * tile, row * (tile + header) + header))
            label = "baseline" if weight == 0 else f"effective w={weight}"
            draw.text((col * tile + 10, row * (tile + header) + 9),
                      f"{label} | {name} | {record['physical_volume_L']:.2f} L", fill="#1c3040")
    BASE.mkdir(parents=True, exist_ok=True)
    figure = BASE / "comparison.png"
    sheet.save(figure)
    (BASE / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    weights = [r["effective_weight"] for r in records]
    axes[0].plot(weights, [r["physical_volume_L"] for r in records], "o-", color="#325f85")
    axes[0].set_ylabel("Final OBJ volume (L)")
    axes[1].plot(weights, [100*r["retention"]["core6"] for r in records], "o-", color="#16816d")
    axes[1].set_ylabel("Dense core >6 mm retained (%)")
    evaluated = [r for r in records if "independent_fea" in r]
    if evaluated:
        axes[2].plot([r["effective_weight"] for r in evaluated],
                     [r["independent_fea"]["compliance"]/1e6 for r in evaluated],
                     "o-", color="#a65a42")
    axes[2].set_ylabel("Independent compliance (×10⁶)")
    for ax in axes:
        ax.set_xlabel("Effective core weight")
        ax.set_xticks(weights)
        ax.grid(alpha=.22)
    fig.savefig(BASE / "trend.png", dpi=180, facecolor="white")
    plt.close(fig)
    table = "".join(
        f"<tr><td>{r['effective_weight']}</td><td>{r['physical_volume_L']:.3f}</td>"
        f"<td>{r['retention']['all']:.1%}</td><td>{r['retention']['core6']:.1%}</td>"
        f"<td>{r['retention']['core8']:.1%}</td>"
        f"<td>{r['independent_fea']['compliance']/1e6:.3f}</td>"
        f"<td>{r['grad_norms'][0]:.3g}</td><td>{r['param_update_norms'][0]:.3g}</td>"
        f"<td><a href='{html.escape(os.path.relpath(r['mesh'], BASE))}'>OBJ</a></td></tr>"
        if r['grad_norms'] and r['param_update_norms'] else
        f"<tr><td>{r['effective_weight']}</td><td>{r['physical_volume_L']:.3f}</td>"
        f"<td>{r['retention']['all']:.1%}</td><td>{r['retention']['core6']:.1%}</td>"
        f"<td>{r['retention']['core8']:.1%}</td>"
        f"<td>{r['independent_fea']['compliance']/1e6:.3f}</td><td>—</td><td>—</td>"
        f"<td><a href='{html.escape(os.path.relpath(r['mesh'], BASE))}'>OBJ</a></td></tr>"
        for r in records if "independent_fea" in r
    )
    on_dir = BASE / "w060_fea_on"
    on_obj = on_dir / "generation/mesh.obj"
    on_fea_path = on_dir / "independent_fea_15mm/summary.json"
    on_section = ""
    if on_obj.exists() and on_fea_path.exists():
        off = next(r for r in records if r["effective_weight"] == 60)
        on_fea = json.loads(on_fea_path.read_text())
        off_fea = off["independent_fea"]
        pair = Image.new("RGB", (tile * 2, tile + header), "#f5f7f8")
        pair_draw = ImageDraw.Draw(pair)
        eye, up = camera_from_elev_azim(center, 2.0, 25, 35)
        for col, (label, path) in enumerate((("FEA OFF", Path(off["mesh"])), ("FEA ON", on_obj))):
            mesh = trimesh.load(path, force="mesh", process=False)
            image = render_lit(mesh, eye, center, up, size=tile, fit_extent=.56,
                               color=(.56, .61, .66))
            pair.paste(Image.fromarray(image).convert("RGB"), (col * tile, header))
            pair_draw.text((col * tile + 10, 9), f"w=60 | {label}", fill="#1c3040")
        pair.save(BASE / "w060_fea_pair.png")
        on_section = (f"<h2>Selected w=60: in-loop FEA check</h2>"
                      f"<p>Both cases use the same Dense cache, seed and 6 mm core guidance. "
                      f"The final meshes were independently evaluated on the same 15 mm FEM grid "
                      f"with simultaneous 800 N −Z and 200 N +Y loads.</p>"
                      f"<table><tr><th>Case</th><th>FEM volume L</th><th>C ×10⁶</th><th>OBJ</th></tr>"
                      f"<tr><td>FEA OFF</td><td>{off_fea['solid_volume_liters']:.3f}</td>"
                      f"<td>{off_fea['compliance']/1e6:.3f}</td>"
                      f"<td><a href='{html.escape(os.path.relpath(off['mesh'], BASE))}'>OBJ</a></td></tr>"
                      f"<tr><td>FEA ON</td><td>{on_fea['solid_volume_liters']:.3f}</td>"
                      f"<td>{on_fea['compliance']/1e6:.3f}</td>"
                      f"<td><a href='w060_fea_on/generation/mesh.obj'>OBJ</a></td></tr></table>"
                      f"<p>FEA ON changes C by "
                      f"{(on_fea['compliance']/off_fea['compliance']-1)*100:+.2f}% "
                      f"relative to FEA OFF; this does not show a meaningful improvement.</p>"
                      f"<img src='w060_fea_pair.png'>")
    (BASE / "index.html").write_text(f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<title>Chair Dense-core weight search</title><style>body{{font:16px system-ui;max-width:1100px;
margin:30px auto;background:#f3f6f8;color:#1c3040}}article{{background:white;padding:24px;border-radius:12px}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd7de;padding:8px;text-align:left}}
img{{max-width:100%}}</style><article><h1>Dense-core guidance weight search</h1>
<p>Same Dense cache, same image and seed, 6 mm core, FEA OFF, 30 sparse steps.
Effective weight = sp_guide_w × sp_dense_core_w. Other active guidance terms are zero.</p>
<table><tr><th>Weight</th><th>Volume L</th><th>Dense cells</th><th>Core &gt;6 mm</th>
<th>Core &gt;8 mm</th><th>C ×10⁶</th><th>First grad norm</th><th>First update norm</th><th>OBJ</th></tr>
{table}</table><h2>Weight response</h2><img src='trend.png'>
<h2>Aligned mesh renders</h2><img src='comparison.png'>
{on_section}
<p><a href='summary.json'>Numeric data</a></p></article></html>""")
    print(figure)
    print(BASE / "summary.json")
    print(BASE / "index.html")


if __name__ == "__main__":
    main()
