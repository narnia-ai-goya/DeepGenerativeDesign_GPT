"""SNAP3D-inspired interface pilot on the selected Dense10/Sparse100 chair.

The input chair is monolithic.  We infer semantic frame-to-seat contacts,
parameterize symmetric structural ligaments at those contacts, and rank them
by independent two-load FEM.  This is not a reproduction of SNAP3D's
detachable peg/socket assembly or rigid-body solver.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pyvista as pv
import trimesh

from make_chair_domain import ROOT
from run_chair_existing_fea_on_015_2026_10_04 import OUT as PRIOR, SPEC


OUT = ROOT / "experiments/chair/sofa_style_2026-09-28/snap3d_selected_10_100_2026-10-04"
PARENT = PRIOR / "diagnostics/fea_log_search_2026-10-04/dense_f00010_sparse_f00100"
BASE_OBJ = PARENT / "independent_fea_15mm/aligned_full.obj"
BASE_FEA = PARENT / "independent_fea_15mm/summary.json"
PYTHON = "/home/goya/miniconda3/envs/direct3ds2/bin/python"
FRONT_A = (.20, -.19, .34)
FRONT_B = (.12, -.08, .52)
REAR_A = (.18, .20, .35)
REAR_B = (.12, .12, .52)
CASES = [
    ("front_r014", ("front",), .014),
    ("rear_r014", ("rear",), .014),
    ("four_r014", ("front", "rear"), .014),
    ("front_r023", ("front",), .023),
    ("rear_r023", ("rear",), .023),
    ("four_r023", ("front", "rear"), .023),
    ("four_r030", ("front", "rear"), .030),
]


def connectors(positions: tuple[str, ...], radius: float) -> list[trimesh.Trimesh]:
    pieces = []
    for kind in positions:
        a, b = (FRONT_A, FRONT_B) if kind == "front" else (REAR_A, REAR_B)
        for side in (-1, 1):
            start = (side * a[0], a[1], a[2])
            end = (side * b[0], b[1], b[2])
            pieces.append(trimesh.creation.cylinder(radius=radius,
                                                     segment=(start, end), sections=24))
    return pieces


def specification_audit(pieces: list[trimesh.Trimesh]) -> dict:
    """Check connector surfaces against the existing voxelized design rules."""
    specification = np.load(SPEC / "voxel.npz")
    points = np.concatenate([piece.sample(5000) for piece in pieces])
    ijk = np.floor((points - specification["origin"]) /
                   specification["pitch_xyz"]).astype(int)
    within_grid = np.all((ijk >= 0) & (ijk < np.asarray(specification["bracket"].shape)), axis=1)
    ijk = np.clip(ijk, 0, np.asarray(specification["bracket"].shape) - 1)
    envelope = specification["bracket"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    keepout = specification["keepout"][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return {"sample_count": len(points),
            "within_grid_fraction": float(np.mean(within_grid)),
            "within_envelope_fraction": float(np.mean(envelope & within_grid)),
            "keepout_overlap_fraction": float(np.mean(keepout & within_grid))}


def render(base: trimesh.Trimesh, row: dict) -> Image.Image:
    pieces = connectors(tuple(row["positions"]), row["radius_m"])
    center = np.array([0., .01, .46])
    camera = (center + np.array([.85, -1.2, .8]), center, (0, 0, 1))
    plotter = pv.Plotter(off_screen=True, window_size=(500, 500))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    plotter.add_mesh(pv.wrap(base), color="#82909c", smooth_shading=True,
                     ambient=.3, diffuse=.64, specular=.1)
    for piece in pieces:
        plotter.add_mesh(pv.wrap(piece), color="#df8648", smooth_shading=True,
                         ambient=.4, diffuse=.6)
    plotter.camera_position = [np.asarray(camera[0]).tolist(),
                               np.asarray(camera[1]).tolist(), list(camera[2])]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = .65
    image = Image.fromarray(plotter.screenshot(return_img=True,
                                               transparent_background=True)).convert("RGBA")
    plotter.close()
    return image


def evaluate(row: dict) -> dict:
    folder = OUT / row["name"]
    fea = folder / "independent_fea_15mm"
    summary = fea / "summary.json"
    if not summary.exists():
        cmd = [PYTHON, str(ROOT / "codebase/evaluate_chair_dense_core_raw_fea_2026_10_04.py"),
               "--source-obj", row["mesh"], "--output-dir", str(fea), "--source-aligned"]
        with (folder / "fea_evaluation.log").open("w") as stream:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            row["fea_error"] = str(folder / "fea_evaluation.log")
            return row
    result = json.loads(summary.read_text())
    row["fea_compliance"] = result["compliance"]
    row["fea_volume_liters"] = result["solid_volume_liters"]
    row["fea_summary"] = str(summary)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = trimesh.load(BASE_OBJ, force="mesh", process=False)
    baseline = json.loads(BASE_FEA.read_text())
    rows = []
    for name, positions, radius in CASES:
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "assembled.obj"
        pieces = connectors(positions, radius)
        if not path.exists():
            mesh = trimesh.boolean.union([base, *pieces], engine="manifold")
            mesh.export(path)
        else:
            mesh = trimesh.load(path, force="mesh", process=False)
        row = {"name": name, "positions": positions, "radius_m": radius,
               "connector_count": len(pieces), "mesh": str(path),
               "watertight": bool(mesh.is_watertight),
               "components": len(mesh.split(only_watertight=False)),
               "obj_volume_liters": float(mesh.volume * 1000)}
        row["specification_audit"] = specification_audit(pieces)
        if not row["watertight"] or row["components"] != 1:
            row["geometry_error"] = "Boolean assembly is not one watertight solid"
        rows.append(row)
        print("generated", name, row["watertight"], row["obj_volume_liters"], flush=True)
    eligible = [r for r in rows if r["watertight"] and r["components"] == 1
                and r["specification_audit"]["within_envelope_fraction"] == 1.0
                and r["specification_audit"]["keepout_overlap_fraction"] == 0.0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluated = list(pool.map(evaluate, eligible))
    lookup = {r["name"]: r for r in evaluated}
    rows = [lookup.get(r["name"], r) for r in rows]
    for row in rows:
        if "fea_compliance" in row:
            row["compliance_change_percent"] = 100 * (row["fea_compliance"] / baseline["compliance"] - 1)
            row["volume_change_percent"] = 100 * (row["fea_volume_liters"] / baseline["solid_volume_liters"] - 1)
            row["cv_ratio"] = (row["fea_compliance"] * row["fea_volume_liters"] /
                               (baseline["compliance"] * baseline["solid_volume_liters"]))
    graph = {"nodes": ["center_seat_back", "left_side_frame", "right_side_frame"],
             "edges": [{"from": "center_seat_back", "to": "left_side_frame", "plane_x_m": -.17},
                       {"from": "center_seat_back", "to": "right_side_frame", "plane_x_m": .17}],
             "interpretation": "semantic contact graph inferred from a single watertight monolithic chair; not detachable assembly",
             "front_anchor_xyz_m": [FRONT_A, FRONT_B], "rear_anchor_xyz_m": [REAR_A, REAR_B]}
    (OUT / "contact_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    (OUT / "metrics.json").write_text(json.dumps({"baseline": baseline, "candidates": rows}, indent=2) + "\n")
    tile, header = 500, 40
    sheet = Image.new("RGBA", (tile * 4, (tile + header) * 2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        img = render(base, row)
        col, line = i % 4, i // 4
        sheet.alpha_composite(img, (col * tile, line * (tile + header) + header))
        c = row.get("fea_compliance")
        label = f"{row['name']} | C={c/1e6:.2f}M" if c else row["name"]
        draw.text((col * tile + 9, line * (tile + header) + 12), label, fill="#203746")
    sheet.save(OUT / "connector_candidates.png")
    valid = [r for r in rows if "fea_compliance" in r]
    best_c = min(valid, key=lambda r: r["fea_compliance"]) if valid else None
    best_cv = min(valid, key=lambda r: r["cv_ratio"]) if valid else None
    table = "".join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['obj_volume_liters']:.3f}</td>"
                    f"<td>{r.get('fea_compliance',float('nan'))/1e6:.3f}</td>"
                    f"<td>{r.get('compliance_change_percent',float('nan')):+.2f}%</td>"
                    f"<td>{r.get('cv_ratio',float('nan')):.4f}</td>"
                    f"<td><a href='{html.escape(r['name'])}/assembled.obj'>OBJ</a></td></tr>" for r in rows)
    finding = (f"Lowest C: {best_c['name']}; lowest C×volume: {best_cv['name']}."
               if best_c and best_cv else "No candidate passed independent FEA.")
    (OUT / "index.html").write_text(f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<title>SNAP3D-inspired chair interface pilot</title><style>
body{{font:16px/1.5 system-ui;max-width:1100px;margin:24px auto;background:#f2f6f8;color:#1c3441}}
section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}
table{{width:100%;border-collapse:collapse}}td,th{{border:1px solid #d0dce2;padding:7px}}
img{{max-width:100%}}</style><h1>Chair interface / connector pilot</h1>
<section><p>The selected Dense-10 / Sparse-100 monolithic chair is the common base.
Symmetric front/rear seat-to-frame ligaments are added at a semantic contact graph,
then each candidate is independently tested on the same 15 mm, two-load FEM grid.
Every candidate is one watertight component; sampled connector surfaces are inside
the existing voxel envelope and outside its keepout. The sampled rule check is not
a continuous geometric proof.
This is a structural adaptation of SNAP3D's contact reasoning and physics feedback,
not a reproduction of detachable peg/socket assembly or rigid-body simulation.</p>
<p>Baseline: {baseline['solid_volume_liters']:.3f} L, C={baseline['compliance']/1e6:.3f}×10⁶.
{finding}</p><p><a href='contact_graph.json'>Contact graph</a> ·
<a href='metrics.json'>All metrics</a> ·
<a href='https://arxiv.org/abs/2609.13146'>SNAP3D paper</a></p></section>
<section><img src='connector_candidates.png'></section>
<section><table><tr><th>Candidate</th><th>OBJ volume L</th><th>C ×10⁶ ↓</th>
<th>ΔC</th><th>C×volume / baseline ↓</th><th>Mesh</th></tr>{table}</table></section></html>""")
    report = ["# SNAP3D-inspired selected-chair interface pilot", "",
              "The input is one monolithic Dense10/Sparse100 chair. Semantic side-frame/seat contacts",
              "are parameterized as symmetric structural ligaments and independently tested on the",
              "same 15 mm FEM with simultaneous 800 N -Z and 200 N +Y loads. This does not implement",
              "SNAP3D's detachable assembly, learned part generator, peg/socket or rigid-body simulation.", "",
              f"Baseline OBJ: {BASE_OBJ}", f"Baseline C: {baseline['compliance']}",
              f"Baseline FEM volume L: {baseline['solid_volume_liters']}", "",
              "| Candidate | OBJ volume L | C ×10⁶ | ΔC | C×V ratio |",
              "|---|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['name']} | {r['obj_volume_liters']:.3f} | "
                      f"{r.get('fea_compliance',float('nan'))/1e6:.3f} | "
                      f"{r.get('compliance_change_percent',float('nan')):+.2f}% | "
                      f"{r.get('cv_ratio',float('nan')):.4f} |")
    report += ["", finding, "", f"HTML: {OUT / 'index.html'}", "",
               "Connector surface samples: all candidates inside the 64³ envelope,",
               "outside keepout, with one watertight component. This is a sampled",
               "specification check, not a continuous geometric proof.", "",
               "SNAP3D source: https://arxiv.org/abs/2609.13146"]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n")
    print(OUT / "index.html", flush=True)


if __name__ == "__main__":
    main()
