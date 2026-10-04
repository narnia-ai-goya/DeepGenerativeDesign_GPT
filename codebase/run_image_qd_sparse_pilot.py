#!/usr/bin/env python3
"""Promote one image-QD elite per occupied cell through sparse and post stages."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from run_connectivity_qd_sampling import (ROOT, BASE, GENERATOR, POST, REMESH, PYTHON,
    dump, generation_env, load_mesh, render_mesh)

OUT = ROOT / "experiments/bracket/image_qd_2026-09-22"


def select() -> list[dict]:
    rows = json.loads((OUT / "image_to_dense.json").read_text())
    cells = {}
    for row in rows:
        cell = tuple(row["cell"])
        score = (row["dense_main_face_fraction"], row["skeleton_thickness_p10_px162"])
        if cell not in cells or score > cells[cell][0]: cells[cell] = (score, row)
    chosen = [value[1] for _, value in sorted(cells.items())]
    dump(OUT / "image_qd_selection.json", {"selected": chosen, "occupied_cells": len(chosen)})
    return chosen


def sparse_one(row: dict, gpu: int) -> dict:
    name = row["archetype"]; case = OUT / "cases" / name; gen = case / "gen"
    marker = case / "sparse_complete.json"
    if marker.exists(): return {"name": name, "ok": True, "skipped": True}
    cfg = json.loads((case / "config.json").read_text())
    cfg["name"] += "_sparse"; cfg["stages"]["mesh"].update({
        "skip_sparse": False, "load_dense_cache": str((case / "dense_cache.npz").resolve()),
        "save_dense_cache": None})
    cp = case / "config_sparse.json"; dump(cp, cfg)
    cmd = [str(PYTHON), str(GENERATOR), "--config", str(cp), "--target-dir",
           str(Path(row["input"]).parent), "--out", str(gen)]
    with (case / "sparse.log").open("w") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu), stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and (gen / "mesh.obj").exists()
    if ok: dump(marker, {"mesh": str((gen / "mesh.obj").resolve())})
    return {"name": name, "ok": ok, "returncode": rc, "gpu": gpu}


def transform_matrix() -> np.ndarray:
    a = load_mesh(BASE / "generation/mesh.obj")
    b = load_mesh(BASE / "generation/mesh_physical_aligned.obj")
    idx = np.arange(0, len(a.vertices), 64)
    return np.linalg.lstsq(np.c_[a.vertices[idx], np.ones(len(idx))], b.vertices[idx], rcond=None)[0]


def post_one(row: dict, transform: np.ndarray) -> bool:
    name = row["archetype"]; case = OUT / "cases" / name; gen = case / "gen"
    cfg = case / "config_sparse.json"; final = gen / "final.obj"
    raw = load_mesh(gen / "mesh.obj")
    aligned = trimesh.Trimesh(np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform,
                              raw.faces.copy(), process=False)
    aligned_path = gen / "mesh_physical_aligned.obj"; aligned.export(aligned_path)
    hybrid = gen / "mesh_bc_preserved.obj"
    with (case / "post.log").open("w") as stream:
        rc = subprocess.run([str(PYTHON), str(POST), "--config", str(cfg), "--in",
             str(aligned_path), "--out", str(hybrid)], cwd=ROOT, stdout=stream,
             stderr=subprocess.STDOUT).returncode
    if rc: return False
    with (case / "remesh.log").open("w") as stream:
        rc = subprocess.run([str(PYTHON), str(REMESH), "--config", str(cfg), "--in",
             str(hybrid), "--out", str(final)], cwd=ROOT, stdout=stream,
             stderr=subprocess.STDOUT).returncode
    return rc == 0 and final.exists()


def main() -> None:
    chosen = select(); status=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs={pool.submit(sparse_one,row,i):row for i,row in enumerate(chosen)}
        for f in concurrent.futures.as_completed(fs):
            r=f.result();status.append(r);print(r,flush=True)
    dump(OUT / "sparse_status.json", status)
    transform=transform_matrix(); results=[]
    for row in chosen:
        if not post_one(row,transform): continue
        name=row["archetype"]; gen=OUT/"cases"/name/"gen"
        raw=load_mesh(gen/"mesh.obj"); final=load_mesh(gen/"final.obj")
        parts=raw.split(only_watertight=False); sizes=sorted([len(p.faces) for p in parts],reverse=True)
        preview=OUT/"report_assets"/f"{name}_final.png";render_mesh(gen/"final.obj",preview,name)
        results.append({**row,"raw_components":len(parts),"raw_main_face_fraction":sizes[0]/sum(sizes),
                        "final_components":len(final.split(only_watertight=False)),
                        "final_watertight":bool(final.is_watertight),
                        "final_volume_mm3":abs(float(final.volume))*1e9,
                        "final_mesh":str((gen/"final.obj").resolve()),"final_preview":str(preview.resolve())})
    dump(OUT/"image_qd_final.json",results)
    cards=''.join(f"<article><div><img src='images_normalized/{Path(r['normalized']).name}'><img src='report_assets/{r['archetype']}_final.png'></div><h2>{r['archetype']} · cell {r['cell']}</h2><p>raw main {100*r['raw_main_face_fraction']:.2f}% · final components {r['final_components']} · watertight {r['final_watertight']}</p><code>{r['final_mesh']}</code></article>" for r in results)
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Image-QD final realizations</title><style>body{{font:15px system-ui;background:#f5f7f9;color:#17212b;margin:30px}}main{{max-width:1500px;margin:auto}}.lead{{background:#e9f6f0;border-left:5px solid #168065;padding:14px}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}article{{background:white;border:1px solid #d5dce2;border-radius:10px;padding:12px}}article div{{display:grid;grid-template-columns:1fr 1fr}}img{{width:100%}}code{{font-size:10px;overflow-wrap:anywhere;display:block}}</style><main><h1>Gradient-free image QD → final 3D</h1><p class="lead">One elite per occupied image cell. Same lr162 sampler and seed; no latent QD gradient and no training.</p><div class="grid">{cards}</div></main></html>'''
    (OUT/"index_final.html").write_text(html);print(OUT/"index_final.html")

if __name__=="__main__":main()
