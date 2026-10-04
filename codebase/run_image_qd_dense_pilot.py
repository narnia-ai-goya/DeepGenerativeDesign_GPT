#!/usr/bin/env python3
"""Realize every initial image-QD archetype with the unchanged lr162 dense sampler."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from run_connectivity_qd_sampling import ROOT, BASE, GENERATOR, PYTHON, dump, generation_env, load_mesh, render_mesh


OUT = ROOT / "experiments/bracket/image_qd_2026-09-22"


def prepare() -> list[dict]:
    archive = json.loads((OUT / "image_archive.json").read_text())
    base = json.loads((BASE / "config.json").read_text())
    jobs = []
    for record in archive["records"]:
        name = record["archetype"]
        case = OUT / "cases" / name; gen = case / "gen"; gen.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base)); cfg["name"] = f"image_qd_dense_{name}"; cfg["seed"] = 42
        cfg["stages"]["mesh"].update({
            "skip_sparse": True, "load_dense_cache": None,
            "save_dense_cache": str((case / "dense_cache.npz").resolve()),
            "shape_qd_w": 0.0, "shape_anchor_w": 0.0, "shape_scaffold_w": 0.0,
            "shape_residual_w": 0.0, "shape_transport_radius": 0.0, "oc_flow_w": 0.0,
        })
        config = case / "config.json"; dump(config, cfg)
        jobs.append({"name": name, "case": case, "gen": gen, "config": config,
                     "input": Path(record["input"]), "image_record": record})
    return jobs


def run_one(job: dict, gpu: int, force: bool) -> dict:
    mesh = job["gen"] / "mesh_dense.obj"
    if mesh.exists() and not force:
        return {"name": job["name"], "ok": True, "skipped": True}
    cmd = [str(PYTHON), str(GENERATOR), "--config", str(job["config"]),
           "--target-dir", str(job["input"].parent), "--out", str(job["gen"])]
    with (job["case"] / "dense.log").open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu), stdout=stream,
                              stderr=subprocess.STDOUT)
    return {"name": job["name"], "ok": proc.returncode == 0 and mesh.exists(),
            "returncode": proc.returncode, "gpu": gpu}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    jobs = prepare(); status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fs = {pool.submit(run_one, j, i, args.force): j for i, j in enumerate(jobs)}
        for f in concurrent.futures.as_completed(fs):
            r = f.result(); status.append(r); print(r, flush=True)
    dump(OUT / "dense_status.json", status)
    records = []
    for job in jobs:
        path = job["gen"] / "mesh_dense.obj"
        if not path.exists(): continue
        mesh = load_mesh(path); parts = mesh.split(only_watertight=False)
        image = OUT / "report_assets" / f"{job['name']}_dense.png"
        render_mesh(path, image, job["name"])
        records.append({**job["image_record"], "dense_mesh": str(path.resolve()),
                        "dense_preview": str(image.resolve()), "dense_components": len(parts),
                        "dense_main_face_fraction": max(len(p.faces) for p in parts) / len(mesh.faces),
                        "dense_volume_mm3": abs(float(mesh.volume)) * 1e9})
    dump(OUT / "image_to_dense.json", records)
    cards = ''.join(f"<article><div><img src='images_normalized/{Path(r['normalized']).name}'><img src='report_assets/{r['archetype']}_dense.png'></div><h2>{r['archetype']}</h2><p>cell {r['cell']} · void {r['void_fraction']:.3f} · branches {r['load_path_branch_clusters']}<br>dense components {r['dense_components']} · main {100*r['dense_main_face_fraction']:.2f}%</p><code>{r['dense_mesh']}</code></article>" for r in records)
    html=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Image QD to dense 3D</title><style>body{{font:15px system-ui;background:#f5f7f9;color:#17212b;margin:30px}}main{{max-width:1500px;margin:auto}}.lead{{background:#e9f6f0;border-left:5px solid #168065;padding:14px}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}article{{background:white;border:1px solid #d5dce2;border-radius:10px;padding:12px}}article div{{display:grid;grid-template-columns:1fr 1fr}}img{{width:100%}}code{{display:block;font-size:10px;overflow-wrap:anywhere}}</style><main><h1>Image-space QD → dense 3D realization</h1><p class="lead">Structured prompt images are the QD phenotypes. Every mesh uses the unchanged lr162 generator, seed 42, with no latent QD guidance and no training.</p><div class="grid">{cards}</div></main></html>'''
    (OUT / "index_dense.html").write_text(html)
    print(OUT / "index_dense.html")


if __name__ == "__main__": main()
