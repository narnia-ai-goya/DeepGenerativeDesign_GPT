#!/usr/bin/env python3
"""Evaluate the dense-only training-free OC-Flow pilot against its fixed-seed baseline."""
from __future__ import annotations

import json
import os
import argparse
from pathlib import Path

import numpy as np
import trimesh

from tsdf_shape_operator import mesh_tsdf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bracket/shape_dqd_ocflow_2026-09-22"
BASELINE = ROOT / "experiments/bracket/shape_dqd_dense_residual_2026-09-22/baseline_rebuilt/mesh_dense.obj"
PILOT = OUT / "niche_01_w40_rel05/mesh_dense.obj"


def load(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def metrics(mesh: trimesh.Trimesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight), "volume_m3": float(abs(mesh.volume)),
        "area_m2": float(mesh.area),
    }


def render(meshes: list[tuple[str, trimesh.Trimesh]], out: Path) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont
    try:
        pv.start_xvfb()
    except Exception:
        pass
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    cards = []
    for name, mesh in meshes:
        plotter = pv.Plotter(off_screen=True, window_size=(500, 420))
        plotter.set_background("#f7f8fa")
        plotter.add_mesh(pv.wrap(mesh), color="#6f8293", smooth_shading=True,
                         ambient=.25, diffuse=.65, specular=.25, specular_power=20)
        plotter.enable_parallel_projection()
        plotter.camera_position = "iso"
        image = Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB")
        plotter.close()
        canvas = Image.new("RGB", (500, 452), "white")
        canvas.paste(image, (0, 32))
        ImageDraw.Draw(canvas).text((10, 8), name, font=font, fill="#17212b")
        cards.append(canvas)
    sheet = Image.new("RGB", (1000, 452), "white")
    for i, card in enumerate(cards): sheet.paste(card, (500 * i, 0))
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=BASELINE)
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument("--baseline-label", default="Dense baseline")
    ap.add_argument("--label", default="OC-Flow, niche 1")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    baseline_path, pilot_path, out = args.baseline.resolve(), args.pilot.resolve(), args.out.resolve()
    if not baseline_path.exists() or not pilot_path.exists():
        raise FileNotFoundError("Expected dense baseline and OC-Flow mesh are missing")
    domain = load(ROOT / "data_real/bracket/original_DesignSpace.stl")
    baseline, pilot = load(baseline_path), load(pilot_path)
    # This is a fixed-frame diagnostic, not a training loss.  It measures real
    # output geometry after envelope masking on a common 96^3 world grid.
    fields = {"baseline": mesh_tsdf(baseline_path, domain.bounds, 96),
              "ocflow": mesh_tsdf(pilot_path, domain.bounds, 96)}
    result = {
        "method": "training-free OC-Flow-style velocity correction",
        "baseline": str(baseline_path), "ocflow": str(pilot_path),
        "baseline_metrics": metrics(baseline), "ocflow_metrics": metrics(pilot),
        "mean_abs_tsdf_delta": float(np.mean(np.abs(fields["baseline"] - fields["ocflow"]))),
        "max_abs_tsdf_delta": float(np.max(np.abs(fields["baseline"] - fields["ocflow"]))),
        "interpretation": "A delta above same-seed replay noise indicates the velocity controller changed the dense mesh; this alone does not establish target-niche attainment.",
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "ocflow_vs_baseline_tsdf.npz", **fields)
    (out / "ocflow_pilot_result.json").write_text(json.dumps(result, indent=2) + "\n")
    render([(args.baseline_label, baseline), (args.label, pilot)], out / "ocflow_vs_baseline.png")
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>OC-Flow dense pilot</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;color:#17212b}}img{{max-width:100%;border:1px solid #d8dfe5}}pre{{background:#f5f7f9;padding:16px;white-space:pre-wrap}}</style>
<h1>Training-free OC-Flow dense pilot</h1><p>Same image, BC/envelope, seed, and dense recipe. The only change is a bounded structural-loss correction to the Flow Matching velocity before each Euler step.</p><img src=\"ocflow_vs_baseline.png\"><pre>{json.dumps(result, indent=2)}</pre>"""
    (out / "index.html").write_text(page)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
