#!/usr/bin/env python3
"""Show input, dense, and unprocessed sparse chair geometry in aligned cameras."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / "experiments/chair/image_concepts_2026-09-27/diagonal_braced"
CASE = BASE / "camera_proj_w50"
VIEWS = (("hero", 28, 40), ("front", 15, 0), ("right", 15, 90), ("top", 85, 0))
INPUTS = {
    "hero": BASE / "input_512.png",
    "front": BASE / "multiview_registered/input/v00_front_lo.png",
    "right": BASE / "multiview_registered/input/v02_right_lo.png",
    "top": BASE / "multiview_registered/input/v_top.png",
}
STAGES = {
    "dense": CASE / "dense/mesh_dense.obj",
    "sparse_raw": CASE / "sparse/mesh.obj",
}


def silhouette_scores(rendered: Image.Image, target: Image.Image) -> dict[str, float]:
    shape = np.asarray(rendered.convert("RGB")).min(axis=2) < 210
    ref = np.asarray(target.convert("RGB")).min(axis=2) < 210
    return {
        "iou": round(float((shape & ref).sum() / max((shape | ref).sum(), 1)), 4),
        "false_positive_fraction": round(float((shape & ~ref).sum() / max(shape.sum(), 1)), 4),
        "false_negative_fraction": round(float((ref & ~shape).sum() / max(ref.sum(), 1)), 4),
    }


def main() -> None:
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    width, header = 512, 36
    sheet = Image.new("RGB", (width * len(VIEWS), (width + header) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (view, _, _) in enumerate(VIEWS):
        sheet.paste(Image.open(INPUTS[view]).convert("RGB"), (j * width, header))
        draw.text((j * width + 10, 10), f"input: {view}", fill="#20252b")
    metrics = {}
    for row, (name, path) in enumerate(STAGES.items(), start=1):
        mesh = trimesh.load(path, force="mesh")
        components = mesh.split(only_watertight=False)
        stats = {
            "mesh": str(path),
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "components": len(components),
            "watertight": bool(mesh.is_watertight),
            "silhouettes": {},
        }
        for j, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(
                render_lit(mesh, eye, center, up, size=width, fit_extent=scale,
                           margin=1.15, color=(.42, .46, .49))
            ).convert("RGB")
            sheet.paste(rendered, (j * width, row * (width + header) + header))
            draw.text((j * width + 10, row * (width + header) + 10),
                      f"{name}: {view}", fill="#20252b")
            if view != "hero":
                stats["silhouettes"][view] = silhouette_scores(
                    rendered, Image.open(INPUTS[view]))
        metrics[name] = stats
    image_path = BASE / "prefea_dense_sparse_comparison.png"
    json_path = BASE / "prefea_dense_sparse_metrics.json"
    html_path = BASE / "prefea_dense_sparse.html"
    sheet.save(image_path)
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    dense, sparse = metrics["dense"], metrics["sparse_raw"]
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair: input → dense → sparse raw</title>
<style>body{{font:16px system-ui,sans-serif;color:#1d2730;background:#f4f6f8;
max-width:2100px;margin:2rem auto;padding:0 1rem}} img{{max-width:100%;height:auto}}
article{{background:white;padding:1rem;border-radius:12px}}
a{{color:#175c96}}</style>
<article><h1>의자 형상: 입력 이미지 → dense → sparse 원본</h1>
<p>FEA와 최종 boolean·remesh 이전 단계입니다. 각 열은 동일한 카메라와 스케일로 렌더했습니다.
첫 열의 hero 입력은 원본 투시 이미지이므로 정합된 측정은 정면·측면·상단 뷰에서만 합니다.</p>
<p>정면 IoU: dense {dense["silhouettes"]["front"]["iou"]:.3f},
sparse {sparse["silhouettes"]["front"]["iou"]:.3f}.
측면 IoU: dense {dense["silhouettes"]["right"]["iou"]:.3f},
sparse {sparse["silhouettes"]["right"]["iou"]:.3f}.
상단 IoU: dense {dense["silhouettes"]["top"]["iou"]:.3f},
sparse {sparse["silhouettes"]["top"]["iou"]:.3f}.</p>
<p><a href="camera_proj_w50/dense/mesh_dense.obj">dense OBJ</a> ·
<a href="camera_proj_w50/sparse/mesh.obj">sparse raw OBJ</a> ·
<a href="prefea_dense_sparse_metrics.json">측정 JSON</a></p>
<a href="prefea_dense_sparse_comparison.png"><img src="prefea_dense_sparse_comparison.png"
alt="Input, dense, and raw sparse renders"></a></article></html>"""
    html_path.write_text(page)
    print(json.dumps({"image": str(image_path), "html": str(html_path),
                      "metrics": str(json_path), "stages": metrics}, indent=2))


if __name__ == "__main__":
    main()
