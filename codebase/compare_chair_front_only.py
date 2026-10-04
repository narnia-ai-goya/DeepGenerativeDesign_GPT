#!/usr/bin/env python3
"""Compare three-view and front-only chair dense/sparse geometry before FEA."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / "experiments/chair/image_concepts_2026-09-27/diagonal_braced"
STUDIES = {
    "front_only": {
        "three_view_dense": BASE / "camera_proj_w50/dense/mesh_dense.obj",
        "three_view_sparse": BASE / "camera_proj_w50/sparse/mesh.obj",
        "front_only_dense": BASE / "front_only_camera_proj_w50/dense/mesh_dense.obj",
        "front_only_sparse": BASE / "front_only_camera_proj_w50/sparse/mesh.obj",
    },
    "envelope": {
        "restricted_dense": BASE / "camera_proj_w50/dense/mesh_dense.obj",
        "restricted_sparse": BASE / "camera_proj_w50/sparse/mesh.obj",
        "full_envelope_dense": BASE / "full_envelope_camera_proj_w50/dense/mesh_dense.obj",
        "full_envelope_sparse": BASE / "full_envelope_camera_proj_w50/sparse/mesh.obj",
    },
}
VIEWS = (("hero", 28, 40, BASE / "input_512.png"),
         ("front", 15, 0, BASE / "multiview_registered/input/v00_front_lo.png"),
         ("right", 15, 90, BASE / "multiview_registered/input/v02_right_lo.png"),
         ("top", 85, 0, BASE / "multiview_registered/input/v_top.png"))


def main(study: str) -> None:
    cases = STUDIES[study]
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    width, header = 512, 36
    sheet = Image.new("RGB", (width * len(VIEWS), (width + header) * 5), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (view, _, _, path) in enumerate(VIEWS):
        sheet.paste(Image.open(path).convert("RGB"), (j * width, header))
        draw.text((j * width + 10, 10), f"input: {view}", fill="#20252b")
    metrics = {}
    for row, (name, path) in enumerate(cases.items(), start=1):
        mesh = trimesh.load(path, force="mesh")
        components = mesh.split(only_watertight=False)
        metrics[name] = {
            "mesh": str(path), "components": len(components),
            "largest_component_fraction": round(max(abs(c.volume) for c in components)
                                                / max(sum(abs(c.volume) for c in components), 1e-12), 8),
            "silhouettes": {},
        }
        for j, (view, elev, azim, target_path) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(
                render_lit(mesh, eye, center, up, size=width, fit_extent=scale,
                           margin=1.15, color=(.42, .46, .49))
            ).convert("RGB")
            sheet.paste(rendered, (j * width, row * (width + header) + header))
            draw.text((j * width + 10, row * (width + header) + 10),
                      f"{name}: {view}", fill="#20252b")
            if view != "hero":
                metrics[name]["silhouettes"][view] = silhouette_scores(
                    rendered, Image.open(target_path))
    stem = "front_only_vs_multiview_prefea" if study == "front_only" else "full_vs_restricted_envelope_prefea"
    image_path = BASE / f"{stem}.png"
    json_path = BASE / f"{stem}.json"
    html_path = BASE / f"{stem}.html"
    sheet.save(image_path)
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    rows = []
    for name, row in metrics.items():
        scores = row["silhouettes"]
        rows.append(f"<tr><td>{name}</td><td>{scores['front']['iou']:.3f}</td>"
                    f"<td>{scores['right']['iou']:.3f}</td>"
                    f"<td>{scores['top']['iou']:.3f}</td>"
                    f"<td>{row['components']}</td></tr>")
    study_note = (
        '<p>제한된 voxel envelope는 34,634칸, 원래 envelope는 45,344칸입니다. '
        'BC, 물리 STL envelope, 입력 이미지, 생성 loss는 동일하고 voxel 허용 영역만 바꿨습니다. '
        '<a href="envelope_mask_comparison.png">좌면 아래 단면에서 추가된 영역 보기</a>.</p>'
        if study == "envelope" else ""
    )
    html_path.write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        f'<title>Chair {study} comparison</title>'
        '<style>body{font:16px system-ui;background:#f4f6f8;color:#1d2730;'
        'max-width:2100px;margin:2rem auto;padding:0 1rem}img{max-width:100%}'
        'article{background:white;padding:1rem;border-radius:12px}'
        'td,th{padding:.4rem;border:1px solid #bbb}table{border-collapse:collapse}</style>'
        f'<article><h1>의자: {study}, FEA 이전 dense/sparse 비교</h1>'
        '<p>첫 행은 입력 이미지입니다. 나머지 행은 표의 순서대로 dense/sparse 원본입니다. '
        'BC·카메라·스케일은 동일합니다.</p>'
        + study_note +
        '<table><tr><th>stage</th><th>front IoU</th><th>right IoU</th>'
        '<th>top IoU</th><th>components</th></tr>'
        + "".join(rows) + '</table><p>'
        f'<a href="{stem}.json">측정 JSON</a> · '
        + ' · '.join(f'<a href="{path.relative_to(BASE)}">{name} OBJ</a>'
                     for name, path in cases.items())
        + f'</p><a href="{stem}.png"><img src="{stem}.png"></a></article></html>')
    print(json.dumps({"image": str(image_path), "report": str(html_path),
                      "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=STUDIES, default="front_only")
    main(parser.parse_args().study)
