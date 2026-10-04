#!/usr/bin/env python3
"""Compare baseline and selected tuned sofa sparse meshes before FEA."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores
from report_chair_sofa_parameter_grid import back_opening_mask

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
OUT = BASE / "tuning_open_arm"
INPUT = BASE / "open_arm/input"
VIEWS = (("front", "v00_front_lo", 15, 0),
         ("right", "v02_right_lo", 15, 90),
         ("top", "v_top", 85, 0))
CASES = {
    "baseline_sparse": BASE / "open_arm/sparse/mesh.obj",
    "p50_v35_pw00_sparse": OUT / "p50_v35_pw00/sparse/mesh.obj",
    "p150_v35_pw00_sparse": OUT / "p150_v35_pw00/sparse/mesh.obj",
}


def main() -> None:
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    targets = {view: Image.open(INPUT / f"{filename}.png").convert("RGB")
               for view, filename, _, _ in VIEWS}
    opening = back_opening_mask(targets["front"])
    available = [(name, path) for name, path in CASES.items() if path.exists()]
    size, header = 512, 36
    sheet = Image.new("RGB", (size * 3, (size + header) * (len(available) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (view, _, _, _) in enumerate(VIEWS):
        sheet.paste(targets[view], (j * size, header))
        draw.text((j * size + 10, 10), f"input: {view}", fill="#263139")
    metrics = {}
    for i, (name, path) in enumerate(available, start=1):
        mesh = trimesh.load(path, force="mesh")
        components = mesh.split(only_watertight=False)
        row = {"mesh": str(path), "components": len(components),
               "largest_component_volume_fraction": round(
                   max(abs(c.volume) for c in components)
                   / max(sum(abs(c.volume) for c in components), 1e-12), 8),
               "volume_litres": round(abs(float(mesh.volume)) * 1000, 3),
               "silhouettes": {}}
        for j, (view, _, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(
                mesh, eye, center, up, size=size, fit_extent=scale,
                margin=1.15, color=(.42, .46, .49))).convert("RGB")
            sheet.paste(rendered, (j * size, i * (size + header) + header))
            draw.text((j * size + 10, i * (size + header) + 10),
                      f"{name}: {view}", fill="#263139")
            row["silhouettes"][view] = silhouette_scores(rendered, targets[view])
            if view == "front":
                actual = np.asarray(rendered).min(axis=2) < 210
                row["back_opening_fill_fraction"] = round(float(actual[opening].mean()), 4)
        metrics[name] = row
    image_path = OUT / "selected_sparse_comparison.png"
    json_path = OUT / "selected_sparse_metrics.json"
    html_path = OUT / "selected_sparse.html"
    sheet.save(image_path)
    json_path.write_text(json.dumps(metrics, indent=2) + "\n")
    table = "".join(
        f'<tr><td>{name}</td><td>{m["silhouettes"]["front"]["iou"]:.3f}</td>'
        f'<td>{m["silhouettes"]["right"]["iou"]:.3f}</td>'
        f'<td>{m["silhouettes"]["top"]["iou"]:.3f}</td>'
        f'<td>{m["back_opening_fill_fraction"]:.1%}</td>'
        f'<td>{m["components"]}</td><td>{m["volume_litres"]:.1f}</td></tr>'
        for name, m in metrics.items())
    links = " · ".join(
        f'<a href="/{path.relative_to(ROOT)}">{name} OBJ</a>'
        for name, path in available)
    html_path.write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Sofa sparse tuning</title>'
        '<style>body{font:16px system-ui;background:#f4f6f8;color:#1d2730;'
        'max-width:1600px;margin:2rem auto;padding:0 1rem}'
        'article{background:white;padding:1rem;border-radius:12px}'
        'img{max-width:100%}td,th{border:1px solid #bbb;padding:.4rem}'
        'table{border-collapse:collapse;background:white}</style>'
        '<article><h1>소파형 열린 팔걸이: sparse 원본 비교</h1>'
        '<p>FEA와 후처리 전 raw sparse 메시입니다. 모든 뷰는 입력과 동일한 카메라와 스케일입니다.</p>'
        '<p>선택 후보는 p50_v35_pw00입니다. 세 뷰의 윤곽 점수가 모두 초기 설정보다 높고 '
        '단일 연결 메시입니다. 등받이 곡면과 팔걸이 접합은 여전히 입력과 다릅니다. '
        '<a href="TUNING_RESULTS.md">실험 해석</a> · '
        '<a href="index.html">dense 전체 탐색</a></p>'
        '<table><tr><th>case</th><th>front IoU</th><th>right IoU</th>'
        '<th>top IoU</th><th>back opening filled</th><th>components</th>'
        '<th>volume L</th></tr>' + table + '</table>'
        f'<p><a href="selected_sparse_metrics.json">측정 JSON</a> · {links}</p>'
        '<a href="selected_sparse_comparison.png"><img src="selected_sparse_comparison.png">'
        '</a></article></html>')
    print(html_path)


if __name__ == "__main__":
    main()
