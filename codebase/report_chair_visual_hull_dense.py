#!/usr/bin/env python3
"""Render and score the visual-hull-conditioned chair dense experiment."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
OUT = BASE / "visual_hull_dense"
CASES = {
    "image-only baseline": BASE / "tuning_open_arm/p50_v35_pw00/dense/mesh_dense.obj",
    "visual hull proxy": OUT / "visual_hull_proxy.obj",
    "anchor100_sc20": OUT / "anchor100_sc20/dense/mesh_dense.obj",
    "anchor300_sc100": OUT / "anchor300_sc100/dense/mesh_dense.obj",
    "anchor2000_sc500": OUT / "anchor2000_sc500/dense/mesh_dense.obj",
}
VIEWS = (("oblique", 25, 35), ("front", 15, 0),
         ("right", 15, 90), ("top", 85, 0))
TARGETS = {"front": "v00_front_lo", "right": "v02_right_lo", "top": "v_top"}


def main() -> None:
    envelope = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = envelope.bounds.mean(axis=0)
    radius = float(np.linalg.norm(envelope.extents)) * 1.5
    scale = float(envelope.extents.max()) / 2
    targets = {view: Image.open(BASE / "open_arm/input" / f"{stem}.png").convert("RGB")
               for view, stem in TARGETS.items()}
    available = [(n, p) for n, p in CASES.items() if p.exists()]
    size, header = 384, 36
    sheet = Image.new("RGB", (size * 4, (size + header) * len(available)), "white")
    draw = ImageDraw.Draw(sheet)
    metrics = {}
    for i, (name, path) in enumerate(available):
        mesh = trimesh.load(path, force="mesh")
        parts = mesh.split(only_watertight=False)
        row = {"mesh": str(path), "components": len(parts),
               "volume_litres": round(abs(float(mesh.volume)) * 1000, 3),
               "silhouettes": {}}
        for j, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(
                mesh, eye, center, up, size=size, fit_extent=scale,
                margin=1.15, color=(.50, .55, .60))).convert("RGB")
            sheet.paste(rendered, (j * size, i * (size + header) + header))
            draw.text((j * size + 10, i * (size + header) + 10),
                      f"{name}: {view}", fill="#263139")
            if view in targets:
                row["silhouettes"][view] = silhouette_scores(
                    rendered, targets[view].resize((size, size)))
        metrics[name] = row
    sheet.save(OUT / "dense_comparison.png")
    (OUT / "dense_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    rows = "".join(
        f'<tr><td><a href="/{path.relative_to(ROOT)}">{name}</a></td>'
        + "".join(f'<td>{m["silhouettes"][v]["iou"]:.3f}</td>'
                  for v in ("front", "right", "top"))
        + f'<td>{m["components"]}</td><td>{m["volume_litres"]:.1f}</td></tr>'
        for name, path in available for m in [metrics[name]])
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Chair dense visual hull experiment</title>'
        '<style>body{font:16px system-ui;background:#f4f6f8;color:#1d2730;'
        'max-width:1600px;margin:2rem auto;padding:0 1rem}'
        'article{background:white;padding:1rem;border-radius:12px}'
        'img{max-width:100%}td,th{border:1px solid #bbb;padding:.4rem}'
        'table{border-collapse:collapse;background:white}</style>'
        '<article><h1>의자 dense: 이미지 외곽선 + 3D 점유 목표</h1>'
        '<p>입력 이미지 세 뷰와 기존 BC/envelope에서 만든 visual hull proxy를 '
        'dense 단계의 형상 조건으로 추가했습니다. FEA와 sparse는 제외했습니다.</p>'
        '<p><strong>판정:</strong> 약한 3D 조건은 등받이를 개선했지만 팔걸이와 접합부를 '
        '복구하지 못했습니다. 가장 강한 조건은 메시를 분리시켰습니다. 따라서 어떤 후보도 '
        '최종 의자로 채택하지 않았습니다.</p>'
        '<table><tr><th>case / OBJ</th><th>front IoU</th><th>right IoU</th>'
        '<th>top IoU</th><th>components</th><th>volume L</th></tr>'
        + rows + '</table><p><a href="dense_metrics.json">측정 JSON</a> · '
        '<a href="visual_hull_proxy_views.png">visual hull 형상</a> · '
        '<a href="STUDY.md">해석</a></p>'
        '<a href="dense_comparison.png"><img src="dense_comparison.png"></a></article></html>')
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
