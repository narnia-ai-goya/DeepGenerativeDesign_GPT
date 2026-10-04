#!/usr/bin/env python3
"""Compare sparse-only ablations from one fixed chair dense cache."""
from __future__ import annotations

import html
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
OUT = BASE / "sparse_artifact_ablation"
INPUT = BASE / "open_arm/input"
CASES = {
    "dense fixed input": BASE / "tuning_open_arm/p50_v35_pw00/dense/mesh_dense.obj",
    "sparse peak80 baseline": BASE / "tuning_open_arm/p50_v35_pw00/sparse/mesh.obj",
    **{name: OUT / name / "sparse/mesh.obj" for name in
       ("guide_off", "peak10", "peak30", "thick0_peak10", "mc040_peak10")},
}
VIEWS = (("oblique", 25, 35), ("front", 15, 0),
         ("right", 15, 90), ("top", 85, 0))
INPUT_NAMES = {"front": "v00_front_lo", "right": "v02_right_lo", "top": "v_top"}


def main() -> None:
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    targets = {view: Image.open(INPUT / f"{stem}.png").convert("RGB")
               for view, stem in INPUT_NAMES.items()}
    opening = back_opening_mask(targets["front"])
    available = [(name, path) for name, path in CASES.items() if path.exists()]
    size, header = 384, 36
    sheet = Image.new("RGB", (size * 4, (size + header) * len(available)), "white")
    draw = ImageDraw.Draw(sheet)
    metrics = {}
    for i, (name, path) in enumerate(available):
        mesh = trimesh.load(path, force="mesh")
        components = mesh.split(only_watertight=False)
        row = {"mesh": str(path), "components": len(components),
               "largest_component_volume_fraction": round(
                   max(abs(c.volume) for c in components)
                   / max(sum(abs(c.volume) for c in components), 1e-12), 8),
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
                target = targets[view].resize((size, size))
                row["silhouettes"][view] = silhouette_scores(rendered, target)
                if view == "front":
                    actual = np.asarray(rendered).min(axis=2) < 210
                    small_opening = np.asarray(Image.fromarray(opening).resize(
                        (size, size), Image.Resampling.NEAREST)).astype(bool)
                    row["back_opening_fill_fraction"] = round(
                        float(actual[small_opening].mean()), 4)
        metrics[name] = row
    sheet.save(OUT / "comparison.png")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    rows = "".join(
        f'<tr><td><a href="/{path.relative_to(ROOT)}">{html.escape(name)}</a></td>'
        + "".join(f'<td>{m["silhouettes"][v]["iou"]:.3f}</td>'
                  for v in ("front", "right", "top"))
        + f'<td>{m["back_opening_fill_fraction"]:.1%}</td>'
        f'<td>{m["components"]}</td><td>{m["volume_litres"]:.1f}</td></tr>'
        for name, path in available for m in [metrics[name]])
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Chair sparse artifact ablation</title>'
        '<style>body{font:16px system-ui;background:#f4f6f8;color:#1d2730;'
        'max-width:1600px;margin:2rem auto;padding:0 1rem}'
        'article{background:white;padding:1rem;border-radius:12px}'
        'img{max-width:100%}td,th{border:1px solid #bbb;padding:.4rem}'
        'table{border-collapse:collapse;background:white}</style>'
        '<article><h1>의자 sparse 형상 이상: 동일 dense에서 파라미터 비교</h1>'
        '<p>FEA와 후처리 전 메시. 모든 경우 동일한 dense cache, 이미지, BC, seed를 사용합니다.</p>'
        '<p>peak80은 기존 sparse, 나머지는 sparse guidance 최대치·두께 제약·'
        'mesh 추출 임계값만 변경했습니다.</p>'
        '<p><strong>결론:</strong> 팔걸이 누락, 잘못된 등받이와 다리 접합은 dense부터 나타납니다. '
        'sparse guidance 상한을 낮춰도 해결되지 않았습니다. mc_threshold 0.4는 윤곽 점수를 조금 '
        '높였지만 형상 오류는 남습니다. FEA는 실행하지 않았습니다.</p>'
        '<table><tr><th>case / OBJ</th><th>front IoU</th><th>right IoU</th>'
        '<th>top IoU</th><th>back opening filled</th><th>components</th>'
        '<th>volume L</th></tr>' + rows + '</table>'
        '<p><a href="metrics.json">측정 JSON</a> · '
        '<a href="manifest.json">설정 차이</a> · '
        '<a href="STUDY.md">결과 해석</a></p>'
        '<a href="comparison.png"><img src="comparison.png"></a></article></html>')
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
