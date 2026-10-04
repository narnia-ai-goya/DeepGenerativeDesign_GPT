#!/usr/bin/env python3
"""Score the sofa dense parameter sweep using aligned views and back-opening void."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
CASE = BASE / "open_arm"
OUT = BASE / "tuning_open_arm"
VIEWS = (("front", "v00_front_lo", 15, 0),
         ("right", "v02_right_lo", 15, 90),
         ("top", "v_top", 85, 0))


def back_opening_mask(front_image: Image.Image) -> np.ndarray:
    foreground = np.asarray(front_image.convert("RGB")).min(axis=2) < 210
    labels, count = label(~foreground)
    candidates = []
    for i in range(1, count + 1):
        yy, xx = np.where(labels == i)
        if (len(xx) > 100 and xx.min() > 0 and yy.min() > 0
                and xx.max() < 511 and yy.max() < 511 and yy.mean() < 260):
            candidates.append((len(xx), i))
    if not candidates:
        raise ValueError("no enclosed back opening found in sofa front input")
    return labels == max(candidates)[1]


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    targets = {name: Image.open(CASE / "input" / f"{filename}.png").convert("RGB")
               for name, filename, _, _ in VIEWS}
    opening = back_opening_mask(targets["front"])
    rows = []
    cards = []
    for name, params in manifest.items():
        mesh_path = OUT / name / "dense/mesh_dense.obj"
        if not mesh_path.exists():
            continue
        mesh = trimesh.load(mesh_path, force="mesh")
        components = mesh.split(only_watertight=False)
        preview = Image.new("RGB", (512 * 3, 548), "white")
        draw = ImageDraw.Draw(preview)
        scores = {}
        for j, (view, _, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(
                mesh, eye, center, up, size=512, fit_extent=scale,
                margin=1.15, color=(.42, .46, .49))).convert("RGB")
            preview.paste(rendered, (j * 512, 36))
            draw.text((j * 512 + 10, 10), f"{name}: {view}", fill="#263139")
            scores[view] = silhouette_scores(rendered, targets[view])
            if view == "front":
                actual = np.asarray(rendered).min(axis=2) < 210
                opening_fill = float(actual[opening].mean())
        image_path = OUT / name / "dense_preview.png"
        preview.save(image_path)
        row = {**params, "name": name, "mesh": str(mesh_path),
               "preview": str(image_path), "components": len(components),
               "largest_component_volume_fraction": round(
                   max(abs(c.volume) for c in components)
                   / max(sum(abs(c.volume) for c in components), 1e-12), 8),
               "volume_litres": round(abs(float(mesh.volume)) * 1000, 3),
               "back_opening_fill_fraction": round(opening_fill, 4),
               "silhouettes": scores}
        rows.append(row)
        cards.append(
            f'<article><h2>{name}</h2><p>front/right/top IoU '
            f'{scores["front"]["iou"]:.3f}/{scores["right"]["iou"]:.3f}/'
            f'{scores["top"]["iou"]:.3f}; back opening filled {opening_fill:.1%}; '
            f'{len(components)} components</p>'
            f'<a href="{name}/dense_preview.png"><img src="{name}/dense_preview.png"></a>'
            f'<p><a href="{name}/dense/mesh_dense.obj">dense OBJ</a> · '
            f'<a href="{name}/config_dense.json">config</a></p></article>')
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    table = "".join(
        f'<tr><td>{r["name"]}</td><td>{r.get("cfg", 9.0):.1f}</td>'
        f'<td>{r["silhouettes"]["front"]["iou"]:.3f}</td>'
        f'<td>{r["silhouettes"]["right"]["iou"]:.3f}</td>'
        f'<td>{r["silhouettes"]["top"]["iou"]:.3f}</td>'
        f'<td>{r["back_opening_fill_fraction"]:.1%}</td>'
        f'<td>{r["components"]}</td><td>{r["volume_litres"]:.1f}</td></tr>'
        for r in rows)
    (OUT / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Sofa open-arm dense parameter grid</title>'
        '<style>body{font:16px system-ui;background:#f4f6f8;color:#1d2730;'
        'max-width:1600px;margin:2rem auto;padding:0 1rem}'
        'article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}'
        'img{max-width:100%}td,th{border:1px solid #bbb;padding:.4rem}'
        'table{border-collapse:collapse;background:white}</style>'
        '<h1>소파형 열린 팔걸이: dense 파라미터 탐색</h1>'
        '<p>이미지 투영 가중치 50/100/150, 체적 목표 0.20/0.35, 기존 support path '
        '가중치 0/3.5를 비교했습니다. CFG는 9를 기본으로 두고 한 경우만 6.5로 낮췄습니다. '
        'volume weight는 50으로 고정했습니다. '
        '모든 경우 FEA OFF, full chair envelope, 동일 BC·이미지·seed입니다. '
        'back opening filled는 정면 이미지에서 닫힌 빈 공간이 메시 투영으로 '
        '메워진 비율이며 낮을수록 좋습니다.</p>'
        '<p><a href="metrics.json">전체 측정 JSON</a> · '
        '<a href="../open_arm/input_contact.png">입력 이미지</a></p>'
        '<table><tr><th>case</th><th>CFG</th><th>front IoU</th><th>right IoU</th>'
        '<th>top IoU</th><th>back opening filled</th><th>components</th>'
        '<th>volume L</th></tr>' + table + '</table>' + "".join(cards) + '</html>')
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
