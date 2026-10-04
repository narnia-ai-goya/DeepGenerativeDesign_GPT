#!/usr/bin/env python3
"""Render sofa-style chair inputs and raw dense/sparse stages before FEA."""
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
VIEWS = (("front", "v00_front_lo", 15, 0),
         ("right", "v02_right_lo", 15, 90),
         ("top", "v_top", 85, 0))
STAGES = (("dense", "dense/mesh_dense.obj"), ("sparse_raw", "sparse/mesh.obj"))


def run_case(name: str, center: np.ndarray, radius: float, scale: float) -> dict:
    case = BASE / name
    available = [(stage, case / relative) for stage, relative in STAGES
                 if (case / relative).exists()]
    size, header = 512, 36
    sheet = Image.new("RGB", (size * 3, (size + header) * (len(available) + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    targets = {}
    for j, (view, filename, _, _) in enumerate(VIEWS):
        targets[view] = Image.open(case / "input" / f"{filename}.png").convert("RGB")
        sheet.paste(targets[view], (j * size, header))
        draw.text((j * size + 10, 10), f"input: {view}", fill="#263139")
    metrics = {"case": str(case), "stages": {}}
    for i, (stage, path) in enumerate(available, start=1):
        mesh = trimesh.load(path, force="mesh")
        components = mesh.split(only_watertight=False)
        stat = {"mesh": str(path), "components": len(components),
                "largest_component_volume_fraction": round(
                    max(abs(c.volume) for c in components)
                    / max(sum(abs(c.volume) for c in components), 1e-12), 8),
                "volume_litres": round(abs(float(mesh.volume)) * 1000, 3),
                "silhouettes": {}}
        for j, (view, _, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(
                render_lit(mesh, eye, center, up, size=size, fit_extent=scale,
                           margin=1.15, color=(.42, .46, .49))
            ).convert("RGB")
            sheet.paste(rendered, (j * size, i * (size + header) + header))
            draw.text((j * size + 10, i * (size + header) + 10),
                      f"{stage}: {view}", fill="#263139")
            stat["silhouettes"][view] = silhouette_scores(rendered, targets[view])
        metrics["stages"][stage] = stat
    sheet.save(case / "prefea_comparison.png")
    (case / "prefea_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    env = trimesh.load(ROOT / "data_real/chair/original_DesignSpace.stl", force="mesh")
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    cards = []
    summary = {}
    for name in ("open_arm", "solid_side"):
        metrics = run_case(name, center, radius, scale)
        summary[name] = metrics
        stage_text = []
        for stage, row in metrics["stages"].items():
            s = row["silhouettes"]
            stage_text.append(
                f"{stage}: front/right/top IoU "
                f"{s['front']['iou']:.3f}/{s['right']['iou']:.3f}/{s['top']['iou']:.3f}; "
                f"{row['components']} components")
        cards.append(
            f'<article><h2>{name}</h2><p>{" · ".join(stage_text)}</p>'
            f'<p><a href="{name}/generated_sheet.png">GPT-generated sheet</a> · '
            f'<a href="{name}/input_contact.png">registered inputs</a> · '
            f'<a href="{name}/envelope_feasibility.png">envelope feasibility</a> · '
            f'<a href="{name}/prefea_metrics.json">metrics</a></p>'
            f'<a href="{name}/prefea_comparison.png">'
            f'<img src="{name}/prefea_comparison.png"></a></article>')
    (BASE / "prefea_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (BASE / "index.html").write_text(
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Sofa-style chair image to 3D pilot</title>'
        '<style>body{font:16px system-ui,sans-serif;background:#f4f6f8;color:#1d2730;'
        'max-width:1600px;margin:2rem auto;padding:0 1rem}'
        'article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}'
        'img{max-width:100%}</style>'
        '<h1>소파형 의자: 입력 이미지 → dense → sparse 원본</h1>'
        '<p>두 이미지 시안은 같은 1인용 의자 BC와 full chair envelope로 생성했습니다. '
        'FEA와 후처리는 수행하지 않았습니다. 모든 메시 이미지는 입력과 같은 카메라로 렌더했습니다.</p>'
        '<p><a href="ASSESSMENT.md">결과 해석과 부족한 부분</a> · '
        '<a href="prompts.md">이미지 프롬프트</a> · '
        '<a href="tuning_open_arm/index.html">열린 팔걸이형 파라미터 탐색</a> · '
        '<a href="tuning_open_arm/selected_sparse.html">선택 sparse 비교</a></p>'
        + "".join(cards) + '</html>')
    print(BASE / "index.html")


if __name__ == "__main__":
    main()
