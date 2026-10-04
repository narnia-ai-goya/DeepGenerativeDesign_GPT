#!/usr/bin/env python3
"""Render and summarize the matched bracket multi-view sparse runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyvista as pv
import trimesh

from run_bracket_multiview_case_study import OUT, ROOT


def render(mesh_path: Path, path: Path, view: str) -> None:
    mesh = pv.read(mesh_path)
    center = mesh.center
    bounds = mesh.bounds
    span = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 750))
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="#8c999f", smooth_shading=False,
                     ambient=0.38, diffuse=0.67, specular=0.32, specular_power=16)
    if view == "xminus":
        camera = (center[0]-1, center[1], center[2]); up = (0, 0, 1)
    elif view == "xplus":
        camera = (center[0]+1, center[1], center[2]); up = (0, 0, 1)
    else:
        camera = (center[0]+0.8, center[1]-0.8, center[2]+0.8); up = (0, 0, 1)
    plotter.camera_position = [camera, center, up]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = span * (0.61 if view != "iso" else 0.80)
    plotter.add_light(pv.Light(position=(center[0]+0.4,center[1]-0.5,center[2]+0.8),
                              focal_point=center, intensity=0.9))
    plotter.screenshot(str(path))
    plotter.close()


def metrics(path: Path) -> dict:
    mesh = trimesh.load(path, force="mesh", process=False)
    parts = mesh.split(only_watertight=False)
    return {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces)),
            "components": int(len(parts)), "volume_cm3": round(abs(float(mesh.volume))*1e6, 1),
            "watertight": bool(mesh.is_watertight)}


def main() -> None:
    rows = json.loads((OUT / "manifest.json").read_text())
    report = OUT / "report"
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    parts = ["""<!doctype html><html lang='ko'><meta charset='utf-8'><title>Bracket multi-view study</title>
<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:30px auto;color:#18212b;background:#f4f6f8}h1,h2{margin-left:12px}.note{margin:12px;padding:16px;background:#fff;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.tile{background:white;padding:10px;border-radius:10px}.tile img{width:100%;display:block}a{word-break:break-all}table{border-collapse:collapse;background:white}th,td{padding:8px;border:1px solid #ddd}</style>
<h1>Bracket multi-view case study</h1><div class='note'>동일한 dense cache·seed·FEA 설정에서 sparse 입력만 top-only → top + ±X side로 변경. Side는 동일 dense mesh의 렌더이며 독립적인 실측/생성 뷰가 아닙니다. 형상 비교는 raw sparse mesh 기준입니다.</div><div class='note'><b>관찰:</b> 세 사례 모두 측면 실루엣이 다소 매끈해지고 체적이 증가했지만, 고정부 근처의 패임은 남았습니다. swept·spine은 연결 성분이 하나로 모였고 scalloped는 극소량의 조각이 늘었습니다. 따라서 multi-view만으로 외벽 결함이 안정적으로 해결됐다고 판단할 수 없습니다.</div>"""]
    for row in rows:
        name = row["name"]
        base = Path(row["baseline_mesh"])
        multi = Path(row["multiview_mesh"])
        if not multi.exists():
            parts.append(f"<h2>{name}: generation failed</h2><p>{row['log']}</p>")
            continue
        row["baseline_metrics"] = metrics(base)
        row["multiview_metrics"] = metrics(multi)
        parts.append(f"<h2>{name}</h2><p class='note'>top-only: {row['baseline_metrics']}<br>multi-view: {row['multiview_metrics']}<br><a href='/{base.relative_to(ROOT)}'>top-only OBJ</a> · <a href='/{multi.relative_to(ROOT)}'>multi-view OBJ</a></p>")
        for label, mesh in (("top_only", base), ("multi_view", multi)):
            for view in ("xminus", "xplus", "iso"):
                render(mesh, assets / f"{name}_{label}_{view}.png", view)
        parts.append("<div class='grid'>")
        for label, img in (("top input", Path(row["input"])/"그림1.png"),
                           ("side input −X", Path(row["input"])/"side_xminus.png"),
                           ("side input +X", Path(row["input"])/"side_xplus.png")):
            local = assets / f"{name}_{img.name}"
            shutil.copyfile(img, local)
            parts.append(f"<div class='tile'><b>{label}</b><img src='assets/{local.name}'></div>")
        for view in ("xminus", "xplus", "iso"):
            for label in ("top_only", "multi_view"):
                parts.append(f"<div class='tile'><b>{label} · {view}</b><img src='assets/{name}_{label}_{view}.png'></div>")
        parts.append("</div>")
    parts.append("</html>")
    (report / "index.html").write_text("\n".join(parts))
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(report / "index.html")


if __name__ == "__main__":
    main()
