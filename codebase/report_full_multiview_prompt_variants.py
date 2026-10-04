#!/usr/bin/env python3
"""Render final prompt variants and write a local comparison page."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyvista as pv
import trimesh
from PIL import Image, ImageDraw

from report_bracket_multiview_case_study import render
from run_full_multiview_prompt_variants import OUT, ROOT


def render_top(mesh_path: Path, output: Path) -> None:
    mesh = pv.read(mesh_path)
    center = mesh.center
    span = max(mesh.bounds[1] - mesh.bounds[0], mesh.bounds[3] - mesh.bounds[2])
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="#8c999f", smooth_shading=False,
                     ambient=0.38, diffuse=0.67, specular=0.32, specular_power=16)
    plotter.camera_position = [(center[0], center[1], center[2] + 1), center, (0, 1, 0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = span * 0.59
    plotter.screenshot(str(output))
    plotter.close()


def summary(mesh_path: Path) -> dict:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    return {"volume_cm3": round(abs(float(mesh.volume)) * 1e6, 1),
            "faces": int(len(mesh.faces)), "watertight": bool(mesh.is_watertight),
            "components": int(len(mesh.split(only_watertight=False)))}


def main() -> None:
    rows = json.loads((OUT / "manifest.json").read_text())
    audits = json.loads((OUT / "bc_audit.json").read_text()) if (OUT / "bc_audit.json").exists() else {}
    report = OUT / "report"
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for row in rows:
        name = row["id"]
        mesh = Path(row["final"])
        if not mesh.exists():
            row["status"] = "missing final mesh"
            continue
        row["metrics"] = summary(mesh)
        row["bc_audit"] = audits.get(name, {})
        row["status"] = ("accepted" if row["bc_audit"].get("fix", 0) >= 0.99
                         and row["bc_audit"].get("load", 0) >= 0.99
                         and row["metrics"]["watertight"] and row["metrics"]["components"] == 1
                         else "rejected_bc")
        shutil.copyfile(Path(row["input_dir"]) / "그림1.png", assets / f"{name}_input.png")
        for view in ("iso", "xminus", "xplus"):
            render(mesh, assets / f"{name}_{view}.png", view)
        render_top(mesh, assets / f"{name}_top.png")
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    for view in ("top", "iso"):
        images = [(row["id"], assets / f"{row['id']}_{view}.png") for row in rows
                  if row.get("status") == "accepted"]
        w, h = ((520, 520) if view == "top" else (640, 480))
        canvas = Image.new("RGB", (w * len(images), h), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(images):
            canvas.paste(Image.open(path).convert("RGB").resize((w, h)), (i*w, 0))
            draw.text((i*w+15, 12), name, fill="black", stroke_width=2, stroke_fill="white")
        canvas.save(assets / f"comparison_{view}.png")
    html = ["""<!doctype html><html lang='ko'><meta charset='utf-8'><title>Bracket full multi-view prompt variants</title>
<style>body{font:16px/1.5 system-ui;max-width:1500px;margin:28px auto;padding:0 16px;background:#f4f6f8;color:#19232b}.note,.card{background:#fff;border:1px solid #d9e0e7;border-radius:10px;padding:16px;margin:12px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%}a{word-break:break-all}table{border-collapse:collapse;background:white}th,td{border:1px solid #ddd;padding:8px 12px}</style>
<h1>Bracket · 동일 레시피, 서로 다른 text-driven 이미지</h1><p class='note'>스타일 라벨은 swept / scalloped / fork / tapered. 정확한 원본 image-generation prompt 문장은 아카이브에 없으므로 라벨을 원문 prompt로 오인하지 않는다. 모든 사례는 동일 seed 42, 3 views, dense+sparse FEA ON, BC dilate 6 mm, 최종 BC union + expanded envelope boolean + 0.75 mm remesh를 사용한다. 측면 뷰는 해당 사례의 기존 top-only dense mesh 렌더다.</p>"""]
    html.append("<h2>최종 mesh 비교</h2><p>Top view</p><img src='assets/comparison_top.png'><p>Isometric view</p><img src='assets/comparison_iso.png'>")
    html.append("<div class='grid'>")
    for row in rows:
        name = row["id"]
        if row.get("status") == "missing final mesh":
            html.append(f"<div class='card'><h2>{name}</h2><p>Failed or incomplete</p></div>")
            continue
        mesh = Path(row["final"])
        link = "/" + str(mesh.relative_to(ROOT))
        m = row["metrics"]
        bc = row["bc_audit"]
        html.append(f"<div class='card'><h2>{name} · {row['status']}</h2><p>{row['descriptor']}</p><img src='assets/{name}_input.png'><p>입력 이미지</p><img src='assets/{name}_top.png'><p>최종 top</p><img src='assets/{name}_iso.png'><p>최종 iso</p><img src='assets/{name}_xplus.png'><p>최종 +X</p><p>체적 {m['volume_cm3']} cm³ · watertight {m['watertight']} · 연결 성분 {m['components']} · fix/load BC {bc.get('fix',0):.2f}/{bc.get('load',0):.2f}</p><a href='{link}'>{mesh}</a></div>")
    html.append("</div></html>")
    (report / "index.html").write_text("\n".join(html))
    print((report / "index.html").resolve())


if __name__ == "__main__":
    main()
