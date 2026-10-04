"""Visual map from SNAP3D's full workflow to our selected monolithic chair.

The color decomposition and exploded view are visual proposals only. The contact
locations, structural ligaments, and FEM numbers come from the actual pilot.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import trimesh

from run_chair_snap3d_interface_selected_2026_10_04 import (
    BASE_OBJ, OUT as PILOT, connectors,
)


OUT = PILOT / "other_stages"
COLORS = {"backrest": "#D5855D", "seat": "#4F9AAB",
          "left_frame": "#7382CA", "right_frame": "#8CB586"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size: int):
    return ImageFont.truetype(FONT, size)


def camera():
    center = np.array([0., .01, .46])
    return [(center + np.array([.85, -1.2, .8])).tolist(), center.tolist(), [0, 0, 1]]


def plotter(size=(700, 680)):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    p.enable_anti_aliasing("msaa")
    p.camera_position = camera()
    p.camera.parallel_projection = True
    p.camera.parallel_scale = .65
    return p


def shot(p, path):
    img = Image.fromarray(p.screenshot(return_img=True, transparent_background=True)).convert("RGBA")
    p.close()
    img.save(path)
    return img


def add_part(p, mesh, color):
    p.add_mesh(pv.wrap(mesh), color=color, smooth_shading=True,
               ambient=.37, diffuse=.62, specular=.13)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = trimesh.load(BASE_OBJ, force="mesh", process=False)
    centers = base.triangles_center
    x, y, z = centers.T
    # This is a deterministic surface-label visualization, not a watertight
    # part reconstruction or a learned semantic segmentation result.
    back = (z > .62) & (y > .055) & (np.abs(x) < .15)
    seat = (z > .45) & (z <= .62) & (np.abs(x) < .225) & (y > -.23) & (y < .21)
    seat &= ~back
    labels = np.where(back, 0, np.where(seat, 1, np.where(x < 0, 2, 3)))
    names = ["backrest", "seat", "left_frame", "right_frame"]
    parts = {}
    for i, name in enumerate(names):
        faces = base.faces[labels == i]
        parts[name] = trimesh.Trimesh(vertices=base.vertices, faces=faces, process=False)

    p = plotter()
    for name in names:
        add_part(p, parts[name], COLORS[name])
    semantic = shot(p, OUT / "01_surface_regions_proposal.png")

    offsets = {"backrest": (0, .055, .12), "seat": (0, 0, .055),
               "left_frame": (-.07, 0, 0), "right_frame": (.07, 0, 0)}
    p = plotter()
    for name in names:
        part = parts[name].copy()
        part.apply_translation(offsets[name])
        add_part(p, part, COLORS[name])
    exploded = shot(p, OUT / "02_exploded_regions_proposal.png")

    p = plotter()
    p.add_mesh(pv.wrap(base), color="#84929B", opacity=.53, smooth_shading=True,
               ambient=.4, diffuse=.6)
    for piece in connectors(("front", "rear"), .023):
        add_part(p, piece, "#EA8740")
    contact = shot(p, OUT / "03_contact_connector_measured.png")

    selected = trimesh.load(PILOT / "rear_r023/assembled.obj", force="mesh", process=False)
    p = plotter()
    add_part(p, selected, "#788996")
    assembled = shot(p, OUT / "04_fea_candidate_measured.png")

    tile_w, tile_h = 640, 730
    sheet = Image.new("RGB", (tile_w * 4, tile_h + 190), "#F4F8FA")
    draw = ImageDraw.Draw(sheet)
    metrics = json.loads((PILOT / "metrics.json").read_text())
    baseline = metrics["baseline"]
    rear = next(r for r in metrics["candidates"] if r["name"] == "rear_r023")
    panels = [
        (semantic, "1  Surface regions", "Proposed semantic labels",
         "Backrest / seat / left / right", "Not separate part meshes"),
        (exploded, "2  Assembly layout", "Illustrative exploded view",
         "Surface patches moved apart", "No penetration correction run"),
        (contact, "3  Contact + connectors", "Measured contact candidates",
         "Left frame - seat - right frame", "Structural ligaments, not sockets"),
        (assembled, "4  Physics feedback", "Rear connector, radius 23 mm",
         f"C  {baseline['compliance']/1e6:.2f} -> {rear['fea_compliance']/1e6:.2f} M",
         "Independent two-load FEM"),
    ]
    for i, (pic, title, subtitle, detail, note) in enumerate(panels):
        x0 = i * tile_w
        draw.rounded_rectangle((x0 + 13, 18, x0 + tile_w - 13, tile_h + 174),
                               18, fill="white", outline="#D9E4E8", width=2)
        draw.text((x0 + 35, 40), title, font=font(29), fill="#1C3441")
        draw.text((x0 + 35, 85), subtitle, font=font(22), fill="#526873")
        rendered = pic.resize((620, 600), Image.Resampling.LANCZOS)
        sheet.paste(rendered, (x0 + 10, 116), rendered)
        draw.text((x0 + 35, 722), detail, font=font(22), fill="#203C49")
        draw.text((x0 + 35, 764), note, font=font(19), fill="#B15A2B" if i < 2 else "#286F67")
    sheet.save(OUT / "snap3d_stages_chair.png")
    (OUT / "index.html").write_text(f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<title>SNAP3D stages mapped to chair</title><style>
body{{font:17px/1.55 system-ui;background:#f4f8fa;color:#1c3441;max-width:1250px;margin:30px auto}}
section{{background:white;padding:20px;margin:20px 0;border-radius:12px}}
img{{max-width:100%}}a{{color:#126987}}</style>
<h1>SNAP3D 단계를 현재 의자에 대응시키기</h1>
<section><img src='snap3d_stages_chair.png' alt='Four stages on the same chair'></section>
<section><h2>실행 상태</h2><p>1–2단계의 색 구분과 분해 배치는 일체형 메시에서 만든
<strong>개념 시각화</strong>입니다. 독립적인 watertight 부품 분할, 부품 간 침투 제거,
peg/socket, 중력 기반 rigid-body 조립 검증은 아직 수행하지 않았습니다.</p>
<p>3–4단계의 연결부 형상은 실제 OBJ에서 생성했고 15 mm FEM, 동시 800 N -Z
좌면 하중 및 200 N +Y 등받이 하중으로 평가했습니다. 기준 C={baseline['compliance']/1e6:.2f}×10⁶,
뒤쪽 23 mm 연결부 C={rear['fea_compliance']/1e6:.2f}×10⁶
({rear['compliance_change_percent']:.2f}%), 체적 {baseline['solid_volume_liters']:.2f}→
{rear['fea_volume_liters']:.2f} L.</p><p><a href='../index.html'>7개 연결부 실험</a> ·
<a href='https://lucytuan.github.io/SNAP3D/'>SNAP3D 공식 페이지</a></p></section></html>""")
    print(OUT / "snap3d_stages_chair.png")


if __name__ == "__main__":
    main()
