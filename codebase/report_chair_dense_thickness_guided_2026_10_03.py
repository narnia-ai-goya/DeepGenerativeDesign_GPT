"""Measure dense thickness effects before and after sparse generation."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_opening, distance_transform_edt, maximum_filter
from scipy.spatial.transform import Rotation
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE, OUT as MAIN
from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

OUT = MAIN / 'dense_thickness_guided_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
CASES = [('vanilla_baseline', MAIN / 'qd_01/generation', 0., 'vanilla'),
         ('guided_zero', OUT / 'guided_zero/generation', 0., 'guided_zero'),
         ('hard_0p01', OUT / 'hard_0p01/generation', .01, 'hard'),
         ('hard_0p03', OUT / 'hard_0p03/generation', .03, 'hard'),
         ('hard_0p06', OUT / 'hard_0p06/generation', .06, 'hard'),
         ('rmin_100', OUT / 'rmin_100/generation', 100., 'rmin'),
         ('rmin_30000', OUT / 'rmin_30000/generation', 30000., 'rmin')]


def align(mesh: trimesh.Trimesh, cal: dict) -> trimesh.Trimesh:
    m = mesh.copy()
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    m.vertices = ((m.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                  cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
    return m


def metrics(mesh: trimesh.Trimesh, origin: np.ndarray, pitch: np.ndarray) -> tuple[dict, np.ndarray]:
    occ64 = voxel_centers_inside(mesh, 64, origin, pitch).astype(bool)
    opened = binary_opening(occ64, structure=np.ones((3, 3, 3), bool))
    p128 = pitch / 2
    occ128 = voxel_centers_inside(mesh, 128, origin, p128).astype(bool)
    dist = distance_transform_edt(occ128)
    z = origin[2] + (np.arange(128) + .5) * p128[2]
    ridge = occ128 & (dist >= maximum_filter(dist, size=3) - 1e-8) & (z >= .61)[None, None, :]
    mm = 2 * dist[ridge] * p128[0] * 1000
    return ({'occupied_voxels_64': int(occ64.sum()),
             'volume_liters_64': float(occ64.sum() * np.prod(pitch) * 1000),
             'opening_killed_fraction_64': float((occ64 & ~opened).sum() / max(1, occ64.sum())),
             'upper_medial_p10_mm_128': float(np.percentile(mm, 10)) if len(mm) else None,
             'upper_medial_median_mm_128': float(np.median(mm)) if len(mm) else None,
             'upper_medial_below_24mm_fraction_128': float((mm < 24).mean()) if len(mm) else None,
             'upper_medial_sample_count': int(len(mm)),
             'watertight': bool(mesh.is_watertight)}, occ64)


def main() -> None:
    spec = np.load(SPEC / 'voxel.npz')
    cal = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    rows = []
    dense_occ = {}
    canvas = Image.new('RGB', (1200, 360 * len(CASES)), '#f6f8fa')
    draw = ImageDraw.Draw(canvas)
    for i, (name, folder, weight, mode) in enumerate(CASES):
        dense_file = folder / 'mesh_dense_raw.obj'
        sparse_file = folder / 'mesh.obj'
        if not dense_file.exists() or not sparse_file.exists():
            continue
        dense = align(trimesh.load(dense_file, force='mesh', process=False), cal)
        raw_sparse = trimesh.load(sparse_file, force='mesh', process=False)
        sparse = align(max(raw_sparse.split(only_watertight=False),
                           key=lambda piece: abs(piece.volume)), cal)
        dense_stat, occ = metrics(dense, spec['origin'], spec['pitch_xyz'])
        sparse_stat, _ = metrics(sparse, spec['origin'], spec['pitch_xyz'])
        dense_occ[name] = occ
        row = {'id': name, 'mode': mode, 'weight': weight,
               'dense_mesh': str(dense_file), 'sparse_mesh': str(sparse_file),
               'dense': dense_stat, 'sparse': sparse_stat}
        rows.append(row)
        for col, mesh in enumerate((dense, sparse)):
            for view, azim in enumerate((0, 90)):
                target = np.array([0., .01, .46])
                eye, up = camera_from_elev_azim(target, 2., 15, azim)
                image = render_lit(mesh, eye, target, up, size=280, fit_extent=.56,
                                   color=(.63, .69, .74))
                x, y = (col * 2 + view) * 300, i * 360 + 25
                canvas.paste(Image.fromarray(image).convert('RGB'), (x, y))
                draw.text((x + 5, i * 360 + 6), f'{name} {"dense" if col == 0 else "sparse"} {"front" if view == 0 else "side"}',
                          fill='#233643')
        print(name, 'dense volume', round(dense_stat['volume_liters_64'], 2),
              'opening loss', round(dense_stat['opening_killed_fraction_64'], 3),
              'upper p10', dense_stat['upper_medial_p10_mm_128'], flush=True)
    baseline = dense_occ.get('vanilla_baseline')
    if baseline is not None:
        for row in rows:
            occ = dense_occ[row['id']]
            row['dense_iou_to_vanilla_64'] = float((occ & baseline).sum() /
                                                     max(1, (occ | baseline).sum()))
    canvas.crop((0, 0, 1200, 360 * len(rows))).save(OUT / 'thickness_comparison.png')
    (OUT / 'thickness_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    cards = []
    for row in rows:
        d, s = row['dense'], row['sparse']
        cards.append(f'<tr><td>{html.escape(row["id"])}</td><td>{d["volume_liters_64"]:.2f}</td>'
                     f'<td>{d["opening_killed_fraction_64"]:.1%}</td>'
                     f'<td>{d["upper_medial_p10_mm_128"]:.1f}</td>'
                     f'<td>{s["upper_medial_p10_mm_128"]:.1f}</td>'
                     f'<td>{row.get("dense_iou_to_vanilla_64", float("nan")):.3f}</td></tr>')
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Dense chair thickness ablation</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;color:#1b2c38}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #cbd7df;padding:7px}a{color:#145e93}</style><h1>의자 dense 두께 loss 비교</h1><p>모든 이미지는 qd_01/seed42. `guided_zero`는 VANILLA를 끄되 두께 가중치 0인 대조군이다. 128³ medial 두께는 약 7.9 mm pitch의 voxel proxy이며, 최소 두께의 제조 보증치가 아니다.</p><table><tr><th>case</th><th>dense volume L</th><th>3³ opening loss</th><th>dense upper p10 mm</th><th>sparse upper p10 mm</th><th>dense IoU to vanilla</th></tr>'''+''.join(cards)+'''</table><img src="thickness_comparison.png"><p><a href="thickness_metrics.json">모든 수치</a></p>''')


if __name__ == '__main__':
    main()
