"""One/two-voxel chair design-envelope expansion, with all BC patches fixed."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label
from skimage.measure import marching_cubes
import trimesh

from chair_qd_long_protocol_2026_10_03 import OUT as MAIN
from evaluate_chair_protected_parts_qd_2026_10_03 import ANCHOR, SPEC, cell_of, descriptor, front_mask, preview
from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402

OUT = MAIN / 'envelope_expansion_2026-10-03'
NAMES = ('qd_01', 'qd_02', 'qd_04', 'random_02', 'random_03')


def mesh_from_occ(occ: np.ndarray, origin: np.ndarray, pitch: np.ndarray) -> trimesh.Trimesh:
    vertices, faces, _, _ = marching_cubes(np.pad(occ, 1), .5)
    vertices = origin + (vertices - .5) * pitch
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = np.load(SPEC / 'voxel.npz')
    env = spec['bracket'].astype(bool)
    bc = spec['bc'].astype(bool)
    keepout = spec['keepout'].astype(bool)
    origin, pitch = spec['origin'], spec['pitch_xyz']
    z = origin[2] + (np.arange(64) + .5) * pitch[2]
    anchor = trimesh.load(ANCHOR, force='mesh', process=False)
    anchor_occ = voxel_centers_inside(anchor, 64, origin, pitch).astype(bool)
    protected = (z <= .61)[None, None, :] | spec['back_load'].astype(bool)
    structure = generate_binary_structure(3, 1)
    envelope_by = {0: env}
    for n in (1, 2):
        expanded = binary_dilation(env, structure=structure, iterations=n)
        expanded &= ~keepout
        expanded &= (z >= 0)[None, None, :]
        expanded |= bc
        envelope_by[n] = expanded
        mesh_from_occ(expanded, origin, pitch).export(OUT / f'envelope_plus_{n}_voxel.obj')
    rows = []
    for name in NAMES:
        mesh = trimesh.load(MAIN / 'mesh_cases' / name / 'aligned_main.obj', force='mesh', process=False)
        generated = voxel_centers_inside(mesh, 64, origin, pitch).astype(bool)
        candidate = (anchor_occ & protected) | (generated & ~protected)
        for n, allowed in envelope_by.items():
            realized = (candidate & allowed) | bc
            repair = float(((realized & ~candidate).sum() + (candidate & ~realized).sum()) /
                           max(1, realized.sum()))
            outside = float((candidate & ~allowed).sum() / max(1, candidate.sum()))
            components = int(label(realized, structure)[1])
            desc = descriptor(front_mask(realized))
            case = OUT / name / f'plus_{n}'
            case.mkdir(parents=True, exist_ok=True)
            final_mesh = mesh_from_occ(realized, origin, pitch)
            final_mesh.export(case / 'combined_voxel.obj')
            preview(final_mesh, case / 'preview.png')
            row = {'id': name, 'expansion_voxels': n,
                   'nominal_expansion_mm': n * float(pitch[0]) * 1000,
                   'envelope_voxels': int(allowed.sum()),
                   'outside_fraction': outside, 'repair_fraction': repair,
                   'components': components,
                   'seat_bc': float((candidate & spec['load']).sum() / spec['load'].sum()),
                   'back_bc': float((candidate & spec['back_load']).sum() / spec['back_load'].sum()),
                   'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
                   'descriptor': desc, 'cell': cell_of(desc),
                   'geometry_gate': bool(outside <= .01 and repair <= .05 and components == 1 and
                                         cell_of(desc) is not None),
                   'mesh': str(case / 'combined_voxel.obj'), 'preview': str(case / 'preview.png')}
            rows.append(row)
            print(name, 'expand', n, 'out', round(outside, 3), 'repair', round(repair, 3),
                  'cell', row['cell'], 'gate', row['geometry_gate'], flush=True)
    (OUT / 'result.json').write_text(json.dumps({
        'status': 'geometric envelope expansion pilot, no rerun of 3D generator or FEA',
        'original_spec': str(SPEC / 'voxel.npz'),
        'expansion': '6-neighbour binary dilation by 1/2 voxels; protected occupant keepout removed; z<0 removed; BC unchanged',
        'voxel_pitch_mm': float(pitch[0]) * 1000,
        'limitations': ['new envelope is voxel-domain geometry only',
                        'not a complete new native-frame spec, envelope STL, or FEA domain',
                        'do not compare old-domain FEA after expanded clipping'],
        'rows': rows}, indent=2) + '\n')
    import html
    cards = []
    for name in NAMES:
        parts = []
        for row in [r for r in rows if r['id'] == name]:
            n = row['expansion_voxels']
            parts.append(f'''<div><h3>+{n} voxel · {n*float(pitch[0])*1000:.1f} mm</h3>
<img src="{name}/plus_{n}/preview.png"><p>outside {row['outside_fraction']:.1%} · repair {row['repair_fraction']:.1%} · mass {row['mass_liters']:.1f} L · cell {row['cell']} · gate {row['geometry_gate']}</p>
<a href="{name}/plus_{n}/combined_voxel.obj">OBJ</a></div>''')
        cards.append('<article><h2>' + html.escape(name) + '</h2><section>' + ''.join(parts) + '</section></article>')
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair envelope expansion pilot</title><style>body{font:16px/1.5 system-ui;max-width:1600px;margin:25px auto;background:#eef2f5;color:#1b2c39}article{background:white;padding:18px;margin:18px 0;border-radius:12px}section{display:flex;gap:12px}section div{width:33%}img{width:100%}a{color:#145e93}</style><h1>의자 envelope 확장: BC/keepout 고정</h1><p>원래 envelope과 +1/+2 voxel(약 15.8/31.6 mm)을 비교한다. 기존 생성 메쉬를 새 허용영역으로만 재평가했으며 생성 단계나 FEA 도메인은 아직 바꾸지 않았다. 따라서 형상 feasibility 진단으로만 해석해야 한다. <a href="result.json">수치</a></p>'''+''.join(cards))


if __name__ == '__main__':
    main()
