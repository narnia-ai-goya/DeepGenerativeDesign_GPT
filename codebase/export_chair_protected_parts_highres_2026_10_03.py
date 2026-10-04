"""Export 128^3 display meshes for the protected-parts chair QD pilot."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'text_reasoned_front_axes_2026-10-03/protected_parts_qd'
SPEC = BASE / 'single_view_spec_2026-10-03'


def upsample(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(mask.astype(bool), 2, axis=0), 2, axis=1), 2, axis=2)


def main():
    report = json.loads((OUT / 'result.json').read_text())
    spec = np.load(SPEC / 'voxel.npz')
    origin, pitch = spec['origin'], spec['pitch_xyz'] / 2
    anchor = trimesh.load(report['fixed_anchor'], force='mesh', process=False)
    base = voxel_centers_inside(anchor, 128, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(128) + .5) * pitch[2]
    protected = (z <= report['cut_z_m'])[None, None, :] | upsample(spec['back_load'])
    env, bc = upsample(spec['bracket']), upsample(spec['bc'])
    results = []
    for row in report['rows']:
        name = row['id']
        mesh = trimesh.load(row['generated_mesh'], force='mesh', process=False)
        generated = voxel_centers_inside(mesh, 128, origin, pitch).astype(bool)
        candidate = (base & protected) | (generated & ~protected)
        final = (candidate & env) | bc
        vertices, faces, _, _ = marching_cubes(np.pad(final, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        case = OUT / 'mesh_cases' / name
        path = case / 'combined_highres128.obj'
        surface.export(path)
        canvas = Image.new('RGB', (1220, 610), '#f6f8fa')
        for i, azim in enumerate((0, 90)):
            target = np.array([0., .01, .46])
            eye, up = camera_from_elev_azim(target, 2., 15, azim)
            rgb = render_lit(surface, eye, target, up, size=600, fit_extent=.56,
                             color=(.64, .69, .73))
            canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 610, 5))
        canvas.save(case / 'combined_highres128_preview.png')
        repair = float(((final & ~candidate).sum() + (candidate & ~final).sum()) / final.sum())
        item = {'id': name, 'display_mesh': str(path), 'faces': int(len(faces)),
                'components': int(label(final)[1]), 'repair_fraction': repair,
                'seat_bc': float((candidate & upsample(spec['load'])).sum() / upsample(spec['load']).sum()),
                'back_bc': float((candidate & upsample(spec['back_load'])).sum() / upsample(spec['back_load']).sum()),
                'fea_note': 'FEA in result.json uses a separate 64^3 voxel representation'}
        results.append(item)
        print(name, 'faces', len(faces), 'components', item['components'],
              'seat/back', round(item['seat_bc'], 3), round(item['back_bc'], 3), flush=True)
    (OUT / 'highres128_manifest.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
