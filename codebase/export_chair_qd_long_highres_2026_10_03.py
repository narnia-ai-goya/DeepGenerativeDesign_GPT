"""Export 128^3 display meshes for strict-eligible long chair QD candidates."""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage.measure import marching_cubes
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE, OUT as MAIN
from evaluate_chair_protected_parts_qd_2026_10_03 import ANCHOR, SPEC

sys.path.insert(0, str(BASE.parent.parent.parent / 'codebase/code'))
sys.path.insert(0, str(BASE.parent.parent.parent / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


def upsample(mask: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(np.repeat(mask.astype(bool), 2, axis=0), 2, axis=1), 2, axis=2)


def export(out) -> None:
    report = json.loads((out / 'result.json').read_text())
    spec = np.load(SPEC / 'voxel.npz')
    origin, pitch = spec['origin'], spec['pitch_xyz'] / 2
    anchor = trimesh.load(ANCHOR, force='mesh', process=False)
    base = voxel_centers_inside(anchor, 128, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(128) + .5) * pitch[2]
    protected = (z <= .61)[None, None, :] | upsample(spec['back_load'])
    env, bc = upsample(spec['bracket']), upsample(spec['bc'])
    manifest = []
    for row in report['rows']:
        if not row['strict_eligible']:
            continue
        name = row['id']
        mesh = trimesh.load(row['aligned_mesh'], force='mesh', process=False)
        generated = voxel_centers_inside(mesh, 128, origin, pitch).astype(bool)
        candidate = (base & protected) | (generated & ~protected)
        final = (candidate & env) | bc
        vertices, faces, _, _ = marching_cubes(np.pad(final, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        folder = out / 'mesh_cases' / name
        path = folder / 'combined_highres128.obj'
        surface.export(path)
        canvas = Image.new('RGB', (1220, 610), '#f6f8fa')
        for i, azim in enumerate((0, 90)):
            target = np.array([0., .01, .46])
            eye, up = camera_from_elev_azim(target, 2., 15, azim)
            rgb = render_lit(surface, eye, target, up, size=600, fit_extent=.56,
                             color=(.64, .69, .73))
            canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 610, 5))
        preview = folder / 'combined_highres128_preview.png'
        canvas.save(preview)
        item = {'id': name, 'display_mesh': str(path), 'preview': str(preview),
                'faces': int(len(faces)), 'components': int(label(final)[1]),
                'fea_note': 'FEA in result.json uses separate 64^3 voxel density, not this display mesh'}
        manifest.append(item)
        print(name, 'highres faces', len(faces), 'components', item['components'], flush=True)
    (out / 'highres128_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    index = out / 'index.html'
    page = index.read_text()
    for item in manifest:
        name = item['id']
        page = page.replace(f'mesh_cases/{name}/combined_preview.png',
                            f'mesh_cases/{name}/combined_highres128_preview.png')
        page = page.replace(f'href="mesh_cases/{name}/combined_voxel.obj">OBJ',
                            f'href="mesh_cases/{name}/combined_highres128.obj">128³ display OBJ')
    index.write_text(page)


if __name__ == '__main__':
    export(MAIN)
    followup = MAIN / 'feedback_round_02'
    if (followup / 'result.json').exists():
        export(followup)
