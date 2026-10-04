#!/usr/bin/env python3
"""Controlled open-arm dense test with only the upper arm/back junction extended."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'local_arm_envelope_2026-10-02'
SOURCE = BASE / 'open_arm'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
CONFIG = BASE / 'fresh_dense_no_prototype_2026-10-02/open_arm/image_strong/config_dense.json'
ENVELOPE_STL = BASE / 'coherent_proxy_dense/envelope.stl'


def _project(mask: np.ndarray, grid: np.ndarray) -> np.ndarray:
    field = torch.from_numpy(mask.astype(np.float32))[None, None]
    with torch.no_grad():
        sampled = F.grid_sample(field, torch.from_numpy(grid.astype(np.float32))[None],
                                mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
    return sampled.amax(dim=0).numpy() > .5


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    src = np.load(BC)
    data = {k: np.array(src[k]) for k in src.files}
    old = data['bracket'].astype(bool)
    origin = data['origin']
    pitch = data['pitch_xyz']
    x, y, z = np.meshgrid(*[origin[i] + (np.arange(64) + .5) * pitch[i]
                             for i in range(3)], indexing='ij')
    # The side-view missing pixels lie at Y=.10-.15, Z=.785-.833 m: the upper
    # arm-to-back transition, not the front of the armrest. BC is unchanged.
    addition = ((np.abs(x) >= .19) & (np.abs(x) <= .28) &
                (y >= .075) & (y <= .16) & (z >= .72) & (z <= .86))
    data['bracket'] = old | addition
    data['design'] = data['bracket'] & ~data['bc'].astype(bool)
    if not np.array_equal(data['bc'], src['bc']):
        raise AssertionError('BC changed')
    if not np.all(data['bc'] <= data['bracket']):
        raise AssertionError('BC outside expanded envelope')
    np.savez_compressed(OUT / 'voxel.npz', **data)

    envelope = trimesh.load(ENVELOPE_STL, force='mesh')
    boxes = []
    for side in (-1, 1):
        box = trimesh.creation.box(extents=(.09, .085, .14))
        box.apply_translation((side * .235, .1175, .79))
        boxes.append(box)
    expanded = trimesh.boolean.union([envelope, *boxes], engine='manifold')
    if not isinstance(expanded, trimesh.Trimesh) or not expanded.is_watertight:
        raise RuntimeError('expanded envelope STL union failed')
    expanded.export(OUT / 'envelope.stl')

    targets = np.load(SOURCE / 'camera_projection_targets.npz')
    scores = {}
    sheet = Image.new('RGB', (3 * 512, 2 * 512), 'white')
    draw = ImageDraw.Draw(sheet)
    for col, view in enumerate(('front', 'right', 'top')):
        target = targets[f'target_{view}'] > .5
        grid = targets[f'camera_grid_{view}']
        for row, mask in enumerate((old, data['bracket'])):
            variant = 'old' if row == 0 else 'expanded'
            projected = _project(mask, grid)
            outside = target & ~projected
            scores[f'{view}_{variant}'] = {
                'target_pixels': int(target.sum()), 'outside_pixels': int(outside.sum()),
                'outside_fraction': round(float(outside.sum() / max(target.sum(), 1)), 4)}
            rgb = np.full((*target.shape, 3), 255, dtype=np.uint8)
            rgb[target] = (67, 91, 105)
            rgb[outside] = (225, 62, 62)
            image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST)
            sheet.paste(image, (col * 512, row * 512))
            draw.text((col * 512 + 12, row * 512 + 12),
                      f'{view} {variant}: {scores[f"{view}_{variant}"]["outside_fraction"]:.1%} outside',
                      fill='black')
    sheet.save(OUT / 'envelope_feasibility_comparison.png')
    metrics = {'old_voxels': int(old.sum()), 'expanded_voxels': int(data['bracket'].sum()),
               'added_voxels': int((data['bracket'] & ~old).sum()),
               'grid_fraction_old': round(float(old.mean()), 4),
               'grid_fraction_expanded': round(float(data['bracket'].mean()), 4),
               'bc_voxels': int(data['bc'].sum()), 'silhouette_feasibility': scores}
    (OUT / 'envelope_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    return metrics


def run(gpu: int):
    case = OUT / 'open_arm_image_strong'
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    cfg['name'] = 'chair_open_arm_local_upper_junction_envelope'
    cfg['stages']['mesh'].update({
        'bracket_occ': str(OUT / 'voxel.npz'), 'bc_proper': str(OUT / 'voxel.npz'),
        'fea_bracket_stl': str(OUT / 'envelope.stl'),
        'save_dense_cache': str(case / 'dense_cache.npz'), 'load_dense_cache': None,
    })
    cfg['stages']['post']['bracket_stl'] = str(OUT / 'envelope.stl')
    (case / 'config_dense.json').write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(case / 'config_dense.json'),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'generation.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (case / 'run.json').write_text(json.dumps({'command': cmd, 'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    report = audit(case / 'generation/mesh_dense.obj', OUT / 'voxel.npz')
    (case / 'bc_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'mesh': str(case / 'generation/mesh_dense.obj'), 'bc_pass': report['bc_geometry_pass'],
            'components': report['mesh_components']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(), indent=2), flush=True)
    if not args.prepare_only:
        print(json.dumps(run(args.gpu), indent=2), flush=True)


if __name__ == '__main__':
    main()
