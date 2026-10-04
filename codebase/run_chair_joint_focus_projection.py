#!/usr/bin/env python3
"""Designer-marked negative-space projection guidance for chair A/B joints."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from pysdf import SDF
from scipy.ndimage import binary_dilation

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30'
ORIGINAL_TARGET = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/camera_projection_targets.npz'
CASES = (('focus_w5', 5.0, 0), ('focus_w20', 20.0, 1))


def prepare_target():
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(ORIGINAL_TARGET)
    arrays = {k: source[k] for k in source.files}
    bc = np.load(BC_STUDY / 'voxel.npz')
    baseline = trimesh.load(SOURCE / 'sparse_pw2_d13/generation/mesh.obj', force='mesh')
    sdf = SDF(baseline.vertices.astype(np.float32), baseline.faces.astype(np.uint32))
    ijk = np.indices((64, 64, 64)).reshape(3, -1).T
    world = (bc['origin'] + (ijk + .5) * bc['pitch_xyz']).astype(np.float32)
    occupancy = (sdf(world) > 0).reshape(64, 64, 64).astype(np.float32)
    focal = {}
    regions = {
        'front': [(18, 26, 36, 47), (38, 46, 36, 47)],
        'right': [(36, 45, 34, 45)],
    }
    diagnostics = {}
    for name, boxes in regions.items():
        camera = torch.from_numpy(arrays[f'camera_grid_{name}'].astype(np.float32))[None]
        projected = F.grid_sample(torch.from_numpy(occupancy)[None, None],
                                  camera, mode='bilinear', align_corners=True)[0, 0].max(0).values.numpy()
        target = arrays[f'target_{name}']
        roi = np.zeros((64, 64), bool)
        for x0, x1, y0, y1 in boxes:
            roi[y0:y1, x0:x1] = True
        false_positive = (projected > .2) & (target < .25) & roi
        focus = binary_dilation(false_positive, iterations=1) & (target < .25) & roi
        if focus.sum() < 2:
            raise RuntimeError(f'{name} has too few negative-space pixels: {focus.sum()}')
        # ImageProjectionLoss clips weights to [0,1] and normalizes by their sum.
        # Keep a weak global reference while making the selected void pixels dominant.
        arrays[f'weight_{name}'] = np.where(
            focus, 1.0, .005 * arrays[f'weight_{name}']).astype(np.float32)
        arrays[f'focus_{name}'] = focus.astype(np.uint8)
        focal[name] = focus
        diagnostics[name] = {'roi_boxes_xyxy': boxes,
                             'false_positive_pixels': int(false_positive.sum()),
                             'focused_empty_pixels': int(focus.sum())}
    path = OUT / 'joint_focus_targets.npz'
    np.savez_compressed(path, **arrays)
    # Diagnostic image: gray target material, red current excess, orange weighted empty pixels.
    sheet = Image.new('RGB', (128 * len(focal), 128), 'white')
    for i, name in enumerate(focal):
        target = arrays[f'target_{name}']
        im = np.full((64, 64, 3), 255, np.uint8)
        im[target > .5] = (90, 100, 110)
        im[focal[name]] = (239, 119, 23)
        sheet.paste(Image.fromarray(im).resize((128, 128), Image.Resampling.NEAREST),
                    (128 * i, 0))
    sheet.save(OUT / 'focus_pixels.png')
    (OUT / 'target_audit.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    return path


def run_one(row, target):
    name, weight, gpu, *options = row
    correct_inside = bool(options[0]) if options else False
    case = OUT / name
    case.mkdir(exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_joint_projection_' + name
    cfg['stages']['mesh'].update(image_proj_target=str(target),
                                sp_image_proj_w=weight,
                                sp_sdf_inside_low=correct_inside,
                                fea_w=0.0, sp_fea_w=0.0)
    path = case / 'config.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(path),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'weight': weight, 'gpu': gpu,
              'correct_inside': correct_inside,
              'exit_code': result.returncode, 'command': cmd,
              'mesh': str(case / 'generation/mesh.obj')}
    if result.returncode == 0:
        record['audit'] = audit(case / 'generation/mesh.obj',
                                BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz')
        (case / 'audit.json').write_text(json.dumps(record['audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    target = prepare_target()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda row: run_one(row, target), CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'bc_pass': r.get('audit', {}).get('bc_geometry_pass'),
                       'components': r.get('audit', {}).get('mesh_components')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
