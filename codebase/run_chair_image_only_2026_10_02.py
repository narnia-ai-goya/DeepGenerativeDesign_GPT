#!/usr/bin/env python3
"""Remove straight support-corridor priors while retaining all chair BC regions."""
from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh
from scipy.ndimage import label
from skimage.measure import marching_cubes

from make_chair_domain import ROOT
import run_chair_alternative_image_2026_10_01 as previous

study = previous.study
SOURCE = previous.OUT / previous.CASE
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/image_only_bc_2026-10-02'
CASE = previous.CASE
study.OUT = OUT


def prepare():
    sub = OUT / CASE
    sub.mkdir(parents=True, exist_ok=True)
    bc = np.load(study.BC_STUDY / 'voxel.npz')
    hull, projected = study.visual_hull(CASE, bc)
    mask = hull | bc['bc'].astype(bool)
    labels, n = label(mask)
    sizes = sorted(np.bincount(labels[labels > 0]).tolist(), reverse=True)
    if n != 1:
        raise RuntimeError(f'image+BC support has {n} connected components: {sizes[:8]}')
    np.savez_compressed(sub / 'prototype.npz', prototypes=mask[None].astype(np.float32))
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1), .5)
    verts = bc['origin'] + (verts - .5) * bc['pitch_xyz']
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(sub / 'prototype.obj')
    cfg = json.loads((SOURCE / 'config_dense.json').read_text())
    cfg['name'] = 'chair_image_only_bc_complex_truss'
    m = cfg['stages']['mesh']
    m['load_path_mask'] = None
    m['shape_anchor_bank'] = str(sub / 'prototype.npz')
    m['save_dense_cache'] = str(sub / 'dense_cache.npz')
    (sub / 'config_dense.json').write_text(json.dumps(cfg, indent=2) + '\n')
    metrics = {
        'image_hull_voxels': int(hull.sum()),
        'mandatory_bc_voxels': int(bc['bc'].sum()),
        'prototype_voxels': int(mask.sum()),
        'outside_image_hull_voxels': int((mask & ~hull).sum()),
        'components': n,
        'component_sizes': sizes,
        'per_view_projected_voxels': projected,
    }
    (sub / 'prototype_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'dense', 'sparse'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'dense':
        study.run_dense(CASE, args.gpu, variant='pw2')
    else:
        study.run_sparse(CASE, args.gpu, bc_dilate_mm=13.0, dense_variant='pw2')


if __name__ == '__main__':
    main()
