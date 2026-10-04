#!/usr/bin/env python3
"""Keep only front-leg load paths; omit the artificial backrest U corridor."""
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
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'front_support_only_2026-10-02'
CASE = previous.CASE
study.OUT = OUT


def prepare():
    sub = OUT / CASE
    sub.mkdir(parents=True, exist_ok=True)
    bc = np.load(study.BC_STUDY / 'voxel.npz')
    hull, _ = study.visual_hull(CASE, bc)
    corridors = np.load(study.BC_STUDY / 'all_support_corridors.npz')['mask'].astype(bool)
    labels, n = label(corridors)
    if n != 3:
        raise RuntimeError(f'expected 3 support-corridor components, got {n}')
    # The broad component is the backrest U; the two smaller L-shaped components
    # attach the front feet to the seat. Keep only those front-foot paths.
    sizes = np.bincount(labels.ravel())
    back = int(np.argmax(sizes[1:]) + 1)
    front = corridors & (labels != back)
    np.savez_compressed(sub / 'front_support_corridors.npz', mask=front)
    mask = hull | bc['bc'].astype(bool) | front
    _, components = label(mask)
    if components != 1:
        raise RuntimeError(f'front-support prototype has {components} components')
    np.savez_compressed(sub / 'prototype.npz', prototypes=mask[None].astype(np.float32))
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1), .5)
    vertices = bc['origin'] + (vertices - .5) * bc['pitch_xyz']
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(sub / 'prototype.obj')
    source = json.loads((BASE / 'image_only_bc_2026-10-02' / CASE / 'config_dense.json').read_text())
    source['name'] = 'chair_front_support_only_complex_truss'
    m = source['stages']['mesh']
    m['load_path_mask'] = str(sub / 'front_support_corridors.npz')
    m['shape_anchor_bank'] = str(sub / 'prototype.npz')
    m['save_dense_cache'] = str(sub / 'dense_cache.npz')
    (sub / 'config_dense.json').write_text(json.dumps(source, indent=2) + '\n')
    stats = {'visual_hull_voxels': int(hull.sum()), 'front_corridor_voxels': int(front.sum()),
             'back_corridor_removed_voxels': int((labels == back).sum()),
             'prototype_voxels': int(mask.sum()), 'components': components}
    (sub / 'prototype_metrics.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'dense', 'sparse'))
    parser.add_argument('--gpu', type=int, default=5)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'dense':
        study.run_dense(CASE, args.gpu, variant='pw2')
    else:
        study.run_sparse(CASE, args.gpu, bc_dilate_mm=13.0, dense_variant='pw2')


if __name__ == '__main__':
    main()
