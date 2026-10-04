#!/usr/bin/env python3
"""Build a small, realization-aware chair QD archive from measured meshes.

Functional-interface preservation is a shared non-design constraint on the
evaluation voxel grid.  Every candidate pays an explicit repair-cost budget;
this is a pilot voxel realization, not a claim that the source OBJ itself is
fully BC-compliant or that it is a finished high-resolution mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'single_view_qd_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
CASES = ['seed_42','seed_43','seed_44','seed_45',
         'guided_43_b2_o2_active','guided_44_b5_o2_active','guided_45_b5_o2_active',
         'image_angular_seed42','image_wing_seed42']
DESCRIPTOR_RANGES = {'side_open_fraction': [.60, .96],
                     'backrest_taper_ratio': [.75, 1.05]}
ARCHIVE_DIMS = [3, 3]
REPAIR_LIMIT = .05


def features(occ: np.ndarray, spec: np.lib.npyio.NpzFile) -> tuple[float,float]:
    x,y,z = [spec['origin'][i]+(np.arange(64)+.5)*spec['pitch_xyz'][i] for i in range(3)]
    side = occ.any(axis=0)
    opening = 1.0 - side[np.ix_((y>=-.16)&(y<=.12), (z>=.61)&(z<=.76))].mean()
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    widths=[]
    for lo,hi in ((.70,.76),(.84,.90)):
        patch=occ&(Z>=lo)&(Z<hi)&(Y>.12)
        xx=X[patch]
        widths.append(float(np.percentile(xx,95)-np.percentile(xx,5)) if len(xx) else 0.)
    taper=widths[1]/widths[0] if widths[0] else 0.
    return float(opening), float(taper)


def cell_for(pair: tuple[float,float]) -> list[int] | None:
    result=[]
    for value,(lo,hi),n in zip(pair,DESCRIPTOR_RANGES.values(),ARCHIVE_DIMS):
        if not lo<=value<=hi:return None
        result.append(min(n-1,int((value-lo)/(hi-lo)*n)))
    return result


def evaluate() -> list[dict]:
    spec=np.load(SPEC/'voxel.npz')
    env=spec['bracket'].astype(bool);bc=spec['bc'].astype(bool)
    structure=np.zeros((3,3,3),bool);structure[1,1,1]=1
    for axis in range(3):
        for sign in (-1,1):
            offset=[1,1,1];offset[axis]+=sign;structure[tuple(offset)]=1
    rows=[]
    for name in CASES:
        case=OUT/name
        if not (case/'occupancy.npz').exists():continue
        original=np.load(case/'occupancy.npz')['occupied'].astype(bool)
        realized=(original&env)|bc
        added=int((realized&~original).sum());removed=int((original&~realized).sum())
        repair_fraction=(added+removed)/max(1,int(realized.sum()))
        labels,n=label(realized,structure=structure)
        counts=np.bincount(labels[labels>0])
        one_component=n==1
        pair=features(realized,spec)
        measured=json.loads((case/'metrics.json').read_text())
        quality_gate=(one_component and repair_fraction<=REPAIR_LIMIT and
                      measured['watertight'] and cell_for(pair) is not None)
        mass=float(realized.sum()*np.prod(spec['pitch_xyz'])*1000)
        vertices,faces,_,_=marching_cubes(np.pad(realized,1),.5)
        vertices=spec['origin']+(vertices-.5)*spec['pitch_xyz']
        realized_mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
        realized_mesh.export(case/'realized_voxel.obj')
        realized_mesh.export(case/'realized_voxel.glb')
        row={**measured,'realized_mass_liters':mass,
             'realized_voxel_mesh':str(case/'realized_voxel.obj'),
             'realized_side_open_fraction':pair[0],
             'realized_backrest_taper_ratio':pair[1],
             'realized_cell':cell_for(pair),
             'realized_components_6conn':int(n),
             'largest_component_voxel_fraction':float(counts.max()/counts.sum()),
             'bc_covered_after_preservation':bool(np.all(realized[bc])),
             'added_functional_voxels':added,'removed_outside_voxels':removed,
             'repair_fraction':repair_fraction,'geometry_gate':bool(quality_gate)}
        np.savez_compressed(case/'realized_occupancy.npz',occupied=realized)
        (case/'qd_metrics.json').write_text(json.dumps(row,indent=2)+'\n')
        rows.append(row)
    (OUT/'geometry_archive.json').write_text(json.dumps({
        'archive_dims':ARCHIVE_DIMS,'descriptor_ranges':DESCRIPTOR_RANGES,
        'repair_limit':REPAIR_LIMIT,'evaluated':len(rows),
        'geometry_feasible':sum(r['geometry_gate'] for r in rows),
        'rows':rows},indent=2)+'\n')
    return rows


if __name__=='__main__':
    rows=evaluate()
    for r in rows:
        print(r['id'],r['realized_cell'],r['geometry_gate'],
              'repair',round(r['repair_fraction'],3),
              'components',r['realized_components_6conn'],
              'mass_L',round(r['realized_mass_liters'],2))
