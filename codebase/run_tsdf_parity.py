#!/usr/bin/env python
"""Measure existing direct-QD meshes with the fixed-frame TSDF operator."""
import json
from pathlib import Path
import numpy as np
import trimesh
from tsdf_shape_operator import mesh_tsdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/bracket/shape_qd_tsdf_parity_2026-09-21'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dom=trimesh.load(ROOT/'data_real/bracket/original_DesignSpace.stl',force='mesh')
    paths={'baseline':ROOT/'experiments/bracket/direct3ds2_lowres_2026-09-15/lr162/generation/mesh.obj',
           'pca_dense_sparse':ROOT/'experiments/bracket/shape_dqd_direct_pilot_2026-09-21/niche_04_sparse/mesh.obj',
           'anchor_support':ROOT/'experiments/bracket/shape_dqd_anchor_support_2026-09-21/niche_01_capped/mesh.obj'}
    fields={}
    for name,path in paths.items():
        print(f'TSDF {name}',flush=True)
        fields[name]=mesh_tsdf(path,dom.bounds,96)
    np.savez_compressed(OUT/'fields.npz',**fields)
    result={}
    for a in fields:
        for b in fields:
            if a<b: result[f'{a}__{b}']=float(np.mean(np.abs(fields[a]-fields[b])))
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
