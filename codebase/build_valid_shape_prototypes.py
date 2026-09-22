#!/usr/bin/env python
"""Freeze condition-compatible bracket shape prototypes from verified result.json files."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

from run_shape_qd_representation_pilot import ROOT, load_mesh, voxel_grid


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'experiments/bracket/shape_qd_valid_bank_2026-09-21')
    ap.add_argument('--niches',type=int,default=6)
    ap.add_argument('--resolution',type=int,default=64)
    ap.add_argument('--seed',type=int,default=20260921)
    a=ap.parse_args(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    dom=load_mesh(ROOT/'data_real/bracket/original_DesignSpace.stl'); lo,hi=dom.bounds
    bc=voxel_grid(load_mesh(ROOT/'data_real/bracket/fixed.stl'),lo,hi,a.resolution)
    bc|=voxel_grid(load_mesh(ROOT/'data_real/bracket/load.stl'),lo,hi,a.resolution)
    rows=[]; fields=[]
    seen=set()
    for rp in sorted((ROOT/'experiments/bracket').glob('**/result.json')):
        try: r=json.loads(rp.read_text()); mp=Path(r['mesh'])
        except Exception: continue
        if not r.get('valid') or not mp.exists() or str(mp.resolve()) in seen: continue
        try:
            m=load_mesh(mp)
            if not m.is_watertight: continue
            f=voxel_grid(m,lo,hi,a.resolution); f[bc]=False
            if f.sum()<64: continue
        except Exception: continue
        seen.add(str(mp.resolve())); fields.append(f.astype(np.float32)); rows.append({'mesh':str(mp.resolve()),'result':str(rp.resolve()),'compliance_J':r.get('compliance_J'),'voxels':int(f.sum())})
    if len(fields)<a.niches: raise RuntimeError(f'only {len(fields)} valid meshes')
    x=np.stack(fields); # Morphology feature: 16³ average occupancy, not a saturated silhouette.
    feat=x.reshape(len(x),16,4,16,4,16,4).mean((2,4,6)).reshape(len(x),-1)
    km=KMeans(n_clusters=a.niches,n_init=64,random_state=a.seed).fit(feat)
    prototypes=[]; members=[]
    for k in range(a.niches):
        ids=np.flatnonzero(km.labels_==k); center=km.cluster_centers_[k]
        medoid=ids[np.argmin(np.linalg.norm(feat[ids]-center,axis=1))]
        prototypes.append(x[medoid]); members.append({'niche':k,'count':int(len(ids)),'medoid':int(medoid),'mesh':rows[medoid]['mesh'],'members':[int(i) for i in ids]})
    np.savez_compressed(out/'valid_shape_prototypes.npz',prototypes=np.stack(prototypes),bc_mask=bc,domain_bounds=np.stack([lo,hi]),labels=km.labels_,features=feat)
    (out/'bank.json').write_text(json.dumps({'resolution':a.resolution,'niches':a.niches,'valid_meshes':len(rows),'records':rows,'niche_members':members},indent=2)+'\n')
    print(out/'valid_shape_prototypes.npz'); print(json.dumps({'valid_meshes':len(rows),'members':[m['count'] for m in members]},indent=2))
if __name__=='__main__': main()
