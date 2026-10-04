#!/usr/bin/env python3
"""Chair pilot: explicit straight MMC components as a dense-stage shape prior.

This is a fixed component layout followed by the existing dense optimizer, not
an MMC topology-optimization solver. FEA is disabled for this geometry pilot.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env

SOURCE = ROOT / "experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced"
BC = ROOT / "experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz"
OUT = ROOT / "experiments/chair/sofa_style_2026-09-28/mmc_dense_pilot_2026-09-29"


def bar(world: np.ndarray, start: tuple[float, float, float],
        end: tuple[float, float, float], width: float, depth: float) -> np.ndarray:
    """Oriented rectangular prism with a straight centerline and square ends."""
    p, q = np.asarray(start), np.asarray(end)
    axis = (q - p) / np.linalg.norm(q - p)
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < .85 else np.array([0., 1., 0.])
    side = np.cross(axis, reference); side /= np.linalg.norm(side)
    third = np.cross(axis, side)
    local = world - (p + q) / 2
    return ((np.abs(local @ axis) <= np.linalg.norm(q-p)/2) &
            (np.abs(local @ side) <= width/2) &
            (np.abs(local @ third) <= depth/2))


def build() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(BC)
    ijk = np.indices((64,64,64)).reshape(3,-1).T
    world = data["origin"] + (ijk + .5) * data["pitch_xyz"]
    mask = np.zeros(len(world), dtype=bool)
    components = []

    def add(name, start, end, width, depth):
        nonlocal mask
        mask |= bar(world, start, end, width, depth)
        components.append(dict(name=name,start=start,end=end,width=width,depth=depth))

    for x in (-.23,.23):
        for y in (-.21,.21):
            add(f"leg_{x:+.2f}_{y:+.2f}",(x,y,.035),(x,y,.495),.070,.070)
    # Flat seat and straight front/side rails. The panel is a component too.
    add("seat_panel",(-.22,0,.477),(.22,0,.477),.455,.045)
    add("front_apron",(-.23,-.21,.445),(.23,-.21,.445),.070,.075)
    for x in (-.23,.23):
        add(f"side_apron_{x:+.2f}",(x,-.21,.447),(x,.21,.447),.065,.070)
        add(f"back_post_{x:+.2f}",(x,.21,.47),(x,.21,.865),.065,.070)
    add("back_top",(-.23,.21,.83),(.23,.21,.83),.085,.075)
    add("back_lower",(-.23,.21,.56),(.23,.21,.56),.060,.060)

    mask = mask.reshape(64,64,64)
    # Existing BC regions retain their exact occupancy and remain mandatory.
    mask |= data["bc"].astype(bool)
    mask &= (data["bracket"].astype(bool) | data["bc"].astype(bool))
    bank = OUT / "mmc_scaffold.npz"
    np.savez_compressed(bank, prototypes=mask[None].astype(np.float32))
    vertices, faces, _, _ = marching_cubes(np.pad(mask,1),.5)
    vertices = data["origin"] + (vertices-.5)*data["pitch_xyz"]
    trimesh.Trimesh(vertices=vertices,faces=faces,process=False).export(OUT / "mmc_scaffold.obj")
    (OUT / "components.json").write_text(json.dumps(components,indent=2)+"\n")
    return bank


def run(gpu: int=2, anchor_w: float=500., scaffold_w: float=80., variant: str='dense'):
    bank=build()
    output=OUT/variant
    config=json.loads((SOURCE / "config_dense_pw2.json").read_text())
    config["name"]="chair_mmc_component_anchor_dense_pilot"
    config["stages"]["mesh"].update(shape_anchor_bank=str(bank),
        shape_anchor_w=anchor_w,shape_scaffold_w=scaffold_w,
        save_dense_cache=str(OUT / f"{variant}_cache.npz"),load_dense_cache=None,
        skip_sparse=True,fea_w=0.,sp_fea_w=0.)
    config_path=OUT / f"config_{variant}.json"
    config_path.write_text(json.dumps(config,indent=2)+"\n")
    command=[str(PYTHON),str(GENERATOR),"--config",str(config_path),
             "--target-dir",str(SOURCE / "input_lr162"),"--out",str(output)]
    with (OUT / f"{variant}.log").open("w") as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),stdout=log,stderr=subprocess.STDOUT)
    (OUT / f"{variant}_run.json").write_text(json.dumps({"command":command,
        "exit_code":result.returncode,"scaffold":str(OUT / "mmc_scaffold.obj"),
        "dense":str(output / "mesh_dense.obj"),"anchor_w":anchor_w,
        "scaffold_w":scaffold_w},indent=2)+"\n")
    return result.returncode


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--anchor-w',type=float,default=500.)
    parser.add_argument('--scaffold-w',type=float,default=80.)
    parser.add_argument('--variant',default='dense')
    args=parser.parse_args()
    raise SystemExit(run(args.gpu,args.anchor_w,args.scaffold_w,args.variant))
