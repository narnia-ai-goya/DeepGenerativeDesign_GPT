#!/usr/bin/env python3
"""Matched Direct3D-S2 generation for every text-latent QD and random image."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from PIL import Image

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR,PYTHON,generation_env
from chair_text_latent_qd_pilot_2026_10_03 import OUT,BASE

TEMPLATE=BASE/'single_view_qd_2026-10-03/image_angular_seed42/config.json'


def generate(row:dict,gpu:int)->dict:
    case=OUT/'mesh_cases'/row['id'];case.mkdir(parents=True,exist_ok=True)
    source=OUT/'images'/f"{row['id']}.png"
    target=case/'input_lr162/v00_front_lo.png';target.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as image:
        image.convert('RGB').resize((162,162),Image.Resampling.LANCZOS).save(target)
    config=json.loads(TEMPLATE.read_text())
    config['name']='chair_text_latent_qd_'+row['id']
    config['stages']['mesh']['save_dense_cache']=str(case/'dense_cache.npz')
    config_path=case/'config.json';config_path.write_text(json.dumps(config,indent=2)+'\n')
    command=[str(PYTHON),str(GENERATOR),'--config',str(config_path),
             '--target-dir',str(target.parent),'--out',str(case/'generation')]
    env=generation_env(gpu);env['VANILLA']='1'
    with (case/'generation.log').open('w') as stream:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    result={'id':row['id'],'method':row['method'],'round':row['round'],
            'gpu':gpu,'command':command,'exit_code':process.returncode,
            'mesh':str(case/'generation/mesh.obj'),'source_image':str(source),
            'input_162':str(target),'VANILLA':'1'}
    (case/'run.json').write_text(json.dumps(result,indent=2)+'\n')
    print(row['id'],'exit',process.returncode,flush=True)
    return result


if __name__=='__main__':
    rows=json.loads((OUT/'observations.json').read_text())
    results=[]
    for pair in (rows[:2],rows[2:]):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results.extend(pool.map(lambda args:generate(*args),zip(pair,(4,5))))
    (OUT/'generation_results.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['exit_code'] for r in results):raise SystemExit(1)
