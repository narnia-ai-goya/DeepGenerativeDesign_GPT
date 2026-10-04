#!/usr/bin/env python3
"""Transfer the two new image-QD cells through the unchanged vanilla chair recipe."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from PIL import Image

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR,PYTHON,generation_env

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
IMAGE_QD=BASE/'single_view_image_qd_2026-10-03'
MESH_QD=BASE/'single_view_qd_2026-10-03'
TEMPLATE=MESH_QD/'image_angular_seed42/config.json'
CASES=(('tall_rect',4),('open_arch',5))


def run(item:tuple[str,int])->dict:
    name,gpu=item
    folder=MESH_QD/f'image_{name}_seed42'
    folder.mkdir(parents=True,exist_ok=True)
    source=IMAGE_QD/'images'/f'{name}.png'
    target=folder/'input_lr162/v00_front_lo.png'
    target.parent.mkdir(parents=True,exist_ok=True)
    with Image.open(source) as image:
        image.convert('RGB').resize((162,162),Image.Resampling.LANCZOS).save(target)
    (folder/'input_image.png').write_bytes(source.read_bytes())
    config=json.loads(TEMPLATE.read_text())
    config['name']=f'chair_image_qd_{name}_seed42'
    config['stages']['mesh']['save_dense_cache']=str(folder/'dense_cache.npz')
    config_path=folder/'config.json'
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    command=[str(PYTHON),str(GENERATOR),'--config',str(config_path),
             '--target-dir',str(target.parent),'--out',str(folder/'generation')]
    env=generation_env(gpu);env['VANILLA']='1'
    with (folder/'generation.log').open('w') as log:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record={'id':name,'gpu':gpu,'source_image':str(source),'input_162':str(target),
            'command':command,'VANILLA':'1','exit_code':process.returncode,
            'mesh':str(folder/'generation/mesh.obj')}
    (folder/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(record,flush=True)
    return record


if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,CASES))
    if any(r['exit_code'] for r in results):raise SystemExit(1)
