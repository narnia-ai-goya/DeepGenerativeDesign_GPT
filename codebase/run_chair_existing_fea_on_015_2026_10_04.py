"""Repeat the image-conditioned chair generator with 15 mm simultaneous-load FEA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from make_chair_domain import ROOT
from run_chair_sparse_fea_loop_2026_10_03 import command
from run_chair_dual_load_fea_2026_10_04 import INPUT, OUT as DUAL, SPEC
from run_connectivity_qd_sampling import generation_env


OLD = DUAL.parent / 'simultaneous_load_dense_sparse_2026-10-04'
OUT = OLD / 'existing_generator_fea_on_015_2026-10-04'


def run_stage(stage: str) -> None:
    case = OUT / stage
    case.mkdir(parents=True, exist_ok=True)
    template = OLD / ('dense_dual' if stage == 'dense' else 'sparse_d1_s1') / 'config.json'
    cfg = json.loads(template.read_text())
    cfg['name'] = f'chair_existing_generator_fea_on_015_{stage}'
    m = cfg['stages']['mesh']
    m['fea_mesh_cache'] = str(SPEC / 'fea_domain/chair_015.msh')
    m['fea_mesh_size'] = .015
    m['fea_combine_loads'] = True
    if stage == 'dense':
        m['skip_sparse'] = True
        m['load_dense_cache'] = None
        m['save_dense_cache'] = str(OUT / 'dense/dense_cache.npz')
        m['fea_w'] = 1e-7
        m['sp_fea_w'] = 0.0
    else:
        cache = OUT / 'dense/dense_cache.npz'
        if not cache.exists():
            raise FileNotFoundError(cache)
        m['skip_sparse'] = False
        m['load_dense_cache'] = str(cache)
        m['save_dense_cache'] = None
        m['fea_w'] = 0.0
        m['sp_fea_w'] = 8e-5
        m['sp_fea_mode'] = 'manual'
        m['sp_fea_every'] = 5
        m['sp_fea_warmup'] = .5
        m['sp_fea_step_size'] = .01
        m['sp_fea_volume_neutral'] = True
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    env = generation_env(6)
    env.update({'VANILLA': '0', 'FEA_LOAD_MAGNITUDE': '800',
                'FEA_MAX_NODE_MAP_DISTANCE': '0.04',
                'BC_SURFACE_DIST': '1', 'BC_DIST': '0.025',
                'FEA_WORK_DIR': str(case / 'fea_work'),
                'FEA_KSP_TYPE': 'gmres', 'FEA_PC_TYPE': 'gamg',
                'FEA_KSP_RTOL': '1e-7', 'FEA_KSP_MAX_IT': '1500',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'MKL_NUM_THREADS': '1'})
    for key in ('FEA_SECOND_LOAD_STL', 'FEA_SECOND_LOAD_MAGNITUDE',
                'FEA_SECOND_LOAD_MODE'):
        env.pop(key, None)
    output = case / 'generation'
    log = case / 'generation.log'
    with log.open('w') as stream:
        proc = subprocess.run(command(config, INPUT, output), cwd=ROOT, env=env,
                              stdout=stream, stderr=subprocess.STDOUT)
    trace = log.read_text(errors='replace')
    tag = '[dense simultaneous FEA step ' if stage == 'dense' else '[sp simultaneous FEA step '
    hits = trace.count(tag)
    record = {'stage': stage, 'exit_code': proc.returncode, 'fea_steps': hits,
              'config': str(config), 'mesh': str(output / 'mesh.obj'),
              'dense_cache': str(OUT / 'dense/dense_cache.npz'),
              'log': str(log), 'fea_mesh': str(SPEC / 'fea_domain/chair_015.msh')}
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    if proc.returncode or hits < (5 if stage == 'dense' else 4):
        raise RuntimeError(f'{stage} failed; inspect {log}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('dense', 'sparse', 'both'), default='both')
    stage = parser.parse_args().stage
    for item in (('dense', 'sparse') if stage == 'both' else (stage,)):
        run_stage(item)
