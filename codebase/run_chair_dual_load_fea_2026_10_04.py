"""Controlled chair study: paired seat/back FEA at Dense and Sparse stages."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from make_chair_domain import ROOT
from run_chair_sparse_fea_loop_2026_10_03 import BASE, SPEC, command
from run_chair_llm_text_round3_2026_10_03 import OUT as PARENT
from run_connectivity_qd_sampling import generation_env


OUT = PARENT / 'dual_load_dense_sparse_2026-10-04'
SIMULTANEOUS = False
SOURCE = PARENT / 'r3_narrow_outerloop'
BACK = SPEC / 'fea_domain_back'
TEMPLATE = SOURCE / 'sparse_config.json'
INPUT = SOURCE / 'input_lr162'


def make_config(stage: str, dense_dual: bool, sparse_dual: bool, case: Path) -> Path:
    cfg = json.loads(TEMPLATE.read_text())
    cfg['name'] = f'chair_dual_load_{stage}_d{int(dense_dual)}_s{int(sparse_dual)}'
    m = cfg['stages']['mesh']
    m.update({'fea_domain_dir': str(SPEC / 'fea_domain'),
              'fea_back_domain_dir': str(BACK) if ((stage == 'dense' and dense_dual) or
                                                        (stage == 'sparse' and sparse_dual)) else None,
              'fea_back_load_mode': 'y', 'fea_back_load_magnitude': 200.0,
              'fea_combine_loads': bool(SIMULTANEOUS and
                  ((stage == 'dense' and dense_dual) or (stage == 'sparse' and sparse_dual))),
              'fea_mesh_cache': str(SPEC / 'fea_domain/chair_035.msh'),
              'fea_mesh_size': .035,
              'fea_bracket_stl': str(SPEC / 'envelope.stl'),
              'fea_node_alignment': str(BASE / 'single_view_spec_2026-10-03/specification.json'),
              'bc_proper': str(SPEC / 'native_frame_spec.npz'),
              'bracket_occ': str(SPEC / 'native_frame_spec.npz'),
              'fix_stl': str(SPEC / 'fea_domain/fixed.stl'),
              'load_stl': str(SPEC / 'fea_domain/load.stl'),
              'load_mode': '-z',
              'bc_w': 5.0, 'out_w': 1.0, 'dw': 0.0,
              'dense_token_policy': 'raw',
              'sp_sdf_smooth_sigma': 3.0,
              'sp_sdf_smooth_volume_match': True})
    if stage == 'dense':
        m.update({'skip_sparse': True, 'load_dense_cache': None,
                  'save_dense_cache': str(case / 'dense_cache.npz'),
                  'fea_w': 1e-8 if dense_dual else 0.0,
                  'fea_every_n': 5, 'fea_warmup': .3,
                  'fea_penal': 2.0, 'sp_fea_w': 0.0})
    else:
        dense_case = OUT / ('dense_dual' if dense_dual else 'dense_control')
        cache = dense_case / 'dense_cache.npz'
        if not cache.exists():
            raise FileNotFoundError(cache)
        m.update({'skip_sparse': False, 'load_dense_cache': str(cache),
                  'save_dense_cache': None, 'fea_w': 0.0,
                  'sp_fea_w': 8e-5 if sparse_dual else 0.0,
                  'sp_fea_mode': 'manual', 'sp_fea_every': 5,
                  'sp_fea_warmup': .5, 'sp_fea_step_size': .01,
                  'sp_fea_volume_neutral': True, 'sparse_steps': 30})
    path = case / 'config.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    return path


def run(stage: str, dense_dual: bool, sparse_dual: bool, gpu: int) -> dict:
    name = (('dense_dual' if dense_dual else 'dense_control') if stage == 'dense' else
            f'sparse_d{int(dense_dual)}_s{int(sparse_dual)}')
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    config = make_config(stage, dense_dual, sparse_dual, case)
    output = case / 'generation'
    log = case / 'generation.log'
    prior = case / 'run.json'
    if prior.exists() and (output / 'mesh.obj').exists():
        result = json.loads(prior.read_text())
        if result['exit_code'] == 0:
            print(name, 'reuse', flush=True)
            return result
    env = generation_env(gpu)
    env.update({'VANILLA': '0', 'FEA_LOAD_MAGNITUDE': '800',
                'FEA_MAX_NODE_MAP_DISTANCE': '0.04',
                'BC_SURFACE_DIST': '1', 'BC_DIST': '0.025',
                'FEA_WORK_DIR': str(case / 'fea_work')})
    for key in ('FEA_SECOND_LOAD_STL', 'FEA_SECOND_LOAD_MAGNITUDE', 'FEA_SECOND_LOAD_MODE'):
        env.pop(key, None)
    with log.open('w') as stream:
        proc = subprocess.run(command(config, INPUT, output), cwd=ROOT, env=env,
                              stdout=stream, stderr=subprocess.STDOUT)
    trace = log.read_text(errors='replace')
    dense_tag = '[dense simultaneous FEA step ' if SIMULTANEOUS else '[dense dual FEA step '
    sparse_tag = '[sp simultaneous FEA step ' if SIMULTANEOUS else '[sp dual FEA step '
    verified = (trace.count(dense_tag) >= 5 if stage == 'dense' and dense_dual else
                trace.count(sparse_tag) == 4 if stage == 'sparse' and sparse_dual else True)
    record = {'name': name, 'stage': stage, 'dense_dual': dense_dual,
              'sparse_dual': sparse_dual, 'gpu': gpu,
              'config': str(config), 'mesh': str(output / 'mesh.obj'),
              'dense_cache': str(case / 'dense_cache.npz') if stage == 'dense' else
                             str(OUT / ('dense_dual' if dense_dual else 'dense_control') / 'dense_cache.npz'),
              'log': str(log), 'exit_code': proc.returncode,
              'fea_logs_verified': verified}
    prior.write_text(json.dumps(record, indent=2) + '\n')
    print(name, 'exit', proc.returncode, 'fea verified', verified, flush=True)
    return record


def main() -> None:
    global OUT, SIMULTANEOUS
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('dense', 'sparse'), required=True)
    p.add_argument('--simultaneous', action='store_true',
                   help='use one FEA solve with seat and back forces applied together')
    args = p.parse_args()
    SIMULTANEOUS = args.simultaneous
    if SIMULTANEOUS:
        OUT = PARENT / 'simultaneous_load_dense_sparse_2026-10-04'
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'dense':
        jobs = [('dense', False, False, 6), ('dense', True, False, 7)]
    else:
        jobs = [('sparse', False, False, 6), ('sparse', False, True, 7),
                ('sparse', True, False, 6), ('sparse', True, True, 7)]
    results = []
    for i in range(0, len(jobs), 2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results.extend(pool.map(lambda job: run(*job), jobs[i:i+2]))
    (OUT / f'{args.stage}_runs.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(x['exit_code'] != 0 or not x['fea_logs_verified'] for x in results):
        raise SystemExit('dual-load stage failed or expected FEA logs absent')


if __name__ == '__main__':
    main()
