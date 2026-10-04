"""Reuse each chair's dense cache and run sparse denoising with in-step FEA."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from PIL import Image

from chair_qd_long_protocol_2026_10_03 import BASE
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env


PARENT = BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
OUT = PARENT / 'sparse_fea_loop_2026-10-03'
SPEC = BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03/envelope_plus16_spec_2026-10-03'
TEMPLATE = BASE / 'single_view_qd_2026-10-03/image_angular_seed42/config.json'


def cache_from_previous(name: str) -> Path | None:
    for number in (1, 2):
        previous = PARENT / f'round_{number:02d}' / name / 'dense_cache.npz'
        if previous.exists():
            return previous
    return None


def command(config: Path, input_dir: Path, output: Path) -> list[str]:
    return [str(PYTHON), str(GENERATOR), '--config', str(config),
            '--target-dir', str(input_dir), '--out', str(output)]


def execute(cmd: list[str], folder: Path, filename: str, gpu: int) -> int:
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    env['FEA_LOAD_MAGNITUDE'] = '800'
    # 35 mm tetrahedra and a registered 15.8 mm voxel grid have a 33.2 mm
    # p99 nearest-map distance. Keep the bound just above this verified value.
    env['FEA_MAX_NODE_MAP_DISTANCE'] = '0.04'
    env['BC_SURFACE_DIST'] = '1'
    env['BC_DIST'] = '0.025'
    with (folder / filename).open('w') as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return proc.returncode


def run(row: dict, round_number: int, gpu: int) -> dict:
    name = row['id']
    case = OUT / f'round_{round_number:02d}' / name
    case.mkdir(parents=True, exist_ok=True)
    prior = case / 'run.json'
    log = case / 'generation.log'
    if prior.exists() and (case / 'generation/mesh.obj').exists() and log.exists():
        saved = json.loads(prior.read_text())
        log_text = log.read_text(errors='replace')
        if saved.get('exit_code') == 0 and log_text.count('[sp FEA step ') >= 4 and '[sp FEA] FAIL' not in log_text:
            print(name, 'sparse FEA already complete; reuse', flush=True)
            return saved
    target = case / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(row['image']) as image:
        image.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    dense_cache = cache_from_previous(name)
    if dense_cache is None:
        dense_cache = case / 'dense_cache.npz'
        dense_config = json.loads(TEMPLATE.read_text())
        dense_config['name'] = f'chair_sparse_fea_loop_dense_{name}'
        dense_config['stages']['mesh'].update({
            'skip_sparse': True, 'save_dense_cache': str(dense_cache),
            'load_dense_cache': None})
        cfg = case / 'dense_config.json'
        cfg.write_text(json.dumps(dense_config, indent=2) + '\n')
        exit_code = execute(command(cfg, target.parent, case / 'dense_generation'),
                            case, 'dense_generation.log', gpu)
        if exit_code or not dense_cache.exists():
            result = {'id': name, 'method': row['method'], 'gpu': gpu, 'stage': 'dense cache',
                      'exit_code': exit_code or 99, 'dense_cache': str(dense_cache)}
            (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
            print(name, 'dense failed', exit_code, flush=True)
            return result
    config = json.loads(TEMPLATE.read_text())
    config['name'] = f'chair_sparse_fea_loop_r{round_number}_{name}'
    mesh = config['stages']['mesh']
    mesh.update({
        'load_dense_cache': str(dense_cache), 'save_dense_cache': None,
        'bc_proper': str(SPEC / 'native_frame_spec.npz'),
        'bracket_occ': str(SPEC / 'native_frame_spec.npz'),
        'fea_domain_dir': str(SPEC / 'fea_domain'),
        'fea_bracket_stl': str(SPEC / 'envelope.stl'),
        'fea_mesh_cache': str(SPEC / 'fea_domain/chair_035.msh'),
        'fea_node_alignment': str(BASE / 'single_view_spec_2026-10-03/specification.json'),
        'fix_stl': str(SPEC / 'fea_domain/fixed.stl'),
        'load_stl': str(SPEC / 'fea_domain/load.stl'),
        # Prior chair sparse-FEA used 3e-12 with the solver's legacy 42,300 N
        # default. Compliance gradients scale as force squared; 8e-9 keeps a
        # comparable update scale at the actual 800 N chair seat load.
        'sp_fea_w': 8e-9, 'sp_fea_mode': 'manual',
        'sp_fea_every': 5, 'sp_fea_warmup': .5,
        'sp_fea_steepness': 4., 'sp_fea_step_size': .01,
        'sp_fea_volume_neutral': True,
        'sp_guide_w': 0., 'sp_lap_w': 0.,
        'fea_w': 0., 'sparse_steps': 30,
    })
    cfg = case / 'config.json'
    cfg.write_text(json.dumps(config, indent=2) + '\n')
    exit_code = execute(command(cfg, target.parent, case / 'generation'),
                        case, 'generation.log', gpu)
    result = {'id': name, 'method': row['method'], 'round': round_number,
              'gpu': gpu, 'source_image': row['image'], 'dense_cache': str(dense_cache),
              'config': str(cfg), 'mesh': str(case / 'generation/mesh.obj'),
              'exit_code': exit_code}
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'sparse FEA generation exit', exit_code, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    parser.add_argument('--pilot', type=str, default='')
    args = parser.parse_args()
    selection = json.loads((OUT / f'round_{args.round:02d}/selection.json').read_text())['selected']
    if args.pilot:
        selection = [r for r in selection if r['id'] == args.pilot]
        if not selection: raise SystemExit('pilot id absent from frozen selection')
    rows = []
    for start in range(0, len(selection), 2):
        batch = selection[start:start + 2]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            rows.extend(pool.map(lambda pair: run(pair[0], args.round, pair[1]),
                                 zip(batch, (6, 7))))
    path = OUT / f'round_{args.round:02d}/generation_results.json'
    path.write_text(json.dumps(rows, indent=2) + '\n')
    if any(r['exit_code'] for r in rows):
        raise SystemExit('Sparse FEA generation failed; see case logs')


if __name__ == '__main__':
    main()
