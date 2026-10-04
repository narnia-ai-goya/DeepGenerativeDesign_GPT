#!/usr/bin/env python3
"""Test whether reusing the top-only dense cache preserves the input openings."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
from pathlib import Path

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


EXP = ROOT / 'experiments/bracket/angular_image_qd_loop_2026-09-24'
OUT = EXP / 'stage_fidelity_2026-09-24/dense_cache_reuse_ablation'
CASES = {
    'triangular_truss': (EXP / 'round_00/triangular_truss', 0),
    'staggered_chevron': (EXP / 'round_01/staggered_chevron', 5),
}


def run(name: str, source: Path, gpu: int) -> dict:
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config_full.json').read_text())
    config['name'] = f'angular_cache_reuse_{name}'
    mesh = config['stages']['mesh']
    mesh['load_dense_cache'] = str((source / 'dense_top_cache.npz').resolve())
    mesh['save_dense_cache'] = None
    mesh['skip_sparse'] = False
    config_path = case / 'config_full.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n')
    generation = case / 'generation'
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config_path),
           '--target-dir', str(source / 'input'), '--out', str(generation)]
    env = generation_env(gpu)
    env['FEA_MAX_NODE_MAP_DISTANCE'] = '0.02'
    with (case / 'generation.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return {'name': name, 'source': str(source), 'case': str(case),
            'config': str(config_path), 'exit_code': result.returncode,
            'generation_mesh': str(generation / 'mesh.obj')}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, name, source, gpu) for name, (source, gpu) in CASES.items()]
        rows = [future.result() for future in futures]
    (OUT / 'run_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    for row in rows:
        print(row, flush=True)


if __name__ == '__main__':
    main()
