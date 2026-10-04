"""Dense-only thickness-weight ablation on one fixed chair image and seed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from chair_qd_long_protocol_2026_10_03 import OUT as MAIN
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

OUT = MAIN / 'dense_thickness_ablation'
SOURCE = MAIN / 'qd_01'
CASES = [('hard_0p01', .01), ('hard_0p05', .05), ('hard_0p20', .20)]


def run(item: tuple[str, float], gpu: int) -> dict:
    name, weight = item
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    cfg['name'] = 'chair_dense_thickness_' + name
    mesh = cfg['stages']['mesh']
    mesh['tw_hard'] = weight
    mesh['tw_soft'] = 0.0
    mesh['tw_rmin'] = 0.0
    mesh['load_dense_cache'] = None
    mesh['save_dense_cache'] = str(folder / 'dense_cache.npz')
    config = folder / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(folder / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (folder / 'generation.log').open('w') as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = {'id': name, 'dense_tw_hard': weight, 'gpu': gpu, 'exit_code': proc.returncode,
           'image': str(MAIN / 'qd_01.png'), 'config': str(config),
           'dense_mesh': str(folder / 'generation/mesh_dense_raw.obj'),
           'sparse_mesh': str(folder / 'generation/mesh.obj'), 'command': command}
    (folder / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, 'exit', proc.returncode, flush=True)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'protocol.json').write_text(json.dumps({
        'status': 'exploratory dense thickness ablation',
        'baseline': str(SOURCE / 'config.json'),
        'fixed_image': str(MAIN / 'qd_01.png'),
        'same_seed': 42,
        'single_changed_parameter': 'stages.mesh.tw_hard',
        'weights': [0.0] + [value for _, value in CASES],
        'dense_voxel_pitch_mm': 15.8125,
        'note': '3x3x3 dense opening-based loss targets approximately one-voxel features; sparse recipe unchanged. Parameter does not impose a hard physical minimum thickness.'}, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda pair: run(*pair), zip(CASES, (0, 1, 2))))
    (OUT / 'generation_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit('Some dense thickness runs failed')


if __name__ == '__main__':
    main()
