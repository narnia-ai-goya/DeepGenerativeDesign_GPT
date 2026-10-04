"""Paired Sparse-only FEA weight sweep for the LLM chair candidate."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from run_chair_llm_text_round3_2026_10_03 import OUT as PARENT
from run_chair_sparse_fea_loop_2026_10_03 import command, execute


OUT = PARENT / 'fea_weight_sweep_2026-10-04'
SOURCE = PARENT / 'r3_narrow_outerloop'
WEIGHTS = {'off': 0.0, 'w10': 8e-8, 'w100': 8e-7,
           'w1000': 8e-6, 'w10000': 8e-5}


def one(name: str, weight: float, gpu: int) -> dict:
    case = OUT / 'round_01' / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_config.json').read_text())
    cfg['name'] = f'chair_round3_fea_weight_{name}'
    mesh = cfg['stages']['mesh']
    mesh['sp_fea_w'] = weight
    mesh['load_dense_cache'] = str(SOURCE / 'dense_cache.npz')
    mesh['save_dense_cache'] = None
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    target = SOURCE / 'input_lr162'
    output = case / 'generation'
    log = case / 'generation.log'
    result = case / 'run.json'
    if result.exists() and (output / 'mesh.obj').exists():
        saved = json.loads(result.read_text())
        if saved['exit_code'] == 0:
            print(name, 'reuse', flush=True)
            return saved
    exit_code = execute(command(config, target, output), case, log.name, gpu)
    record = {'id': name, 'sp_fea_w': weight, 'gpu': gpu,
              'config': str(config), 'mesh': str(output / 'mesh.obj'),
              'log': str(log), 'exit_code': exit_code}
    result.write_text(json.dumps(record, indent=2) + '\n')
    print(name, 'exit', exit_code, flush=True)
    return record


def main() -> None:
    folder = OUT / 'round_01'
    folder.mkdir(parents=True, exist_ok=True)
    original = json.loads((PARENT / 'round_01/selection.json').read_text())['selected'][0]
    selection = {'round': 1, 'selected': [dict(original, id=name, method=f'Sparse FEA weight {name}')
                                          for name in WEIGHTS]}
    (folder / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = list(pool.map(lambda x: one(*x), [('off', 0.0, 6), ('w10', 8e-8, 7)]))
    middle = one('w100', 8e-7, 6)
    with ThreadPoolExecutor(max_workers=2) as pool:
        high = list(pool.map(lambda x: one(*x),
                             [('w1000', 8e-6, 6), ('w10000', 8e-5, 7)]))
    records = first + [middle] + high
    (OUT / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
    if any(row['exit_code'] for row in records):
        raise SystemExit('See individual generation logs')


if __name__ == '__main__':
    main()
