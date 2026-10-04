"""Generate fresh LLM-authored chair candidates; never select old pool images."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from PIL import Image

from run_chair_sparse_fea_loop_2026_10_03 import BASE, SPEC, TEMPLATE, command, execute


OUT = BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03/llm_text_proposal_round_03_2026-10-03'


def run(row: dict, stage: str, gpu: int) -> dict:
    case = OUT / row['id']
    case.mkdir(exist_ok=True)
    target = case / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(exist_ok=True)
    with Image.open(row['image']) as source:
        source.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    cache = case / 'dense_cache.npz'
    cfg = json.loads(TEMPLATE.read_text())
    cfg['name'] = f"chair_llm_round3_{row['id']}_{stage}"
    mesh = cfg['stages']['mesh']
    if stage == 'dense':
        output = case / 'dense_generation'
        mesh.update({'skip_sparse': True, 'save_dense_cache': str(cache), 'load_dense_cache': None})
    else:
        if not cache.exists():
            raise FileNotFoundError(f'missing Dense cache: {cache}')
        output = case / 'sparse_generation'
        mesh.update({
            'skip_sparse': False, 'load_dense_cache': str(cache), 'save_dense_cache': None,
            'bc_proper': str(SPEC / 'native_frame_spec.npz'),
            'bracket_occ': str(SPEC / 'native_frame_spec.npz'),
            'fea_domain_dir': str(SPEC / 'fea_domain'),
            'fea_bracket_stl': str(SPEC / 'envelope.stl'),
            'fea_mesh_cache': str(SPEC / 'fea_domain/chair_035.msh'),
            'fea_node_alignment': str(BASE / 'single_view_spec_2026-10-03/specification.json'),
            'fix_stl': str(SPEC / 'fea_domain/fixed.stl'),
            'load_stl': str(SPEC / 'fea_domain/load.stl'),
            'sp_fea_w': 8e-9, 'sp_fea_mode': 'manual',
            'sp_fea_every': 5, 'sp_fea_warmup': .5,
            'sp_fea_steepness': 4., 'sp_fea_step_size': .01,
            'sp_fea_volume_neutral': True,
            'sp_sdf_smooth_sigma': 3., 'sp_sdf_smooth_volume_match': True,
            'sp_guide_w': 0., 'sp_lap_w': 0., 'fea_w': 0., 'sparse_steps': 30,
        })
    config = case / f'{stage}_config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    log = case / f'{stage}_generation.log'
    result = case / f'{stage}_run.json'
    mesh_path = output / 'mesh.obj'
    if result.exists() and mesh_path.exists():
        saved = json.loads(result.read_text())
        trace = log.read_text(errors='replace')
        valid = ('[skip_sparse]' in trace if stage == 'dense' else
                 trace.count('[sp FEA step ') == 4 and '[sparse SDF smooth]' in trace)
        if saved['exit_code'] == 0 and valid:
            print(row['id'], stage, 'reuse', flush=True)
            return saved
    code = execute(command(config, target.parent, output), case, log.name, gpu)
    record = {'id': row['id'], 'stage': stage, 'gpu': gpu,
              'image': row['image'], 'config': str(config), 'mesh': str(mesh_path),
              'dense_cache': str(cache), 'exit_code': code}
    result.write_text(json.dumps(record, indent=2) + '\n')
    print(row['id'], stage, 'exit', code, flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('dense', 'sparse'), required=True)
    parser.add_argument('--ids', default='')
    args = parser.parse_args()
    screen = json.loads((OUT / 'image_screening.json').read_text())
    rows = [r for r in screen if r['functional_image_gate']]
    if args.ids:
        selected = set(args.ids.split(','))
        rows = [r for r in rows if r['id'] in selected]
    results = []
    for start in range(0, len(rows), 2):
        batch = rows[start:start + 2]
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            results.extend(pool.map(lambda pair: run(pair[0], args.stage, pair[1]),
                                    zip(batch, (6, 7))))
    (OUT / f'{args.stage}_runs.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(r['exit_code'] for r in results):
        raise SystemExit(f'{args.stage} generation failed')


if __name__ == '__main__':
    main()
