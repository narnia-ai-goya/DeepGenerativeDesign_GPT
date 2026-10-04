#!/usr/bin/env python3
"""Generate a side frame and central body independently from edited chair images."""
from __future__ import annotations

import argparse
import json
import subprocess

from PIL import Image

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'semantic_parts_from_image_2026-10-02'
REF = BASE / 'vanilla_direct3ds2_2026-10-02/front/config_dense.json'
CASES = {
    'side_frame': ('side_frame_imagegen.png', 'v02_right_lo'),
    'seat_back': ('seat_back_imagegen.png', 'v00_front_lo'),
}
EXTRA_VIEW = {
    'side_frame': ('side_frame_front_imagegen.png', 'v00_front_lo'),
    'seat_back': ('seat_back_right_imagegen.png', 'v02_right_lo'),
}


def prepare(case: str, two_view: bool) -> tuple[str, str, str]:
    variant = case + ('_two_view' if two_view else '')
    target = OUT / variant / 'input'
    target.mkdir(parents=True, exist_ok=True)
    # The generator reads <view>.png from --target-dir.
    low = OUT / variant / 'input_lr162'
    low.mkdir(parents=True, exist_ok=True)
    images = [CASES[case]] + ([EXTRA_VIEW[case]] if two_view else [])
    for filename, view in images:
        source = OUT / 'inputs' / filename
        with Image.open(source) as image:
            image = image.convert('RGB')
            image.resize((512, 512), Image.Resampling.LANCZOS).save(target / f'{view}.png')
            image.resize((162, 162), Image.Resampling.LANCZOS).save(low / f'{view}.png')
    views = ','.join(view for _, view in images)
    cfg = json.loads(REF.read_text())
    cfg['name'] = f'chair_semantic_part_{variant}_vanilla'
    cfg['views'] = views
    cfg['stages']['mesh'].update({
        'views': views, 'n_views': len(images),
        'cfg': 7.0, 'dense_steps': 50,
        'skip_sparse': False, 'sparse_steps': 30, 'sp_cfg': 7.0,
        'bc_w': 0.0, 'out_w': 0.0, 'dw': 0.0, 'vw': 0.0,
        'pw': 0.0, 'cw': 0.0, 'sw': 0.0, 'tw': 0.0,
        'fea_w': 0.0, 'image_proj_w': 0.0,
        'shape_anchor_w': 0.0, 'shape_scaffold_w': 0.0, 'shape_qd_w': 0.0,
        'sp_guide_w': 0.0, 'sp_guide_w_peak': 0.0, 'sp_thick_w': 0.0,
        'sp_fea_w': 0.0, 'sp_image_proj_w': 0.0,
        'sp_bc_w': 0.0, 'sp_bc_buffer_w': 0.0,
        'sp_out_w': 0.0, 'sp_design_w': 0.0, 'sp_pool_anchor_w': 0.0,
        'sp_support_halo_vox': 0,
        'force_bc_solid': False, 'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
        'dense_keep_bc_components': False,
        'dense_keep_largest_mesh_component': False,
        'load_dense_cache': None,
        'save_dense_cache': str(OUT / variant / 'dense_cache.npz'),
    })
    config = OUT / variant / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    return str(config), str(low), variant


def run(case: str, gpu: int, two_view: bool) -> None:
    config, target, variant = prepare(case, two_view)
    folder = OUT / variant
    command = [str(PYTHON), str(GENERATOR), '--config', config,
               '--target-dir', target, '--out', str(folder / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (folder / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (folder / 'run.json').write_text(json.dumps({
        'command': command, 'gpu': gpu,
        'images': [str(OUT / 'inputs' / f) for f, _ in
                   ([CASES[case]] + ([EXTRA_VIEW[case]] if two_view else []))],
        'exit_code': result.returncode,
    }, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(folder / 'generation/mesh.obj')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--two-view', action='store_true')
    args = parser.parse_args()
    run(args.case, args.gpu, args.two_view)
