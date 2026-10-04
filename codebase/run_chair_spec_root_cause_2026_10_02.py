#!/usr/bin/env python3
"""Controlled one-view chair dense ablation of envelope and BC gradients."""
from __future__ import annotations

import argparse
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'spec_root_cause_2026-10-02'
SOURCE = BASE / 'open_arm/input_lr162'
REF = BASE / 'minimal_direct3ds2_2026-10-02/config.json'
CASES = {
    'out_only': {'out_w': 100.0, 'bc_w': 0.0},
    'bc_only': {'out_w': 0.0, 'bc_w': 5.0},
    'out_bc': {'out_w': 100.0, 'bc_w': 5.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    base = json.loads(REF.read_text())
    for name in args.cases:
        case = OUT / name
        case.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base))
        cfg['name'] = f'chair_spec_root_cause_{name}'
        mesh = cfg['stages']['mesh']
        mesh.update(CASES[name])
        mesh.update({
            'skip_sparse': True,
            'cfg': 7.0,
            'image_proj_w': 0.0,
            'pw': 0.0,
            'cw': 0.0,
            'fea_w': 0.0,
            'load_path_mask': None,
            'force_bc_solid': False,
            'force_envelope_clip': False,
            'restrict_active_to_envelope': False,
            'save_dense_cache': str(case / 'dense_cache.npz'),
            'load_dense_cache': None,
        })
        config = case / 'config.json'
        config.write_text(json.dumps(cfg, indent=2) + '\n')
        command = [str(PYTHON), str(GENERATOR), '--config', str(config),
                   '--target-dir', str(SOURCE), '--out', str(case / 'generation')]
        with (case / 'generation.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=generation_env(args.gpu),
                                    stdout=log, stderr=subprocess.STDOUT)
        (case / 'run.json').write_text(json.dumps({
            'command': command, 'exit_code': result.returncode,
            'baseline': str(REF), 'changes': CASES[name],
        }, indent=2) + '\n')
        print(name, result.returncode, case / 'generation/mesh_dense_raw.obj', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
