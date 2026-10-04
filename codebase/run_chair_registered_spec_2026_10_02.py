#!/usr/bin/env python3
"""Test chair specification guidance after mapping fixed physical masks to D3D-S2 frame."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'registered_spec_2026-10-02'
PHYSICAL = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
REFERENCE = BASE / 'minimal_direct3ds2_2026-10-02'
IMAGE = BASE / 'open_arm/input_lr162'
CASES = {
    'gentle': (5.0, 1.0),
    'medium': (20.0, 2.0),
    'strong': (100.0, 5.0),
}


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(PHYSICAL)
    raw = trimesh.load(REFERENCE / 'generation/mesh_dense_raw.obj', force='mesh')
    envelope = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    source_center = raw.bounds.mean(axis=0)
    physical_center = envelope.bounds.mean(axis=0)
    rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    scale = 0.94
    origin, pitch = d['origin'], d['pitch_xyz']
    axes = [origin[i] + (np.arange(64) + .5) * pitch[i] for i in range(3)]
    native = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    phys = (native - source_center) @ rotation.T * scale + physical_center
    index = ((phys - origin) / pitch - 0.5).transpose(3, 0, 1, 2)
    fields = {k: np.asarray(d[k]) for k in d.files if k not in
              {'bracket', 'design', 'bc', 'fix', 'load', 'back_load'}}
    for key in ('bracket', 'fix', 'load', 'back_load'):
        fields[key] = (map_coordinates(d[key].astype(np.uint8), index,
                                       order=0, mode='constant', cval=0) > 0)
    fields['bc'] = fields['fix'] | fields['load'] | fields['back_load']
    fields['bracket'] |= fields['bc']
    fields['design'] = fields['bracket'] & ~fields['bc']
    np.savez_compressed(OUT / 'native_frame_spec.npz', **fields)
    calibration = {
        'physical_spec': str(PHYSICAL),
        'native_frame_spec': str(OUT / 'native_frame_spec.npz'),
        'native_to_physical': {
            'source_center_m': source_center.tolist(),
            'physical_center_m': physical_center.tolist(),
            'rotation_x_degrees': 90,
            'uniform_scale': scale,
        },
        'voxel_counts': {k: {'physical': int(np.asarray(d[k]).sum()),
                             'native': int(fields[k].sum())}
                         for k in ('bracket', 'bc', 'fix', 'load', 'back_load')},
        'sampling': 'native 64^3 center -> physical center -> nearest physical mask voxel',
        'model_training': False,
        'fea': False,
    }
    (OUT / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    print(json.dumps(calibration, indent=2), flush=True)


def run(names: list[str], gpu: int) -> None:
    baseline = json.loads((REFERENCE / 'config.json').read_text())
    for name in names:
        out = OUT / name
        out.mkdir(exist_ok=True)
        cfg = json.loads(json.dumps(baseline))
        cfg['name'] = f'chair_registered_spec_{name}'
        m = cfg['stages']['mesh']
        m.update({
            'bracket_occ': str(OUT / 'native_frame_spec.npz'),
            'bc_proper': str(OUT / 'native_frame_spec.npz'),
            'out_w': CASES[name][0], 'bc_w': CASES[name][1],
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
            'save_dense_cache': str(out / 'dense_cache.npz'),
            'load_dense_cache': None,
        })
        config = out / 'config.json'
        config.write_text(json.dumps(cfg, indent=2) + '\n')
        command = [str(PYTHON), str(GENERATOR), '--config', str(config),
                   '--target-dir', str(IMAGE), '--out', str(out / 'generation')]
        with (out / 'generation.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                    stdout=log, stderr=subprocess.STDOUT)
        (out / 'run.json').write_text(json.dumps({'command': command,
            'exit_code': result.returncode, 'weights': CASES[name]}, indent=2) + '\n')
        print(name, result.returncode, out / 'generation/mesh_dense_raw.obj', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


def prepare_native_stls() -> dict[str, str]:
    config = json.loads((OUT / 'strong/config.json').read_text())['stages']['mesh']
    calibration = json.loads((OUT / 'calibration.json').read_text())['native_to_physical']
    source = np.asarray(calibration['source_center_m'])
    center = np.asarray(calibration['physical_center_m'])
    rotation = Rotation.from_euler('x', calibration['rotation_x_degrees'], degrees=True).as_matrix()
    scale = float(calibration['uniform_scale'])
    result = {}
    for key, source_stl in [('fix_stl', config['fix_stl']),
                            ('load_stl', config['load_stl']),
                            ('fea_bracket_stl', config['fea_bracket_stl'])]:
        mesh = trimesh.load(source_stl, force='mesh')
        mesh.vertices = (mesh.vertices - center) @ rotation / scale + source
        target = OUT / ('native_frame_' + key + '.stl')
        mesh.export(target)
        result[key] = str(target)
    (OUT / 'native_stls.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def sparse(gpu: int, variant: str) -> None:
    name = 'strong_sparse_' + variant
    out = OUT / name
    out.mkdir(exist_ok=True)
    cfg = json.loads((OUT / 'strong/config.json').read_text())
    cfg['name'] = f'chair_registered_spec_{name}'
    m = cfg['stages']['mesh']
    m.update({
        'load_dense_cache': str(OUT / 'strong/dense_cache.npz'),
        'save_dense_cache': None,
        'skip_sparse': False,
        'sparse_steps': 30,
        'sp_cfg': 7.0,
        'sp_guide_w': 0.0,
        'sp_guide_w_peak': 0.0,
        'sp_fea_w': 0.0,
        'sp_image_proj_w': 0.0,
        'sp_bc_w': 0.0,
        'sp_bc_buffer_w': 0.0,
        'sp_out_w': 0.0,
        'sp_design_w': 0.0,
        'sp_pool_anchor_w': 0.0,
        'sp_thick_w': 0.0,
        'sp_support_halo_vox': 0,
        'force_bc_solid': False,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
    })
    if variant in ('soft_bc', 'soft_sign', 'soft_sign_bc_only',
                   'soft_sign_bc15', 'soft_sign_bc_halo1', 'soft_sign_bc_halo2',
                   'soft_halo1_fix13'):
        m.update({'sp_guide_w': 5.0, 'sp_guide_w_peak': 5.0,
                  'sp_bc_w': 5.0, 'sp_out_w': 2.0})
        if variant != 'soft_bc':
            m['sp_sdf_inside_low'] = True
        if variant == 'soft_sign_bc_only':
            m['sp_out_w'] = 0.0
        elif variant in ('soft_sign_bc15', 'soft_sign_bc_halo1',
                         'soft_sign_bc_halo2', 'soft_halo1_fix13'):
            m['sp_out_w'] = 0.0
            m['sp_bc_w'] = 15.0
        if variant in ('soft_sign_bc_halo1', 'soft_sign_bc_halo2', 'soft_halo1_fix13'):
            masks = np.load(OUT / 'native_frame_spec.npz')
            bank = OUT / 'bc_active_halo_bank.npz'
            np.savez_compressed(bank, prototypes=masks['bc'][None].astype(np.float32))
            m.update({'shape_anchor_bank': str(bank), 'shape_qd_target': 0,
                      'shape_anchor_expand_vox': 2 if variant == 'soft_sign_bc_halo2' else 1,
                      'shape_anchor_max_added': 0,
                      'shape_qd_w': 0.0, 'shape_anchor_w': 0.0,
                      'sp_shape_qd_w': 0.0, 'sp_shape_anchor_w': 0.0})
        if variant == 'soft_halo1_fix13':
            m.update(prepare_native_stls())
            m.update({'force_bc_solid': True, 'force_bc_fix_only': True,
                      'force_envelope_clip': True, 'force_bc_dilate_mm': 13.0,
                      'force_load_dilate_mm': 6.0})
    elif variant in ('hard_bc', 'hard_bc_fix13'):
        m.update(prepare_native_stls())
        m.update({'force_bc_solid': True, 'force_envelope_clip': True,
                  'force_bc_dilate_mm': 13.0 if variant == 'hard_bc_fix13' else 6.0,
                  'force_load_dilate_mm': 6.0})
    config = out / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(IMAGE), '--out', str(out / 'generation')]
    with (out / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (out / 'run.json').write_text(json.dumps({'command': command,
        'exit_code': result.returncode, 'dense_source': str(OUT / 'strong/dense_cache.npz')}, indent=2) + '\n')
    print(name, result.returncode, out / 'generation/mesh.obj', flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'sparse'))
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--variant', choices=('vanilla', 'soft_bc', 'soft_sign',
                                              'soft_sign_bc_only', 'soft_sign_bc15',
                                              'soft_sign_bc_halo1', 'soft_sign_bc_halo2',
                                              'soft_halo1_fix13',
                                              'hard_bc', 'hard_bc_fix13'), default='vanilla')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        if not (OUT / 'native_frame_spec.npz').exists():
            prepare()
        run(args.cases, args.gpu)
    else:
        sparse(args.gpu, args.variant)


if __name__ == '__main__':
    main()
