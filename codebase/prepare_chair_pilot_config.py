#!/usr/bin/env python3
"""Prepare isolated chair-pilot dense and sparse configurations."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/chair/pilot_2026-09-25'
DOMAIN = ROOT / 'data_real/chair'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / 'codebase/configs/bracket.json').read_text())
    cfg.update(name='chair_neutral_pilot', domain='chair', seed=42,
               views='v00_front_lo,v02_right_lo,v_top')
    cfg['stages']['prep'].update(edge_envelope_mm=25.0, edge_peg_mm=15.0,
                                res=64, mode='uniscale', margin=.05,
                                dilate_fix=1, dilate_load=1, dilate_bracket=0)
    mesh = cfg['stages']['mesh']
    mesh.update(n_views=3, bc_w=5.0, out_w=20.0, dw=0.0, cw=0.0,
                fea_w=0.0, sp_fea_w=0.0, load_mode='-z',
                aug_lag=False, vw=0.0, vol_target=.25,
                sp_aug_lag=False, sp_vw=0.0, sp_rmin_w=0.0,
                sp_thick_w=10.0, sp_interior_w=0.0, sp_normal_fd_w=0.0,
                sp_lap_w=0.0, sp_hole_w=0.0,
                sparse_steps=30, dense_steps=50, mc_threshold=.30,
                dense_index_threshold=.10, no_grad_normalize=False,
                bc_proper=str(DOMAIN / 'voxel.npz'),
                bracket_occ=str(DOMAIN / 'voxel.npz'),
                fea_domain_dir=str(DOMAIN / 'fea_domain'),
                fea_bracket_stl=str(DOMAIN / 'original_DesignSpace.stl'),
                fea_mesh_cache=str(DOMAIN / 'fea_shared.msh'),
                fix_stl=str(DOMAIN / 'fixed_remesh.stl'),
                load_stl=str(DOMAIN / 'load_remesh.stl'),
                fea_mesh_size=.035,
                force_bc_solid=True, force_bc_dilate_mm=0.0,
                force_envelope_clip=True, restrict_active_to_envelope=True,
                restrict_dilate_vox=0.0, restrict_dilate_thin=2.0,
                fea_node_alignment=None,
                image_proj_target=None, image_proj_w=0.0,
                skip_sparse=True, load_dense_cache=None,
                save_dense_cache=str(OUT / 'dense_cache.npz'),
                views='v00_front_lo,v02_right_lo,v_top')
    cfg['stages']['post'].update(
        bracket_stl=str(DOMAIN / 'original_DesignSpace_remesh.stl'),
        fix=str(DOMAIN / 'fixed_remesh.stl'), load=str(DOMAIN / 'load_remesh.stl'),
        peg_dilate_mm=0.0, edge_mm=20.0, laplacian_iters=0)
    cfg['stages']['fea'].update(
        fix_stl=str(DOMAIN / 'fixed_remesh.stl'),
        load_stl=str(DOMAIN / 'load_remesh.stl'),
        tet_size=.030, force_dir='0,0,-1', force_N=800,
        E_GPa=10.0, nu=.30, target_edge_mm=20.0, sane_c=5.0)
    cfg['pilot_note'] = ('New chair geometry and BC; neutral rendered reference; '
                         'no in-loop FEA until independent solver preflight; '
                         'not an LMTO reproduction.')
    (OUT / 'config_dense.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
    full = json.loads(json.dumps(cfg))
    full['name'] = 'chair_neutral_pilot_sparse'
    full['stages']['mesh'].update(skip_sparse=False,
                                 load_dense_cache=str(OUT / 'dense_cache.npz'),
                                 save_dense_cache=None)
    (OUT / 'config_sparse.json').write_text(json.dumps(full, indent=2, ensure_ascii=False) + '\n')
    print(OUT / 'config_dense.json')
    print(OUT / 'config_sparse.json')


if __name__ == '__main__':
    main()
