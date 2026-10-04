#!/usr/bin/env python3
"""Compare existing chair sparse candidates on the same local void and connectivity tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from make_chair_domain import ROOT
from report_chair_joint_focus_projection import BC, SOURCE, local_counts

STUDY = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = STUDY / 'joint_candidate_audit_2026-10-01'
CASES = (
    ('baseline', SOURCE / 'sparse_pw2_d13/generation/mesh.obj', SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    *[(name, STUDY / 'sparse_style_image_guidance_2026-09-29' / name / 'generation/mesh.obj',
       STUDY / 'sparse_style_image_guidance_2026-09-29' / name / 'bc_audit.json')
      for name in ('thick_3', 'thick_6', 'thick_6_proj_0p5', 'thick_6_proj_2')],
    ('no_guidance', STUDY / 'joint_stage_diagnostic_2026-09-30/unguided/generation/mesh.obj',
     STUDY / 'joint_stage_diagnostic_2026-09-30/unguided/final_audit.json'),
    *[(name, STUDY / 'thickness_sign_diagnostic_2026-09-30' / name / 'generation/mesh.obj',
       STUDY / 'thickness_sign_diagnostic_2026-09-30' / name / 'final_audit.json')
      for name in ('corrected_w3', 'corrected_w10')],
    *[(name, STUDY / 'joint_focus_projection_2026-09-30' / name / 'generation/mesh.obj',
       STUDY / 'joint_focus_projection_2026-09-30' / name / 'audit.json')
      for name in ('corrected_focus_w5', 'corrected_focus_w20')],
    *[(name, STUDY / 'direct_negative_space_2026-09-30' / name / 'generation/mesh.obj',
       STUDY / 'direct_negative_space_2026-09-30' / name / 'audit.json')
      for name in ('direct_w1', 'direct_w5', 'direct_late_w10')],
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bc = np.load(BC)
    query = (bc['origin'] + (np.indices((64, 64, 64)).reshape(3, -1).T + .5)
             * bc['pitch_xyz']).astype(np.float32)
    focus = np.load(STUDY / 'joint_focus_projection_2026-09-30/joint_focus_targets.npz')
    rows = []
    for name, mesh_path, audit_path in CASES:
        if not mesh_path.is_file():
            continue
        mesh = trimesh.load(mesh_path, force='mesh')
        audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
        local = local_counts(mesh, focus, query)
        components = audit.get('mesh_components')
        if components is None:
            components = len(mesh.split(only_watertight=False))
        row = {'name': name, 'mesh': str(mesh_path),
               'components': components,
               'bc_pass': audit.get('bc_geometry_pass'),
               'front_excess': local['front']['excess_pixels'],
               'right_excess': local['right']['excess_pixels']}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    valid = [r for r in rows if r['bc_pass'] and r['components'] == 1]
    valid.sort(key=lambda r: (r['front_excess'] + r['right_excess'], r['right_excess']))
    (OUT / 'connected_rank.json').write_text(json.dumps(valid, indent=2) + '\n')


if __name__ == '__main__':
    main()
