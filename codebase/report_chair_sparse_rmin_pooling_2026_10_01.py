#!/usr/bin/env python3
"""Render and score the sparse r_min max-pooling ablation."""
from __future__ import annotations

import shutil

from make_chair_domain import ROOT
import report_chair_joint_focus_projection as report

STUDY = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = STUDY / 'sparse_rmin_pooling_2026-10-01'
FOCUS = STUDY / 'joint_focus_projection_2026-09-30'
SIGN = STUDY / 'thickness_sign_diagnostic_2026-09-30/corrected_w10'
report.OUT = OUT
report.ROWS = (
    ('original baseline (legacy sign)', report.SOURCE / 'sparse_pw2_d13/generation/mesh.obj',
     report.SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    ('corrected-sign control, rmin off', SIGN / 'generation/mesh.obj', SIGN / 'final_audit.json'),
    *[(name, OUT / name / 'generation/mesh.obj', OUT / name / 'audit.json')
      for name in ('solid_k2_w1', 'dual_k2_w1', 'dual_k4_w1', 'dual_k2_w3')],
)


def main():
    shutil.copy2(FOCUS / 'joint_focus_targets.npz', OUT / 'joint_focus_targets.npz')
    shutil.copy2(FOCUS / 'focus_pixels.png', OUT / 'focus_pixels.png')
    report.main()
    index = OUT / 'index.html'
    page = index.read_text().replace('Chair A/B local negative-space guidance',
                                     'Chair sparse r_min morphological pooling')
    page = page.replace('Only image projection guidance changes.',
        'The r_min loss pools the 512-sparse SDF onto a 128-cubed grid and applies max-pool erosion/dilation. All new cases use the negative-inside sign and the same dense cache, seed, BC, envelope, and FEA-off settings.')
    index.write_text(page)


if __name__ == '__main__':
    main()
