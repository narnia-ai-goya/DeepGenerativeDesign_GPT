#!/usr/bin/env python3
"""Render and score the sparse decoded-SDF mean-pool anchor sweep."""
from __future__ import annotations

import shutil

from make_chair_domain import ROOT
import report_chair_joint_focus_projection as report

STUDY = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = STUDY / 'sparse_mean_anchor_2026-10-01'
FOCUS = STUDY / 'joint_focus_projection_2026-09-30'
PILOT = STUDY / 'sparse_pooling_2026-10-01/anchor_long'
report.OUT = OUT
report.ROWS = (
    ('baseline anchor w0', report.SOURCE / 'sparse_pw2_d13/generation/mesh.obj',
     report.SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    ('prior anchor w10', PILOT / 'generation/mesh.obj', PILOT / 'audit.json'),
    *[(name, OUT / name / 'generation/mesh.obj', OUT / name / 'audit.json')
      for name in ('anchor_w50_k4', 'anchor_w200_k4',
                   'anchor_w100_k8', 'anchor_w100_k2')],
)


def main():
    shutil.copy2(FOCUS / 'joint_focus_targets.npz', OUT / 'joint_focus_targets.npz')
    shutil.copy2(FOCUS / 'focus_pixels.png', OUT / 'focus_pixels.png')
    report.main()
    index = OUT / 'index.html'
    page = index.read_text().replace('Chair A/B local negative-space guidance',
                                     'Chair sparse SDF mean-pool anchor sweep')
    page = page.replace('Only image projection guidance changes.',
        'Only decoded sparse SDF mean-pool anchor weight and block-size schedule change. Latent pooling, dense cache, seed, BC, envelope and FEA-off settings are held fixed.')
    index.write_text(page)


if __name__ == '__main__':
    main()
