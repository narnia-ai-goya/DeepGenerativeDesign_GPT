#!/usr/bin/env python3
"""Render and score the chair sparse pooling ablation."""
from __future__ import annotations

import shutil

from make_chair_domain import ROOT
import report_chair_joint_focus_projection as report

OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_pooling_2026-10-01'
FOCUS = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30'
report.OUT = OUT
report.ROWS = (
    ('baseline: early soft pool', report.SOURCE / 'sparse_pw2_d13/generation/mesh.obj',
     report.SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    *[(name, OUT / name / 'generation/mesh.obj', OUT / name / 'audit.json')
      for name in ('no_pool', 'soft_long', 'hard_long', 'anchor_long')],
)


def main():
    shutil.copy2(FOCUS / 'joint_focus_targets.npz', OUT / 'joint_focus_targets.npz')
    shutil.copy2(FOCUS / 'focus_pixels.png', OUT / 'focus_pixels.png')
    report.main()
    index = OUT / 'index.html'
    page = index.read_text().replace('Chair A/B local negative-space guidance',
                                     'Chair sparse pooling ablation')
    page = page.replace('Only image projection guidance changes.',
        'Only sparse pooling strategy changes. All variants use the same dense cache, seed, BC, envelope, sparse guidance, and 30 steps.')
    index.write_text(page)


if __name__ == '__main__':
    main()
