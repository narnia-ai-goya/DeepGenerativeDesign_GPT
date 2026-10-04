#!/usr/bin/env python3
"""Compare direct sparse empty-space guidance against the chair baseline."""
from __future__ import annotations

import shutil

from make_chair_domain import ROOT
import report_chair_joint_focus_projection as report

DIRECT = ROOT / 'experiments/chair/sofa_style_2026-09-28/direct_negative_space_2026-09-30'
FOCUS = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30'
report.OUT = DIRECT
report.ROWS = (
    ('baseline', report.SOURCE / 'sparse_pw2_d13/generation/mesh.obj',
     report.SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    ('direct empty-space w1', DIRECT / 'direct_w1/generation/mesh.obj',
     DIRECT / 'direct_w1/audit.json'),
    ('direct empty-space w5', DIRECT / 'direct_w5/generation/mesh.obj',
     DIRECT / 'direct_w5/audit.json'),
    ('direct late w10', DIRECT / 'direct_late_w10/generation/mesh.obj',
     DIRECT / 'direct_late_w10/audit.json'),
)


def main():
    shutil.copy2(FOCUS / 'joint_focus_targets.npz', DIRECT / 'joint_focus_targets.npz')
    shutil.copy2(FOCUS / 'focus_pixels.png', DIRECT / 'focus_pixels.png')
    report.main()
    index = DIRECT / 'index.html'
    page = index.read_text().replace('Only image projection guidance changes.',
        'The image-empty pixels are back-projected into the design envelope and penalized directly on the 512³ sparse decoder SDF; the baseline lacks this loss.')
    index.write_text(page)


if __name__ == '__main__':
    main()
