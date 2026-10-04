#!/usr/bin/env python3
"""Compare legacy, unguided, and sign-corrected sparse chair surfaces."""
from __future__ import annotations

from make_chair_domain import ROOT
import report_chair_joint_stage_diagnostic as report

OLD = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_stage_diagnostic_2026-09-30'
NEW = ROOT / 'experiments/chair/sofa_style_2026-09-28/thickness_sign_diagnostic_2026-09-30'
report.OUT = NEW
report.ROWS = (
    ('legacy guided: pre-refiner', OLD / 'guided/generation/mesh_pre_refiner_world.obj'),
    ('legacy guided: final', OLD / 'guided/generation/mesh.obj'),
    ('no guidance: final', OLD / 'unguided/generation/mesh.obj'),
    ('corrected sign w10: pre-refiner', NEW / 'corrected_w10/generation/mesh_pre_refiner_world.obj'),
    ('corrected sign w10: final', NEW / 'corrected_w10/generation/mesh.obj'),
    ('corrected sign w3: final', NEW / 'corrected_w3/generation/mesh.obj'),
)


def main():
    report.main()
    index = NEW / 'index.html'
    page = index.read_text()
    page = page.replace('Same dense cache, seed, 30 sparse steps, 13 mm BC and envelope. Guided uses sp_guide_w=10; unguided uses zero. Both include the original BC hard constraint and use FEA off.',
        'Same dense cache, seed, 30 sparse steps, 13 mm BC and envelope. Legacy and corrected runs differ only in sparse SDF inside/outside convention and indicated thickness weight; the unguided control sets guidance weight to zero. FEA off throughout.')
    index.write_text(page)


if __name__ == '__main__':
    main()
