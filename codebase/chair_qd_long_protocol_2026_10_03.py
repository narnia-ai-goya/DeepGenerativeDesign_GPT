"""Freeze the prospective image-to-3D chair QD comparison before generation."""
from __future__ import annotations

import json
from pathlib import Path

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
REFERENCE = BASE / 'text_latent_qd_2026-10-03/images/faceted__curved__standard.png'

COMMON = (
    'Edit the attached chair reference into one complete, centered FRONT orthographic studio render. '
    'Use black metallic material with realistic highlights on a pure white background. '
    'Preserve camera, scale, seat and front lip, all four legs and four feet, under-seat braces, '
    'and the solid central load-bearing backrest patch. Keep the left/right sides symmetric. '
    'Change only the upper backrest silhouette and the paired arm-to-back openings. '
    'The four feet must remain visible. No text, dimensions, extra parts, floating bars, or cut-off object.'
)

CASES = [
    {'id': 'qd_01', 'arm': 'closed', 'back': 'medium', 'target_cell': [0, 1],
     'edit': 'Make broad paired triangular arm gussets so the arm-to-back white openings are small. Make the backrest top medium width, close to the lower backrest width: clearly wider than the narrow tapered variant but narrower than the strongly flared variant. Crisp flat top and gently slanted sides.'},
    {'id': 'qd_02', 'arm': 'medium', 'back': 'wide', 'target_cell': [1, 2],
     'edit': 'Make the backrest upper edge distinctly wide and flared, nearly the full outer chair width, with angular widening sides. Show two medium-sized white arm-to-back openings: visibly larger than small gusset slots, visibly smaller than fully open arm loops. Keep a solid central backrest.'},
    {'id': 'qd_03', 'arm': 'open', 'back': 'narrow', 'target_cell': [2, 0],
     'edit': 'Make a markedly narrow tapered backrest top with straight angled sides. Move the slim arm rails outward symmetrically to create very large open triangular white spaces between each arm and the backrest. Keep the central backrest solid.'},
    {'id': 'qd_04', 'arm': 'open', 'back': 'wide', 'target_cell': [2, 2],
     'edit': 'Make a broad flared backrest upper edge, close to outer chair width, while retaining a solid central backrest. Bow thin paired arm rails outward far enough for large visible white arm-to-back openings on both sides, without changing the feet or seat.'},
    {'id': 'random_01', 'arm': 'unspecified', 'back': 'unspecified',
     'edit': 'Try a clean trapezoidal upper backrest and smooth curved arm rails while keeping the exact specification-owned base.'},
    {'id': 'random_02', 'arm': 'unspecified', 'back': 'unspecified',
     'edit': 'Try a straight architectural upper backrest with sharp corner transitions and slim angular arm rails.'},
    {'id': 'random_03', 'arm': 'unspecified', 'back': 'unspecified',
     'edit': 'Try a softly curved, wing-like upper backrest with flowing paired arm supports, keeping the chair realistic.'},
    {'id': 'random_04', 'arm': 'unspecified', 'back': 'unspecified',
     'edit': 'Try a compact faceted upper backrest and visibly structural diagonal support at each arm.'},
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        'status': 'prospective frozen protocol; candidate images and 3D results not yet generated',
        'frozen_reference': str(REFERENCE),
        'image_model': 'built-in imagegen; model identifier and deterministic seed not exposed',
        'common_prompt': COMMON,
        'cases': [{**case, 'full_prompt': COMMON + ' ' + case['edit']} for case in CASES],
        'comparison': '4 archive-targeted shape prompts vs 4 fixed non-targeted shape prompts',
        'same_3d_generator': 'Direct3D-S2, 162px FRONT input, original single-view config and seed 42',
        'composition': 'shared 64^3 BC-bearing anchor at z<=0.61m plus exact back-load voxels; remaining generated geometry',
        'quality': 'two independent 35mm linear-elastic FEM proxy solves, 800N -Z seat and 200N +Y back; lower worst normalized compliance preferred',
        'archive': {'dimensions': [3, 3], 'ranges': {'front_arm_aperture_fraction': [0, .45], 'front_upper_span_ratio': [.5, 1.2]},
                    'initial_occupied_cells': 5, 'frozen_before_new_candidates': True},
        'gates': {'seat_coverage_min': .5, 'back_coverage_min': .9, 'outside_max': .01,
                  'repair_max': .05, 'components': 1, 'source_watertight': True, 'fea_both_valid': True},
        'primary_endpoints': ['incremental occupied 3D descriptor cells from shared 5-cell archive',
                              'number of valid candidates', 'archive best worst-compliance ratio per cell'],
        'limitations': ['small 4+4 sample and unseeded image model; not proof of superiority',
                        'specification-owned base shared across all candidates',
                        'FEA of repaired voxel density proxy, not direct raw OBJ'],
    }
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(OUT / 'protocol.json')


if __name__ == '__main__':
    main()
