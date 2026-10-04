#!/usr/bin/env python3
"""Try a registered angular/truss chair image with the established BC recipe."""
from __future__ import annotations

import argparse
import json

from make_chair_domain import ROOT
import run_chair_backrest_multi_examples as study

REGISTERED = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/alternative_image_2026-10-01'
CASE = 'complex_truss_armchair'

# Reuse exactly the established visual-hull, BC, and dense/sparse recipe.
# The source case table in the original pilot lists only three named examples;
# bind its accessors to this already registered fourth image set.
study.OUT = OUT
study.image_source = lambda case: REGISTERED
study.projection_target = lambda case: REGISTERED / 'camera_projection_targets.npz'
study.input_dir = lambda case: REGISTERED / 'input_lr162'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'dense', 'sparse'))
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.stage == 'prepare':
        study.prepare(CASE)
        path = OUT / CASE / 'prototype_metrics.json'
        print(json.dumps(json.loads(path.read_text()), indent=2))
    elif args.stage == 'dense':
        study.run_dense(CASE, args.gpu, variant='pw2')
    else:
        study.run_sparse(CASE, args.gpu, bc_dilate_mm=13.0, dense_variant='pw2')


if __name__ == '__main__':
    main()
