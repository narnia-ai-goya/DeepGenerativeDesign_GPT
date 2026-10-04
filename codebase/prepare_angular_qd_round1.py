#!/usr/bin/env python3
"""Prepare the adaptive off-axis image candidate using the established angular recipe."""
import json
from pathlib import Path

from PIL import Image


ROOT = Path('/home/goya/SDL/3d_qd/experiments/bracket/angular_image_qd_loop_2026-09-24')
SOURCE = Path('/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654/exec-0b997904-eea0-494a-9828-ee057ec2c7b2.png')
OLD = ROOT / 'round_00/triangular_truss'
NEW = ROOT / 'round_01/off_axis_spine'


def main() -> None:
    target = NEW / 'input'
    target.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE).convert('RGB')
    image.save(target / 'source_full.png')
    image.resize((512, 512), Image.Resampling.LANCZOS).save(target / '그림1.png')
    for kind in ('dense_top', 'full'):
        old = OLD / f'config_{kind}.json'
        config = json.loads(old.read_text())
        body = json.dumps(config, ensure_ascii=False, indent=2)
        body = body.replace(str(OLD), str(NEW)).replace('triangular_truss', 'off_axis_spine')
        (NEW / f'config_{kind}.json').write_text(body + '\n')
    manifest = {
        'round': 1,
        'candidate': 'off_axis_spine',
        'parent_shape_niche': 'angular baseline; motivated by round-0 regular truss/X-brace layouts',
        'source_reference': '/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/angular_full.png',
        'source_generated': str(SOURCE),
        'prompt': 'Edit the supplied engineering reference into a new, distinctly different bracket concept image for a 3D shape-diversity experiment. Preserve the exact locations, sizes, and colors of all four red fixed-boundary-condition bores and both green loaded bores. Show a single oblique 3D metallic bracket render on a clean white background, same camera and scale as reference, with no text or labels. Geometry should be intentionally angular: create a long off-axis zigzag spine connecting the loaded region to the four anchors, with asymmetric polygonal cutouts and sharply chamfered straight-edged ribs. Make the outer perimeter locally tapered and faceted, but retain robust material around every colored bore and continuous load paths. Avoid smooth organic forms, rounded decorative ribs, fractured walls, extra holes at anchor collars, or detached components. This concept should be visibly distinct from a regular triangular truss and from a centered X-brace.',
    }
    (NEW / 'candidate.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(NEW)


if __name__ == '__main__':
    main()
