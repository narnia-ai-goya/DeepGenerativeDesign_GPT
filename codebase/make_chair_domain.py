#!/usr/bin/env python3
"""Build a neutral, single-load chair design envelope and physical BC solids.

Coordinates are metres: X left/right, Y front/back (+Y is back), Z up.
This is a new chair *pilot* domain, not a reproduction of LMTO's Fig. C.13.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_real/chair'


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(x1-x0, y1-y0, z1-z0))
    mesh.apply_translation(((x0+x1)/2, (y0+y1)/2, (z0+z1)/2))
    return mesh


def foot(x: float, y: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=.055, height=.070, sections=64)
    mesh.apply_translation((x, y, .035))
    return mesh


def export_mesh(mesh: trimesh.Trimesh, name: str) -> dict:
    path = OUT / name
    mesh.export(path)
    return {'path': str(path), 'bounds_m': mesh.bounds.tolist(),
            'volume_litres': float(mesh.volume * 1000), 'watertight': bool(mesh.is_watertight),
            'components': len(mesh.split(only_watertight=False)),
            'vertices': len(mesh.vertices), 'faces': len(mesh.faces)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Broad material permission below the seat. QD/TO may remove most of it.
    underseat = box(-.27, .27, -.26, .26, .025, .460)
    # Continuous usable seat surface; only its central patch is non-design material.
    seat = box(-.27, .27, -.26, .26, .430, .500)
    # High rear volume and two side volumes allow semantic back/wing/arm variants.
    back = box(-.27, .27, .145, .28, .455, .920)
    left_wing = box(-.28, -.19, -.15, .23, .465, .710)
    right_wing = box(.19, .28, -.15, .23, .465, .710)
    feet = [foot(x, y) for y in (-.18, .18) for x in (-.195, .195)]
    fixed = trimesh.util.concatenate(feet)
    # One downward seat load, so the existing one-load FEA path can be preflighted.
    load = box(-.18, .18, -.15, .15, .455, .500)
    envelope = trimesh.boolean.union(
        [underseat, seat, back, left_wing, right_wing, *feet], engine='manifold')
    if not isinstance(envelope, trimesh.Trimesh):
        raise RuntimeError('manifold union did not return a mesh')
    # Explicit occupant clearance reference. It is disjoint from the envelope.
    keepout = box(-.18, .18, -.18, .12, .500, .920)
    records = {
        'name': 'chair_concept_pilot_01',
        'units': 'm', 'axes': {'x': 'left-right', 'y': 'front-back; +Y=back', 'z': 'up'},
        'intent': 'broad monolithic chair volume for penguin/avocado-inspired structural QD',
        'status': 'geometry/BC pilot; not an LMTO exact replication or FEA-validated chair',
        'load_case': {'patch': 'central seat, -Z distributed load', 'force_N_proposed': 800,
                      'backrest_load': 'not yet represented'},
        'fixed_case': 'four floor-contact discs at x=±0.195 m, y=±0.18 m',
        'seat_height_m': .500, 'minimum_seat_patch_m': [.360, .300],
        'envelope': export_mesh(envelope, 'original_DesignSpace.stl'),
        'fixed': export_mesh(fixed, 'fixed.stl'),
        'load': export_mesh(load, 'load.stl'),
        'occupant_keepout': export_mesh(keepout, 'occupant_keepout.stl'),
    }
    assert records['envelope']['watertight'] and records['envelope']['components'] == 1
    assert records['fixed']['watertight'] and records['fixed']['components'] == 4
    assert records['load']['watertight'] and records['load']['components'] == 1
    (OUT / 'domain.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(OUT / 'domain.json')


if __name__ == '__main__':
    main()
