#!/usr/bin/env python3
"""Create minimal four-leg fix-to-seat support corridors on the chair 64^3 grid."""
from __future__ import annotations

import json

import numpy as np

from make_chair_domain import ROOT

OUT = ROOT / "experiments/chair/sofa_style_2026-09-28/visual_hull_dense/rear_projection"


def main() -> None:
    d = np.load(ROOT / "data_real/chair/voxel.npz")
    origin, pitch = d["origin"], d["pitch_xyz"]
    coord = [origin[i] + (np.arange(64) + .5) * pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*coord, indexing="ij")
    mask = np.zeros((64, 64, 64), dtype=bool)
    for cx in (-.195, .195):
        for cy in (-.180, .180):
            mask |= (((x - cx) ** 2 + (y - cy) ** 2) <= .03 ** 2) & (z >= .04) & (z <= .48)
    mask &= d["bracket"].astype(bool)
    np.savez_compressed(OUT / "four_leg_corridors.npz", mask=mask)
    metrics = {"path": str(OUT / "four_leg_corridors.npz"),
               "radius_mm": 30, "z_m": [.04, .48],
               "foot_centers_xy_m": [[cx, cy] for cx in (-.195, .195)
                                     for cy in (-.180, .180)],
               "voxels": int(mask.sum()),
               "overlap_fix_voxels": int((mask & d["fix"].astype(bool)).sum()),
               "overlap_load_voxels": int((mask & d["load"].astype(bool)).sum())}
    (OUT / "four_leg_corridors.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
