#!/usr/bin/env python3
"""Render the chair envelope, fixed/load solids, and occupant keep-out."""
from __future__ import annotations

from pathlib import Path

import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_real/chair'


def main() -> None:
    envelope = pv.read(OUT / 'original_DesignSpace.stl')
    fixed = pv.read(OUT / 'fixed.stl')
    load = pv.read(OUT / 'load.stl')
    keepout = pv.read(OUT / 'occupant_keepout.stl')
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(1800, 640))
    views = [('ISO · overall', (1.25, -1.5, 1.1), (0, 0, 1)),
             ('FRONT · X/Z', (0, -2, .55), (0, 0, 1)),
             ('SIDE · Y/Z', (2, 0, .55), (0, 0, 1))]
    for i, (label, eye, up) in enumerate(views):
        plotter.subplot(0, i)
        plotter.set_background('#f7f8fa')
        plotter.add_mesh(envelope, color='#8293a1', opacity=.24, name='envelope')
        plotter.add_mesh(fixed, color='#d14b3f', opacity=1, name='fixed')
        plotter.add_mesh(load, color='#29a86b', opacity=1, name='load')
        plotter.add_mesh(keepout.outline(), color='#4b82c7', line_width=2.2,
                         opacity=.9, name='occupant clearance')
        plotter.add_text(label, position='upper_left', font_size=13, color='#1f2933')
        if i == 0:
            plotter.add_text('GRAY envelope   RED fixed   GREEN load   BLUE clearance',
                             position='lower_left', font_size=9, color='#1f2933')
        plotter.camera_position = [eye, (0, 0, .45), up]
        plotter.camera.parallel_projection = True
        plotter.camera.parallel_scale = .59
    image = OUT / 'domain_preview.png'
    plotter.screenshot(str(image))
    plotter.close()
    print(image)


if __name__ == '__main__':
    main()
