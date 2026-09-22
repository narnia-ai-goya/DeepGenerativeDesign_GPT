"""Render lit views with pyvista — matches the camera convention used by
cond_render_masks_occ.py so the masks and the lit images are pixel-aligned.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import trimesh
os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv


def camera_from_elev_azim(center, radius, elev_deg, azim_deg):
    e = np.deg2rad(elev_deg); a = np.deg2rad(azim_deg)
    x = radius * np.cos(e) * np.sin(a)
    y = -radius * np.cos(e) * np.cos(a)
    z = radius * np.sin(e)
    eye = center + np.array([x, y, z])
    # If elev is near-vertical, the default up=+z is parallel to view → use +y.
    # Use +y so world +x maps to image right (no L/R mirror).
    if abs(elev_deg) > 80:
        up = np.array([-np.sin(a), np.cos(a), 0])
    else:
        up = np.array([0, 0, 1])
    return eye, up


def view_projected_extent(vertices, center, eye, up, margin=1.05):
    """Half-extent in the camera image plane for a tight fit.
    Returns max(half_width, half_height) of the projected bbox."""
    fwd = center - eye; fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, up); right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, fwd); true_up = true_up / (np.linalg.norm(true_up) + 1e-12)
    rel = vertices - center
    u = rel @ right; v = rel @ true_up
    half_w = float(np.max(np.abs(u)))
    half_h = float(np.max(np.abs(v)))
    return max(half_w, half_h) * margin


def render_lit(meshes, eye, center, up, size=1024, fit_extent=None, margin=1.15,
               perspective=False, fov=30.0, color=(0.85, 0.85, 0.87)):
    """High-quality lit render: parallel projection (default), anti-aliased,
    Phong shading, multi-light setup for clearer surface detail.

    If perspective=True, use perspective projection (fov degrees) so nearer
    parts appear larger for a stronger sense of depth. Framing auto-fits the
    whole mesh via reset_camera, then shrinks by margin.
    """
    if isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.enable_anti_aliasing('msaa')
    pl.remove_all_lights()
    # 3-light rig
    pl.add_light(pv.Light(position=tuple(eye + np.array([0, 0, abs(np.linalg.norm(eye))*0.5])),
                          focal_point=tuple(center), intensity=0.6, light_type='scene light'))
    pl.add_light(pv.Light(position=tuple(eye), focal_point=tuple(center),
                          intensity=0.5, light_type='headlight'))
    pl.add_light(pv.Light(position=tuple(center + (eye - center) * 0.5 + np.array([0, 0, -np.linalg.norm(eye)])),
                          focal_point=tuple(center), intensity=0.25, light_type='scene light'))
    for m in meshes:
        pv_mesh = pv.wrap(m).compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)
        pl.add_mesh(pv_mesh, color=color, smooth_shading=True,
                     ambient=0.25, diffuse=0.65, specular=0.15, specular_power=20)
    pl.background_color = 'white'
    pl.camera_position = [eye.tolist(), center.tolist(), up.tolist()]
    if perspective:
        pl.camera.parallel_projection = False
        pl.camera.view_angle = float(fov)
        pl.reset_camera()              # auto-fit the whole mesh, keeping view direction
        pl.camera.zoom(1.0 / margin)   # leave `margin` of empty space
    else:
        pl.camera.parallel_projection = True
        if fit_extent is not None:
            pl.camera.parallel_scale = float(fit_extent) * margin
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bracket', required=True, help='design-space STL (always rendered)')
    ap.add_argument('--fix', required=True, help='fix BC STL — used for camera framing only by default')
    ap.add_argument('--load', required=True, help='load BC STL — used for camera framing only by default')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--include-bc', action='store_true',
                    help='render fix+load together with design body (old behavior). '
                         'Default: only design body rendered, BC peg pixels stay null/blank.')
    ap.add_argument('--tight-fit', action='store_true',
                    help='per-view auto-fit (object fills frame) instead of consistent scale. '
                         'Larger style features for small objects like caliper.')
    ap.add_argument('--perspective', action='store_true',
                    help='perspective-projection render (stronger depth). default is orthographic')
    ap.add_argument('--fov', type=float, default=30.0,
                    help='field of view (degrees) when perspective; larger = more perspective distortion')
    args = ap.parse_args()

    br = trimesh.load(args.bracket, force='mesh')
    fix = trimesh.load(args.fix, force='mesh')
    load = trimesh.load(args.load, force='mesh')
    if args.include_bc:
        meshes = [br, fix, load]
    else:
        meshes = [br]

    all_v = np.vstack([br.vertices, fix.vertices, load.vertices])
    center = (all_v.min(axis=0) + all_v.max(axis=0)) / 2
    extent = all_v.max(axis=0) - all_v.min(axis=0)
    radius = float(np.linalg.norm(extent)) * 1.5
    # parallel-projection scale = half of the longest extent
    fit_extent = float(extent.max()) / 2

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    azim_names = ["front", "front_right", "right", "back_right",
                  "back", "back_left", "left", "front_left"]
    azims = [0, 45, 90, 135, 180, 225, 270, 315]
    elev_specs = [("lo", 15), ("hi", 45)]
    idx = 0
    from PIL import Image
    all_v_for_fit = all_v
    for elev_tag, elev in elev_specs:
        for az_name, az in zip(azim_names, azims):
            name = f"v{idx:02d}_{az_name}_{elev_tag}"
            eye, up = camera_from_elev_azim(center, radius, elev, az)
            fe = view_projected_extent(all_v_for_fit, center, eye, up) if args.tight_fit else fit_extent
            img = render_lit(meshes, eye, center, up, args.size, fit_extent=fe, perspective=args.perspective, fov=args.fov)
            Image.fromarray(img).save(out / f'{name}.png')
            idx += 1
            print(f'  {name}')
    eye, up = camera_from_elev_azim(center, radius, 85, 0)
    fe = view_projected_extent(all_v_for_fit, center, eye, up) if args.tight_fit else fit_extent
    img = render_lit(meshes, eye, center, up, args.size, fit_extent=fe, perspective=args.perspective, fov=args.fov)
    Image.fromarray(img).save(out / 'v_top.png')
    print(f'  v_top')
    # bottom view (-85 elev)
    eye, up = camera_from_elev_azim(center, radius, -85, 0)
    fe = view_projected_extent(all_v_for_fit, center, eye, up) if args.tight_fit else fit_extent
    img = render_lit(meshes, eye, center, up, args.size, fit_extent=fe, perspective=args.perspective, fov=args.fov)
    Image.fromarray(img).save(out / 'v_bottom.png')
    print(f'  v_bottom')
    # equatorial views (elev=0) — useful for axially symmetric parts (wheel etc.)
    azim_names = ["front", "front_right", "right", "back_right",
                  "back", "back_left", "left", "front_left"]
    azims = [0, 45, 90, 135, 180, 225, 270, 315]
    for az_name, az in zip(azim_names, azims):
        eye, up = camera_from_elev_azim(center, radius, 0, az)
        fe = view_projected_extent(all_v_for_fit, center, eye, up) if args.tight_fit else fit_extent
        img = render_lit(meshes, eye, center, up, args.size, fit_extent=fe, perspective=args.perspective, fov=args.fov)
        Image.fromarray(img).save(out / f'v_{az_name}.png')
        print(f'  v_{az_name}')
    print(f'DONE -> {out}')


if __name__ == '__main__':
    main()
