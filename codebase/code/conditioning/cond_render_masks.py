"""Occlusion-aware BC masks v2 — single-scene rendering (camera-consistent).

v1 (cond_render_masks_occ.py) problem: bracket/fix/load were each rendered in a
separate Plotter, so pyvista fit the camera to each mesh's bounds and the scale
differed per mesh. In practice the BC pads ended up outside the bracket
silhouette (~2x zoom).

v2 guarantees:
  1. Put all meshes in a single Plotter scene and extract per-mesh depth via
     visibility toggling -> physically identical camera (scale/position cannot
     drift).
  2. Camera framing is based on the combined bounds of all meshes -> no mesh is
     clipped out of frame.
  3. If a silhouette touches the frame border after rendering, grow the margin
     automatically and retry.
  4. Supports an arbitrary number of BC meshes (beyond --fix/--load).

Outputs (compatible with build_aligned_masks.py, all white=positive):
  <view>_bracket.png   combined silhouette (design + visible BC)
  <view>_fix.png       visible fix (white=fix)
  <view>_load.png      visible load (white=load)
  <view>_bc.png        visible fix|load, dilated (white=BC)
  <view>_stylable.png  combined_sil AND NOT bc
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import numpy as np
import trimesh
from PIL import Image, ImageFilter

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv


def camera_from_elev_azim(center: np.ndarray, radius: float,
                          elev_deg: float, azim_deg: float):
    e = np.deg2rad(elev_deg); a = np.deg2rad(azim_deg)
    x = radius * np.cos(e) * np.sin(a)
    y = -radius * np.cos(e) * np.cos(a)
    z = radius * np.sin(e)
    eye = center + np.array([x, y, z])
    if abs(elev_deg) > 80:
        up = np.array([-np.sin(a), np.cos(a), 0])
    else:
        up = np.array([0, 0, 1])
    return eye, up


def view_projected_extent(vertices, center, eye, up, margin=1.05):
    """Half-extent in the camera image plane (for tight-fit)."""
    fwd = center - eye; fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(fwd, up); right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, fwd); true_up = true_up / (np.linalg.norm(true_up) + 1e-12)
    rel = vertices - center
    half_w = float(np.max(np.abs(rel @ right)))
    half_h = float(np.max(np.abs(rel @ true_up)))
    return max(half_w, half_h) * margin


ID_COLORS = {'br': (255, 0, 0), 'fix': (0, 255, 0), 'load': (0, 0, 255)}


def _set_camera(pl, eye, center, up, parallel_scale, perspective, fov, margin):
    """Same camera setup as cond_render_pv.render_lit (pixel alignment)."""
    pl.camera_position = [eye.tolist(), center.tolist(), up.tolist()]
    if perspective:
        pl.camera.parallel_projection = False
        pl.camera.view_angle = float(fov)
        pl.reset_camera()
        pl.camera.zoom(1.0 / margin)
    else:
        pl.camera.parallel_projection = True
        pl.camera.parallel_scale = parallel_scale


def depth_per_mesh(meshes: dict[str, trimesh.Trimesh], eye, center, up,
                   parallel_scale: float, size: int,
                   perspective: bool = False, fov: float = 30.0, margin: float = 1.15):
    """Extract per-mesh silhouette + frontmost ID from a single scene via
    visibility toggling.

    Occlusion is decided by a **flat-color ID render**, not by comparing depth
    values: rendering the three meshes in their own primary colors (ambient=1,
    no lighting) in one scene lets the OpenGL z-buffer decide the frontmost mesh
    per pixel directly, so depth-sign convention (negative distance, etc.)
    misinterpretation is impossible by construction.

    If perspective=True, render with the same perspective camera as cond_render_pv.
    Returns: (sils: {name: bool HxW}, frontmost: {name: bool HxW})
    """
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    actors = {}
    for n, m in meshes.items():
        c = ID_COLORS[n]
        actors[n] = pl.add_mesh(
            pv.wrap(m), color=[v / 255 for v in c],
            ambient=1.0, diffuse=0.0, specular=0.0, smooth_shading=False)
    pl.remove_all_lights()
    pl.background_color = 'black'
    _set_camera(pl, eye, center, up, parallel_scale, perspective, fov, margin)
    pl.screenshot(return_img=False)          # force first render (absorb camera reset)
    _set_camera(pl, eye, center, up, parallel_scale, perspective, fov, margin)

    # 1) frontmost ID — render all meshes at once, classify by color (dominant channel)
    for a in actors.values():
        a.SetVisibility(True)
    pl.render()
    img = pl.screenshot(return_img=True)[..., :3].astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    fg = img.max(axis=-1) > 64                       # exclude background (black)
    frontmost = {
        'br':   fg & (r >= g) & (r >= b),
        'fix':  fg & (g > r) & (g >= b),
        'load': fg & (b > r) & (b > g),
    }

    # 2) per-mesh full silhouette — visibility toggle (depth used only for hit/no-hit)
    sils = {}
    for target in meshes:
        for n, a in actors.items():
            a.SetVisibility(n == target)
        pl.render()
        depth = pl.get_image_depth(fill_value=np.nan)
        sils[target] = ~np.isnan(depth)
    pl.close()
    return sils, frontmost


def touches_border(sil: np.ndarray) -> bool:
    return bool(sil[0, :].any() or sil[-1, :].any() or
                sil[:, 0].any() or sil[:, -1].any())


def render_view_robust(meshes, eye, center, up, base_scale, size,
                       max_retry=3, grow=1.2, perspective=False, fov=30.0, margin=1.15):
    """Grow the margin automatically until every mesh silhouette fits in frame.
    With perspective, reset_camera handles fitting so clip retries rarely occur."""
    scale = base_scale
    for attempt in range(max_retry + 1):
        sils, front = depth_per_mesh(meshes, eye, center, up, scale, size,
                                     perspective=perspective, fov=fov, margin=margin)
        clipped = [n for n, sil in sils.items() if touches_border(sil)]
        if not clipped:
            return sils, front, scale
        print(f'    WARN: {clipped} touches frame border @scale {scale:.4g} '
              f'-> retry x{grow}', flush=True)
        scale *= grow
        margin *= grow
    print(f'    WARN: still clipped after {max_retry} retries — using last', flush=True)
    return sils, front, scale


def standard_views():
    azim_names = ["front", "front_right", "right", "back_right",
                  "back", "back_left", "left", "front_left"]
    azims = [0, 45, 90, 135, 180, 225, 270, 315]
    views, idx = [], 0
    for elev_tag, elev in [("lo", 15), ("hi", 45)]:
        for az_name, az in zip(azim_names, azims):
            views.append((f"v{idx:02d}_{az_name}_{elev_tag}", elev, az))
            idx += 1
    views.append(("v_top", 85, 0))
    views.append(("v_bottom", -85, 0))
    # equatorial views (elev=0) — useful for axially symmetric parts (wheel etc.)
    for az_name, az in zip(azim_names, azims):
        views.append((f"v_{az_name}", 0, az))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", required=True, help="design-space STL (the body)")
    ap.add_argument("--fix", required=True)
    ap.add_argument("--load", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--dilate-bc", type=int, default=8)
    ap.add_argument("--margin", type=float, default=1.15,
                    help="parallel_scale margin (same default as cond_render_pv)")
    ap.add_argument("--tight-fit", action='store_true',
                    help="per-view auto-fit (based on combined bounds, so unlike v1 it keeps meshes aligned)")
    ap.add_argument("--views", default=None,
                    help="comma-sep view name filter (e.g. v_top,v00_front_lo). Default: all 18 views")
    ap.add_argument("--perspective", action='store_true',
                    help="generate masks with the same perspective camera as cond_render_pv --perspective")
    ap.add_argument("--fov", type=float, default=30.0)
    args = ap.parse_args()

    meshes = {
        'br':   trimesh.load(args.bracket, force='mesh'),
        'fix':  trimesh.load(args.fix, force='mesh'),
        'load': trimesh.load(args.load, force='mesh'),
    }
    for n, m in meshes.items():
        if m.vertices.shape[0] == 0:
            raise SystemExit(f'ERROR: mesh "{n}" is empty')

    # camera framing: combined bounds of all meshes -> no mesh leaves the frame
    all_v = np.vstack([m.vertices for m in meshes.values()])
    center = (all_v.min(axis=0) + all_v.max(axis=0)) / 2
    extent = all_v.max(axis=0) - all_v.min(axis=0)
    radius = float(np.linalg.norm(extent)) * 1.5
    fit_extent = float(extent.max()) / 2

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    views = standard_views()
    if args.views:
        keep = set(args.views.split(','))
        views = [v for v in views if v[0] in keep]
        if not views:
            raise SystemExit(f'ERROR: no view matches filter {sorted(keep)}')

    for name, elev, azim in views:
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        half = (view_projected_extent(all_v, center, eye, up)
                if args.tight_fit else fit_extent)
        sils, front, used_scale = render_view_robust(
            meshes, eye, center, up, half * args.margin, args.size,
            perspective=args.perspective, fov=args.fov, margin=args.margin)
        sil_br, sil_fix, sil_ld = sils['br'], sils['fix'], sils['load']

        # visibility = frontmost from ID render (z-buffer decides directly, no depth-sign interpretation)
        visible_fix = front['fix'] & sil_fix
        visible_ld  = front['load'] & sil_ld
        visible_bc = visible_fix | visible_ld
        combined_sil = sil_br | visible_fix | visible_ld

        bc_img = (visible_bc.astype(np.uint8) * 255)
        if args.dilate_bc > 0:
            bc_img = np.array(Image.fromarray(bc_img, 'L').filter(
                ImageFilter.MaxFilter(2 * args.dilate_bc + 1)))

        stylable = ((combined_sil) & (bc_img == 0)).astype(np.uint8) * 255
        Image.fromarray(stylable, 'L').save(out_dir / f"{name}_stylable.png")
        Image.fromarray(bc_img, 'L').save(out_dir / f"{name}_bc.png")
        Image.fromarray(combined_sil.astype(np.uint8) * 255, 'L').save(out_dir / f"{name}_bracket.png")
        Image.fromarray(visible_fix.astype(np.uint8) * 255, 'L').save(out_dir / f"{name}_fix.png")
        Image.fromarray(visible_ld.astype(np.uint8) * 255, 'L').save(out_dir / f"{name}_load.png")
        print(f"  {name}: stylable {(stylable>0).mean()*100:.1f}%  "
              f"BC {(bc_img>0).mean()*100:.1f}%  "
              f"(occluded_fix={int(sil_fix.sum() - visible_fix.sum())}, "
              f"occluded_load={int(sil_ld.sum() - visible_ld.sum())})", flush=True)
    print(f'\nDONE -> {out_dir}')


if __name__ == '__main__':
    main()
