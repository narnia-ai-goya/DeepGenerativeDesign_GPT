#!/usr/bin/env python3
"""Make a softer open-back chair reference in the existing chair BC frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT, foot

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim
sys.path.insert(0, str(ROOT / 'codebase'))
from diagnose_bc_preservation import field, interior_points, load_mesh


OUT = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27'
DOMAIN = ROOT / 'data_real/chair'
VIEWS = [('v00_front_lo', 15, 0), ('v02_right_lo', 15, 90),
         ('v04_back_lo', 15, 180), ('v06_left_lo', 15, 270),
         ('v_top', 85, 0), ('v_bottom', -85, 0)]


def closed_loft(rings: np.ndarray) -> trimesh.Trimesh:
    """Triangulate equal-length, counterclockwise rings with end caps."""
    ring_count, n, _ = rings.shape
    vertices = np.concatenate((rings.reshape(-1, 3),
                               rings[0].mean(axis=0, keepdims=True),
                               rings[-1].mean(axis=0, keepdims=True)), axis=0)
    faces = []
    for k in range(ring_count - 1):
        for j in range(n):
            a = k*n + j
            b = k*n + (j+1) % n
            c = (k+1)*n + j
            d = (k+1)*n + (j+1) % n
            faces.extend(((a, b, d), (a, d, c)))
    lower_center, upper_center = ring_count*n, ring_count*n+1
    for j in range(n):
        q = (j+1) % n
        faces.append((lower_center, q, j))
        faces.append((upper_center, (ring_count-1)*n+j, (ring_count-1)*n+q))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=True)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def rounded_rectangle(half_x: float, half_y: float,
                      radius: float, arc_steps: int = 12) -> np.ndarray:
    points = []
    for cx, cy, theta0 in ((half_x-radius, half_y-radius, 0),
                           (-half_x+radius, half_y-radius, 90),
                           (-half_x+radius, -half_y+radius, 180),
                           (half_x-radius, -half_y+radius, 270)):
        # Start at rightmost point of each corner and go CCW.
        for theta in np.linspace(theta0, theta0+90, arc_steps, endpoint=False):
            angle = np.deg2rad(theta)
            points.append((cx+radius*np.cos(angle), cy+radius*np.sin(angle)))
    return np.asarray(points, dtype=float)


def seat_shell() -> trimesh.Trimesh:
    levels = ((.431, .235, .225, .065),
              (.448, .260, .250, .075),
              (.482, .264, .252, .080),
              (.500, .245, .233, .077))
    rings = []
    for z, hx, hy, radius in levels:
        xy = rounded_rectangle(hx, hy, radius)
        rings.append(np.column_stack((xy, np.full(len(xy), z))))
    return closed_loft(np.asarray(rings))


def tube(centers: np.ndarray, radii_u: np.ndarray, radii_v: np.ndarray,
         preferred_u: np.ndarray, sections: int = 32) -> trimesh.Trimesh:
    centers = np.asarray(centers, dtype=float)
    circles = []
    angles = np.linspace(0, 2*np.pi, sections, endpoint=False)
    for i, center in enumerate(centers):
        tangent = centers[min(i+1, len(centers)-1)] - centers[max(i-1, 0)]
        tangent /= np.linalg.norm(tangent)
        u = preferred_u - tangent*np.dot(preferred_u, tangent)
        u /= np.linalg.norm(u)
        v = np.cross(tangent, u)
        rings = center + (radii_u[i]*np.cos(angles))[:, None]*u \
                       + (radii_v[i]*np.sin(angles))[:, None]*v
        circles.append(rings)
    return closed_loft(np.asarray(circles))


def leg_part(x_sign: int, y_sign: int) -> trimesh.Trimesh:
    zs = np.linspace(.025, .455, 11)
    t = (zs-.025)/(.455-.025)
    # Four subtly splayed, waisted legs. Their feet remain the unchanged BC solids.
    xs = x_sign*(.195 + .018*t + .006*np.sin(np.pi*t))
    ys = y_sign*(.180 + .025*t)
    centers = np.column_stack((xs, ys, zs))
    radii = .043 - .010*np.sin(np.pi*t) + .006*t
    return tube(centers, radii, radii*.92, np.array([1., 0., 0.]))


def back_upright(sign: int) -> trimesh.Trimesh:
    zs = np.linspace(.465, .860, 12)
    t = (zs-.465)/(.860-.465)
    xs = sign*(.205 - .016*t)
    ys = .211 + .012*t - .006*np.sin(np.pi*t)
    centers = np.column_stack((xs, ys, zs))
    widths = .041 - .008*np.sin(np.pi*t)
    depths = .032 - .004*np.sin(np.pi*t)
    return tube(centers, widths, depths, np.array([1., 0., 0.]))


def arch_bar(z_edge: float, rise: float, width: float,
             depth: float) -> trimesh.Trimesh:
    xs = np.linspace(-.207, .207, 23)
    t = xs/.207
    centers = np.column_stack((xs, np.full_like(xs, .217),
                               z_edge + rise*(1-t*t)))
    radii_y = np.full_like(xs, depth)
    radii_z = width*(.92 + .08*(1-t*t))
    return tube(centers, radii_y, radii_z, np.array([0., 1., 0.]))


def make_reference() -> trimesh.Trimesh:
    parts = [seat_shell(),
             *[foot(x, y) for y in (-.180, .180) for x in (-.195, .195)],
             *[leg_part(xs, ys) for ys in (-1, 1) for xs in (-1, 1)],
             back_upright(-1), back_upright(1),
             arch_bar(.852, .030, .026, .033),
             arch_bar(.660, .012, .020, .024)]
    reference = trimesh.boolean.union(parts, engine='manifold')
    if not isinstance(reference, trimesh.Trimesh):
        raise RuntimeError('manifold did not return a mesh')
    return reference


def validate(mesh: trimesh.Trimesh) -> dict:
    env = load_mesh(DOMAIN / 'original_DesignSpace.stl')
    fix = load_mesh(DOMAIN / 'fixed.stl')
    load = load_mesh(DOMAIN / 'load.stl')
    keepout = load_mesh(DOMAIN / 'occupant_keepout.stl')
    rng = np.random.default_rng(42)
    mesh_sdf = field(mesh)
    envelope_sdf = field(env)
    bc_containment = {}
    for name, source in (('fix', fix), ('load', load)):
        points = interior_points(source, 12000, rng)
        bc_containment[name] = float((mesh_sdf(points) > -0.003).mean())
    clearance = interior_points(keepout, 12000, rng)
    occupied_clearance = float((mesh_sdf(clearance) > 0).mean())
    mesh_points = interior_points(mesh, 20000, rng)
    outside_env = float((envelope_sdf(mesh_points) < -0.002).mean())
    result = {'mesh': str(OUT / 'reference.obj'),
              'watertight': bool(mesh.is_watertight),
              'components': len(mesh.split(only_watertight=False)),
              'volume_litres': float(mesh.volume*1000),
              'bc_containment': bc_containment,
              'occupant_keepout_occupied_fraction': occupied_clearance,
              'sampled_outside_envelope_fraction': outside_env,
              'bounds_m': mesh.bounds.tolist()}
    result['passed'] = bool(result['watertight'] and result['components'] == 1
                            and min(bc_containment.values()) >= .999
                            and occupied_clearance == 0 and outside_env == 0)
    return result


def render_metal(mesh: trimesh.Trimesh, eye: np.ndarray, center: np.ndarray,
                 up: np.ndarray, scale: float, size: int = 512) -> np.ndarray:
    """Use crease-aware normals so the flat seat does not show fan triangles."""
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.enable_anti_aliasing('msaa')
    plotter.remove_all_lights()
    distance = float(np.linalg.norm(eye))
    plotter.add_light(pv.Light(position=tuple(eye+np.array([0, 0, distance*.5])),
                                 focal_point=tuple(center), intensity=.6,
                                 light_type='scene light'))
    plotter.add_light(pv.Light(position=tuple(eye), focal_point=tuple(center),
                                 intensity=.5, light_type='headlight'))
    plotter.add_light(pv.Light(position=tuple(center+(eye-center)*.5
                                                +np.array([0, 0, -distance])),
                                 focal_point=tuple(center), intensity=.25,
                                 light_type='scene light'))
    surface = pv.wrap(mesh).compute_normals(
        point_normals=True, cell_normals=False, auto_orient_normals=True,
        split_vertices=True, feature_angle=15)
    plotter.add_mesh(surface, color=(.19, .22, .25), smooth_shading=True,
                     ambient=.25, diffuse=.65, specular=.15, specular_power=20)
    plotter.background_color = 'white'
    plotter.camera_position = [eye.tolist(), center.tolist(), up.tolist()]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = scale*1.15
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return image


def render_views(mesh: trimesh.Trimesh) -> None:
    inp = OUT / 'input'
    inp.mkdir(exist_ok=True)
    env = load_mesh(DOMAIN / 'original_DesignSpace.stl')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    scale = float(env.extents.max())/2
    sheet = Image.new('RGB', (512*len(VIEWS), 548), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, elev, azim) in enumerate(VIEWS):
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        image = Image.fromarray(render_metal(mesh, eye, center, up, scale)).convert('RGB')
        image.save(inp / f'{name}.png')
        sheet.paste(image, (512*i, 36))
        draw.text((512*i+12, 10), name, fill='#263139')
    sheet.save(OUT / 'contact.png')
    hero_eye, hero_up = camera_from_elev_azim(center, radius, 28, 40)
    Image.fromarray(render_metal(mesh, hero_eye, center, hero_up, scale, 768)).save(
        OUT / 'hero.png')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = make_reference()
    reference.export(OUT / 'reference.obj')
    check = validate(reference)
    (OUT / 'validation.json').write_text(json.dumps(check, indent=2) + '\n')
    render_views(reference)
    print(json.dumps(check, indent=2), flush=True)
    if not check['passed']:
        raise SystemExit('reference failed geometry/BC validation')


if __name__ == '__main__':
    main()
