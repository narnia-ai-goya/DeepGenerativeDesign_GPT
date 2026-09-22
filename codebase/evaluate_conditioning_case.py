"""Compare CAD-conditioned meshes with explicit geometric and image-space proxies.

No learned style score is claimed. Projected holes are 2D connected voids, not
3D handles. Medial thickness is a voxel EDT proxy; BC proximity is a geometric
diagnostic, not evidence that the FEM boundary condition is physically correct.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import trimesh
from pysdf import SDF

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import pyvista as pv
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code' / 'conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit
from cond_render_masks import depth_per_mesh, standard_views


def hole_stats(solid, pixel_mm, min_pixels=32):
    holes = ndi.binary_fill_holes(solid) & ~solid
    labels, _ = ndi.label(holes)
    sizes = np.bincount(labels.ravel())[1:]
    keep = sizes >= min_pixels
    return {'count': int(keep.sum()), 'area_mm2': float(sizes[keep].sum()*pixel_mm**2),
            'median_area_mm2': float(np.median(sizes[keep])*pixel_mm**2) if keep.any() else None}


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def image_metrics(solid, domain, bc, pixel_mm):
    rim = domain & ~ndi.binary_erosion(domain, iterations=2)
    return {**{'projected_holes_'+k: v for k, v in hole_stats(solid, pixel_mm).items()},
            'outside_cad_fraction': ratio((solid & ~domain).sum(), solid.sum()),
            'cad_rim_retention': ratio((solid & rim).sum(), rim.sum()),
            'visible_bc_retention': ratio((solid & bc).sum(), bc.sum()),
            'solid_fraction_in_cad': ratio((solid & domain).sum(), domain.sum())}


def surface_samples(mesh, n=12000):
    rng = np.random.default_rng(42)
    tri = mesh.triangles[rng.choice(len(mesh.faces), n, p=mesh.area_faces/mesh.area)]
    uv = rng.random((n, 2))
    flip = uv.sum(1) > 1
    uv[flip] = 1-uv[flip]
    return tri[:, 0] + uv[:, :1]*(tri[:, 1]-tri[:, 0]) + uv[:, 1:]*(tri[:, 2]-tri[:, 0])


def sdf(mesh):
    return SDF(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32))


def voxel_metrics(mesh, allowed, pitch_mm):
    pitch = pitch_mm / 1000
    lo = mesh.bounds[0] - 2*pitch
    shape = np.ceil((mesh.bounds[1]-lo)/pitch).astype(int)+3
    indices = np.indices(shape, dtype=np.int32).reshape(3, -1).T
    xyz = np.ascontiguousarray(lo + indices*pitch, dtype=np.float32)
    field = sdf(mesh)
    inside = np.zeros(len(xyz), dtype=bool)
    for start in range(0, len(xyz), 200000):
        inside[start:start+200000] = field(xyz[start:start+200000], n_threads=4) > 0
    occ = inside.reshape(shape)
    dist = ndi.distance_transform_edt(occ, sampling=pitch_mm)
    # Ridge samples approximate medial thickness; report grid pitch explicitly.
    ridge = occ & (dist >= ndi.maximum_filter(dist, size=3))
    thickness = 2*dist[ridge]
    points = xyz[inside]
    allowed_inside = np.zeros(len(points), dtype=bool)
    for part in allowed:
        sf = sdf(part)
        for start in range(0, len(points), 200000):
            allowed_inside[start:start+200000] |= sf(points[start:start+200000], n_threads=4) >= -pitch/2
    return {'pitch_mm': pitch_mm, 'occupied_voxels': int(inside.sum()),
            'containment_fraction_half_voxel_tolerance': float(allowed_inside.mean()) if len(points) else None,
            'medial_thickness_p10_mm': float(np.percentile(thickness, 10)) if len(thickness) else None,
            'medial_thickness_median_mm': float(np.median(thickness)) if len(thickness) else None,
            'medial_samples_below_3mm_fraction': float((thickness < 3).mean()) if len(thickness) else None,
            'medial_samples': int(len(thickness))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mesh', type=Path, required=True)
    ap.add_argument('--conditioning', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference-conditioning', type=Path)
    ap.add_argument('--domain-dir', type=Path, default=Path(__file__).resolve().parents[1]/'data_real/caliper')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--pitch-mm', type=float, default=0.75)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(a.mesh, force='mesh')
    cad = {k: trimesh.load(a.domain_dir/n, force='mesh') for k,n in
           [('br','original_DesignSpace.stl'),('fix','fixed.stl'),('load','load.stl')]}
    allv = np.vstack([m.vertices for m in cad.values()])
    center = (allv.min(0)+allv.max(0))/2
    extent = allv.max(0)-allv.min(0)
    radius = float(np.linalg.norm(extent))*1.5
    scale = float(extent.max())/2*1.15
    pixel_mm = 2*scale/a.size*1000
    frame = {v:(el,az) for v,el,az in standard_views()}
    views = ['v00_front_lo','v02_right_lo','v04_back_lo','v06_left_lo','v_top','v_bottom']
    parts = mesh.split(only_watertight=False)
    report = {'mesh': str(a.mesh.resolve()), 'conditioning':str(a.conditioning.resolve()),
              'watertight':bool(mesh.is_watertight),'components':len(parts),
              'vertices':len(mesh.vertices),'faces':len(mesh.faces),
              'volume_mm3':float(abs(mesh.volume)*1e9),
              'signed_volume_mm3':float(mesh.volume*1e9),
              'surface_area_mm2':float(mesh.area*1e6),
              'euler_number':int(mesh.euler_number),
              'genus_sum_if_closed':float((2*len(parts)-mesh.euler_number)/2) if mesh.is_watertight else None,
              'views':{}}
    for v in views:
        el,az=frame[v]; eye,up=camera_from_elev_azim(center,radius,el,az)
        sils,front=depth_per_mesh(cad,eye,center,up,scale,a.size)
        domain=sils['br']|front['fix']|front['load']; bc=front['fix']|front['load']
        pl=pv.Plotter(off_screen=True,window_size=(a.size,a.size))
        pl.add_mesh(pv.wrap(mesh),color='black',ambient=1,diffuse=0,specular=0)
        pl.background_color='white'; pl.camera_position=[eye.tolist(),center.tolist(),up.tolist()]
        pl.camera.parallel_projection=True; pl.camera.parallel_scale=scale
        pl.screenshot(return_img=False)
        pl.camera_position=[eye.tolist(),center.tolist(),up.tolist()]; pl.camera.parallel_scale=scale
        pl.render(); solid=~np.isnan(pl.get_image_depth(fill_value=np.nan)); pl.close()
        data={'mesh':image_metrics(solid,domain,bc,pixel_mm)}
        for label,folder in [('input',a.conditioning),('common_reference',a.reference_conditioning)]:
            if folder is None or not (folder/f'{v}.png').exists(): continue
            with Image.open(folder/f'{v}.png') as im:
                rgb=np.asarray(im.convert('RGB').resize((a.size,a.size),Image.Resampling.LANCZOS))
            target=rgb.min(axis=2)<245
            data[label]=image_metrics(target,domain,bc,pixel_mm)
            data[label]['mesh_foreground_iou']=ratio((solid&target).sum(),(solid|target).sum())
        report['views'][v]=data
        Image.fromarray(render_lit(mesh,eye,center,up,size=a.size,
                                  fit_extent=float(extent.max())/2,
                                  color=(0.12, 0.12, 0.13))).save(a.out/f'{v}.png')
    field=sdf(mesh)
    report['boundary_surface_proximity']={}
    for name in ['fix','load']:
        distances=np.abs(field(np.ascontiguousarray(surface_samples(cad[name]),dtype=np.float32),n_threads=4))*1000
        report['boundary_surface_proximity'][name]={'median_mm':float(np.median(distances)),
             'within_1_5mm_fraction':float((distances<=1.5).mean()),'samples':len(distances)}
    report['voxel']=voxel_metrics(mesh,list(cad.values()),a.pitch_mm)
    report['notes']=['Volume is the magnitude of the signed mesh volume; inspect winding/cavities if signed volume is negative.',
       'Projected holes are 2D enclosed voids >=32 pixels, not 3D handles.',
       'Input foreground uses RGB min<245 after analysis-only resizing; bright surfaces can be misclassified.',
       'Voxel thickness uses EDT ridge diameters, not exact wall thickness or a manufacturing certificate.',
       'Boundary proximity samples entire fixture/load surfaces, not only physical contact facets.',
       'All views use the source CAD camera frame; no image alignment hides camera drift.']
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    mesh.export(a.out/'final.stl')
    eye,up=camera_from_elev_azim(center,radius,15,315)
    Image.fromarray(render_lit(mesh,eye,center,up,size=1024,
                              fit_extent=float(extent.max())/2,
                              color=(0.12, 0.12, 0.13))).save(a.out/'final_preview.png')
    print(a.out/'metrics.json',flush=True)


if __name__=='__main__':
    main()
