"""Turn a multi-body CAD assembly export into the three-STL domain contract run_prep.py expects.

`run_prep.py` (Stage 0) assumes a domain is given as three files — `original_DesignSpace.stl`
(a single watertight solid envelope), `fixed.stl`, `load.stl`. A CAD assembly exported to OBJ/STL
usually is not that: the caliper input shipped here (`caliper_base.obj` + `fix.obj` + `load.obj`,
Autodesk ATF export) arrives as 265 separate near-closed shells with no watertight body among
them, and its "base" is the finished part's own thin walls rather than a design space.

This script bridges that gap in three steps:

  1. **solidify** — split into connected components, drop fragments below `--min-frac` of the
     bounding diagonal, run pymeshfix per component (each is nearly closed: the caliper's shells
     miss only 0.1-1.8% of their edges), and boolean-union the repaired parts with manifold3d.
     Applied to all three inputs.
  2. **envelope** — rasterize the solidified part onto a voxel grid, morphologically **close**
     (`--close-mm`) to merge neighbouring bodies into one solid, **dilate** (`--dilate-mm`) to open
     up design freedom, keep the largest connected component, then marching-cubes it back to a
     surface. The radii are what control whether functional gaps stay open: closing only bridges
     gaps narrower than ~2x its radius, so a brake caliper's rotor slot (32 mm here) survives an
     8 mm closing while the split walls of one caliper half merge.
  3. **coarsen** — re-run marching cubes on the grid downsampled by `--coarsen` (block max). This
     is the step that makes the result usable, not a cosmetic one: gmsh's in-loop domain mesher
     keeps the input surface as its boundary, so a 1.5 mm-edge envelope forces ~344k tets no
     matter what `fea_mesh_size` says, and every attempt to coarsen an already-fine surface
     (quadric decimation, isotropic remesh + pymeshfix) reintroduces self-intersections that gmsh
     rejects. Marching cubes on a coarse grid is self-intersection-free by construction.

Scaling: `--scale` multiplies the output (the pipeline works in metres, CAD exports usually do
not). Everything downstream — the 64^3 grid pitch, `bc_dist`, tet sizes, E and the 1000 N load —
assumes metres, so this has to be right; it is a stated input, not something the script infers.

The load surface can be filtered to one side with `--load-x-sign`: a fixed caliper's pistons push
in opposing directions, which a single uniform force vector cannot express, so one bank is
selected and `stages.fea.force_dir` states which way it pushes.

Usage (the caliper domain shipped in data_real/caliper was produced with exactly this):
  python code/domain_from_assembly.py --in-dir data_real/caliper \
      --envelope caliper_base.obj --fix fix.obj --load load.obj \
      --scale 0.005 --close-mm 8 --dilate-mm 4 --coarsen 3 --load-x-sign +

Reports every intermediate count so the result can be judged: component counts, watertightness,
envelope volume as a multiple of the part, and what fraction of the fixture/load surfaces ended up
inside the envelope (they must be, or boundary-condition detection finds nothing).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import trimesh


def solidify(path, min_frac, label):
    """Repair a fragmented multi-body surface into one watertight (possibly multi-shell) solid."""
    import pymeshfix
    m = trimesh.load(str(path), force='mesh', process=False)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    diag = float(np.linalg.norm(m.bounds[1] - m.bounds[0]))
    comps = [c for c in m.split(only_watertight=False)
             if np.linalg.norm(c.bounds[1] - c.bounds[0]) > min_frac * diag]
    ok = []
    for c in comps:
        try:
            mf = pymeshfix.MeshFix(c.vertices.astype(np.float64), c.faces.astype(np.int32))
            mf.repair()
            r = trimesh.Trimesh(np.asarray(mf.points), np.asarray(mf.faces), process=True)
            if not r.is_watertight or len(r.faces) < 12:
                continue
            if r.volume < 0:
                r.invert()
            if r.volume > 1e-9:
                ok.append(r)
        except Exception:
            pass
    if not ok:
        raise SystemExit(f'{label}: nothing repairable in {path}')
    u = manifold_union(ok)
    eb = bad_edges(u)
    print(f'  {label:9} components {len(comps)} → repaired {len(ok)} → union '
          f'F={len(u.faces):,} shells={len(u.split(only_watertight=False))} '
          f'non-manifold edges={eb} vol={abs(u.volume):.2f}', flush=True)
    if eb:
        print(f'  WARNING: {label} still has {eb} non-manifold edges. manifold3d rejects such a '
              f'mesh outright (returns an EMPTY manifold), which silently empties the whole '
              f'post-processing boolean — see post_hybrid_union_clip.py.')
    return u


def bad_edges(m):
    """Edges not shared by exactly two faces — what makes manifold3d refuse a mesh."""
    _, cnt = np.unique(m.edges_sorted, axis=0, return_counts=True)
    return int((cnt != 2).sum())


def manifold_union(parts):
    """Union a list of solids INSIDE manifold3d and read the result back.

    Folding trimesh's `.union(engine='manifold')` pairwise looks equivalent but is not: each step
    round-trips through trimesh, and a step that throws used to be skipped silently, leaving
    overlapping shells behind. Overlaps are exactly what later breaks — the isotropic remesh in
    run_prep turns coincident surfaces into non-manifold edges (measured on the caliper fixture:
    50 bad edges before the remesh, 277 after), and from there manifold3d returns an empty
    manifold for the peg and the design-domain clip intersects to nothing.

    Doing the whole union in manifold3d guarantees the output is a valid manifold or the status
    says why."""
    import manifold3d as m3
    def to_mf(t):
        return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, dtype=np.float32),
                                   tri_verts=np.asarray(t.faces, dtype=np.uint32)))
    acc = to_mf(parts[0])
    for t in parts[1:]:
        acc = acc + to_mf(t)
    if acc.status() != m3.Error.NoError or acc.is_empty():
        raise SystemExit(f'manifold3d union failed: status={acc.status()} empty={acc.is_empty()}')
    mesh = acc.to_mesh()
    # Read the result back WITHOUT welding. manifold3d keeps shells that merely touch as separate
    # vertex sets; merging coincident vertices across them fuses those contacts into edges shared
    # by four faces, which is exactly the non-manifold input the next boolean then refuses.
    return trimesh.Trimesh(vertices=mesh.vert_properties.copy(), faces=mesh.tri_verts.copy(),
                           process=False)


def voxel_solidify(mesh, pitch, label, close_vox=1.0):
    """Rasterize a multi-shell solid and marching-cubes it back into clean manifold bodies.

    Needed because manifold3d's union leaves shells that merely *touch* as separate vertex sets:
    valid for manifold3d, but `run_prep.py` welds the STL it reads (raw CAD exports are triangle
    soup, so welding is its job), and welding a contact turns it into an edge shared by four faces
    — non-manifold. The next boolean then refuses the peg and silently intersects to nothing.

    Going through a voxel grid removes contacts by construction: touching bodies become one
    connected region, and marching cubes emits a single manifold surface per region. Marching cubes
    is also what post_hybrid_union_clip.py already uses for its dilated peg collars, so the pegs
    stay consistent with how they are consumed."""
    from scipy import ndimage as ndi
    from skimage import measure
    occ, lo = rasterize(mesh, pitch, pad=pitch * 4)
    # One-voxel closing first: bodies that touch only diagonally stay separate 6-connected regions,
    # but marching cubes still emits surfaces that meet at those corners, and welding them is again
    # non-manifold. Closing turns a diagonal contact into one solid region.
    occ = ndi.binary_closing(occ, ball(close_vox))
    v, f, *_ = measure.marching_cubes(np.pad(occ.astype(np.float32), 2), level=0.5,
                                      spacing=(pitch,) * 3)
    v += lo - 2 * pitch
    m = trimesh.Trimesh(v, f, process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    print(f'  {label:9} voxel {pitch:.4f} u → F={len(m.faces):,} '
          f'bodies={len(m.split(only_watertight=False))} watertight={m.is_watertight} '
          f'non-manifold edges={bad_edges(m)} vol={abs(m.volume):.2f}', flush=True)
    return m


def rasterize(mesh, pitch, pad):
    """Occupancy grid of a (possibly multi-shell) solid: OR of each shell's inside test."""
    from pysdf import SDF
    lo = mesh.bounds[0] - pad
    hi = mesh.bounds[1] + pad
    axes = [np.arange(lo[i], hi[i] + 1e-9, pitch) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    occ = np.zeros(len(pts), bool)
    for c in mesh.split(only_watertight=False):
        if len(c.faces) < 12:
            continue
        try:
            occ |= SDF(c.vertices.astype(np.float32), c.faces.astype(np.uint32))(pts) > 0
        except Exception:
            pass
    return occ.reshape(tuple(len(a) for a in axes)), lo


def ball(r):
    n = int(np.ceil(r))
    a = np.arange(-n, n + 1)
    return np.linalg.norm(np.stack(np.meshgrid(a, a, a, indexing='ij'), -1), axis=-1) <= r


def part_occupancy(path, grid):
    """Solid occupancy of the raw assembly surface: rasterize it, then flood-fill the interior.

    NOT the per-component `solidify()` route. That repairs each shell on its own and drops the ones
    pymeshfix cannot close (135 of 227 on the caliper), so what gets dilated afterwards is a partial
    reconstruction — holes where shells went missing, bumps along the survivors' borders. Voxelizing
    the whole surface and flood-filling from outside uses every triangle and needs no shell to be
    closed: the arch's own walls seal the interior, and enclosed cavities become solid, which is what
    a design space wants anyway."""
    m = trimesh.load(str(path), force='mesh', process=False)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    pitch = float((m.bounds[1] - m.bounds[0]).max() / grid)
    vg = m.voxelized(pitch).fill()
    occ = vg.matrix.astype(bool)
    print(f'  occupancy  pitch={pitch:.4f} u  grid={occ.shape}  '
          f'filled volume={occ.sum() * pitch ** 3:.1f} u3  '
          f'(surface F={len(m.faces):,}, {len(m.split(only_watertight=False))} shells)', flush=True)
    return occ, vg.transform[:3, 3], pitch, m


def envelope(occ, org, pitch, close_v, dilate_v, smooth_v, edge_u, cavity=None):
    """occupancy -> (optional close/dilate) -> gaussian -> marching cubes -> Taubin -> remesh.

    Order matters, and getting it wrong is what produced a terraced blob the first time round.
    Marching cubes on a COARSE grid bakes the voxel staircase into the geometry, and an isotropic
    remesh cannot undo it: remeshing redistributes triangles while preserving the surface, so the
    steps survive at 66k faces exactly as they were at 18k. So: run marching cubes on the FINE grid,
    take the staircase out with a gaussian on the occupancy field plus Taubin smoothing (which is
    volume-preserving, unlike plain Laplacian), and only then remesh down to a CAD-like triangle
    count. Measured on the caliper: 8.6k faces, 22k in-loop tets — the same bracket sits at."""
    from scipy import ndimage as ndi
    from skimage import measure
    import pymeshlab
    E = occ
    if close_v > 0:
        E = ndi.binary_closing(E, ball(close_v))
    if dilate_v > 0:
        E = ndi.binary_dilation(E, ball(dilate_v))
    lab, n = ndi.label(E)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    E = lab == sizes.argmax()
    if n > 1:
        print(f'  envelope  {n} components -> largest keeps '
              f'{E.sum() / max(lab.astype(bool).sum(), 1) * 100:.1f}% of the volume', flush=True)
    field = ndi.gaussian_filter(E.astype(np.float32), smooth_v) if smooth_v > 0 else E.astype(np.float32)
    if cavity is not None:
        # Carve the cavity AFTER the gaussian, not before. Smoothing a wall thinner than its own
        # radius erases it: carving first and smoothing second punched the caliper's outer surface
        # open above the piston bores (genus 4 -> 8, holes the real part does not have). Forcing the
        # field to "outside" here keeps the cavity crisp and leaves the smoothed outer skin intact.
        field = np.where(cavity, 0.0, field)
    v, f, *_ = measure.marching_cubes(np.pad(field, 2), level=0.5, spacing=(pitch,) * 3)
    v += org - 2 * pitch                      # padded grid index -> world (keeps the input frame)
    m = trimesh.Trimesh(v, f, process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m = max(m.split(only_watertight=False), key=lambda c: len(c.faces))
    m.fix_normals()
    m = trimesh.smoothing.filter_taubin(m, iterations=8)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
    ms.meshing_isotropic_explicit_remeshing(iterations=8, adaptive=False,
                                            targetlen=pymeshlab.PureValue(edge_u))
    cm = ms.current_mesh()
    r = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=True)
    r.merge_vertices()
    r.remove_unreferenced_vertices()
    r = max(r.split(only_watertight=False), key=lambda c: len(c.faces))
    r.fix_normals()
    return r, E.sum() * pitch ** 3


def inside_frac(dom, mesh, stride):
    from pysdf import SDF
    sdf = SDF(dom.vertices.astype(np.float32), dom.faces.astype(np.uint32))
    return float((sdf(mesh.vertices[::stride].astype(np.float32)) > 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-dir', required=True, help='directory holding the assembly exports')
    ap.add_argument('--envelope', required=True, help='design-space / part body file (in --in-dir)')
    ap.add_argument('--fix', required=True, help='fixture surface file')
    ap.add_argument('--load', required=True, help='load surface file')
    ap.add_argument('--out-dir', default=None, help='defaults to --in-dir (writes the three '
                                                    'pipeline STLs; inputs are left untouched)')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='multiply the output into metres (mm input → 0.001)')
    ap.add_argument('--grid', type=int, default=192,
                    help='rasterization resolution along the longest axis')
    ap.add_argument('--close-mm', type=float, default=0.0,
                    help='morphological closing radius in MILLIMETRES. Merges neighbouring bodies; '
                         'keep below half the narrowest functional gap that must stay open, or that '
                         'gap gets bridged shut. Usually unnecessary once the interior is flood-filled')
    ap.add_argument('--dilate-mm', type=float, default=0.0,
                    help='outward offset, in MILLIMETRES — extra design freedom beyond the part. '
                         'Check what it costs: on the caliper 4 mm sealed the 32 mm rotor slot and '
                         'the whole envelope became a brick')
    ap.add_argument('--smooth-mm', type=float, default=3.0,
                    help='gaussian applied to the occupancy field before marching cubes, in '
                         'MILLIMETRES. This is what removes the voxel staircase; a later remesh '
                         'cannot, since remeshing preserves the surface it is given')
    ap.add_argument('--edge-mm', type=float, default=4.0,
                    help='isotropic remesh target edge for the envelope, in MILLIMETRES. Sets the '
                         'triangle count, and with it the in-loop tet count — gmsh keeps this '
                         'surface as its boundary. The shipped domains sit at 340-10k faces')
    ap.add_argument('--peg-pitch-mm', type=float, default=0.8,
                    help='voxel pitch for re-solidifying the fixture/load surfaces, in MILLIMETRES. '
                         'Small enough not to matter for boundary-condition proximity tests, and it '
                         'is what guarantees the pegs survive being welded by run_prep.py')
    ap.add_argument('--peg-close-vox', type=float, default=1.0,
                    help='closing radius in VOXELS applied to the peg grid. 1 removes diagonal-only '
                         'contacts; raise it when the pegs arrive as many fragments that still touch '
                         'after marching cubes (the remesh in run_prep then welds them into '
                         'non-manifold edges and manifold3d rejects the peg)')
    ap.add_argument('--min-frac', type=float, default=0.02,
                    help='drop components whose diagonal is under this fraction of the whole')
    ap.add_argument('--load-as-cavity', action='store_true',
                    help='the load interface is a HOLE, not a boss: subtract it from the design '
                         'domain so the generator can never fill it. Pair with '
                         'post_hybrid_union_clip --load-cavity (set stages.post.load_cavity)')
    ap.add_argument('--load-clearance-mm', type=float, default=0.0,
                    help='extra clearance carved around the load cavity, in MILLIMETRES')
    ap.add_argument('--load-x-sign', default='both', choices=['both', '+', '-'],
                    help="keep only the load bodies whose centre has this x sign (a single force "
                         "vector cannot express two banks pushing apart)")
    args = ap.parse_args()

    ind = Path(args.in_dir)
    out = Path(args.out_dir) if args.out_dir else ind
    out.mkdir(parents=True, exist_ok=True)
    mm = args.scale * 1000.0            # input unit → mm, for reporting

    print(f'[1/3] solidify  (drop fragments under {args.min_frac:.0%} of the diagonal)')
    part = solidify(ind / args.envelope, args.min_frac, 'envelope')
    fix = solidify(ind / args.fix, args.min_frac, 'fix')
    load = solidify(ind / args.load, args.min_frac, 'load')

    if args.load_x_sign != 'both':
        want = 1 if args.load_x_sign == '+' else -1
        parts = [c for c in load.split(only_watertight=False)
                 if np.sign(c.bounds.mean(0)[0]) == want]
        if not parts:
            raise SystemExit(f'no load body on the {args.load_x_sign}x side')
        load = manifold_union(parts) if len(parts) > 1 else parts[0]
        print(f'  load      kept the {args.load_x_sign}x bank: {len(parts)} bodies, '
              f'F={len(load.faces):,} non-manifold edges={bad_edges(load)}')

    peg_pitch = args.peg_pitch_mm / mm                 # mm → input units
    fix = voxel_solidify(fix, peg_pitch, 'fix', args.peg_close_vox)
    load = voxel_solidify(load, peg_pitch, 'load', args.peg_close_vox)

    print(f'[2/3] envelope  (close {args.close_mm:.1f} + dilate {args.dilate_mm:.1f} + '
          f'smooth {args.smooth_mm:.1f} mm, remesh edge {args.edge_mm:.1f} mm)')
    occ, org, pitch, raw = part_occupancy(ind / args.envelope, args.grid)
    cavity = None
    if args.load_as_cavity:
        # The interior flood-fill in part_occupancy fills EVERY enclosed void, which is right for
        # casting cavities but wrong for a functional one: a brake caliper's piston bores are voids
        # the pistons need, and filling them hands the generator 18% of the part as material it must
        # not place. Carve them back out here, with clearance, so the design domain never contained
        # them in the first place (post_hybrid_union_clip --load-cavity then enforces it exactly).
        from pysdf import SDF
        clr = args.load_clearance_mm / mm
        axes = [org[i] + np.arange(occ.shape[i]) * pitch for i in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
        inside = np.zeros(len(P), bool)
        for c in load.split(only_watertight=False):
            if len(c.faces) < 12:
                continue
            try:
                inside |= SDF(c.vertices.astype(np.float32),
                              c.faces.astype(np.uint32))(P) > -clr
            except Exception:
                pass
        cavity = inside.reshape(occ.shape)
        print(f'  cavity     load volume to carve: {(occ & cavity).sum() * pitch**3:.1f} u3 of '
              f'{occ.sum() * pitch**3:.1f} u3 '
              f'({(occ & cavity).sum()/max(occ.sum(),1)*100:.1f}%), clearance '
              f'{args.load_clearance_mm:.1f} mm — applied after smoothing', flush=True)
    dom, vol = envelope(occ, org, pitch, args.close_mm / mm / pitch, args.dilate_mm / mm / pitch,
                        (args.smooth_mm / mm) / pitch, args.edge_mm / mm, cavity=cavity)
    print(f'  envelope  F={len(dom.faces):,} watertight={dom.is_watertight} '
          f'bodies={len(dom.split(only_watertight=False))} non-manifold edges={bad_edges(dom)}  '
          f'volume {abs(dom.volume) / max(abs(raw.volume), 1e-12):.2f}x the raw part bbox-volume')
    d = (dom.bounds - raw.bounds) * mm
    print(f'  alignment  bbox delta vs the input part: min={np.round(d[0], 2)} '
          f'max={np.round(d[1], 2)} mm  (a uniform shift here would mean the grid origin was lost)')
    fi, li = inside_frac(dom, fix, 80), inside_frac(dom, load, 150)
    print(f'  containment  fix {fi*100:.1f}% / load {li*100:.1f}% of vertices inside the envelope '
          f'(the shipped domains sit at 7-33%: BC pegs are meant to poke out)')

    print(f'[3/3] write  (scale x{args.scale})')
    for name, m in (('original_DesignSpace', dom), ('fixed', fix), ('load', load)):
        s = m.copy()
        s.apply_scale(args.scale)
        p = out / f'{name}.stl'
        s.export(str(p))
        e = (s.bounds[1] - s.bounds[0]) * 1000
        print(f'  {p}  F={len(s.faces):,}  extent={np.round(e, 1)} mm')
    print(f'\nNow run:  python run_prep.py --domain {out.name}')


if __name__ == '__main__':
    main()
