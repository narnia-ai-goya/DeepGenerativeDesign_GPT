"""Final post-processing: SDF union (body ∪ dilated-BC ∪ pegs) + MESH-level envelope clip.

Produces the "hybrid" mesh that:
  - fills the BC (fix ∪ load) region and the peg STLs via a continuous SDF union
    (voxel-robust, no gaps at the body↔peg seam), and
  - clips to the envelope at the MESH level via manifold3d, so the envelope
    boundary follows the STL facets exactly (no marching-cubes staircase there).

Pipeline:
  1. Mesh-level union (manifold3d), body kept exact:
       union = body ∪ fix.stl ∪ load.stl (+ locally MC'd dilated-peg collars)
  2. Mesh clip  : manifold3d intersection
       final = union  ∩  (envelope.stl ∪ fix.stl ∪ load.stl)
     — extras kept in the clip reference so BC pegs protruding past the envelope survive.
  3. keep largest connected component.

Usage:
  python post_hybrid_union_clip.py --in mesh.obj --out mesh_hybrid.obj \
      --bracket-stl data_real/bracket/original_DesignSpace_remesh.stl \
      --fix data_real/bracket/fixed_remesh.stl --load data_real/bracket/load_remesh.stl
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh


def _clean(m):
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fix_inversion(m)
    if not m.is_watertight:
        try:
            trimesh.repair.fill_holes(m)
        except Exception:
            pass
    return m


def _mf(m):
    import manifold3d as m3d
    return m3d.Manifold(m3d.Mesh(
        vert_properties=np.asarray(m.vertices, dtype=np.float32),
        tri_verts=np.asarray(m.faces, dtype=np.uint32)))


def _ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def spike_trim(mesh, pitch=0.0007, radius=3, remesh_edge=0.0017, laplacian_iters=1):
    """Voxel morphological OPENING to strip thin in-plane flaps, then MC + isotropic remesh +
    Laplacian. Preserves rings/bars and does NOT close holes (opening only erodes-then-dilates
    solid). Used for thin lattices (link) whose web leaves sharp protruding tips after the union."""
    from scipy import ndimage
    from skimage import measure
    import pymeshlab
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda c: len(c.vertices))
    vg = mesh.voxelized(pitch).fill()
    vol = vg.matrix.astype(bool)
    op = ndimage.binary_opening(vol, structure=_ball(radius))
    lab, n = ndimage.label(op)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        op = (lab == (int(np.argmax(sizes)) + 1))
    op = np.pad(op, 2)
    v, f, _, _ = measure.marching_cubes(op.astype(float), 0.5)
    # Map grid indices back through the voxel grid's own transform (scale AND translation), undoing
    # the pad first. `v * pitch` would scale correctly but silently drop the grid origin, planting
    # the trimmed mesh at ~[0,0,0] instead of where it came from. That shift is what breaks the
    # boundary conditions downstream: the fixture/load STLs stay put, so a domain whose coordinates
    # are not already near the origin loses every BC facet (measured on the caliper: the part moved
    # +56/+86/-31 mm and the a-posteriori FEA reported "no fix facets found"). A domain that happens
    # to sit near the origin, like the link, only moved ~1 mm and hid the bug.
    v = trimesh.transform_points(v - 2.0, vg.transform)
    tm = trimesh.Trimesh(v, f, process=False)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(tm.vertices, tm.faces))
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(remesh_edge),
                                            iterations=10, adaptive=False)
    cm = ms.current_mesh()
    tm = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=False)
    if laplacian_iters > 0:
        tm = trimesh.smoothing.filter_laplacian(tm, lamb=0.5, iterations=laplacian_iters)
    return tm


def wedge_trim(mesh, radius=0.0004, sphere_subdiv=1):
    """Morphological OPENING at the *wedge* scale to strip knife-edge slivers left by the envelope
    clip. Removes material thinner than 2*radius and nothing else.

    Done in the mesh domain with manifold3d Minkowski ops, NOT on a voxel grid. The voxel route was
    tried first and is unusable here: trimesh's `voxelized()` marks every voxel the surface passes
    through as occupied, so `fill()` returns the body dilated by ~half a voxel, and the MC round-trip
    hands back a part inflated by that much over its entire area. On a thin-walled body that is not a
    rounding error — measured on this caliper at pitch 0.4 mm, two styles came back +10.2% and +9.4%
    in volume while the opening itself had removed only 0.18% and 0.14% of voxels. The Minkowski
    route on the same mesh costs -0.07%, i.e. it removes the wedges and nothing else.

    Why this is separate from spike_trim
    ------------------------------------
    spike_trim targets protruding flaps on thin lattices: pitch 0.7 mm, ball radius 3 → it erases
    features up to ~4 mm, which is more than the caliper's median wall (3.2 mm), and it finishes
    with a Laplacian pass that shrinks volume. What the envelope clip produces is a different
    defect at a different scale: where the generated surface runs nearly *tangent* to the envelope,
    `union ^ clip_ref` leaves wedges 0.03-0.25 mm thick (measured: the a-posteriori von Mises
    maximum sits 0.03-0.25 mm from the surface, 8-60 mm away from any BC facet, and reaches
    1.4-2.4 GPa — 4-8x the yield of the steel being modelled). Linear tetrahedra turn those wedges
    into stress singularities, so the reported compliance does not converge: refining one caliper
    style from 30k to 346k elements moved it 287 -> 477 -> 618 mJ, still climbing 30% per step,
    while a clean style moved 23.9 -> 27.6 -> 29.5 mJ (increments 15% -> 7%).

    The defect scales with how *tangentially* the body meets the envelope, which is why it is a
    smooth-envelope problem: the fraction of delivered surface lying within 0.5 mm of the envelope
    is 2.8% (link), 17-19% (motor_mount), 47-48% (bracket) and 57-58% (caliper), and the caliper's
    envelope is a marching-cubes SDF isosurface (mean facet angle 1.98 deg) rather than CAD faces
    (18.69 deg) — so its clip band is both the largest and the only one that is curved.

    Element quality is NOT the cause and is deliberately not touched here: the style with the worst
    tetrahedron (normalised radius ratio 9.6e-06) is the *best*-behaved one, and bracket has the
    same degenerate-element rate as the divergent caliper style while staying at 37 MPa.

    The opening is volume-monotone by construction (erode-then-dilate can only remove), so the
    delivered volume stays interpretable. Rounding convex edges by `radius` is the intended cost — a
    0.1 mm knife edge is not castable or machinable anyway.

    A large volume drop here is a *result*, not a failure: it means the body was only held together
    by ligaments thinner than 2*radius. Measured at 0.8 mm on this caliper, `de_crackle` lost 46.7%
    of its volume while `de_kagome` and `de_trilattice` lost 0.18% and 0.14% — crackle's 751 mJ was
    a near-severed structure, not a stiffness difference.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda c: len(c.vertices))
    import manifold3d as m3d
    ball = trimesh.creation.icosphere(subdivisions=sphere_subdiv, radius=radius)
    v0 = abs(mesh.volume)
    opened = m3d.Manifold.minkowski_sum(
        m3d.Manifold.minkowski_difference(_mf(mesh), _mf(ball)), _mf(ball))
    bm = opened.to_mesh()
    out = trimesh.Trimesh(bm.vert_properties.copy(), bm.tri_verts.copy(), process=True)
    out.merge_vertices()
    # The Minkowski sum tessellates each swept face separately, so it hands back the body plus a
    # cloud of dust shells (measured: 1,755 shells of which the largest held 98.12 of 98.16 cm³).
    # Same reason the caller strips shells after the boolean — a tet mesher chokes on the dust.
    shells = out.split(only_watertight=False)
    if len(shells) > 1:
        v_all = sum(abs(s.volume) for s in shells)
        out = max(shells, key=lambda s: abs(s.volume))
        out.remove_unreferenced_vertices()
        dropped = (v_all - abs(out.volume)) * 1e9
        # Dust and severed fragments look the same here and both must go (a fragment that no longer
        # touches the body carries no load), but they mean different things — so report the volume.
        # Dust is ~0 mm³; a large number means the opening disconnected real structure, i.e. the body
        # was held together by ligaments thinner than 2*radius. Measured: de_kagome/de_trilattice
        # dropped ~0.05 cm³ of dust, de_crackle dropped 39 cm³ (46.6% of itself).
        kind = ('dust' if dropped < 100 else
                f'SEVERED structure — ligaments thinner than {2*radius*1e3:.2f}mm')
        print(f'  [wedge-trim] {len(shells)} shells → 1, discarded {dropped:,.0f} mm³ ({kind})',
              flush=True)
    if out.is_watertight and out.volume < 0:
        out.invert()
    print(f'  [wedge-trim] Minkowski opening r={radius*1e3:.2f}mm '
          f'(removes < {2*radius*1e3:.2f}mm): vol {v0*1e9:,.0f} → {abs(out.volume)*1e9:,.0f} mm³ '
          f'({(abs(out.volume)/max(v0,1e-30)-1)*100:+.2f}%)  F={len(out.faces):,}', flush=True)
    return out


def main():
    # paper ref: main text, post-hoc boolean refinement that forces the design-domain and
    #            boundary-condition constraints to hold exactly; Supplementary Evaluation
    #            Protocol, a posteriori verification chain (interface fusion + design-domain
    #            intersection).
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='run config JSON; stages.post fills bracket_stl / '
                                                   'fix / load / peg_dilate_mm (CLI still overrides)')
    ap.add_argument('--in', dest='inp', required=True, help='raw sparse mesh.obj')
    ap.add_argument('--out', required=True, help='output hybrid mesh path')
    ap.add_argument('--bracket-stl', default='data_real/bracket/original_DesignSpace_remesh.stl',
                    help='envelope STL for mesh clip. Default = re-tessellated (uniform, watertight) '
                         'version → cleaner boolean seam. Falls back to original if missing.')
    ap.add_argument('--fix', default='data_real/bracket/fixed_remesh.stl')
    ap.add_argument('--load', default='data_real/bracket/load_remesh.stl')
    ap.add_argument('--peg-dilate-mm', type=float, default=0.0,
                    help='dilate the fix/load peg SDFs by this many mm before the union — fills a '
                         'smooth collar around the pegs at POST time (no BC baking at generation). '
                         'The final clip (envelope ∪ raw pegs) trims any dilated mass outside.')
    ap.add_argument('--load-cavity', action='store_true',
                    help='treat the load surface as a CAVITY, not a peg: skip its union and subtract '
                         'it from the result, so the space it encloses stays empty. The default peg '
                         'treatment is right when the load interface is a boss or pin sticking out of '
                         'the envelope (every shipped domain: bracket load peg sits 99.6%% outside its '
                         'design domain). It is wrong when the interface is a hole — a brake '
                         "caliper's piston bores are load surfaces AND functional voids, and unioning "
                         'them fills the bores solid (measured: 99.8%% of the bore volume became '
                         'material, 18%% of the part). The fixture is still treated as a peg.')
    ap.add_argument('--spike-trim', action='store_true',
                    help='final voxel morphological OPENING to remove thin in-plane flaps/spikes '
                         '(thin lattices, e.g. link: the bow-tie web leaves sharp protruding tips). '
                         'Opening erodes-then-dilates solid → drops the flaps without closing holes; '
                         'then MC + isotropic remesh + Laplacian. Off by default (bracket/mm need it not).')
    ap.add_argument('--trim-pitch', type=float, default=0.0007, help='spike-trim voxel pitch (m)')
    ap.add_argument('--trim-radius', type=int, default=3, help='spike-trim opening ball radius (voxels)')
    ap.add_argument('--trim-remesh-edge', type=float, default=0.0017, help='spike-trim remesh edge (m)')
    ap.add_argument('--trim-laplacian-iters', type=int, default=1, help='spike-trim Laplacian iters')
    ap.add_argument('--wedge-trim', action='store_true',
                    help='remove knife-edge slivers left by the envelope clip (see wedge_trim). '
                         'Needed when the envelope is a smooth isosurface rather than CAD faces, '
                         'because the body then meets it tangentially over a wide band. Independent '
                         'of --spike-trim: different scale, different defect.')
    ap.add_argument('--wedge-radius', type=float, default=0.0004,
                    help='wedge-trim opening radius in METRES; removes material thinner than 2x this '
                         '(0.4mm → <0.8mm). Keep it well under the design median wall thickness.')
    ap.add_argument('--wedge-sphere-subdiv', type=int, default=1,
                    help='icosphere subdivisions for the Minkowski structuring element (1 = 42 verts; '
                         'higher is rounder but the Minkowski cost grows fast)')
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_config import apply_config
    apply_config(ap, stage='post')          # stages.post → argparse defaults (paths/peg/res)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: union (no envelope clip) ─────────────────────────────────────
    # Union only: body ∪ fix.stl ∪ load.stl. The 64³ voxel BC mass is deliberately NOT
    # unioned in — it is blocky and produces an angular protrusion below the peg, while
    # the BC region is already forced solid at generation (STL-SDF) and the actual peg
    # STLs are unioned here, so the voxel mass would be redundant.
    env = _clean(trimesh.load(args.bracket_stl, force='mesh', process=False))
    fix = _clean(trimesh.load(args.fix, force='mesh', process=False))
    load = _clean(trimesh.load(args.load, force='mesh', process=False))

    # ── Step 1 (default): MESH-LEVEL union — body untouched (no resample/staircase).
    # Dilated-peg collar pieces are MC'd from LOCAL fine SDF grids (peg bbox only,
    # ~0.25mm pitch) and unioned with the body + raw pegs in manifold3d.
    print('[1/3] mesh-level union (body kept exact)...', flush=True)
    body = trimesh.load(args.inp, force='mesh', process=True)
    body.merge_vertices()
    _bc = body.split(only_watertight=False)
    if len(_bc) > 1:
        body = max(_bc, key=lambda c: len(c.faces))
        body.remove_unreferenced_vertices()
    body = _clean(body)
    union_mf = _mf(body) + _mf(fix)
    if not args.load_cavity:
        union_mf = union_mf + _mf(load)     # peg: the load interface is material
    if args.peg_dilate_mm > 0:
        # ROBUST collar: the pegs can be MULTI-component solids
        # (load.stl = TWO pins!) and the raw STLs are unwelded soup — pysdf signs
        # break on soup (~170 fragments) and a largest-component filter then threw
        # away one pin's collar entirely (asymmetric fill, load +y vs -y).
        # Fix: (1) WELD the peg (process=True rebuild) → pysdf signs are clean;
        #      (2) keep EVERY component above a volume threshold (multi-pin collars
        #          survive, sub-mm dust dies) and union them all.
        from pysdf import SDF as _PSDF
        from skimage.measure import marching_cubes as _mc
        dil = args.peg_dilate_mm / 1000.0
        _pegs = [('fix', fix)] + ([] if args.load_cavity else [('load', load)])
        for peg_name, peg in _pegs:
            peg_w = trimesh.Trimesh(peg.vertices.copy(), peg.faces.copy(), process=True)
            peg_w.merge_vertices()
            pad = dil + 0.002
            lo = peg_w.bounds[0] - pad; hi = peg_w.bounds[1] + pad
            ext = hi - lo
            target_pitch = 0.00025                       # 0.25mm
            n = np.clip(np.ceil(ext / target_pitch).astype(int) + 1, 32, 448)
            pitch = ext / (n - 1)
            axes = [np.linspace(lo[i], hi[i], n[i]) for i in range(3)]
            Q = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
            d = _PSDF(peg_w.vertices.astype(np.float32), peg_w.faces.astype(np.uint32))(Q)
            vol = (d + dil).reshape(*n)                  # inside(>0) = dilated peg
            v, f, _, _ = _mc(vol, level=0.0)
            v = lo + v * pitch                           # grid idx → world
            piece = trimesh.Trimesh(v, f, process=True)
            piece.merge_vertices()
            _pc = piece.split(only_watertight=False)
            _kept = [c for c in _pc if abs(c.volume) * 1e9 > 1.0]   # > 1 mm³
            if not _kept:
                _kept = [max(_pc, key=lambda c: len(c.faces))]
            print(f'  dilated {peg_name}: {len(_pc)} comps -> keep {len(_kept)} '
                  f'(vols mm3: {[round(abs(c.volume)*1e9) for c in _kept]})', flush=True)
            for piece_c in _kept:
                piece_c.remove_unreferenced_vertices()
                if piece_c.is_watertight and piece_c.volume < 0:
                    piece_c.invert()
                union_mf = union_mf + _mf(piece_c)

    # ── Step 2: mesh-level intersect with (envelope ∪ fix ∪ load) ────────────
    # paper ref: main text, design-domain intersection of the fused body with the envelope
    #            (plus BC pegs) so the design-domain constraint holds exactly.
    print('[2/3] mesh-level envelope clip (manifold3d)...', flush=True)
    clip_ref = _mf(env) + _mf(fix)
    if not args.load_cavity:
        clip_ref = clip_ref + _mf(load)             # peg: keep mass that protrudes past the envelope
    final_mf = union_mf ^ clip_ref                  # ^ = intersection in manifold3d
    if args.load_cavity:
        # Force the cavity open. The design domain should already exclude it, but the generated body
        # can bulge in and the boolean is what makes the constraint exact — same reason the envelope
        # clip exists.
        final_mf = final_mf - _mf(load)
        print(f'  [load-cavity] subtracted the load volume '
              f'({abs(load.volume)*1e6:.1f} cm³) — bores stay open', flush=True)

    # ── Step 3: keep largest connected manifold (watertight-preserving) ──────
    # Use manifold3d decompose (NOT trimesh split) so the result stays a valid
    # watertight manifold. Sort by ABSOLUTE volume — the SDF-union body often has
    # inverted winding (negative volume), which decompose/volume would otherwise
    # mis-rank, dropping the real bracket for a tiny junk shell.
    parts = sorted(final_mf.decompose(), key=lambda p: abs(p.volume()), reverse=True)
    print(f'[3/3] {len(parts)} manifolds → largest by |volume|', flush=True)
    big = parts[0] if parts else final_mf
    bm = big.to_mesh()
    final = trimesh.Trimesh(vertices=bm.vert_properties.copy(),
                            faces=bm.tri_verts.copy(), process=True)
    final.merge_vertices()   # without merging, dust shells share stale vertex refs and split() sees ONE comp
    # manifold3d keeps interior micro-void shells as part of one manifold; strip
    # everything but the largest shell so the saved mesh is a single clean body.
    _fc = final.split(only_watertight=False)
    if len(_fc) > 1:
        n0 = len(_fc)
        final = max(_fc, key=lambda c: len(c.faces))
        final.remove_unreferenced_vertices()
        print(f'  [largest-shell] {n0} shells -> 1 (dust/micro-void removed)', flush=True)
    if final.is_watertight and final.volume < 0:
        final.invert()   # outward normals (positive signed volume)

    if args.wedge_trim:
        print('[wedge-trim] opening at the clip-wedge scale → remove knife-edge slivers', flush=True)
        v0 = abs(final.volume)
        final = wedge_trim(final, radius=args.wedge_radius, sphere_subdiv=args.wedge_sphere_subdiv)
        if final.is_watertight and final.volume < 0:
            final.invert()
        print(f'  [wedge-trim] vol {v0*1e9:,.0f} → {abs(final.volume)*1e9:,.0f} mm³ '
              f'({(abs(final.volume)/v0-1)*100:+.2f}%)', flush=True)

    if args.spike_trim:
        print('[spike-trim] voxel opening → remove thin flaps → remesh + smooth', flush=True)
        final = spike_trim(final, pitch=args.trim_pitch, radius=args.trim_radius,
                           remesh_edge=args.trim_remesh_edge, laplacian_iters=args.trim_laplacian_iters)
        if final.is_watertight and final.volume < 0:
            final.invert()

    final.export(str(out))
    print(f'[saved] {out}  V={len(final.vertices):,} F={len(final.faces):,}  '
          f'vol={abs(final.volume)*1e9:,.0f} mm³  watertight={final.is_watertight}', flush=True)


if __name__ == '__main__':
    main()
