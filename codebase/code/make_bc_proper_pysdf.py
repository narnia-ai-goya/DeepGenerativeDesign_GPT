"""Build the boundary-condition / design-domain voxel grid from STLs using pysdf —
voxel-center-inside-STL test. Produces data_real/<domain>/voxel.npz, which generation reads as
both `bc_proper` and `bracket_occ` (configs/<domain>.json, stages.mesh).

Why the center test: a conservative rasterizer (`trimesh.voxelized(pitch).fill()`) marks a
voxel TRUE whenever its bounding box overlaps the surface, which inflates the envelope by
~1 voxel pitch on every side (+30% volume observed). Testing the voxel CENTER against the
STL with pysdf (inside > 0) is an exact center-rasterization: envelope volume ≈ STL volume
within voxel discretization. The script prints both so the difference is visible.

One-time input preparation, NOT part of a reproduction run: voxel.npz is already shipped.
The exact per-domain commands the shipped grids were produced with are in code/README.md
("Input preparation"); they differ only in `--margin` (0.05 for bracket/link, 0.25 for the
motor mount) and the bracket's `--dilate-fix/--dilate-load 2`. All three use
`--res 64 --dilate-bracket 1`.

Usage:
  # from the package root, generation env active (needs pysdf, trimesh, scipy)
  python code/make_bc_proper_pysdf.py --bracket data_real/link/original_DesignSpace.stl \
      --fix data_real/link/fixed.stl --load data_real/link/load.stl \
      --res 64 --dilate-bracket 1 --out data_real/link/voxel.npz

Naming note: the NPZ key `bracket`, the CLI flags `--dilate-bracket` / `--bracket-occ` and the
config key `stages.mesh.bracket_occ` all mean the DESIGN-DOMAIN ENVELOPE of whichever domain is being
built — the name is a leftover from the bracket domain this script was first written for. On the
caliper, `bracket` holds the caliper envelope, not a bracket. Renaming would require regenerating the
three shipped voxel.npz files and touching every config plus the `d['bracket']` read in
generate_with_physics_guidance.py, so the name is kept and documented instead.

Output fields (NPZ):
  bracket   (R,R,R) bool — envelope mask  (voxel center inside envelope STL)
  bc        (R,R,R) bool — (fix ∪ load) ∩ bracket
  design    (R,R,R) bool — bracket ∩ ~bc
  fix       (R,R,R) bool — fix-only voxels
  load      (R,R,R) bool — load-only voxels
  pitch_xyz (3,)    f64  — voxel pitch per axis (aniso) or uniform
  origin    (3,)    f64  — world-frame origin (grid (0,0,0) center is origin + 0.5*pitch)
  pitch     scalar f64   — backwards-compat: pitch_xyz[0]
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import trimesh
from pysdf import SDF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_config import apply_config


def voxel_centers_inside(stl_mesh, R, origin, pitch_xyz):
    """Return (R,R,R) bool mask where voxel CENTER is inside the STL.
    pysdf convention: inside > 0.
    Voxel center: origin + (idx + 0.5) * pitch_xyz."""
    ax = (np.arange(R, dtype=np.float64) + 0.5) * pitch_xyz[0] + origin[0]
    ay = (np.arange(R, dtype=np.float64) + 0.5) * pitch_xyz[1] + origin[1]
    az = (np.arange(R, dtype=np.float64) + 0.5) * pitch_xyz[2] + origin[2]
    X, Y, Z = np.meshgrid(ax, ay, az, indexing='ij')
    qpts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    sdf = SDF(stl_mesh.vertices.astype(np.float32), stl_mesh.faces.astype(np.uint32))
    d = sdf(qpts).reshape(R, R, R)
    return d > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="run config JSON; stages.prep fills res / margin / "
                                                  "mode / dilate_fix / dilate_load / dilate_bracket "
                                                  "(CLI still overrides)")
    ap.add_argument("--bracket", default="data_real/bracket/original_DesignSpace.stl",
                    help="design-domain STL. The RAW (not remeshed) one: its bbox fixes this grid's "
                         "pitch and origin, i.e. the generation frame.")
    ap.add_argument("--fix",     default="data_real/bracket/fixed.stl")
    ap.add_argument("--load",    default="data_real/bracket/load.stl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=64,
                    help="grid resolution per axis (default 64)")
    ap.add_argument("--mode", choices=['uniscale', 'aniso'], default='uniscale',
                    help="uniscale: cubic grid covering envelope; "
                         "aniso: per-axis pitch matching bbox extents/R")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="fraction of envelope extent to pad around bbox")
    ap.add_argument("--dilate-fix", type=int, default=0,
                    help="binary dilation iterations on fix-region (post-rasterize)")
    ap.add_argument("--dilate-load", type=int, default=0)
    ap.add_argument("--dilate-bracket", type=int, default=0,
                    help="binary dilation iterations on envelope mask (ensures STL ⊆ envelope-voxel)")
    apply_config(ap, stage='prep')   # stages.prep → argparse defaults (res / margin / mode / dilations)
    args = ap.parse_args()

    br = trimesh.load(args.bracket, force='mesh', process=False)
    fx = trimesh.load(args.fix,     force='mesh', process=False)
    ld = trimesh.load(args.load,    force='mesh', process=False)
    print(f"[load] bracket V={len(br.vertices):,}  fix V={len(fx.vertices):,}  load V={len(ld.vertices):,}")

    # ── Compute grid origin and pitch ────────────────────────────────────────
    extents = br.bounds[1] - br.bounds[0]
    pad = args.margin * extents.max()
    bb_min = br.bounds[0] - pad
    bb_max = br.bounds[1] + pad

    R = args.res
    if args.mode == 'uniscale':
        # Cubic grid: padded longest extent / res = pitch, cube centred on the STL bbox
        longest = (bb_max - bb_min).max()
        pitch = longest / R
        pitch_xyz = np.array([pitch, pitch, pitch], dtype=np.float64)
        # Center the bracket bbox inside the cube
        cube_size = R * pitch
        center = (bb_min + bb_max) / 2.0
        origin = center - cube_size / 2.0
    else:  # aniso
        pitch_xyz = (bb_max - bb_min) / R
        origin = bb_min

    print(f"[grid] mode={args.mode}  R={R}  pitch_xyz={pitch_xyz*1000} mm  origin={origin*1000} mm")
    print(f"[grid] grid extent = {R*pitch_xyz*1000} mm  vs STL extent {extents*1000} mm")

    # ── Rasterize via pysdf voxel-center-inside ──────────────────────────────
    print(f"[raster] envelope...")
    br_v = voxel_centers_inside(br, R, origin, pitch_xyz)
    print(f"  envelope voxels={int(br_v.sum()):,}  voxel_vol={(pitch_xyz.prod()*1e9):.3f}mm³  → {int(br_v.sum())*pitch_xyz.prod()*1e9:,.0f} mm³")
    print(f"  STL volume reference = {br.volume*1e9:,.0f} mm³  (diff = {(int(br_v.sum())*pitch_xyz.prod()-br.volume)*1e9:+,.0f} mm³)")

    print(f"[raster] fix...")
    fix_v = voxel_centers_inside(fx, R, origin, pitch_xyz)
    print(f"  fix voxels={int(fix_v.sum()):,}  → {int(fix_v.sum())*pitch_xyz.prod()*1e9:,.0f} mm³  "
          f"(STL fix vol={fx.volume*1e9:,.0f} mm³)")

    print(f"[raster] load...")
    load_v = voxel_centers_inside(ld, R, origin, pitch_xyz)
    print(f"  load voxels={int(load_v.sum()):,}  → {int(load_v.sum())*pitch_xyz.prod()*1e9:,.0f} mm³  "
          f"(STL load vol={ld.volume*1e9:,.0f} mm³)")

    # ── Optional dilation ────────────────────────────────────────────────────
    if args.dilate_bracket > 0:
        from scipy.ndimage import binary_dilation
        br_v = binary_dilation(br_v, iterations=args.dilate_bracket)
        new_vol = int(br_v.sum()) * pitch_xyz.prod() * 1e9
        print(f"[dilate_bracket=+{args.dilate_bracket}] envelope → {int(br_v.sum()):,} voxels  "
              f"= {new_vol:,.0f} mm³  (STL ≤ envelope guaranteed)")
    if args.dilate_fix > 0:
        from scipy.ndimage import binary_dilation
        fix_v = binary_dilation(fix_v, iterations=args.dilate_fix) & br_v
    if args.dilate_load > 0:
        from scipy.ndimage import binary_dilation
        load_v = binary_dilation(load_v, iterations=args.dilate_load) & br_v

    # ── Compose masks ────────────────────────────────────────────────────────
    # The envelope must CONTAIN the boundary conditions. Without this line the BC voxels that fall
    # outside the rasterised envelope are dropped from `bc` (so bc_w never forces them solid) AND
    # land in out_t = 1 - bracket, where out_w drives them EMPTY — while post unions the same fix/
    # load STLs back in as material at those coordinates (`clip_ref = env ∪ fix ∪ load`). The
    # generator is taught "outside" exactly where post puts material.
    # Measured on the shipped grids: the fraction of BC voxels being driven empty was 0.0% (bracket,
    # which hides it because --dilate-fix/load 2 pushes its BC inside the envelope), 85.5%
    # (motor_mount), 82.3% (link) and 16.0% (caliper), against out_w=30 vs bc_w=10.
    # `design` is unaffected by construction: (br ∪ BC) & ~BC == br & ~BC.
    br_v = br_v | fix_v | load_v
    bc = (fix_v | load_v) & br_v
    design = br_v & ~bc

    print(f"\n[summary]")
    print(f"  bracket = {br_v.sum():>7,} voxels")
    print(f"  bc      = {bc.sum():>7,} voxels   ({bc.sum()/br_v.sum()*100:.1f}% of bracket)")
    print(f"    fix   = {fix_v.sum():>7,}")
    print(f"    load  = {load_v.sum():>7,}")
    print(f"  design  = {design.sum():>7,} voxels")

    # ── Save (NB: backwards-compat pitch scalar + new pitch_xyz / origin) ────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             bracket=br_v, bc=bc, design=design,
             fix=fix_v, load=load_v,
             pitch=float(pitch_xyz[0]),
             pitch_xyz=pitch_xyz.astype(np.float64),
             origin=origin.astype(np.float64))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
