"""FEA compliance on TET mesh built directly from polished surface mesh.

Pipeline:
  1. pymeshfix: repair non-manifold/holes/duplicate-faces → watertight surface
  2. gmsh: surface → tetrahedral volume mesh
  3. dolfinx: linear elasticity, Dirichlet at fix.stl facets, Neumann (traction) at load.stl facets
  4. compute compliance = u·F, u_max, σ_vm, and strain-energy concentration
  5. write VTU fields and a JSON summary

Run in `fenics` conda env.
"""
import argparse, os, sys
import numpy as np
import trimesh
import gmsh
import pymeshfix

from behavior_descriptors import strain_energy_concentration


def repair_to_watertight(mesh_path, out_path, decimate_target_faces=40000, taubin_iters=5):
    m = trimesh.load(mesh_path, force='mesh', process=False)
    # 1) Decimate input surface
    if len(m.faces) > decimate_target_faces:
        try:
            m = m.simplify_quadric_decimation(face_count=decimate_target_faces)
            print(f'  decimated to F={len(m.faces):,} (target={decimate_target_faces})')
        except Exception as e:
            print(f'  decimation failed: {e}')
    # 2) Taubin smoothing — volume-preserving, removes voxel staircase, helps gmsh
    if taubin_iters > 0:
        try:
            m = trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=taubin_iters)
            print(f'  Taubin smoothed ({taubin_iters} iters)')
        except Exception as e:
            print(f'  Taubin failed: {e}')
    # 3) pymeshfix repair → watertight
    mf = pymeshfix.MeshFix(m.vertices.astype(np.float64), m.faces.astype(np.int32))
    mf.repair()
    # pymeshfix 0.18: results in .points (vertices) and .faces (triangles, already (F,3) int)
    verts = np.asarray(mf.points)
    faces = np.asarray(mf.faces)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 4)[:, 1:]   # vtk-style (3, v0, v1, v2) per row
    print(f'  pymeshfix output: verts={verts.shape}  faces={faces.shape}')
    fixed = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    fixed.export(out_path)
    print(f'  repair: V {len(m.vertices):,}→{len(fixed.vertices):,}, '
          f'F {len(m.faces):,}→{len(fixed.faces):,}, '
          f'watertight={fixed.is_watertight}')
    return fixed


def surface_to_tet(stl_path, msh_path, mesh_size=0.01, algo3d=10):
    """Tet meshing via gmsh — no surface classification (single closed shell).
    algo3d: 10=HXT (fast, but occasionally crashes with 'double free'), 1=Delaunay (robust)."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Verbosity', 0)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMax', mesh_size)
        gmsh.option.setNumber('Mesh.CharacteristicLengthMin', mesh_size * 0.3)
        gmsh.option.setNumber('Mesh.Algorithm3D', int(algo3d))
        gmsh.merge(stl_path)
        surfs = gmsh.model.getEntities(2)
        sl = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfs])
        vol = gmsh.model.geo.addVolume([sl])
        gmsh.model.geo.synchronize()
        # Tag volume + surface so dolfinx.io.gmshio.read_from_msh finds cell_information
        gmsh.model.addPhysicalGroup(3, [vol], tag=1, name='Vol')
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surfs], tag=2, name='Surf')
        gmsh.model.mesh.generate(3)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()


def main():
    # paper ref: Supplementary, Evaluation Protocol Details, a posteriori verification
    #            (linear elasticity, E = 110 GPa, nu = 0.3, 1000 N load, fixture clamped;
    #            Gmsh tet + DOLFINx).
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', required=True, help='polished surface mesh.obj')
    ap.add_argument('--fix-stl', required=True)
    ap.add_argument('--load-stl', required=True)
    ap.add_argument('--out', required=True, help='output dir')
    ap.add_argument('--mesh-size', type=float, default=0.005, help='target tet edge length (m)')
    ap.add_argument('--tet-algo', type=int, default=10, help='gmsh Algorithm3D: 10=HXT, 1=Delaunay(robust)')
    ap.add_argument('--force-N', type=float, default=1000.0)
    ap.add_argument('--force-dir', default='0,0,-1',
                    help="load direction: a vector like '0,0,-1', or 'spread:x' for an opposed load "
                         "whose sign flips across the load surface's mid-plane along that axis "
                         "(a fixed brake caliper's two piston banks push its halves apart)")
    ap.add_argument('--E-GPa', type=float, default=110.0)
    ap.add_argument('--nu', type=float, default=0.34)
    ap.add_argument('--bc-radius', type=float, default=0.01, help='facets within this radius of BC peg surface treated as BC')
    ap.add_argument('--no-repair', action='store_true',
                    help='skip decimate+Taubin+pymeshfix; feed the mesh straight to gmsh. '
                         'Use for already-clean watertight manifolds (e.g. manifold3d hybrid) — '
                         'pymeshfix otherwise discards peg regions and breaks BC facet detection.')
    args = ap.parse_args()

    out = args.out
    os.makedirs(out, exist_ok=True)

    fixed_path = os.path.join(out, 'mesh_fixed.stl')
    if args.no_repair:
        print('[1/3] no-repair: feeding mesh directly to gmsh (assumed clean watertight)')
        fixed = trimesh.load(args.mesh, force='mesh', process=False)
        fixed.merge_vertices()
        if fixed.is_watertight and fixed.volume < 0:
            fixed.invert()
        fixed.export(fixed_path)
        print(f'  V={len(fixed.vertices):,} F={len(fixed.faces):,} watertight={fixed.is_watertight}')
    else:
        print('[1/3] repair non-manifold → watertight')
        fixed = repair_to_watertight(args.mesh, fixed_path)
    if not fixed.is_watertight:
        print('  WARN: still not watertight after repair — tet mesh may fail')

    print('[2/3] gmsh tet mesh')
    msh_path = os.path.join(out, 'tet.msh')
    try:
        surface_to_tet(fixed_path, msh_path, mesh_size=args.mesh_size, algo3d=args.tet_algo)
        print(f'  wrote {msh_path}')
    except Exception as e:
        print(f'  gmsh failed: {e}')
        return 2

    print('[3/3] dolfinx linear elasticity')
    import dolfinx, ufl
    from dolfinx import fem, io, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    msh, ct, ft = io.gmshio.read_from_msh(msh_path, MPI.COMM_WORLD, gdim=3)
    print(f'  tet mesh: nodes={msh.geometry.x.shape[0]:,}  cells={msh.topology.index_map(3).size_global:,}')

    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    fcoords = dolfinx.mesh.compute_midpoints(msh, fdim, facets)

    fix_mesh = trimesh.load(args.fix_stl, force='mesh', process=False)
    load_mesh = trimesh.load(args.load_stl, force='mesh', process=False)
    tree_f = cKDTree(fix_mesh.vertices)
    tree_l = cKDTree(load_mesh.vertices)
    d_f, _ = tree_f.query(fcoords)
    d_l, _ = tree_l.query(fcoords)
    fix_mask = d_f < args.bc_radius
    load_mask = (d_l < args.bc_radius) & ~fix_mask
    fix_facets  = facets[fix_mask]
    load_facets = facets[load_mask]
    print(f'  fix facets: {len(fix_facets):,}, load facets: {len(load_facets):,}')

    if len(fix_facets) == 0:
        print(f'  ERROR: no fix facets found (bc_radius={args.bc_radius*1000:.1f}mm)')
        return 3
    if len(load_facets) == 0:
        print(f'  ERROR: no load facets found')
        return 3

    V = fem.functionspace(msh, ('Lagrange', 1, (3,)))
    nu = args.nu; E = args.E_GPa * 1e9
    lam = E * nu / ((1+nu) * (1-2*nu))
    mu  = E / (2*(1+nu))

    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u):
        I = ufl.Identity(3)
        return lam * ufl.tr(epsilon(u)) * I + 2*mu * epsilon(u)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    fix_dofs = fem.locate_dofs_topological(V, fdim, fix_facets)
    u_D = np.zeros(3, dtype=default_scalar_type)
    bc_dir = fem.dirichletbc(u_D, fix_dofs, V)

    LOAD_TAG = 1
    facet_tags = dolfinx.mesh.meshtags(msh, fdim, load_facets.astype(np.int32),
                                       np.full(load_facets.size, LOAD_TAG, dtype=np.int32))
    ds = ufl.Measure('ds', domain=msh, subdomain_data=facet_tags)
    load_area = fem.assemble_scalar(fem.form(1 * ds(LOAD_TAG)))
    if str(args.force_dir).startswith('spread:'):
        # Opposed load: the traction flips sign across the load surface's mid-plane, so two facing
        # banks are pushed APART rather than both being pushed the same way. This is the load case a
        # fixed brake caliper actually sees — its pistons press the pads inward against the rotor, so
        # the reaction on the caliper body spreads the two halves outward. A single uniform vector
        # cannot express it: it would push one bank into the rotor and the other away from it.
        # The mid-plane comes from the load STL's own bounding box, which is exact for two banks.
        ax = {'x': 0, 'y': 1, 'z': 2}[str(args.force_dir).split(':')[1].strip()[-1]]
        split = float(load_mesh.bounds.mean(0)[ax])
        mag = args.force_N / max(load_area, 1e-12)
        xco = ufl.SpatialCoordinate(msh)
        e = [0.0, 0.0, 0.0]; e[ax] = 1.0
        sgn = ufl.conditional(xco[ax] > split, 1.0, -1.0)
        t = ufl.as_vector([sgn * mag * e[0], sgn * mag * e[1], sgn * mag * e[2]])
        n_pos = int((fcoords[load_mask][:, ax] > split).sum())
        print(f'  load area: {load_area*1e6:.1f} mm²  traction: {mag*1e-6:.3f} MPa  '
              f'spread about {"xyz"[ax]}={split*1000:.1f} mm '
              f'({n_pos} facets +, {len(load_facets)-n_pos} facets -)')
        fdir = np.array(e)                      # recorded direction axis (sign is per-facet)
    else:
        fdir = np.array([float(x) for x in args.force_dir.split(',')])
        t_vec = fdir / (np.linalg.norm(fdir) + 1e-12) * args.force_N / max(load_area, 1e-12)
        print(f'  load area: {load_area*1e6:.1f} mm²  traction: {np.linalg.norm(t_vec)*1e-6:.3f} MPa')
        t = fem.Constant(msh, default_scalar_type(t_vec))
    L = ufl.dot(t, v) * ds(LOAD_TAG)

    prob = LinearProblem(a, L, bcs=[bc_dir], u=None,
                         petsc_options={'ksp_type':'cg', 'pc_type':'hypre',
                                        'pc_hypre_type':'boomeramg', 'ksp_rtol':1e-8})
    u_sol = prob.solve()
    u_node = u_sol.x.array.reshape(-1, 3)
    u_mag = np.linalg.norm(u_node, axis=1)
    u_max = float(u_mag.max())

    # paper ref: Supplementary, Evaluation Protocol Details, a posteriori compliance from the
    #            linear-elastic verification solve.
    sef = fem.form(ufl.inner(sigma(u_sol), epsilon(u_sol)) * ufl.dx)
    compliance = float(fem.assemble_scalar(sef))

    # von Mises at cell level (DG0). dolfinx 0.9 API: interpolate(expr, cells)
    DG0 = fem.functionspace(msh, ('DG', 0))
    s = sigma(u_sol)
    dev = s - (1/3) * ufl.tr(s) * ufl.Identity(3)
    vm_expr = ufl.sqrt(1.5 * ufl.inner(dev, dev))
    vm_fn = fem.Function(DG0)
    expr = fem.Expression(vm_expr, DG0.element.interpolation_points())
    n_cells_local = msh.topology.index_map(msh.topology.dim).size_local
    cells = np.arange(n_cells_local, dtype=np.int32)
    vm_fn.interpolate(expr, cells, cells)
    vm = vm_fn.x.array
    vm_max = float(vm.max())
    vm_mean = float(vm.mean())

    # A mechanics behavior descriptor distinct from compliance. Compliance is
    # total strain energy; this records how concentrated that energy is in
    # space after normalizing by each tet's volume.
    energy_density_expr = ufl.inner(sigma(u_sol), epsilon(u_sol))
    energy_density_fn = fem.Function(DG0)
    energy_expr = fem.Expression(energy_density_expr, DG0.element.interpolation_points())
    energy_density_fn.interpolate(energy_expr, cells, cells)
    energy_density = np.maximum(energy_density_fn.x.array[:n_cells_local], 0.0)

    cells_arr = msh.geometry.dofmap.reshape(-1, 4)
    points_arr = msh.geometry.x
    local_tets = cells_arr[:n_cells_local]
    a_xyz, b_xyz, c_xyz, d_xyz = [points_arr[local_tets[:, i]] for i in range(4)]
    cell_volume = np.abs(np.einsum('ij,ij->i', b_xyz - a_xyz,
                                    np.cross(c_xyz - a_xyz, d_xyz - a_xyz))) / 6.0
    cell_energy = energy_density * cell_volume
    # Verification currently runs in serial. Keep the calculation explicit so
    # a future distributed runner cannot silently report a partition-local BD.
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError('strain-energy concentration export currently requires one MPI rank')
    energy_concentration = strain_energy_concentration(cell_energy, cell_volume)

    print(f'  u_max: {u_max*1e3:.4f} mm  vm_max: {vm_max/1e6:.2f} MPa  vm_mean: {vm_mean/1e6:.2f} MPa')
    print(f'  compliance: {compliance:.4e} J')
    print(f'  strain-energy concentration: {energy_concentration:.4f}')

    # Write both VTX (ADIOS2 .bp) and VTU (single-file XML) for easier visualization
    with io.VTXWriter(MPI.COMM_WORLD, os.path.join(out, 'tet_u.bp'), [u_sol]) as f: f.write(0.0)
    with io.VTXWriter(MPI.COMM_WORLD, os.path.join(out, 'tet_vm.bp'), [vm_fn]) as f: f.write(0.0)
    # Export von Mises as VTU via meshio for pyvista compatibility
    try:
        import meshio
        vm_cell = vm_fn.x.array
        mio = meshio.Mesh(points=points_arr,
                          cells=[("tetra", cells_arr)],
                          cell_data={"von_mises_Pa": [vm_cell],
                                     "strain_energy_density_J_m3": [energy_density_fn.x.array],
                                     "strain_energy_J": [energy_density_fn.x.array * cell_volume]})
        mio.write(os.path.join(out, 'tet_vm.vtu'))
        u_pt = u_sol.x.array.reshape(-1, 3)
        u_mag_pt = np.linalg.norm(u_pt, axis=1)
        mio2 = meshio.Mesh(points=points_arr,
                           cells=[("tetra", cells_arr)],
                           point_data={"u_magnitude_m": u_mag_pt,
                                       "u_xyz_m": u_pt})
        mio2.write(os.path.join(out, 'tet_u.vtu'))
        print('  → tet_vm.vtu + tet_u.vtu')
    except Exception as e:
        print(f'  vtu export failed: {e}')

    import json
    json.dump({'u_max': u_max, 'vm_max': vm_max, 'vm_mean': vm_mean,
               'compliance': compliance, 'force_N': args.force_N,
               'strain_energy_concentration': energy_concentration,
               'strain_energy_concentration_definition':
                   '1-exp(-KL(element_energy_share || element_volume_share))',
               'force_dir': str(args.force_dir), 'n_tet_cells': msh.topology.index_map(3).size_global,
               'n_tet_nodes': msh.geometry.x.shape[0]},
              open(os.path.join(out, 'fea_tet_summary.json'), 'w'), indent=2)
    print(f'  → {out}')


if __name__ == '__main__':
    sys.exit(main() or 0)
