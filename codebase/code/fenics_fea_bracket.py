"""FEniCS-based FEA for bracket/caliper topology optimization.

Standalone — must be run with fenics conda env:
    python fenics_fea_bracket.py ...

Adapted from an earlier FEniCS FEA implementation with:
  - DOMAIN_DIR via --domain-dir CLI arg (was hardcoded)
  - Mesh cache: build once, reuse on subsequent calls (--mesh-cache)
  - Compliance + per-element sensitivity (∂C/∂ρ) output

Usage:
    python fenics_fea_bracket.py \\
        --domain-dir data_real/bracket/fea_domain \\
        --density rho.npy --nodes nodes.npy \\
        --output dc.npy --mesh-size 0.006 --mesh-cache /tmp/bracket_fea.msh
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmshio
import gmsh
from scipy.spatial import cKDTree
import trimesh


def build_mesh(domain_dir: str, mesh_size: float, out_msh: str):
    """Generate the in-loop tet mesh from original_DesignSpace.stl via gmsh.

    Two recipes, tried in order:

      1. reparametrize — classifySurfaces(40 deg) + createGeometry() rebuilds CAD-like patches from
         the triangle soup before meshing. This is what the coarse CAD tessellations want (the
         shipped design domains are 340-10k faces) and it gives the cleanest patch boundaries.
      2. discrete surface — merge the STL and build the volume straight off its discrete surface
         entities, skipping reparametrization. Same recipe fea_tet_from_mesh.py uses, and it
         handles surfaces that (1) cannot parametrize: a dense or organically smooth boundary
         makes the 40 deg classification emit patches whose topology gmsh then rejects
         ("Wrong topology of boundary mesh for parametrization").

    Recipe 1 is kept first so every domain that already meshed through it keeps producing a
    byte-identical mesh; 2 only runs when 1 raises."""
    def _mesh(reparametrize):
        gmsh.initialize()
        try:
            gmsh.option.setNumber('General.Verbosity', 0)
            gmsh.merge(f'{domain_dir}/original_DesignSpace.stl')
            if reparametrize:
                gmsh.model.mesh.classifySurfaces(40*np.pi/180, True, True)
                gmsh.model.mesh.createGeometry()
                gmsh.model.geo.synchronize()
            surfaces = gmsh.model.getEntities(2)
            sl = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
            vol = gmsh.model.geo.addVolume([sl])
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(3, [vol], tag=1)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMax', mesh_size)
            gmsh.option.setNumber('Mesh.CharacteristicLengthMin', mesh_size * 0.3)
            gmsh.model.mesh.generate(3)
            gmsh.write(out_msh)
        finally:
            gmsh.finalize()

    try:
        _mesh(reparametrize=True)
    except Exception as e:
        print(f'[mesh] reparametrization failed ({e}) — retrying on the discrete surface',
              flush=True)
        _mesh(reparametrize=False)


def run_fea(domain_dir: str, density_path: str, nodes_path: str, output_path: str,
            mesh_size: float = 0.006, mesh_cache: str = '/tmp/bracket_fea.msh',
            penal: float = 3.0, E0: float = 1.0, nu: float = 0.3,
            load_magnitude: float = -5000.0, vtk_out: str = None, geom_out: str = None):

    # Load mesh — supports .msh (gmsh) or .npz (nodes+tets dict)
    if mesh_cache.endswith('.npz'):
        import basix.ufl
        data = np.load(mesh_cache)
        nodes_npz = data['nodes']
        tets_npz = data['tets'].astype(np.int64)
        coord_elem = basix.ufl.element('Lagrange', 'tetrahedron', 1, shape=(3,))
        domain = mesh.create_mesh(MPI.COMM_WORLD, tets_npz, nodes_npz, coord_elem)
        print(f'loaded npz mesh: {len(nodes_npz)} nodes, {len(tets_npz)} tets')
    else:
        # File-lock around build+check so concurrent FEA workers don't all race
        # to build the same msh. First holder builds; rest wait then read cache.
        # Local 'os' is shadowed later in this function → import under alias here.
        import fcntl, os as _os_lock
        lock_path = mesh_cache + '.lock'
        Path(mesh_cache).parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, 'w') as _lk:
            fcntl.flock(_lk, fcntl.LOCK_EX)
            try:
                if not Path(mesh_cache).exists():
                    print(f'building tet mesh @ size {mesh_size} → {mesh_cache}')
                    tmp_path = mesh_cache + f'.tmp.{_os_lock.getpid()}.msh'  # keep .msh so gmsh.write infers the format
                    build_mesh(domain_dir, mesh_size, tmp_path)
                    _os_lock.replace(tmp_path, mesh_cache)
                else:
                    print(f'reusing cached mesh {mesh_cache}')
            finally:
                fcntl.flock(_lk, fcntl.LOCK_UN)
        domain, _, _ = gmshio.read_from_msh(mesh_cache, MPI.COMM_WORLD, 0, gdim=3)
    coords = domain.geometry.x
    n_nodes = len(coords)
    n_cells = domain.topology.index_map(3).size_local

    # Load density — auto-detect input mode:
    #   len(density) == n_cells  →  tet-element ρ (mesh-based TO, preferred)
    #   else                      →  external-node ρ (voxel-based, legacy)
    ext_density = np.load(density_path)
    domain.topology.create_connectivity(3, 0)
    cell_to_vertex = domain.topology.connectivity(3, 0)

    if len(ext_density) == n_cells:
        elem_density = np.clip(ext_density, 1e-3, 1.0)
        node_density = None
        ext_nodes = None
        ext_mode = 'cell'
        print(f'cell-based ρ mode: n_cells={n_cells}')
    elif len(ext_density) == n_nodes:
        node_density = np.clip(ext_density, 1e-3, 1.0)
        elem_density = np.zeros(n_cells)
        for c in range(n_cells):
            verts = cell_to_vertex.links(c)
            elem_density[c] = node_density[verts].mean()
        ext_nodes = coords.copy()  # nodes are dolfinx coords (assumed same order)
        ext_mode = 'node'
        print(f'node-based ρ mode: n_nodes={n_nodes}')
    else:
        ext_nodes = np.load(nodes_path)
        tree = cKDTree(ext_nodes)
        _, idx = tree.query(coords)
        node_density = np.clip(ext_density[idx], 1e-3, 1.0)
        elem_density = np.zeros(n_cells)
        for c in range(n_cells):
            verts = cell_to_vertex.links(c)
            elem_density[c] = node_density[verts].mean()
        ext_mode = 'legacy'
        print(f'legacy voxel-based ρ mode: n_ext={len(ext_density)}, n_cells={n_cells}')

    # SIMP material per element
    # paper ref: main text, in-loop linear-elastic FEM with SIMP-penalized modulus E(rho) = E0 * rho^p.
    import os as _os
    E_MIN_FENICS = float(_os.environ.get('FENICS_E_MIN', '1e-3'))   # default; 1e-9 numerically unstable for soft ρ
    E_elem = E_MIN_FENICS + np.clip(elem_density, 1e-3, 1.0) ** penal * (E0 - E_MIN_FENICS)

    # Function spaces
    V = fem.functionspace(domain, ('Lagrange', 1, (3,)))
    u_tr = ufl.TrialFunction(V)
    v_te = ufl.TestFunction(V)

    Q = fem.functionspace(domain, ('DG', 0))
    E_func = fem.Function(Q)
    E_func.x.array[:] = E_elem

    mu_e = E_func / (2 * (1 + nu))
    lmbda_e = E_func * nu / ((1 + nu) * (1 - 2 * nu))

    def sigma(u):
        return lmbda_e * ufl.nabla_div(u) * ufl.Identity(3) + 2 * mu_e * ufl.sym(ufl.nabla_grad(u))

    a = ufl.inner(sigma(u_tr), ufl.sym(ufl.nabla_grad(v_te))) * ufl.dx

    # BCs from fix.stl / load.stl
    fixed_mesh = trimesh.load(f'{domain_dir}/fixed.stl')
    load_mesh  = trimesh.load(f'{domain_dir}/load.stl')

    tree_f = cKDTree(fixed_mesh.vertices)
    tree_l = cKDTree(load_mesh.vertices)

    import os
    bc_dist = float(os.environ.get('BC_DIST', '0.006'))   # mesh coord unit; cube-space mesh uses ~0.07

    # Dirichlet: ANY vertex within bc_dist (relaxed from ALL 3)
    # Use SURFACE-distance (closest point on face) instead of vertex-distance for better coverage.
    if os.environ.get('BC_SURFACE_DIST', '0') == '1':
        from trimesh.proximity import closest_point
        _, d_f_surf, _ = closest_point(fixed_mesh, coords)
        fixed_node_mask = d_f_surf < bc_dist
    else:
        d_f_pts, _ = tree_f.query(coords)
        fixed_node_mask = d_f_pts < bc_dist
    fixed_node_ids = np.where(fixed_node_mask)[0].astype(np.int32)
    fdim = domain.topology.dim - 1
    # Direct DOF spec via vertex IDs (Lagrange-1 → DOFs collocated at vertices)
    domain.topology.create_connectivity(0, 3)
    bc_dofs = fem.locate_dofs_topological(V, 0, fixed_node_ids)
    bc = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((0, 0, 0))), bc_dofs, V)
    print(f'Dirichlet: {len(fixed_node_ids)} nodes (3 DOFs each = {3*len(fixed_node_ids)})')

    # Load — force on load region (style: indicator function)
    if os.environ.get('BC_SURFACE_DIST', '0') == '1':
        from trimesh.proximity import closest_point
        _, d_l_surf, _ = closest_point(load_mesh, coords)
        load_node_ids = np.where(d_l_surf < bc_dist)[0]
    else:
        d_l_pts, _ = tree_l.query(coords)
        load_node_ids = np.where(d_l_pts < bc_dist)[0]
    W = fem.functionspace(domain, ('Lagrange', 1))
    load_indicator = fem.Function(W)
    load_indicator.x.array[:] = 0.0
    load_indicator.x.array[load_node_ids] = 1.0
    load_area = fem.assemble_scalar(fem.form(load_indicator * ufl.dx))
    f_total = abs(load_magnitude) if load_magnitude < 0 else load_magnitude
    f_mag = f_total / max(load_area, 1e-10)

    # LOAD MODE — set from the config's stages.mesh.load_mode, exported by the generation
    # script as the LOAD_MODE env var. Must match the direction the a-posteriori FEA verifies
    # (stages.fea.force_dir); run_from_image.py enforces that.
    #   x, -x, y, -y, z, -z : axis-aligned (what every shipped domain uses)
    #   diag                : (1,-1,-1)/sqrt(3) — fallback when LOAD_MODE is unset
    load_mode = os.environ.get('LOAD_MODE', 'diag')
    if load_mode.startswith('spread:'):
        # Opposed load — see fea_tet_from_mesh.py for why: the sign flips across the load surface's
        # mid-plane so two facing banks are pushed apart, which is what a fixed brake caliper's
        # pistons do to its halves. Built as a nodal function because the sign varies in space,
        # unlike the single fem.Constant the axis modes use.
        ax_idx = {'x': 0, 'y': 1, 'z': 2}[load_mode.split(':')[1].strip()[-1]]
        split = float(load_mesh.bounds.mean(0)[ax_idx])
        sign_fn = fem.Function(W)
        sign_fn.x.array[:] = np.where(coords[:, ax_idx] > split, 1.0, -1.0)
        comp = [None, None, None]
        for k in range(3):
            comp[k] = (f_mag * sign_fn * load_indicator) if k == ax_idx else \
                      fem.Constant(domain, PETSc.ScalarType(0.0)) * load_indicator
        f = ufl.as_vector(comp)
        n_pos = int((coords[load_node_ids, ax_idx] > split).sum())
        dir_desc = (f'spread about {"xyz"[ax_idx]}={split*1000:.1f}mm '
                    f'({n_pos}+/{len(load_node_ids)-n_pos}- nodes)')
    elif load_mode in ('-y','y','-z','z','-x','x'):
        sign = -1.0 if load_mode.startswith('-') else 1.0
        axis_map = {'x':0, 'y':1, 'z':2}
        ax_idx = axis_map[load_mode[-1]]
        vec = [0.0, 0.0, 0.0]; vec[ax_idx] = sign
        fx_c = fem.Constant(domain, PETSc.ScalarType(f_mag * vec[0]))
        fy_c = fem.Constant(domain, PETSc.ScalarType(f_mag * vec[1]))
        fz_c = fem.Constant(domain, PETSc.ScalarType(f_mag * vec[2]))
        f = ufl.as_vector([fx_c * load_indicator, fy_c * load_indicator, fz_c * load_indicator])
        dir_desc = f'axis {load_mode}'
    else:  # diag
        fx_r, fy_r, fz_r = 1.0, -1.0, -1.0
        norm_d = np.sqrt(fx_r**2 + fy_r**2 + fz_r**2)
        fx_c = fem.Constant(domain, PETSc.ScalarType(f_mag * fx_r / norm_d))
        fy_c = fem.Constant(domain, PETSc.ScalarType(f_mag * fy_r / norm_d))
        fz_c = fem.Constant(domain, PETSc.ScalarType(f_mag * fz_r / norm_d))
        f = ufl.as_vector([fx_c * load_indicator, fy_c * load_indicator, fz_c * load_indicator])
        dir_desc = 'diag (1,-1,-1)/sqrt(3)'
    L = ufl.dot(f, v_te) * ufl.dx
    print(f'Load: {len(load_node_ids)} nodes, load_area={load_area*1e6:.2f} cm³, total F={f_total:.0f} N, dir={dir_desc}')

    # Solve
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})
    uh = problem.solve()

    # paper ref: main text, compliance C = f^T u from the in-loop linear-elastic FEM solve.
    compliance = fem.assemble_scalar(fem.form(ufl.dot(f, uh) * ufl.dx))

    # Optional VTK output (displacement + density; vm stress = ||u|| magnitude proxy)
    if vtk_out:
        from dolfinx.io import VTKFile
        uh.name = 'displacement'
        E_func.name = 'density_E'
        # u magnitude as Q (DG0) scalar — proxy for stress (high u = compliant region)
        u_mag = fem.Function(Q); u_mag.name = 'u_magnitude'
        # interpolate from u vector field to scalar magnitude
        domain.topology.create_connectivity(3, 0)
        u_arr = uh.x.array.reshape(-1, 3)
        # avg vertex magnitudes to cell
        c2v = domain.topology.connectivity(3, 0)
        u_cell = np.zeros(n_cells, dtype=np.float64)
        for c in range(n_cells):
            vv = c2v.links(c)
            mags = np.linalg.norm(u_arr[vv], axis=1)
            u_cell[c] = mags.mean()
        u_mag.x.array[:] = u_cell
        with VTKFile(MPI.COMM_WORLD, vtk_out, 'w') as vfile:
            vfile.write_mesh(domain)
            vfile.write_function([uh], 0.0)
        with VTKFile(MPI.COMM_WORLD, vtk_out.replace('.pvd','_E.pvd'), 'w') as vfile:
            vfile.write_mesh(domain)
            vfile.write_function([E_func], 0.0)
        with VTKFile(MPI.COMM_WORLD, vtk_out.replace('.pvd','_umag.pvd'), 'w') as vfile:
            vfile.write_mesh(domain)
            vfile.write_function([u_mag], 0.0)
        print(f'  VTK: {vtk_out} + _E + _umag')

    # True SIMP adjoint:
    #   dC/dρ_e = -p · ρ_e^(p-1) · (E₀−E_min) · u_eᵀ K_e^{base} u_e
    # We compute u_eᵀ K_e u_e via DG0-projected strain energy density ψ_h,
    #   u_eᵀ K_e u_e = ∫_e σ:ε dx ≈ 2 ψ_h · V_e
    # K_e^{base} (E=1) form:  u_eᵀ K_e^{base} u_e = (u_eᵀ K_e u_e) / E_e
    #   with E_e = E_min + ρ^p (E0 − E_min).
    # Substituting → dc_e = -p · ρ_e^(p-1) · (E₀−E_min)/E_e · (2 ψ_h V_e).
    # Element-integrated strain energy via DG0 test function:
    #   W_e = ∫_e σ:ε dx = u_eᵀ K_e u_e
    strain_density = ufl.inner(sigma(uh), ufl.sym(ufl.nabla_grad(uh)))
    v0 = ufl.TestFunction(Q)
    we_form = fem.form(v0 * strain_density * ufl.dx)
    from dolfinx.fem.petsc import assemble_vector
    we_vec = assemble_vector(we_form)
    we_vec.assemble()
    u_e_K_u = we_vec.array.copy()  # shape (n_cells,)

    # element volume — needed for cell_volumes diagnostic only
    cell_volumes = np.zeros(n_cells)
    cell_centroids = np.zeros((n_cells, 3))
    for c in range(n_cells):
        verts = cell_to_vertex.links(c)
        v_coords = coords[verts]
        a, b, c2, d = v_coords[0], v_coords[1], v_coords[2], v_coords[3]
        cell_volumes[c] = abs(np.dot(b - a, np.cross(c2 - a, d - a))) / 6.0
        cell_centroids[c] = v_coords.mean(axis=0)
    if geom_out is not None:
        tets_fe = np.zeros((n_cells, 4), dtype=np.int64)
        for c in range(n_cells):
            tets_fe[c] = cell_to_vertex.links(c)
        np.savez_compressed(geom_out, centroids=cell_centroids, volumes=cell_volumes,
                            nodes=np.asarray(coords), tets=tets_fe)
        print(f'  → {geom_out}  (per-cell centroid+volume+connectivity, FEniCS order, n={n_cells})')
    rho_clip = np.clip(elem_density, 1e-3, 1.0)
    E_min = E_MIN_FENICS
    E_e_val = E_min + rho_clip ** penal * (E0 - E_min)
    dc_elem = -penal * rho_clip ** (penal - 1) * (E0 - E_min) / np.maximum(E_e_val, 1e-12) * u_e_K_u

    # Output dc — same shape as input ρ.
    #
    # The density forward map above is
    #   rho_elem[c] = mean(rho_node[v] for v in cell[c])
    # and, in legacy voxel mode,
    #   rho_node[v] = rho_ext[idx[v]].
    # Its exact adjoint therefore distributes dc_elem/4 to the incident FEM
    # nodes, then scatter-adds those nodal values back through idx.  The old
    # implementation averaged incident sensitivities and sampled them in the
    # reverse nearest-neighbour direction (FEM -> every external voxel).  That
    # was not the adjoint of the forward map and broadcast spurious gradients
    # to external voxels which did not contribute to the FEM solve.
    if ext_mode == 'cell':
        dc_out = dc_elem
    else:
        dc_node_arr = np.zeros(n_nodes)
        for c in range(n_cells):
            verts = cell_to_vertex.links(c)
            share = dc_elem[c] / max(1, len(verts))
            for vi in verts:
                dc_node_arr[vi] += share
        if ext_mode == 'node':
            dc_out = dc_node_arr
        else:
            dc_out = np.zeros(len(ext_density), dtype=np.float64)
            np.add.at(dc_out, idx, dc_node_arr)

    np.save(output_path, dc_out)
    info = {
        'compliance': float(compliance),
        'n_nodes': int(n_nodes),
        'n_cells': int(n_cells),
        'n_ext_nodes': 0 if ext_nodes is None else int(len(ext_nodes)),
        'penal': float(penal),
        'mean_density': float(elem_density.mean()),
        'mode': 'mesh' if ext_nodes is None else 'voxel',
        'mean_W_e': float(u_e_K_u.mean()),
        'max_W_e': float(u_e_K_u.max()),
        'sensitivity_mapping': 'exact_forward_adjoint',
        'sensitivity_nonzero': int(np.count_nonzero(dc_out)),
    }
    info_path = output_path.replace('.npy', '_info.json')
    with open(info_path, 'w') as fp:
        json.dump(info, fp, indent=2)
    print(f'FEA done: C={compliance:.6e}  n_nodes={n_nodes}  n_cells={n_cells}  mean_rho={elem_density.mean():.3f}  mean_W_e={u_e_K_u.mean():.3e}')
    print(f'  → {output_path}  (sensitivity, shape={dc_out.shape})')
    print(f'  → {info_path}    (compliance, info)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain-dir', required=True,
                    help='dir containing original_DesignSpace.stl, fixed.stl, load.stl')
    ap.add_argument('--density', default=None,
                    help='input ρ: length n_cells → mesh-based; else legacy node-based '
                         '(not needed with --build-mesh-only)')
    ap.add_argument('--nodes', default=None,
                    help='external node coordinates (.npy, only needed for legacy voxel mode)')
    ap.add_argument('--output', default=None,
                    help='output sensitivity ∂C/∂ρ on external nodes (.npy); '
                         'not needed with --build-mesh-only')
    ap.add_argument('--build-mesh-only', action='store_true',
                    help='only build (or reuse) the tet mesh cache from --domain-dir at --mesh-size '
                         'and exit — no density, no solve. Used by run_prep.py to build the per-domain '
                         'in-loop tet mesh once, so every prompt shares it instead of rebuilding.')
    ap.add_argument('--mesh-size', type=float, default=0.006)
    ap.add_argument('--mesh-cache', default='/tmp/bracket_fea.msh')
    ap.add_argument('--penal', type=float, default=3.0)
    ap.add_argument('--E0', type=float, default=1.0)
    ap.add_argument('--nu', type=float, default=0.3)
    ap.add_argument('--load-magnitude', type=float, default=42300.0,
                    help='total force magnitude in N (default = 42300, a reference scale). In cube space mesh, smaller (e.g. 1.0) is more numerically stable.')
    ap.add_argument('--vtk-out', default=None,
                    help='write displacement + vm + E to .pvd (also creates _vm.pvd, _E.pvd)')
    ap.add_argument('--geom-out', default=None,
                    help='save per-cell centroid+volume (FEniCS cell order) to .npz for mesh-based TO filtering')
    args = ap.parse_args()

    if args.build_mesh_only:
        if Path(args.mesh_cache).exists():
            print(f'reusing cached mesh {args.mesh_cache}')
        else:
            print(f'building tet mesh @ size {args.mesh_size} from {args.domain_dir} → {args.mesh_cache}')
            Path(args.mesh_cache).parent.mkdir(parents=True, exist_ok=True)
            build_mesh(args.domain_dir, args.mesh_size, args.mesh_cache)
        import meshio as _mio
        _m = _mio.read(args.mesh_cache)
        _nc = sum(len(b.data) for b in _m.cells if b.type == 'tetra')
        print(f'[mesh] nodes={len(_m.points):,}  tets={_nc:,}  size={Path(args.mesh_cache).stat().st_size/1e6:.2f} MB')
        raise SystemExit(0)
    if not args.density or not args.output:
        ap.error('--density and --output are required unless --build-mesh-only is given')

    run_fea(args.domain_dir, args.density, args.nodes, args.output,
            mesh_size=args.mesh_size, mesh_cache=args.mesh_cache,
            penal=args.penal, E0=args.E0, nu=args.nu,
            load_magnitude=args.load_magnitude, vtk_out=args.vtk_out, geom_out=args.geom_out)
