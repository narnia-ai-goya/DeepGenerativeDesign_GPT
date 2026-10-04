"""Direct3D-S2 + FlowDPS at dense stage.

Hooks into dense sampling loop to inject gradient guidance via bracket voxel mask:
  - Decode current latent x0 → 64³ occupancy (logit)
  - Multi-region BCE: out=0, BC=1, design=light fill
  - Backprop to latent + flow step

Sparse512/1024 stages run vanilla (cleaner refinement on bracketed structure).
"""
import os
# Defensive: ensure CUDA alloc config is set before torch is imported via sys.path
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8'
)
import sys
# Locate the Direct3D-S2 backbone by absolute path (cwd-independent): D3DS2_ROOT env, else the
# directory that contains this package (…/<pkg>/code/this_file → <pkg> → D3DS2_ROOT).
_D3DS2_ROOT = os.environ.get('D3DS2_ROOT') or os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_D3DS2_ROOT, 'external', 'Direct3D-S2'))
import argparse
import json
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image
from direct3d_s2.pipeline import Direct3DS2Pipeline
from direct3d_s2.modules import sparse as sp
from direct3d_s2.utils import sort_block, normalize_mesh, mesh2index
from scipy.ndimage import binary_erosion
from sparse_wall_ray_loss import prepare_side_fronts, sparse_side_front_loss
from sparse_side_depth_loss import dense_side_depth_reference, sparse_side_depth_loss


@contextmanager
def temporary_fea_load(mode: str, magnitude: float):
    """Select one synchronous FEniCS load case without leaking it to the next call."""
    old_mode = os.environ.get('LOAD_MODE')
    old_magnitude = os.environ.get('FEA_LOAD_MAGNITUDE')
    os.environ['LOAD_MODE'] = str(mode)
    os.environ['FEA_LOAD_MAGNITUDE'] = str(magnitude)
    try:
        yield
    finally:
        if old_mode is None:
            os.environ.pop('LOAD_MODE', None)
        else:
            os.environ['LOAD_MODE'] = old_mode
        if old_magnitude is None:
            os.environ.pop('FEA_LOAD_MAGNITUDE', None)
        else:
            os.environ['FEA_LOAD_MAGNITUDE'] = old_magnitude


def paired_fea_compliance(occ_logits, mask, nodes, seat_domain, back_domain,
                          mesh_cache, mesh_size, penal, reference,
                          back_mode='y', back_magnitude=200.0,
                          bc_mask_np=None, verbose=False):
    """Balanced seat/back compliance; keeps the single-load path unchanged."""
    from fea_compliance_loss import fea_compliance_loss
    seat = fea_compliance_loss(occ_logits, mask, nodes, seat_domain, mesh_cache,
                               mesh_size=mesh_size, penal=penal,
                               bc_mask_np=bc_mask_np, verbose=verbose)
    if os.environ.get('FEA_SECOND_LOAD_STL'):
        # Both physical forces were applied in this one FEA solve.
        return seat, None, None
    if not back_domain:
        return seat, seat, None
    with temporary_fea_load(back_mode, back_magnitude):
        back = fea_compliance_loss(occ_logits, mask, nodes, back_domain, mesh_cache,
                                   mesh_size=mesh_size, penal=penal,
                                   bc_mask_np=bc_mask_np, verbose=verbose)
    if not torch.isfinite(seat).item() or not torch.isfinite(back).item():
        raise ValueError(f'non-finite paired FEA: seat={seat.item()} back={back.item()}')
    if reference.get('back_scale') is None:
        # Scale the backrest's initial compliance to the seat's initial value.
        # The resulting objective has the same magnitude as the old seat-only
        # objective, while both load cases receive equal initial importance.
        reference['back_scale'] = float((seat.detach() / back.detach()).item())
        print(f'  [dual FEA] initial back scale={reference["back_scale"]:.6g} '
              f'(seat={seat.item():.4e}, back={back.item():.4e})', flush=True)
    combined = .5 * (seat + reference['back_scale'] * back)
    return combined, seat, back


def fea_nodes_in_domain_frame(nodes: np.ndarray, alignment_path: str | None,
                              label: str) -> np.ndarray:
    """Map generation-grid nodes into the physical FEA-domain frame.

    Some legacy bracket recipes decode in an anisotropically aligned ``dense``
    frame but solve FEA on the original physical CAD frame.  Density values stay
    in their original voxel order; only their world coordinates supplied to the
    FEA nearest-node transfer need the inverse affine map.
    """
    if not alignment_path:
        return nodes
    path = Path(alignment_path)
    payload = json.loads(path.read_text())
    registration = payload.get('native_frame_registration') or payload.get('rigid_uniform')
    if registration:
        # The chair generator's registered native frame differs from its physical
        # FEM domain by a rotation and uniform scale. Mapping the voxel *nodes*
        # (not reordering the density array) keeps the sparse-grid gradient in
        # native coordinates while transferring material to physical FEM cells.
        from scipy.spatial.transform import Rotation
        source = np.asarray(registration['source_center_m'], dtype=np.float64)
        physical = np.asarray(registration['physical_center_m'], dtype=np.float64)
        scale = float(registration['uniform_scale'])
        angle = float(registration['rotation_x_degrees'])
        if source.shape != (3,) or physical.shape != (3,) or scale <= 0:
            raise ValueError(f'invalid native-frame registration in {path}')
        rotation = Rotation.from_euler('x', angle, degrees=True).as_matrix()
        transformed = ((np.asarray(nodes, dtype=np.float64) - source) @ rotation.T *
                       scale + physical)
        print(f'  [FEA frame] {label}: rigid registration {path.name}; '
              f'bbox mm {transformed.min(axis=0)*1000} → {transformed.max(axis=0)*1000}',
              flush=True)
        return transformed
    affine = payload.get("affine_forward")
    if not affine:
        raise ValueError(f"{path} has no affine_forward mapping")
    src = np.asarray(affine["src_min"], dtype=np.float64)
    dst = np.asarray(affine["dst_min"], dtype=np.float64)
    scale = np.asarray(affine["scale"], dtype=np.float64)
    if src.shape != (3,) or dst.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError(f"invalid affine_forward in {path}")
    transformed = (np.asarray(nodes, dtype=np.float64) - dst) / scale + src
    print(f"  [FEA frame] {label}: inverse affine {path.name}; "
          f"bbox mm {transformed.min(axis=0)*1000} → {transformed.max(axis=0)*1000}",
          flush=True)
    return transformed


# ============== Sparse GuideFlow3D-style guidance helpers (soft + anneal) ==============
def coarsen_sparse_to_dense(sparse_sdf, sparse_coords, K, R=512, default=1.0):
    """Pool sparse SDF into dense (R/K)³ grid via mean over K×K×K super-voxels.

    Memory-safe: count tensor is computed under no_grad (counts don't need gradient).
    Only sum_t / mean_t need autograd for backprop through pooling.
    """
    Rc = R // K
    coords = (sparse_coords[..., 1:].long() // K).clamp(0, Rc - 1)
    s = sparse_sdf.squeeze(-1) if sparse_sdf.dim() > 1 else sparse_sdf
    with torch.no_grad():
        flat = (coords[:, 0] * Rc + coords[:, 1]) * Rc + coords[:, 2]
        cnt_t = torch.zeros(Rc**3, device=s.device, dtype=s.dtype)
        cnt_t.scatter_add_(0, flat, torch.ones_like(s))
        mask_flat = (cnt_t > 0)
        cnt_safe = cnt_t.clamp(min=1.0)
    sum_t = torch.zeros(Rc**3, device=s.device, dtype=s.dtype)
    sum_t.scatter_add_(0, flat, s)
    mean_t = sum_t / cnt_safe
    # Where count=0, use default (empty)
    mean_t = torch.where(mask_flat, mean_t, torch.full_like(mean_t, default))
    return mean_t.reshape(Rc, Rc, Rc), mask_flat.reshape(Rc, Rc, Rc)


def avg_pool_anchor_loss(sparse_sdf, sparse_coords, K, R=512):
    """L2 anchor: push each sparse voxel toward MEAN of its K×K×K super-voxel.

    Effect: progressive smoothing within K×K×K blocks. K=4 → coarse mass, K=1 → no force.
    Memory-safe: target (super-voxel mean) computed under no_grad.
    """
    Rc = R // K
    coords = (sparse_coords[..., 1:].long() // K).clamp(0, Rc - 1)
    s = sparse_sdf.squeeze(-1) if sparse_sdf.dim() > 1 else sparse_sdf
    flat = (coords[:, 0] * Rc + coords[:, 1]) * Rc + coords[:, 2]
    with torch.no_grad():
        cnt_t = torch.zeros(Rc**3, device=s.device, dtype=s.dtype)
        cnt_t.scatter_add_(0, flat, torch.ones_like(s))
        sum_t = torch.zeros(Rc**3, device=s.device, dtype=s.dtype)
        sum_t.scatter_add_(0, flat, s.detach())
        mean_flat = sum_t / cnt_t.clamp(min=1.0)
        target_per_vox = mean_flat[flat]   # gather mean back to each sparse voxel
    return ((s - target_per_vox) ** 2).mean()


def dense_lap_loss(dense_sdf, mask=None):
    """L2 Laplacian² loss on a dense SDF grid (any resolution).

    ∇²s computed via 6-neighbor FD (axis-aligned, isotropic).
    """
    import torch.nn.functional as _F
    pad = _F.pad(dense_sdf[None, None], (1,1,1,1,1,1), mode='replicate')
    lap = (pad[:,:,2:,1:-1,1:-1] + pad[:,:,:-2,1:-1,1:-1]
         + pad[:,:,1:-1,2:,1:-1] + pad[:,:,1:-1,:-2,1:-1]
         + pad[:,:,1:-1,1:-1,2:] + pad[:,:,1:-1,1:-1,:-2]
         - 6.0 * pad[:,:,1:-1,1:-1,1:-1]).squeeze(0).squeeze(0)
    if mask is not None:
        pen = (lap ** 2) * mask.float()
        n = mask.float().sum().clamp(min=1.0)
        return pen.sum() / n
    return (lap ** 2).mean()


def thickness_per_voxel_loss(sparse_sdf, target_norm=0.06, sharpness=10.0, mc_threshold=0.2):
    """SOFT inside mask + softplus penalty — fully differentiable.
    Original implementation (retained for ablation compatibility) — the
    a convention-corrected variant exists in an earlier implementation.
    mc_threshold accepted for call-site compat (unused here)."""
    # paper ref: Supplementary, Sparse-Stage Loss Terms, term (2) Per-voxel thickness (L_th).
    s = sparse_sdf.squeeze(-1) if sparse_sdf.dim() > 1 else sparse_sdf
    inside_soft = torch.sigmoid(-s * sharpness)
    penalty = F.softplus((s + target_norm) * 10.0) / 10.0
    return (penalty * inside_soft).mean()


# ─────────────────────────────────────────────────────────────────────
# TO-style volume constraint utilities (Augmented Lagrangian + Heaviside)
# ─────────────────────────────────────────────────────────────────────
def heaviside_projection(rho, eta, beta):
    """tanh-based smooth Heaviside projection (TO standard, Wang 2011).
       rho ∈ [0,1] → rho_phys ∈ [0,1] with sharp transition at η.
       beta ↑ → sharper. η is the threshold.
    Differentiable w.r.t. rho. eta, beta passed as float (no gradient).
    """
    eta_t = torch.tensor(eta, device=rho.device, dtype=rho.dtype)
    beta_t = torch.tensor(beta, device=rho.device, dtype=rho.dtype)
    num = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (rho - eta_t))
    den = torch.tanh(beta_t * eta_t) + torch.tanh(beta_t * (1.0 - eta_t))
    return num / den.clamp_min(1e-8)


def bisect_eta_for_volume(rho_detached, vol_target, beta, max_iter=15, lo=0.05, hi=0.95):
    """Find η ∈ [lo, hi] such that mean(heaviside(rho, η, β)) = vol_target via bisection.
       rho_detached: 1-D tensor with no grad. Returns float η.
    """
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        v = heaviside_projection(rho_detached, mid, beta).mean().item()
        if v > vol_target:
            lo = mid   # η ↑ → vol ↓ → push η higher to reduce vol
        else:
            hi = mid
    return 0.5 * (lo + hi)


def topopt_rmin_loss(sparse_sdf, sparse_coords, grid_res=128, r_min_voxels=3,
                     void_weight=0.3, sharpness=8.0, mc_threshold=0.2):
    """r_min filter with SOFT masks (sigmoid) + softplus penalty.
    Original implementation (retained for ablation compatibility) — the
    a convention-corrected variant exists in an earlier implementation.
    mc_threshold accepted for call-site compat (unused here)."""
    # paper ref: Supplementary, Sparse-Stage Loss Terms, term (1) Minimum feature size r_min (L_{r_min});
    #            also main text r_min and Table 2.
    device = sparse_sdf.device
    factor = 512 // grid_res
    coords = (sparse_coords[..., 1:].float() / factor).long().clamp(0, grid_res - 1)
    sdf_grid = torch.full((grid_res,)*3, 1.0, device=device, dtype=sparse_sdf.dtype)
    sdf_grid.index_put_(
        (coords[..., 0], coords[..., 1], coords[..., 2]),
        sparse_sdf.squeeze(-1), accumulate=False
    )
    K = r_min_voxels
    sdf_eroded = F.max_pool3d(
        sdf_grid[None, None], kernel_size=2*K+1, stride=1, padding=K
    )[0, 0]
    inside_soft = torch.sigmoid(-sdf_grid * sharpness)
    thin_solid = F.softplus(sdf_eroded * 10.0) / 10.0 * inside_soft

    sdf_dilated = -F.max_pool3d(
        (-sdf_grid)[None, None], kernel_size=2*K+1, stride=1, padding=K
    )[0, 0]
    outside_soft = torch.sigmoid(sdf_grid * sharpness)
    thin_void = F.softplus(-sdf_dilated * 10.0) / 10.0 * outside_soft

    return thin_solid.mean() + void_weight * thin_void.mean()


def topology_preservation_loss(sparse_sdf, sparse_coords, dense_active_64,
                                interior_thresh=2.0, exterior_thresh=2.0,
                                sharpness=5.0, sdf_scale=20.0,
                                softmin_beta=10.0, cone_cos=0.7071,
                                no_diagonal=False,
                                interior_w=0.0, interior_erode=2,
                                interior_depth_graded=False,
                                interior_depth_slope=2.0,
                                _cache={}):
    """NORMAL-DIRECTION loss — **diagonal-aware 45° cone**.

    Each dense shell_in cell has an outward-direction set D and two kinds of cone:
      (a) for each d∈D, a 45° axis cone around ±d
      (b) cells with |D|≥2 also get a 45° diagonal cone around d_mean = Σd/|Σd|
    If a sparse voxel's in-cell rel falls inside some cone:
      - +d cone (outside cone)  → target s>0
      - -d cone (inside  cone)  → target s<0
      - inside no cone          → unconstrained (target=0)
    With multiple valid cones, apply a **softmin penalty** (satisfying one cone is enough).
    shell_out (dense=0, adjacent to solid) is simply s>0.
    """
    from scipy.ndimage import binary_erosion, binary_dilation

    key = (id(dense_active_64), no_diagonal, interior_erode, interior_depth_graded, interior_depth_slope)
    if key not in _cache:
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np  = d_np & ~binary_erosion(d_np)
        shell_out_np = binary_dilation(d_np) & ~d_np
        # deep interior = dense=1 and at least N voxels inward (1-2 layer thick features are automatically empty)
        if interior_depth_graded:
            # depth-graded weight: shell (depth 0) = 0, depth k = 2*k via iterated erosion.
            depth_w_np = np.zeros_like(d_np, dtype=np.float32)
            prev = d_np.copy()
            for k in range(1, 33):   # cap at depth 32
                erod = binary_erosion(d_np, iterations=k)
                layer = prev & ~erod
                if layer.sum() == 0:
                    break
                depth_w_np[layer] = interior_depth_slope * k
                prev = erod
            # innermost still-solid voxels (eroded past loop) → max weight from last k
            if prev.any():
                depth_w_np[prev] = interior_depth_slope * (k + 1)
            deep_int_np = depth_w_np   # NOT binary — continuous weight
            print(f'  [interior depth-graded] max depth_w = {depth_w_np.max():.0f}, '
                  f'mean over active = {depth_w_np[d_np].mean():.2f}', flush=True)
        else:
            deep_int_np = binary_erosion(d_np, iterations=max(1, interior_erode)).astype(np.float32)

        def nbr_empty(axis, shift):
            rolled = np.roll(d_np, -shift, axis=axis)
            sl = [slice(None)] * 3
            sl[axis] = -1 if shift == 1 else 0
            rolled[tuple(sl)] = True
            return shell_in_np & ~rolled
        # 6 axis outward masks
        m_axis = np.stack([
            nbr_empty(0,  1), nbr_empty(0, -1),
            nbr_empty(1,  1), nbr_empty(1, -1),
            nbr_empty(2,  1), nbr_empty(2, -1),
        ]).astype('float32')   # (6, 64, 64, 64)
        dirs = np.array([[ 1,0,0],[-1,0,0],
                         [ 0,1,0],[ 0,-1,0],
                         [ 0,0,1],[ 0,0,-1]], dtype='float32')
        # per-cell d_mean (sum of outward unit vectors, normalized)
        sum_dir = np.zeros((64, 64, 64, 3), dtype='float32')
        for i in range(6):
            sum_dir += m_axis[i, ..., None] * dirs[i]
        sum_norm = np.linalg.norm(sum_dir, axis=-1, keepdims=True)
        d_mean = sum_dir / (sum_norm + 1e-9)             # zero if cell not shell_in
        n_out_per_cell = m_axis.sum(0)                   # 0~6
        diag_valid = (n_out_per_cell >= 2).astype('float32')
        _cache[key] = (
            torch.from_numpy(m_axis     ).to(sparse_sdf.device),
            torch.from_numpy(dirs       ).to(sparse_sdf.device),
            torch.from_numpy(d_mean     ).to(sparse_sdf.device),  # (64,64,64,3)
            torch.from_numpy(diag_valid ).to(sparse_sdf.device),
            torch.from_numpy(shell_out_np.astype('float32')).to(sparse_sdf.device),
            torch.from_numpy(deep_int_np.astype('float32')).to(sparse_sdf.device),
        )
    m_axis_t, dirs_t, dmean_t, diag_v_t, so_t, di_t = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    rel = (coords_512 - coords_64 * 8).float() - 3.5
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    rel_norm = rel.norm(dim=-1).clamp(min=1e-6)
    s = sparse_sdf.squeeze(-1)
    sdf_s = s * sdf_scale

    pen_per_dir, val_per_dir = [], []

    # (a) 6 axis cones
    for d_idx in range(6):
        is_out = m_axis_t[d_idx][cz, cy, cx]
        cos_d = (rel * dirs_t[d_idx]).sum(-1) / rel_norm        # (B, M)
        # +cone (target +1) / -cone (target -1)
        in_pcone = (cos_d >  cone_cos).float()
        in_mcone = (cos_d < -cone_cos).float()
        target_sign = in_pcone - in_mcone
        pen = F.softplus(-target_sign * sdf_s) / sdf_scale
        valid = is_out * (in_pcone + in_mcone)
        pen_per_dir.append(pen); val_per_dir.append(valid)

    # (b) diagonal cone (only |D|≥2 cells)
    if not no_diagonal:
        dm = dmean_t[cz, cy, cx]                                # (B, M, 3)
        cos_dm = (rel * dm).sum(-1) / rel_norm
        in_p_dm = (cos_dm >  cone_cos).float()
        in_m_dm = (cos_dm < -cone_cos).float()
        target_dm = in_p_dm - in_m_dm
        pen_dm = F.softplus(-target_dm * sdf_s) / sdf_scale
        valid_dm = diag_v_t[cz, cy, cx] * (in_p_dm + in_m_dm)
        pen_per_dir.append(pen_dm); val_per_dir.append(valid_dm)

    pen_stack = torch.stack(pen_per_dir, dim=0)                 # (7, B, M)
    val_stack = torch.stack(val_per_dir, dim=0)
    log_val = torch.log(val_stack.clamp(min=1e-12))
    softmin_pen = -torch.logsumexp(-pen_stack * softmin_beta + log_val, dim=0) / softmin_beta
    any_valid = (val_stack.sum(0) > 0)
    l_in = softmin_pen[any_valid].mean() if any_valid.any() else torch.zeros((), device=s.device)

    is_so = so_t[cz, cy, cx]
    pen_out = F.softplus(-sdf_s) / sdf_scale
    l_out = (pen_out * is_so).sum() / is_so.sum().clamp(min=1.0)

    # deep interior: voxels surviving erosion(N) of dense=1 have their sparse SDF forced s<0 (solid)
    # NOTE: l_di is returned separately, so the caller multiplies it by sp_interior_w (independent of sp_hole_w)
    # paper ref: Supplementary, Sparse-Stage Loss Terms, term (3) Density projection / interior anchor
    #            (L_int, average-pool anchor).
    is_di = di_t[cz, cy, cx]
    pen_di = F.softplus(sdf_s) / sdf_scale     # penalized when s>0
    if is_di.sum() > 0:
        l_di = (pen_di * is_di).sum() / is_di.sum()
    else:
        l_di = torch.zeros((), device=s.device)
    return (l_in + l_out), l_di


# alias for callsite compat
topology_no_hole_loss = topology_preservation_loss


def topology_normal_lipschitz_loss(sparse_sdf, sparse_coords,
                                    dense_active_64, dense_normal_64,
                                    cone_cos=0.5, p=3.0, _cache={}):
    """**Lipschitz-style normal-bound** topology loss with **punitive hinge penalty**.

    paper ref: Supplementary, Sparse-Stage Loss Terms, term (4) Surface-normal consistency (L_nfd).

    At each dense shell_in cell, force the sparse SDF's **gradient direction** to lie within a
    cone of the dense MC outward normal `n_d` (angle ≤ acos(cone_cos)).

    From the sparse voxel rel position r ∈ [-3.5, 3.5]³ (zero-mean):
        Σ s_i · r_i  ∝  ∇s   (under zero-mean rel, the moment is proportional to the gradient)
    Normalize this moment into a per-cell unit vector v_cell, then on cone violation apply a **polynomial hinge**:
        pen = max(0, cone_cos - v_cell·n_d) ** p

    p=3 (cubic) default — exactly 0 inside the cone, rising steeply on violation (punitive):
        cos=cone_cos: pen=0
        cos=0      : pen=cone_cos^p   (e.g. 0.5^3=0.125)
        cos=-1     : pen=(cone_cos+1)^p  (e.g. 1.5^3=3.375)
    """
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        _cache[key] = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
    shell_in_t = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    rel = (coords_512 - coords_64 * 8).float() - 3.5     # rel ∈ [-3.5, 3.5]
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    is_shell = shell_in_t[cz, cy, cx]                    # only shell_in cells count

    # Per-cell SDF moment: Σ s_i · r_i (zero-mean rel → ∝ ∇s)
    cell_id = (cz * 64 + cy) * 64 + cx                    # flat 64³ index, (M,)
    sr = (s.unsqueeze(-1) * rel) * is_shell.unsqueeze(-1) # (M, 3) — only shell rows non-zero

    bucket = torch.zeros(64*64*64, 3, device=s.device, dtype=s.dtype)
    bucket.scatter_add_(0, cell_id.unsqueeze(-1).expand(-1, 3), sr)

    n_v = bucket.norm(dim=-1).clamp(min=1e-6)              # (64³,)
    v_unit = bucket / n_v.unsqueeze(-1)

    # Reference dense MC outward normal per cell (flat)
    n_d_flat = dense_normal_64.view(-1, 3)
    n_d_norm = n_d_flat.norm(dim=-1).clamp(min=1e-6)
    n_d_unit = n_d_flat / n_d_norm.unsqueeze(-1)

    # cos(angle): aligned with n_d → cos > cone_cos → no penalty
    cos_angle = (v_unit * n_d_unit).sum(-1)               # (64³,)

    # Active cells = those with shell sparse voxels AND valid dense normal
    has_sdf_var = n_v > 1e-4
    has_normal  = n_d_norm > 1e-4
    active = has_sdf_var & has_normal

    # Punitive polynomial hinge: zero inside cone, grows fast outside
    violation = (cone_cos - cos_angle).clamp(min=0.0)     # ≥ 0
    pen = violation ** p
    if active.sum() > 0:
        return (pen * active.float()).sum() / active.float().sum()
    return torch.zeros((), device=s.device)


def topology_normal_fd_loss(sparse_sdf, sparse_coords,
                              dense_active_64, dense_normal_64,
                              cone_cos=0.5, p=3.0,
                              default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                              _cache={}):
    """**Per-voxel local FD** ∇s vs dense MC normal angle bound.

    paper ref: Supplementary, Sparse-Stage Loss Terms, term (4) Surface-normal consistency (L_nfd).

    Compute ∇s directly via each sparse voxel's 6-neighbor central difference:
        ∂s/∂z ≈ (s(v+e_z) - s(v-e_z)) / 2
    If a neighbor is not in the sparse index, use a default:
        - cell's dense_active=True   → s_default = -1 (deep solid)
        - dense_active=False         → s_default = +1 (deep empty)
    Per-voxel penalty: relu(cone_cos - g·n_d)^p (aggregated over shell-cell voxels only).

    Unlike the cell-mean moment, evaluation is per voxel, so even a few violating voxels in a cell are caught directly.
    Directly targets mid-cell thin-spoke fragmentation in the wheel.
    """
    # Cache shell mask & default SDF per dense cell (depends only on dense_active_64)
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        _cache[key] = (shell_in_t, def_sdf_64)
    shell_in_t, def_sdf_64 = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()                # (M, 3) [Z, Y, X]
    coords_64 = (coords_512 // 8).clamp(0, 63)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    M = s.shape[0]
    is_shell = shell_in_t[cz, cy, cx]

    # Build idx_vol covering bbox of sparse coords (padded +/- 1 for neighbor query)
    z_min = (coords_512[..., 0].min() - 1).item()
    y_min = (coords_512[..., 1].min() - 1).item()
    x_min = (coords_512[..., 2].min() - 1).item()
    z_max = (coords_512[..., 0].max() + 1).item()
    y_max = (coords_512[..., 1].max() + 1).item()
    x_max = (coords_512[..., 2].max() + 1).item()
    D = z_max - z_min + 1
    H = y_max - y_min + 1
    W = x_max - x_min + 1
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    lcz = coords_512[..., 0] - z_min
    lcy = coords_512[..., 1] - y_min
    lcx = coords_512[..., 2] - x_min
    idx_vol[lcz, lcy, lcx] = torch.arange(M, dtype=torch.int32, device=s.device)

    def _neighbor_sdf(dz, dy, dx):
        nz = (lcz + dz).clamp(0, D - 1)
        ny = (lcy + dy).clamp(0, H - 1)
        nx = (lcx + dx).clamp(0, W - 1)
        idx = idx_vol[nz, ny, nx]                              # (M,) int32
        n64_z = ((nz + z_min) // 8).clamp(0, 63).long()
        n64_y = ((ny + y_min) // 8).clamp(0, 63).long()
        n64_x = ((nx + x_min) // 8).clamp(0, 63).long()
        default = def_sdf_64[n64_z, n64_y, n64_x]
        present = (idx >= 0)
        sdf_sparse = s[idx.clamp(min=0).long()]                # masked below
        return torch.where(present, sdf_sparse, default)

    s_zp = _neighbor_sdf(+1, 0, 0); s_zm = _neighbor_sdf(-1, 0, 0)
    s_yp = _neighbor_sdf(0, +1, 0); s_ym = _neighbor_sdf(0, -1, 0)
    s_xp = _neighbor_sdf(0, 0, +1); s_xm = _neighbor_sdf(0, 0, -1)
    grad = torch.stack([
        (s_zp - s_zm) * 0.5,
        (s_yp - s_ym) * 0.5,
        (s_xp - s_xm) * 0.5,
    ], dim=-1)                                                 # (M, 3)
    grad_mag = grad.norm(dim=-1).clamp(min=1e-6)
    g_unit = grad / grad_mag.unsqueeze(-1)

    n_d_vox = dense_normal_64[cz, cy, cx]                      # (M, 3)
    n_d_norm = n_d_vox.norm(dim=-1).clamp(min=1e-6)
    n_d_unit = n_d_vox / n_d_norm.unsqueeze(-1)

    cos_angle = (g_unit * n_d_unit).sum(-1)
    active = is_shell * (grad_mag > 1e-4).float() * (n_d_norm > 1e-4).float()

    violation = (cone_cos - cos_angle).clamp(min=0.0)
    pen = violation ** p
    if active.sum() > 0:
        return (pen * active).sum() / active.sum()
    return torch.zeros((), device=s.device)


def topology_normal_anisotropy_loss(sparse_sdf, sparse_coords,
                                     dense_active_64, dense_normal_64,
                                     nb_radius=1, surface_band=0.2,
                                     lam3_threshold=0.10, p=3.0,
                                     default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                                     _cache={}):
    """**Refined noise detector via orientation-tensor eigenvalue**.

    Key difference (vs topology_normal_coherence_loss):
      1) Surface-band filter: use only voxels/neighbors with |s_v|<band, |s_nb|<band (excludes ∇s≈0 noise from deep solid/empty)
      2) use the **λ_3 of the orientation tensor M = Σ n n^T** instead of the Mean Resultant Length R
         - λ_3 = degree of 3D spread
         - flat (1 cluster): λ_3 ≈ 0
         - edge/dihedral (2 clusters): the orientation tensor is sign-agnostic, so λ_3 ≈ 0 is possible
         - corner (3 clusters): λ_3 medium
         - true noise (isotropic random): λ_3 ≈ 1/3 (after normalization)

    Penalty: relu(λ_3_normalized - lam3_threshold)^p
    """
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        _cache[key] = (shell_in_t, def_sdf_64)
    shell_in_t, def_sdf_64 = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    M_n = s.shape[0]
    is_shell = shell_in_t[cz, cy, cx]

    z_min = (coords_512[..., 0].min() - nb_radius - 1).item()
    y_min = (coords_512[..., 1].min() - nb_radius - 1).item()
    x_min = (coords_512[..., 2].min() - nb_radius - 1).item()
    z_max = (coords_512[..., 0].max() + nb_radius + 1).item()
    y_max = (coords_512[..., 1].max() + nb_radius + 1).item()
    x_max = (coords_512[..., 2].max() + nb_radius + 1).item()
    D = z_max - z_min + 1; H = y_max - y_min + 1; W = x_max - x_min + 1
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    lcz = coords_512[..., 0] - z_min
    lcy = coords_512[..., 1] - y_min
    lcx = coords_512[..., 2] - x_min
    idx_vol[lcz, lcy, lcx] = torch.arange(M_n, dtype=torch.int32, device=s.device)

    def _sdf_at(dz, dy, dx):
        nz = (lcz + dz).clamp(0, D - 1)
        ny = (lcy + dy).clamp(0, H - 1)
        nx = (lcx + dx).clamp(0, W - 1)
        idx = idx_vol[nz, ny, nx]
        n64_z = ((nz + z_min) // 8).clamp(0, 63).long()
        n64_y = ((ny + y_min) // 8).clamp(0, 63).long()
        n64_x = ((nx + x_min) // 8).clamp(0, 63).long()
        default = def_sdf_64[n64_z, n64_y, n64_x]
        present = (idx >= 0)
        return torch.where(present, s[idx.clamp(min=0).long()], default)

    def _grad_at(dz, dy, dx):
        gz = (_sdf_at(dz + 1, dy, dx) - _sdf_at(dz - 1, dy, dx)) * 0.5
        gy = (_sdf_at(dz, dy + 1, dx) - _sdf_at(dz, dy - 1, dx)) * 0.5
        gx = (_sdf_at(dz, dy, dx + 1) - _sdf_at(dz, dy, dx - 1)) * 0.5
        return torch.stack([gz, gy, gx], dim=-1)

    # Active mask: shell + surface-band centered voxels only
    is_surface = (s.abs() < surface_band)
    active_mask = (is_shell > 0.5) & is_surface                       # (M_n,) bool
    if active_mask.sum() == 0:
        return torch.zeros((), device=s.device)
    active_idx = torch.where(active_mask)[0]                          # (K,) indices

    # Build orientation tensor M = Σ w_i * n_i * n_i^T (surface-band neighbors only)
    # — compute only for active voxels (K << M_n) — significantly less eigvalsh cost
    K = active_idx.shape[0]
    M_tensor = torch.zeros(K, 3, 3, device=s.device, dtype=torch.float32)
    w_total = torch.zeros(K, device=s.device, dtype=torch.float32)

    # Pre-sliced active center coords for re-using _grad_at / _sdf_at:
    # We can call those globally then index by active_idx.
    for dz in range(-nb_radius, nb_radius + 1):
        for dy in range(-nb_radius, nb_radius + 1):
            for dx in range(-nb_radius, nb_radius + 1):
                s_nb = _sdf_at(dz, dy, dx)                            # (M_n,)
                w_i = (s_nb.abs() < surface_band).to(torch.float32)
                g = _grad_at(dz, dy, dx)                              # (M_n, 3)
                gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                n = (g / gn).to(torch.float32)                        # fp32 unit normal
                outer = n.unsqueeze(-1) * n.unsqueeze(-2)             # (M_n, 3, 3)
                # accumulate only at active voxels
                outer_act = outer[active_idx] * w_i[active_idx].unsqueeze(-1).unsqueeze(-1)
                M_tensor = M_tensor + outer_act
                w_total = w_total + w_i[active_idx]

    # Normalize: trace ≈ 1 after normalize
    w_safe = w_total.clamp(min=1.0).unsqueeze(-1).unsqueeze(-1)
    M_norm = M_tensor / w_safe                                        # (K, 3, 3)
    # Numerical jitter for eigvalsh stability (cusolver fails on near-singular)
    eye3 = torch.eye(3, device=s.device, dtype=torch.float32) * 1e-6
    M_norm = M_norm + eye3.unsqueeze(0)
    # Symmetrize (fp16→fp32 drift correction)
    M_norm = 0.5 * (M_norm + M_norm.transpose(-1, -2))

    eigvals = torch.linalg.eigvalsh(M_norm)                           # (K, 3) ascending
    lam3 = eigvals[:, 0]                                              # smallest = 3D spread

    has_enough = (w_total >= 3.0).to(torch.float32)                   # ≥3 surface neighbors
    violation = (lam3 - lam3_threshold).clamp(min=0.0)
    pen = (violation ** p) * has_enough
    n_active = has_enough.sum()
    if n_active > 0:
        return (pen.sum() / n_active).to(s.dtype)
    return torch.zeros((), device=s.device)


def topology_normal_coherence_loss(sparse_sdf, sparse_coords,
                                    dense_active_64, dense_normal_64,
                                    nb_radius=1, R_threshold=0.7, p=3.0,
                                    default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                                    _cache={}):
    """**Local normal-coherence (Mean Resultant Length)** loss.

    Collect each sparse voxel's ∇s and that of its (2·radius+1)³ neighbors, unit-normalize them, then
    compute the Mean Resultant Length R = ‖mean(unit normals)‖.
    If R is below threshold (= normals are scattered = noise), apply a cubic penalty.

    Principle:
        - flat/smooth surface: normals cluster in one direction → R ≈ 1, no penalty
        - jitter (noise): normals scatter isotropically → R ≈ 0, penalty
    → independent of curvature itself. Preserves natural surfaces while catching only oscillation.

    Default for missing neighbors: dense_active=True → s=-1 (deep solid),
    else s=+1 (deep empty). FD ∇s is computed from that default.
    """
    # Cache shell mask + default SDF
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        _cache[key] = (shell_in_t, def_sdf_64)
    shell_in_t, def_sdf_64 = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    M = s.shape[0]
    is_shell = shell_in_t[cz, cy, cx]

    # build idx_vol
    z_min = (coords_512[..., 0].min() - nb_radius - 1).item()
    y_min = (coords_512[..., 1].min() - nb_radius - 1).item()
    x_min = (coords_512[..., 2].min() - nb_radius - 1).item()
    z_max = (coords_512[..., 0].max() + nb_radius + 1).item()
    y_max = (coords_512[..., 1].max() + nb_radius + 1).item()
    x_max = (coords_512[..., 2].max() + nb_radius + 1).item()
    D = z_max - z_min + 1; H = y_max - y_min + 1; W = x_max - x_min + 1
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    lcz = coords_512[..., 0] - z_min
    lcy = coords_512[..., 1] - y_min
    lcx = coords_512[..., 2] - x_min
    idx_vol[lcz, lcy, lcx] = torch.arange(M, dtype=torch.int32, device=s.device)

    def _sdf_at(dz, dy, dx):
        nz = (lcz + dz).clamp(0, D - 1)
        ny = (lcy + dy).clamp(0, H - 1)
        nx = (lcx + dx).clamp(0, W - 1)
        idx = idx_vol[nz, ny, nx]
        n64_z = ((nz + z_min) // 8).clamp(0, 63).long()
        n64_y = ((ny + y_min) // 8).clamp(0, 63).long()
        n64_x = ((nx + x_min) // 8).clamp(0, 63).long()
        default = def_sdf_64[n64_z, n64_y, n64_x]
        present = (idx >= 0)
        return torch.where(present, s[idx.clamp(min=0).long()], default)

    # Step 1: compute each voxel's ∇s (6-neighbor central FD)
    def _grad_at(dz, dy, dx):
        # ∇s at voxel offset (dz,dy,dx) — used for neighborhood normal field
        gz = (_sdf_at(dz + 1, dy, dx) - _sdf_at(dz - 1, dy, dx)) * 0.5
        gy = (_sdf_at(dz, dy + 1, dx) - _sdf_at(dz, dy - 1, dx)) * 0.5
        gx = (_sdf_at(dz, dy, dx + 1) - _sdf_at(dz, dy, dx - 1)) * 0.5
        return torch.stack([gz, gy, gx], dim=-1)  # (M, 3)

    # Step 2: sum ∇s_unit over the (2r+1)³ neighbors
    sum_unit = torch.zeros(M, 3, device=s.device, dtype=s.dtype)
    cnt = 0
    for dz in range(-nb_radius, nb_radius + 1):
        for dy in range(-nb_radius, nb_radius + 1):
            for dx in range(-nb_radius, nb_radius + 1):
                g = _grad_at(dz, dy, dx)
                gn = g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                sum_unit = sum_unit + g / gn
                cnt += 1

    # Step 3: R = ‖mean‖
    mu = sum_unit / float(cnt)             # (M, 3)
    R = mu.norm(dim=-1)                    # (M,) in [0, 1]

    active = is_shell

    # cubic hinge: R < threshold → noise → penalty
    violation = (R_threshold - R).clamp(min=0.0)
    pen = violation ** p
    if active.sum() > 0:
        return (pen * active).sum() / active.sum()
    return torch.zeros((), device=s.device)


def topology_off_diagonal_hessian_loss(sparse_sdf, sparse_coords, dense_active_64,
                                        surface_band=0.2, p=1.0,
                                        default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                                        _cache={}):
    """Simplified Off-Diagonal Weingarten loss (FlatCAD-inspired, axis-aligned FD).

    FlatCAD (Yin et al., CGF 2025) regularizes |S_12| (off-diagonal shape operator entry) which
    is equivalent to suppressing (κ_2-κ_1)^2/8 — developability prior. The full version uses
    random tangent-frame angles + Hessian-vector products. Here we use the axis-aligned
    approximation:

        H_xy ≈ [s(+x+y) - s(+x-y) - s(-x+y) + s(-x-y)] / 4
        H_yz, H_xz analogously.

    Penalty = mean(|H_xy|^p + |H_yz|^p + |H_xz|^p) over surface-band shell voxels.

    No second-order autograd required (each term is a 4-sample FD of stored sdf values).
    """
    R64, R512 = 64, 512
    factor = R512 // R64  # 8

    da_key = id(dense_active_64)
    if da_key not in _cache:
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        _cache[da_key] = (shell_in_t, def_sdf_64)
    shell_in_t, def_sdf_64 = _cache[da_key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // factor).clamp(0, R64 - 1)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]
    is_shell = shell_in_t[cz, cy, cx]
    s = sparse_sdf.squeeze(-1)

    is_surface = s.abs() < surface_band
    active_mask = (is_shell > 0.5) & is_surface
    active_idx = torch.where(active_mask)[0]
    if active_idx.numel() == 0:
        return torch.zeros((), device=s.device)

    # Build idx_vol covering bbox of sparse coords (padded ±1)
    z0 = int(coords_512[:, 0].min().item()) - 1
    y0 = int(coords_512[:, 1].min().item()) - 1
    x0 = int(coords_512[:, 2].min().item()) - 1
    z1 = int(coords_512[:, 0].max().item()) + 2
    y1 = int(coords_512[:, 1].max().item()) + 2
    x1 = int(coords_512[:, 2].max().item()) + 2
    D, H, W = z1 - z0, y1 - y0, x1 - x0
    M_n = sparse_sdf.shape[0]
    lcz = coords_512[:, 0] - z0
    lcy = coords_512[:, 1] - y0
    lcx = coords_512[:, 2] - x0
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    idx_vol[lcz, lcy, lcx] = torch.arange(M_n, dtype=torch.int32, device=s.device)

    def _sdf_at(dz, dy, dx):
        nz = lcz[active_idx] + dz
        ny = lcy[active_idx] + dy
        nx = lcx[active_idx] + dx
        valid = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nz_c = nz.clamp(0, D - 1); ny_c = ny.clamp(0, H - 1); nx_c = nx.clamp(0, W - 1)
        idx = idx_vol[nz_c, ny_c, nx_c]
        present = (idx >= 0) & valid
        nc_z = ((coords_512[active_idx, 0] + dz) // factor).clamp(0, R64 - 1)
        nc_y = ((coords_512[active_idx, 1] + dy) // factor).clamp(0, R64 - 1)
        nc_x = ((coords_512[active_idx, 2] + dx) // factor).clamp(0, R64 - 1)
        default = def_sdf_64[nc_z, nc_y, nc_x]
        sdf_sparse = s[idx.clamp(min=0).long()]
        return torch.where(present, sdf_sparse, default)

    # Mixed partials (coords are [z, y, x])
    H_xy = (_sdf_at(0, +1, +1) - _sdf_at(0, +1, -1)
            - _sdf_at(0, -1, +1) + _sdf_at(0, -1, -1)) / 4.0
    H_yz = (_sdf_at(+1, +1, 0) - _sdf_at(+1, -1, 0)
            - _sdf_at(-1, +1, 0) + _sdf_at(-1, -1, 0)) / 4.0
    H_xz = (_sdf_at(+1, 0, +1) - _sdf_at(+1, 0, -1)
            - _sdf_at(-1, 0, +1) + _sdf_at(-1, 0, -1)) / 4.0

    # paper ref: Supplementary, Sparse-Stage Loss Terms, term (5) Edge-aware tangential Laplacian (L_lap).
    penalty = (H_xy.abs() ** p + H_yz.abs() ** p + H_xz.abs() ** p).mean()
    return penalty


def topology_edt_anchored_loss(sparse_sdf, sparse_coords, dense_active_64,
                                tol=0.05, p=1.0, scale=1.0,
                                _cache={}):
    """Voxel-wise deviation from dense MC ideal SDF (= signed Euclidean Distance Transform).

    For each sparse voxel v:
        ref_at_v = signed_EDT(dense_active_64)[v // 8] * scale
        dev      = |sparse_sdf(v) - ref_at_v|
        penalty  = max(0, dev - tol)^p

    Tolerance band `tol` prevents penalizing tiny deviations (numerical noise).
    Returns mean over all sparse voxels.
    """
    R64, R512 = 64, 512
    factor = R512 // R64
    da_key = ('edt', id(dense_active_64), float(scale))
    if da_key not in _cache:
        from scipy.ndimage import distance_transform_edt as _edt
        d_np = dense_active_64.cpu().numpy().astype(bool)
        edt_in = _edt(d_np)            # depth inside solid (voxel units)
        edt_out = _edt(~d_np)          # depth into empty space
        ref_sdf_64 = (edt_in - edt_out).astype('float32') * float(scale)
        _cache[da_key] = torch.from_numpy(ref_sdf_64).to(sparse_sdf.device)
    ref_t = _cache[da_key]
    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // factor).clamp(0, R64 - 1)
    s = sparse_sdf.squeeze(-1)
    ref_at = ref_t[coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]]
    dev = (s - ref_at).abs()
    return torch.clamp(dev - tol, min=0).pow(p).mean()


def topology_snapshot_anchor_loss(sparse_sdf, snap_sdf, anchor_weight_per_voxel, p=1.0):
    """Anchor sparse_sdf to snapshot at step 0, weighted by depth.

    anchor_weight_per_voxel: (M,) — 1 at deep interior, 0 at surface
        (sparse voxels close to surface are free, deep voxels are pulled back to snapshot)
    """
    if snap_sdf is None:
        return torch.zeros((), device=sparse_sdf.device)
    s = sparse_sdf.squeeze(-1)
    snap = snap_sdf.squeeze(-1) if snap_sdf.dim() > 1 else snap_sdf
    dev = (s - snap).abs() * anchor_weight_per_voxel
    return dev.pow(p).mean()


def topology_dense_frame_odw_loss(sparse_sdf, sparse_coords, dense_active_64, dense_normal_64,
                                   surface_band=0.2, p=1.0,
                                   default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                                   _cache={}):
    """Dense-frame Off-diagonal Weingarten (DF-ODW) — rotation-invariant FlatCAD variant.

    Uses dense MC normal `n_d = dense_normal_64[v_64]` to define the tangent frame
    (e_1, e_2) ⊥ n_d. Computes the world Hessian (6 unique entries via FD) and
    projects: S_12 = e_1^T H e_2. Penalty = mean |S_12|^p over shell + surface-band voxels.

    Compared to axis-aligned `topology_off_diagonal_hessian_loss`:
    - Rotation-invariant (works for arbitrary surface orientation)
    - Same neighbor-lookup cost (axial 6 + diagonal 12 = 18 FD samples)
    - Frame comes from dense MC normal (deterministic, no random sampling)
    """
    R64, R512 = 64, 512
    factor = R512 // R64

    da_key = ('df_odw', id(dense_active_64), id(dense_normal_64))
    if da_key not in _cache:
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        _cache[da_key] = (shell_in_t, def_sdf_64)
    shell_in_t, def_sdf_64 = _cache[da_key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // factor).clamp(0, R64 - 1)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]
    is_shell = shell_in_t[cz, cy, cx]
    s = sparse_sdf.squeeze(-1)

    is_surface = s.abs() < surface_band
    n_d_full = dense_normal_64.to(sparse_sdf.device).float()[cz, cy, cx]   # (M, 3)
    n_norm = n_d_full.norm(dim=-1)
    has_normal = n_norm > 1e-4
    active_mask = (is_shell > 0.5) & is_surface & has_normal
    active_idx = torch.where(active_mask)[0]
    if active_idx.numel() == 0:
        return torch.zeros((), device=s.device)

    # Build idx_vol covering bbox of sparse coords (padded ±1)
    z0 = int(coords_512[:, 0].min().item()) - 1
    y0 = int(coords_512[:, 1].min().item()) - 1
    x0 = int(coords_512[:, 2].min().item()) - 1
    z1 = int(coords_512[:, 0].max().item()) + 2
    y1 = int(coords_512[:, 1].max().item()) + 2
    x1 = int(coords_512[:, 2].max().item()) + 2
    D, H, W = z1 - z0, y1 - y0, x1 - x0
    M_n = sparse_sdf.shape[0]
    lcz = coords_512[:, 0] - z0
    lcy = coords_512[:, 1] - y0
    lcx = coords_512[:, 2] - x0
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    idx_vol[lcz, lcy, lcx] = torch.arange(M_n, dtype=torch.int32, device=s.device)

    def _sdf_at(dz, dy, dx):
        nz = lcz[active_idx] + dz
        ny = lcy[active_idx] + dy
        nx = lcx[active_idx] + dx
        valid = (nz >= 0) & (nz < D) & (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        nz_c = nz.clamp(0, D - 1); ny_c = ny.clamp(0, H - 1); nx_c = nx.clamp(0, W - 1)
        idx = idx_vol[nz_c, ny_c, nx_c]
        present = (idx >= 0) & valid
        nc_z = ((coords_512[active_idx, 0] + dz) // factor).clamp(0, R64 - 1)
        nc_y = ((coords_512[active_idx, 1] + dy) // factor).clamp(0, R64 - 1)
        nc_x = ((coords_512[active_idx, 2] + dx) // factor).clamp(0, R64 - 1)
        default = def_sdf_64[nc_z, nc_y, nc_x]
        sdf_sparse = s[idx.clamp(min=0).long()]
        return torch.where(present, sdf_sparse, default)

    s0 = s[active_idx]
    # World Hessian — 6 unique entries (coords are [z, y, x])
    s_xp = _sdf_at(0, 0, +1); s_xm = _sdf_at(0, 0, -1)
    s_yp = _sdf_at(0, +1, 0); s_ym = _sdf_at(0, -1, 0)
    s_zp = _sdf_at(+1, 0, 0); s_zm = _sdf_at(-1, 0, 0)
    H_xx = s_xp - 2 * s0 + s_xm
    H_yy = s_yp - 2 * s0 + s_ym
    H_zz = s_zp - 2 * s0 + s_zm
    H_xy = (_sdf_at(0, +1, +1) - _sdf_at(0, +1, -1)
            - _sdf_at(0, -1, +1) + _sdf_at(0, -1, -1)) / 4.0
    H_yz = (_sdf_at(+1, +1, 0) - _sdf_at(+1, -1, 0)
            - _sdf_at(-1, +1, 0) + _sdf_at(-1, -1, 0)) / 4.0
    H_xz = (_sdf_at(+1, 0, +1) - _sdf_at(+1, 0, -1)
            - _sdf_at(-1, 0, +1) + _sdf_at(-1, 0, -1)) / 4.0

    # Build tangent frame (e_1, e_2) ⊥ n_d using the most-stable axis pick.
    n_hat = n_d_full[active_idx] / n_norm[active_idx].unsqueeze(-1).clamp(min=1e-6)
    # Pick world axis with smallest |n_hat component| as helper to construct e_1 via Gram-Schmidt
    abs_n = n_hat.abs()
    # Helper axis matrix (K, 3): unit vector along smallest abs component
    min_axis = abs_n.argmin(dim=-1)
    helper = torch.zeros_like(n_hat)
    helper[torch.arange(len(min_axis), device=helper.device), min_axis] = 1.0
    e1 = helper - (helper * n_hat).sum(-1, keepdim=True) * n_hat
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    e2 = torch.cross(n_hat, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # S_12 = e_1^T H e_2  (note H is symmetric so H_yx=H_xy etc)
    e1x, e1y, e1z = e1[..., 0], e1[..., 1], e1[..., 2]
    e2x, e2y, e2z = e2[..., 0], e2[..., 1], e2[..., 2]
    # H @ e2 = (H_xx*e2x + H_xy*e2y + H_xz*e2z,  H_xy*e2x + H_yy*e2y + H_yz*e2z,  H_xz*e2x + H_yz*e2y + H_zz*e2z)
    He2_x = H_xx * e2x + H_xy * e2y + H_xz * e2z
    He2_y = H_xy * e2x + H_yy * e2y + H_yz * e2z
    He2_z = H_xz * e2x + H_yz * e2y + H_zz * e2z
    S12 = e1x * He2_x + e1y * He2_y + e1z * He2_z

    # paper ref: Supplementary, Sparse-Stage Loss Terms, term (5) Edge-aware tangential Laplacian (L_lap).
    penalty = S12.abs().pow(p).mean()
    return penalty


def topology_lap_smooth_loss(sparse_sdf, sparse_coords,
                              dense_active_64, dense_normal_64,
                              edge_sigma=0.3, stride=1,
                              default_solid_sdf=-1.0, default_empty_sdf=+1.0,
                              _cache={}):
    """**Edge-aware tangential Laplacian** smoothness loss.

    paper ref: Supplementary, Sparse-Stage Loss Terms, term (5) Edge-aware tangential Laplacian (L_lap).

    L2 of the SDF tangent-plane Laplacian (= Laplace-Beltrami) at each shell-cell sparse voxel:
        ∇²s_tan = trace(H) - n_d^T H n_d
    where H is 3×3 Hessian via 19 neighbor FD (6 axial + 12 in-plane diagonals + self).

    Edge mask per dense 64³ cell:
        edge_score(c) = max over 6 neighbor cells of (1 - cos(n_d(c), n_d(c')))
        w(c) = exp(-edge_score² / sigma²)
    Sharp dihedral edge (= abrupt normal-direction change between adjacent cells) → w ≈ 0 → Laplacian exempt.
    Flat region (adjacent cell normals agree) → w ≈ 1 → penalize even small roughness.

    Penalize only the tangential part → preserve natural surface curvature (across-surface SDF transition),
    catching only micro patterns on the surface.
    """
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        shell_in_t = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
        d_t = dense_active_64.to(sparse_sdf.device).float()
        def_sdf_64 = default_solid_sdf * d_t + default_empty_sdf * (1.0 - d_t)
        # Edge mask: max angle deviation of dense normal vs 6 neighbor cells
        n_d_64 = dense_normal_64.to(sparse_sdf.device)
        n_norm = n_d_64.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        n_unit = n_d_64 / n_norm
        valid = (n_norm.squeeze(-1) > 1e-4).float()
        max_dev = torch.zeros(64, 64, 64, device=sparse_sdf.device)
        for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            n_shift = torch.roll(n_unit,  shifts=(dz,dy,dx), dims=(0,1,2))
            v_shift = torch.roll(valid,   shifts=(dz,dy,dx), dims=(0,1,2))
            cos = (n_unit * n_shift).sum(-1).clamp(-1.0, 1.0)
            dev = (1.0 - cos) * valid * v_shift
            max_dev = torch.maximum(max_dev, dev)
        edge_weight_64 = torch.exp(-max_dev * max_dev / (edge_sigma * edge_sigma))
        _cache[key] = (shell_in_t, def_sdf_64, edge_weight_64)
    shell_in_t, def_sdf_64, edge_weight_64 = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    M = s.shape[0]
    is_shell = shell_in_t[cz, cy, cx]

    # Build idx_vol over bbox (re-used pattern from topology_normal_fd_loss)
    z_min = (coords_512[..., 0].min() - 1).item()
    y_min = (coords_512[..., 1].min() - 1).item()
    x_min = (coords_512[..., 2].min() - 1).item()
    z_max = (coords_512[..., 0].max() + 1).item()
    y_max = (coords_512[..., 1].max() + 1).item()
    x_max = (coords_512[..., 2].max() + 1).item()
    D = z_max - z_min + 1; H = y_max - y_min + 1; W = x_max - x_min + 1
    idx_vol = torch.full((D, H, W), -1, dtype=torch.int32, device=s.device)
    lcz = coords_512[..., 0] - z_min
    lcy = coords_512[..., 1] - y_min
    lcx = coords_512[..., 2] - x_min
    idx_vol[lcz, lcy, lcx] = torch.arange(M, dtype=torch.int32, device=s.device)

    def _nb(dz, dy, dx):
        nz = (lcz + dz).clamp(0, D - 1)
        ny = (lcy + dy).clamp(0, H - 1)
        nx = (lcx + dx).clamp(0, W - 1)
        idx = idx_vol[nz, ny, nx]
        n64_z = ((nz + z_min) // 8).clamp(0, 63).long()
        n64_y = ((ny + y_min) // 8).clamp(0, 63).long()
        n64_x = ((nx + x_min) // 8).clamp(0, 63).long()
        default = def_sdf_64[n64_z, n64_y, n64_x]
        present = (idx >= 0)
        sdf_sparse = s[idx.clamp(min=0).long()]
        return torch.where(present, sdf_sparse, default)

    # 6 axial neighbors (±stride voxel — stride=1 default, >1 for multi-res "wider stencil")
    K = max(1, int(stride))
    s_zp = _nb(+K, 0, 0); s_zm = _nb(-K, 0, 0)
    s_yp = _nb(0, +K, 0); s_ym = _nb(0, -K, 0)
    s_xp = _nb(0, 0, +K); s_xm = _nb(0, 0, -K)

    # 12 in-plane diagonal neighbors (for mixed second derivatives)
    s_pp_zy = _nb(+K, +K, 0); s_pm_zy = _nb(+K, -K, 0)
    s_mp_zy = _nb(-K, +K, 0); s_mm_zy = _nb(-K, -K, 0)
    s_pp_zx = _nb(+K, 0, +K); s_pm_zx = _nb(+K, 0, -K)
    s_mp_zx = _nb(-K, 0, +K); s_mm_zx = _nb(-K, 0, -K)
    s_pp_yx = _nb(0, +K, +K); s_pm_yx = _nb(0, +K, -K)
    s_mp_yx = _nb(0, -K, +K); s_mm_yx = _nb(0, -K, -K)

    # Hessian components (h = K voxel) — divide by K² for scale-correct second derivative
    _K2 = float(K * K)
    Hzz = (s_zp - 2.0 * s + s_zm) / _K2
    Hyy = (s_yp - 2.0 * s + s_ym) / _K2
    Hxx = (s_xp - 2.0 * s + s_xm) / _K2
    Hzy = (s_pp_zy + s_mm_zy - s_pm_zy - s_mp_zy) * 0.25 / _K2
    Hzx = (s_pp_zx + s_mm_zx - s_pm_zx - s_mp_zx) * 0.25 / _K2
    Hyx = (s_pp_yx + s_mm_yx - s_pm_yx - s_mp_yx) * 0.25 / _K2

    lap_full = Hzz + Hyy + Hxx

    n_d_vox = dense_normal_64[cz, cy, cx]
    n_d_norm = n_d_vox.norm(dim=-1).clamp(min=1e-6)
    n_d_unit = n_d_vox / n_d_norm.unsqueeze(-1)
    nz_, ny_, nx_ = n_d_unit[..., 0], n_d_unit[..., 1], n_d_unit[..., 2]
    # n_d^T H n_d
    n_H_n = (nz_*nz_*Hzz + ny_*ny_*Hyy + nx_*nx_*Hxx
             + 2.0*nz_*ny_*Hzy + 2.0*nz_*nx_*Hzx + 2.0*ny_*nx_*Hyx)
    lap_tan = lap_full - n_H_n   # tangential Laplacian (Laplace-Beltrami)

    w_voxel = edge_weight_64[cz, cy, cx]
    active = is_shell * (n_d_norm > 1e-4).float()

    pen = (lap_tan ** 2) * w_voxel * active
    if active.sum() > 0:
        return pen.sum() / active.sum()
    return torch.zeros((), device=s.device)


def topology_no_overhang_loss(sparse_sdf, sparse_coords,
                               dense_active_64, dense_normal_64,
                               p=3.0, _cache={}):
    """**No-overhang / locally-convex-from-outside** topology loss.

    For the two tangent-plane directions t1, t2 (orthogonal to n_d) of each dense shell_in cell,
    force the sparse SDF's second curvature ∂²s/∂t² to be **≥ 0**.

    Math:
        - s convex along the tangent direction (∂²s/∂t² > 0) ⇔ the surface bends inward in that direction
          ⇔ the surface is **convex** seen from outside ⇔ **no overhang**
        - concave (∂²s/∂t² < 0) ⇔ the surface bends outward ⇔ overhang

    In-cell quadratic fit: s_c(p) ≈ a·p + b·p²  → 2b = ∂²s/∂t²
    As a moment ratio over the zero-mean distribution:
        2b ≈ Σ(p_c² · s_c) / Σ(p_c⁴)

    On violation (2b < 0), polynomial hinge: pen = max(0, -2b)^p
    """
    key = id(dense_active_64)
    if key not in _cache:
        from scipy.ndimage import binary_erosion
        d_np = dense_active_64.cpu().numpy().astype(bool)
        shell_in_np = d_np & ~binary_erosion(d_np)
        _cache[key] = torch.from_numpy(shell_in_np.astype('float32')).to(sparse_sdf.device)
    shell_in_t = _cache[key]

    coords_512 = sparse_coords[..., 1:].long()
    coords_64 = (coords_512 // 8).clamp(0, 63)
    rel = (coords_512 - coords_64 * 8).float() - 3.5
    cz, cy, cx = coords_64[..., 0], coords_64[..., 1], coords_64[..., 2]

    s = sparse_sdf.squeeze(-1)
    is_shell = shell_in_t[cz, cy, cx]
    cell_id = (cz * 64 + cy) * 64 + cx

    # Per-voxel n_d (use dense MC outward normal of its cell)
    n_d_vox = dense_normal_64[cz, cy, cx]                       # (M, 3)
    n_d_norm = n_d_vox.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    n_d_unit = n_d_vox / n_d_norm

    # Build 2 tangent dirs ⟂ n_d (Gram-Schmidt with whichever world axis is "least parallel")
    abs_n = n_d_unit.abs()
    min_axis = abs_n.argmin(dim=-1, keepdim=True)               # (M, 1)
    helper = torch.zeros_like(n_d_unit).scatter_(-1, min_axis, 1.0)
    t1 = torch.cross(n_d_unit, helper, dim=-1)
    t1 = t1 / t1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    t2 = torch.cross(n_d_unit, t1, dim=-1)                      # already unit

    N = 64*64*64
    def _cell_sum(vals):
        # vals: (M,) — weighted by is_shell so non-shell rows don't contribute
        buf = torch.zeros(N, device=s.device, dtype=s.dtype)
        buf.scatter_add_(0, cell_id, vals * is_shell)
        return buf

    cnt = _cell_sum(torch.ones_like(s))
    cnt_safe = cnt.clamp(min=1.0)
    sum_s = _cell_sum(s)                                        # (N,)
    mean_s = sum_s / cnt_safe

    losses = []
    for t in (t1, t2):
        p_t = (rel * t).sum(-1)                                 # (M,)
        mean_p = _cell_sum(p_t) / cnt_safe
        p_c = (p_t - mean_p[cell_id]) * is_shell                # (M,) zero-mean per-cell
        # Unbiased least-squares estimator for c in s ≈ a + b·p_c + c·p_c²:
        #   c = (Σ p_c²·s - mean_s · Σ p_c²) / (Σ p_c⁴ - (Σ p_c²)²/M)
        # ∂²s/∂t² = 2c
        sum_p2  = _cell_sum(p_c * p_c)                          # Σ p_c²
        sum_p4  = _cell_sum(p_c * p_c * p_c * p_c)              # Σ p_c⁴
        sum_p2s = _cell_sum(p_c * p_c * s)                      # Σ p_c²·s
        num   = sum_p2s - mean_s * sum_p2
        denom = (sum_p4 - sum_p2 * sum_p2 / cnt_safe).clamp(min=1e-6)
        two_b = 2.0 * num / denom                               # ≈ ∂²s/∂t²
        active = cnt > 4.0
        violation = (-two_b).clamp(min=0.0)                     # > 0 if concave (overhang)
        pen = violation ** p
        if active.any():
            losses.append((pen * active.float()).sum() / active.float().sum())

    if losses:
        return sum(losses) / len(losses)
    return torch.zeros((), device=s.device)


def weight_anneal(step, total_steps, w_base, w_peak, ramp_start=0.66):
    """Cosine ramp w_base → w_peak from ramp_start*total to total."""
    phase = max(0.0, (step / total_steps - ramp_start) / (1.0 - ramp_start))
    phase = min(1.0, phase)
    smooth = 0.5 * (1.0 - np.cos(np.pi * phase))
    return w_base + (w_peak - w_base) * smooth


def n_inner_anneal(step, total_steps, n_base=1, n_late=3, ramp_start=0.66):
    """More inner SGD steps at later phase."""
    return n_late if step >= ramp_start * total_steps else n_base
# ===============================================================================


def encode_multi_view(pipe, images, conditioner, do_classifier_free_guidance=True):
    """Encode multi-view images and fuse via token concatenation.

    images: tensor (N, C, H, W) — N views
    Returns cond, uncond: each (1, N*T, D)
    """
    cond_per_view = conditioner(images[:, :3])  # (N, T, D)
    if isinstance(cond_per_view, tuple):
        cond_per_view, _ = cond_per_view
    N, T, D = cond_per_view.shape
    cond = cond_per_view.reshape(1, N * T, D)
    if do_classifier_free_guidance:
        uncond = torch.zeros_like(cond)
    else:
        uncond = None
    return cond, uncond


def reachability_loss(occ_logits, bc_mask, envelope_mask=None,
                       reach_threshold=0.1, n_iters=100, verbose=True,
                       reach_kernel=41):
    """BC connectivity loss — bigkernel + low threshold version.

    paper ref: main text, geometric guidance reachability term (weight cw).

    Key params:
      reach_threshold=0.1 (vs 0.5): looser definition of "solid for reach"
      reach_kernel: gap-detection max_pool3d kernel size (odd int). default=41
                    (NEW HEAD). OLD 1c27bc9 used 21. Smaller kernel → more
                    localized gap detection → mesh allowed to be more sparse/thin
                    → stronger FEA gradient differential effect.
      n_iters: flood-fill propagation steps. Each step dilates the reached set by one
                    voxel (max_pool3d k=3), so this caps the reachable distance at
                    n_iters voxels. 100 covers the 64³ grid diagonal (111) only just —
                    a BC island further than 100 voxels of solid path away is reported
                    unreached even when it is connected. Exposed as --reach-iters.
      envelope_mask: restrict to bracket envelope
    """
    _rk = int(reach_kernel)
    if _rk % 2 == 0:
        _rk = _rk + 1  # max_pool3d needs odd kernel
    _rp = _rk // 2
    occ_p = torch.sigmoid(occ_logits)
    with torch.no_grad():
        bc_f = bc_mask.float()
        env_f = envelope_mask.float() if envelope_mask is not None else torch.ones_like(bc_f)
        bc_idx = bc_mask.nonzero()
        if len(bc_idx) == 0:
            return torch.tensor(0.0, device=occ_p.device)
        seed = torch.zeros_like(bc_f)
        ci = bc_idx.float().mean(dim=0).long()
        seed[ci[0], ci[1], ci[2]] = 1.0
        if bc_f[ci[0], ci[1], ci[2]] < 0.5:
            seed[bc_idx[0, 0], bc_idx[0, 1], bc_idx[0, 2]] = 1.0
        solid_in_env = (occ_p > reach_threshold).float() * env_f
        reach_through = torch.clamp(bc_f + solid_in_env, max=1.0)
        reached = seed.clone()
        for _ in range(n_iters):
            r4 = reached.unsqueeze(0).unsqueeze(0)
            prop = F.max_pool3d(r4, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0)
            reached = torch.maximum(reached, prop * reach_through)
        unreached_bc = bc_f * (1 - reached)
        n_unreached = float(unreached_bc.sum().item())
        if n_unreached < 1.0:
            if verbose:
                print(f"    [reach] all BC reached (thresh={reach_threshold}) → loss=0", flush=True)
            return torch.tensor(0.0, device=occ_p.device)
        # kernel size from arg (default 41 = NEW HEAD, 21 = OLD 1c27bc9 paper-claim setup)
        reached_near = F.max_pool3d(reached.unsqueeze(0).unsqueeze(0),
                                      kernel_size=_rk, stride=1, padding=_rp).squeeze(0).squeeze(0)
        unreached_near = F.max_pool3d(unreached_bc.unsqueeze(0).unsqueeze(0),
                                        kernel_size=_rk, stride=1, padding=_rp).squeeze(0).squeeze(0)
        gap = reached_near * unreached_near * (occ_p < reach_threshold).float() * env_f
        n_gap = float(gap.sum().item())
        if n_gap < 1.0:
            if verbose:
                print(f"    [reach] {int(n_unreached)} BC unreached but no gap → loss=0", flush=True)
            return torch.tensor(0.0, device=occ_p.device)
        if verbose:
            print(f"    [reach] {int(n_unreached)} BC unreached, {int(n_gap)} gap → push solid", flush=True)
    bce_solid = F.binary_cross_entropy_with_logits(occ_logits, torch.ones_like(occ_logits), reduction='none')
    return (bce_solid * gap).sum() / gap.sum()




def thinness_loss(occ_logits, min_thick=3):
    """DEPRECATED — this is actually a boundary (shell) loss, not thickness.
    Penalizes (occ - eroded) = all surface voxels. Use thickness_kill_loss instead.
    Kept for reproducibility (--tw).
    """
    occ_p = torch.sigmoid(occ_logits)
    eroded = -F.max_pool3d(-occ_p.unsqueeze(0).unsqueeze(0),
                            kernel_size=min_thick, stride=1,
                            padding=min_thick // 2).squeeze()
    if eroded.shape != occ_p.shape:
        eroded = F.interpolate(eroded[None, None], size=occ_p.shape, mode='trilinear',
                                align_corners=False).squeeze()
    thin = (occ_p - eroded).clamp_min(0.0)
    return thin.mean()


def dense_rmin_filter_loss(occ_logits, r_min=2, threshold=0.5, normalize=False):
    """TopOpt-style r_min density filter at dense stage.
    Average-pool (radius r_min) → blurred local density. Solid voxels with
    blurred < threshold get penalty. Gradient pushes empty neighbors UP to raise
    local density (= thicken thin features, growth-oriented unlike opening which kills).

    normalize: divide by the voxel count so the return is a MEAN, not a SUM. The historical form is
      a sum over the whole 64^3 grid, which makes its magnitude scale with grid size and swamps the
      other terms: measured on the caliper with tw_rmin=0.025, r_min=3, threshold=0.5, the term came
      to 102.4 against a dense total of 25.14 — 407% of the loss, versus 90% for the 3-region BCE.
      With every voxel of a 28-32%-occupancy part failing a local-density-0.5 test, the penalty is
      also saturated, so sweeping r_min over 1/2/3 moved the delivered median wall thickness by only
      2.92-3.20 mm (9%). Normalising makes the weight interpretable and the term comparable to the
      sparse-stage losses, which sit at 0.001-0.7% of their total. Off by default so the three
      shipped domains reproduce byte-identically.
    """
    occ_p = torch.sigmoid(occ_logits)
    k = 2 * r_min + 1
    blurred = F.avg_pool3d(occ_p.unsqueeze(0).unsqueeze(0),
                            kernel_size=k, stride=1, padding=r_min,
                            count_include_pad=False).squeeze()
    if blurred.shape != occ_p.shape:
        blurred = F.interpolate(blurred[None, None], size=occ_p.shape, mode='trilinear',
                                 align_corners=False).squeeze()
    deficit = F.relu(threshold - blurred)
    out = (occ_p * deficit).sum()
    if normalize:
        out = out / occ_p.numel()
    return out


def thickness_kill_loss(occ_logits, kill_size=1):
    """Opening-based thickness penalty: features ≤ (2·kill_size+1) voxels die under opening.
      kill_size=1  →  kernel=3  →  1-vox features penalized (2+ survives)
      kill_size=2  →  kernel=5  →  ≤2-vox features penalized (3+ survives)
    Gradient (via argmin/argmax chains) pushes BOTH self (down) and empty neighbors (up)
    so thin features grow into their empty neighborhood rather than just dying.
    """
    occ_p = torch.sigmoid(occ_logits)
    k = 2 * kill_size + 1
    # soft erosion (min-pool via negated max-pool)
    eroded = -F.max_pool3d(-occ_p.unsqueeze(0).unsqueeze(0),
                            kernel_size=k, stride=1, padding=k // 2).squeeze()
    # soft dilation (max-pool back)
    opened = F.max_pool3d(eroded.unsqueeze(0).unsqueeze(0),
                           kernel_size=k, stride=1, padding=k // 2).squeeze()
    if opened.shape != occ_p.shape:
        opened = F.interpolate(opened[None, None], size=occ_p.shape, mode='trilinear',
                                align_corners=False).squeeze()
    # voxels that died in opening = thin features (≤ kernel size).
    # Return raw thin-voxel count (mass). Each spike contributes ~1.
    # Use tw_hard~0.5-2.0 for strong-enough penalty (other dense losses are O(1)).
    killed = (occ_p - opened).clamp_min(0.0)
    return killed.sum()


def dense_flowdps(pipe, image, bracket_mask, design_mask, bc_mask,
                  num_inference_steps=50, guidance_scale=7.0,
                  eta=300.0, bc_w=3.0, out_w=10.0, dw=0.0,
                  inner_steps=3, mc_threshold=0.1, bce_boundary=0.5, bce_mode='recentered', seed=42,
                  use_multi_view=False,
                  vw=0.0, vol_target=0.4,
                  vol_projection=False, vol_proj_w=5.0,
                  aug_lag=False, lambda_init=0.0, mu_aug_lag=50.0, lambda_alpha=1.0,
                  heaviside_proj=False, beta_init=2.0, beta_max=20.0, eta_bisect_iter=15,
                  sw=0.0, symmetry_axis=0,
                  pw=0.0, load_path_mask=None,
                  cw=0.0, tw=0.0, min_thick=3, cw_warmup=0.5, reach_kernel=41, rmin_normalize=False,
                  reach_threshold=0.1, reach_iters=100,
                  tw_hard=0.0, tw_soft=0.0,
                  tw_rmin=0.0, rmin_radius=2, rmin_thresh=0.5,
                  fea_w=0.0, fea_every_n=5, fea_warmup=0.3,
                  fea_domain_dir=None, fea_nodes=None, fea_mesh_cache='/tmp/bracket_fea.msh',
                  fea_mesh_size=0.006, fea_penal=3.0,
                  fea_back_domain_dir=None, fea_back_load_mode='y',
                  fea_back_load_magnitude=200.0,
                  fea_mesh_size_fine=None, fea_mesh_cache_fine='/tmp/bracket_fea_fine.msh',
                  fea_fine_warmup=0.8,
                  snapshot_every=0, snapshot_dir=None,
                  env_origin=None, env_pitch=None,
                  loss_csv=None,
                  dense_opt='flowdps', dense_lr=5e-3, grad_normalize=True,
                  shape_qd_archive=None, shape_qd_target=-1, shape_qd_w=0.0,
                  shape_qd_warmup=0.35, shape_anchor_bank=None, shape_anchor_w=0.0,
                  shape_scaffold_w=0.0, shape_scaffold_pool=4,
                  shape_residual_w=0.0, shape_residual_pool=4,
                  shape_residual_delta=0.15, shape_residual_neutral_w=0.05,
                  shape_transport_radius=0.0, shape_transport_dims=2,
                  shape_transport_ridge=1e-3, shape_transport_max_rel=0.05,
                  shape_transport_every=5, shape_transport_max_updates=1,
                  image_proj_target=None, image_proj_w=0.0, image_proj_warmup=0.35,
                  oc_flow_w=0.0, oc_flow_warmup=0.35,
                  oc_flow_max_rel=0.5,
                  env_excess_w=0.0, env_excess_tol=0.002,
                  dense_token_policy='legacy'):
    """Dense stage with FlowDPS gradient guidance.

    Core 3-region BCE:
      out_w   — outside bracket → empty
      bc_w    — BC region (fix+load peg) → solid
      dw      — design region → solid (soft)

    Engineering extensions (set weight >0 to activate):
      vw + vol_target — soft mass budget on design region (sigmoid(occ).mean() → vol_target)
      sw + symmetry_axis — bilateral symmetry on bracket (X-axis default)
      pw + load_path_mask — load corridor voxels prefer solid (pre-computed mask)

    bracket_mask: (R, R, R) bool — full domain (BC ∪ design)
    design_mask: (R, R, R) bool — interior (free region)
    bc_mask: (R, R, R) bool — boundary (force solid)
    load_path_mask: (R, R, R) bool or None — voxels along fix↔load corridor

    Returns: latent_index for next stage (sparse512).
    """
    dual_fea_reference = {}
    device = pipe.device
    vae = pipe.dense_vae
    dit = pipe.dense_dit
    scheduler = pipe.dense_scheduler

    # Get cond (single-view or multi-view fusion)
    if use_multi_view and image.shape[0] > 1:
        cond, uncond = encode_multi_view(pipe, image, pipe.dense_image_encoder, do_classifier_free_guidance=True)
        print(f"  multi-view cond fused: {cond.shape} from {image.shape[0]} views")
    else:
        cond, uncond = pipe.encode_image(image[:1], pipe.dense_image_encoder,
                                          do_classifier_free_guidance=True)
    batch_size = cond.shape[0]

    latent_shape = (batch_size, *dit.latent_shape)
    print(f"latent shape: {latent_shape}")
    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, dtype=pipe.dtype, device=device, generator=gen)

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Pre-stage masks (matched to decoder output res)
    # The dense decoder produces 64^3 occupancy.
    R = 64
    bracket_t = torch.from_numpy(bracket_mask.astype(np.float32)).to(device)
    image_projection = None
    if image_proj_w > 0:
        if not image_proj_target:
            raise ValueError('image projection guidance needs --image-proj-target')
        from image_projection_loss import ImageProjectionLoss
        image_projection = ImageProjectionLoss(image_proj_target, device, mc_threshold)
        print(f"  image projection: target={image_proj_target} w={image_proj_w} "
              f"warmup={image_proj_warmup}", flush=True)
    shape_qd = None
    if shape_qd_w > 0:
        if not shape_qd_archive or shape_qd_target < 0:
            raise ValueError('shape-QD needs --shape-qd-archive and --shape-qd-target')
        from shape_qd_loss import FrozenShapePCA
        shape_qd = FrozenShapePCA(shape_qd_archive, shape_qd_target, device)
        print(f"  Shape-QD: target niche={shape_qd_target} w={shape_qd_w} warmup={shape_qd_warmup}")
    shape_anchor = None
    if shape_anchor_w > 0:
        if not shape_anchor_bank or shape_qd_target < 0:
            raise ValueError('shape anchor needs --shape-anchor-bank and --shape-qd-target')
        from shape_qd_loss import PrototypeShapeAnchor
        shape_anchor = PrototypeShapeAnchor(shape_anchor_bank, shape_qd_target, device)
        print(f"  Shape anchor: target niche={shape_qd_target} w={shape_anchor_w}")
    shape_scaffold = None
    if shape_scaffold_w > 0:
        if not shape_anchor_bank or shape_qd_target < 0:
            raise ValueError('shape scaffold needs --shape-anchor-bank and --shape-qd-target')
        from shape_qd_loss import MacroShapeScaffold
        shape_scaffold = MacroShapeScaffold(shape_anchor_bank, shape_qd_target, device, shape_scaffold_pool)
        print(f"  Shape scaffold: target niche={shape_qd_target} w={shape_scaffold_w} pool={shape_scaffold_pool}")
    shape_residual = None
    if shape_residual_w > 0:
        if not shape_anchor_bank or shape_qd_target < 0:
            raise ValueError('shape residual needs --shape-anchor-bank and --shape-qd-target')
        from shape_qd_loss import ContrastiveMacroMorphology
        shape_residual = ContrastiveMacroMorphology(
            shape_anchor_bank, shape_qd_target, device, shape_residual_pool,
            shape_residual_delta, shape_residual_neutral_w)
        print(f"  Shape residual: target niche={shape_qd_target} w={shape_residual_w} "
              f"pool={shape_residual_pool} delta={shape_residual_delta}")
    shape_transport = None
    if shape_transport_radius > 0:
        if not shape_qd_archive or shape_qd_target < 0:
            raise ValueError('shape transport needs --shape-qd-archive and --shape-qd-target')
        if dense_opt != 'flowdps':
            raise ValueError('shape transport currently requires --dense-opt flowdps')
        if shape_transport_every < 1 or shape_transport_max_updates < 1:
            raise ValueError('shape transport interval and update count must be positive')
        from shape_qd_loss import LocalPcaTransport
        shape_transport = LocalPcaTransport(
            shape_qd_archive, shape_qd_target, device, shape_transport_dims,
            shape_transport_radius, shape_transport_ridge, shape_transport_max_rel)
        print(f"  Shape transport: target niche={shape_qd_target} dims={shape_transport_dims} "
              f"radius={shape_transport_radius} max_rel={shape_transport_max_rel} "
              f"every={shape_transport_every} max_updates={shape_transport_max_updates}")
    # OC-Flow-style controller: do not perturb z after the model velocity has
    # already been evaluated.  Instead, turn the differentiable structural loss
    # into a bounded correction of that velocity before the Euler scheduler step.
    # For z_next = z + (sigma_next-sigma) * v and sigma_next-sigma < 0 during
    # denoising, adding +grad(L) to v moves z_next in -grad(L), as desired.
    oc_flow_active = oc_flow_w > 0
    if oc_flow_active:
        if dense_opt != 'flowdps':
            raise ValueError('--oc-flow-w currently requires --dense-opt flowdps')
        if not 0.0 <= oc_flow_warmup <= 1.0:
            raise ValueError('--oc-flow-warmup must be in [0, 1]')
        if oc_flow_max_rel <= 0:
            raise ValueError('--oc-flow-max-rel must be positive')
        if shape_transport is not None:
            raise ValueError('OC-Flow and shape transport cannot be enabled together')
        print(f"  OC-Flow velocity control: w={oc_flow_w} warmup={oc_flow_warmup} "
              f"max_rel={oc_flow_max_rel} (one loss-gradient per flow step)")
    design_t = torch.from_numpy(design_mask.astype(np.float32)).to(device)
    bc_t = torch.from_numpy(bc_mask.astype(np.float32)).to(device)
    out_t = 1.0 - bracket_t
    n_bc = bc_t.sum().clamp_min(1.0); n_out = out_t.sum().clamp_min(1.0); n_des = design_t.sum().clamp_min(1.0)
    n_bracket = bracket_t.sum().clamp_min(1.0)   # envelope total voxel count (BC + design)
    if env_excess_w < 0 or not 0 <= env_excess_tol < 1:
        raise ValueError('envelope excess weight/tolerance must be nonnegative and tolerance < 1')
    if dense_token_policy not in ('legacy', 'raw'):
        raise ValueError('dense_token_policy must be legacy or raw')

    # Engineering extras
    if pw > 0 and load_path_mask is not None:
        lp_t = torch.from_numpy(load_path_mask.astype(np.float32)).to(device)
        n_lp = lp_t.sum().clamp_min(1.0)
    else:
        lp_t = None
    if sw > 0:
        print(f"  symmetry loss active: sw={sw}, axis={symmetry_axis}")
    if vw > 0:
        print(f"  volume target active: vw={vw}, target={vol_target}")
    if pw > 0 and lp_t is not None:
        print(f"  load-path active: pw={pw}, corridor voxels={int(lp_t.sum().item())}")

    # latents_scale and latents_shift for x0 decoding
    latents_scale = vae.latents_scale
    latents_shift = vae.latents_shift

    # Optional GuideFlow-style optimizer on z (Tweedie x_0 kept; manual normalized
    # step replaced by Adam/SGD with persistent state). dense_opt='flowdps' is
    # current behavior; 'adam'/'sgd' enables persistent optimizer momentum.
    z_param = None; z_opt = None
    if dense_opt in ('adam', 'sgd'):
        z_param = nn.Parameter(latents.detach().to(torch.float32))
        if dense_opt == 'adam':
            z_opt = torch.optim.Adam([z_param], lr=dense_lr, betas=(0.9, 0.99))
        else:
            z_opt = torch.optim.SGD([z_param], lr=dense_lr, momentum=0.9)
        print(f"  dense_opt={dense_opt}  lr={dense_lr}  (persistent optimizer on z)", flush=True)

    # Dense Aug Lag state (inequality V ≤ vol_target on design_t)
    _dense_aug_lag_lambda = [float(lambda_init)] if aug_lag else [0.0]
    if aug_lag:
        print(f"  [dense aug-lag] inequality V ≤ {vol_target} on design_t. "
              f"lambda_init={_dense_aug_lag_lambda[0]:.2f}  mu={mu_aug_lag:.1f}  alpha={lambda_alpha:.2f}", flush=True)
    if heaviside_proj:
        print(f"  [dense heaviside] beta_init={beta_init:.1f}  beta_max={beta_max:.1f}  "
              f"bisect_iter={eta_bisect_iter}", flush=True)

    for i, t in enumerate(timesteps):
        t_now = float(t.item()) / 1000.0  # FlowMatch t is in [0, 1000] typically
        t_inf = torch.tensor([t.item()], dtype=latents.dtype, device=device)
        _last_vol_loss = None   # set in inner loop when vw>0, written to CSV at outer step
        _env_ratio_last = None

        # Sync latents ←→ z_param at outer-step entry when in GuideFlow mode.
        if z_param is not None:
            latents = z_param.detach().to(pipe.dtype)

        # Standard CFG flow prediction (no grad)
        with torch.no_grad():
            noise_pred_c = dit(x=latents, t=t_inf, cond=cond)
            noise_pred_u = dit(x=latents, t=t_inf, cond=uncond)
            noise_pred = noise_pred_u + guidance_scale * (noise_pred_c - noise_pred_u)

        # FlowDPS gradient via dense decoder occupancy.  OC-Flow evaluates one
        # gradient only: repeated inner iterations would see identical latents
        # because it intentionally delays the update until scheduler.step().
        _vanilla = os.environ.get('VANILLA', '0') == '1'
        _guidance_inner_steps = 1 if oc_flow_active else inner_steps
        _oc_grad = None
        for inner in range(0 if _vanilla else _guidance_inner_steps):
            # In GuideFlow mode, z_param itself is the leaf; in FlowDPS mode,
            # z_in is a fresh leaf each inner step.
            if z_param is not None:
                z_opt.zero_grad()
                z_in = z_param   # leaf w/ requires_grad=True via nn.Parameter
            else:
                z_in_holder = latents.detach().clone().to(torch.float32).requires_grad_(True)
                z_in = z_in_holder
            with torch.enable_grad():
                # Flow Tweedie x0 estimate (approx for FlowMatch)
                x0 = z_in - t_now * noise_pred.detach().to(torch.float32)
                # Unscale to VAE latent space
                x0_unscaled = x0 / latents_scale + latents_shift
                # Decode
                with torch.amp.autocast('cuda', enabled=False):
                    occ_logits = vae.decoder(x0_unscaled.to(pipe.dtype)).to(torch.float32)
                # occ_logits shape: (B, 1, R, R, R)
                occ_logits = occ_logits[0, 0]  # (R, R, R)
                # If decoder res != R, resize
                if occ_logits.shape != bracket_t.shape:
                    occ_logits = F.interpolate(occ_logits[None,None], size=bracket_t.shape,
                                               mode='trilinear', align_corners=False)[0,0]
                # Multi-region BCE (core).
                # paper ref: main text, Geometric Guidance, two-region BCE with recentered
                #            decision boundary tau=0.1.
                # bce_boundary τ: shift logits by logit(τ) so the loss's implicit decision
                # boundary sits at sigmoid=τ instead of 0.5 — set τ = active-token threshold
                # (e.g. 0.1) to ALIGN the BCE with the dense→sparse extraction cut: outside
                # voxels are then pushed decisively below the cut (no 0.1..0.5 gray zone).
                # RECENTERED BCE: linear remap of the probability
                # axis so the balance point sits at τ with the SPANS preserved (1-τ each
                # side): virtual empty target = 2τ-1 (τ=0.1 → -0.8), solid target = 1.
                #   p' = (p - (2τ-1)) / (2(1-τ));  τ=0.5 → p'=p (standard BCE).
                # Empty keeps pushing without saturating at p≈τ (target below 0); solid
                # pushes toward 1 at full strength, symmetric about the τ center.
                if abs(bce_boundary - 0.5) > 1e-9 and bce_mode == 'inout':
                    # FINAL design: boundary τ=0.1 with per-region intent —
                    #   outer/empty : recentered remap p'=(p-(2τ-1))/(2(1-τ)) → virtual
                    #                 target 2τ-1 (=-0.8), balance at τ, keeps pushing
                    #                 below τ without early saturation.
                    #   inner/solid : STANDARD BCE toward 1 — full-strength push, no
                    #                 weakening (the symmetric variants hurt attachment).
                    _pp = torch.sigmoid(occ_logits)
                    _pr = ((_pp - (2.0 * bce_boundary - 1.0)) / (2.0 * (1.0 - bce_boundary))).clamp(1e-6, 1.0 - 1e-6)
                    bce_empty = F.binary_cross_entropy(_pr, torch.zeros_like(_pr), reduction='none')
                    bce_solid = F.binary_cross_entropy_with_logits(occ_logits, torch.ones_like(occ_logits), reduction='none')
                elif abs(bce_boundary - 0.5) > 1e-9 and bce_mode == 'shift':
                    # LEGACY symmetric logit-shift: o' = o - logit(τ) on BOTH
                    # terms. Boundary moves to τ but the solid push weakens near τ.
                    _shift = float(np.log(bce_boundary / (1.0 - bce_boundary)))
                    _o_sh = occ_logits - _shift
                    bce_solid = F.binary_cross_entropy_with_logits(_o_sh, torch.ones_like(_o_sh), reduction='none')
                    bce_empty = F.binary_cross_entropy_with_logits(_o_sh, torch.zeros_like(_o_sh), reduction='none')
                elif abs(bce_boundary - 0.5) > 1e-9:
                    _pp = torch.sigmoid(occ_logits)
                    _pr = ((_pp - (2.0 * bce_boundary - 1.0)) / (2.0 * (1.0 - bce_boundary))).clamp(1e-6, 1.0 - 1e-6)
                    bce_solid = F.binary_cross_entropy(_pr, torch.ones_like(_pr), reduction='none')
                    bce_empty = F.binary_cross_entropy(_pr, torch.zeros_like(_pr), reduction='none')
                else:
                    bce_solid = F.binary_cross_entropy_with_logits(occ_logits, torch.ones_like(occ_logits), reduction='none')
                    bce_empty = F.binary_cross_entropy_with_logits(occ_logits, torch.zeros_like(occ_logits), reduction='none')
                loss = (out_w * (bce_empty * out_t).sum() / n_out
                        + bc_w * (bce_solid * bc_t).sum() / n_bc
                        + dw * (bce_solid * design_t).sum() / n_des)
                if env_excess_w > 0:
                    # One-sided envelope constraint: do not penalize sub-threshold
                    # probability or a sample that is already within tolerance.
                    _excess_occ = (torch.sigmoid(occ_logits) - mc_threshold).clamp_min(0.0)
                    _env_ratio = (_excess_occ * out_t).sum() / _excess_occ.sum().clamp_min(1.0)
                    loss = loss + env_excess_w * (_env_ratio - env_excess_tol).clamp_min(0.0).square()
                    _env_ratio_last = float(_env_ratio.detach())

                # Engineering extras
                if vol_projection:
                    # Hard vol projection: dynamic per-step binary target at exact target_frac
                    with torch.no_grad():
                        occ_prob_d = torch.sigmoid(occ_logits)
                        design_voxels = occ_prob_d[design_t > 0.5]
                        n_des_v = design_voxels.numel()
                        if n_des_v > 0:
                            k = max(1, int(vol_target * n_des_v))
                            threshold = torch.kthvalue(design_voxels.flatten(), n_des_v - k + 1).values
                            binary_target = (occ_prob_d > threshold).float()
                        else:
                            binary_target = torch.zeros_like(occ_logits)
                    bce_proj = F.binary_cross_entropy_with_logits(occ_logits, binary_target, reduction='none')
                    loss = loss + vol_proj_w * (bce_proj * design_t).sum() / n_des
                elif aug_lag:
                    # TO-style Aug Lag inequality V ≤ vol_target on bracket_t (= envelope mask),
                    # with optional Heaviside projection + beta-continuation.
                    # NOTE: mask change design_t → bracket_t (envelope = BC + entire design)
                    # vol_target meaning: fraction of the envelope (BC ~10% + design mass combined).
                    occ_prob = torch.sigmoid(occ_logits)
                    if heaviside_proj:
                        sf_h = i / max(1, len(timesteps) - 1)
                        beta_cur = beta_init * (beta_max / max(beta_init, 1e-3)) ** sf_h
                        with torch.no_grad():
                            rho_env = occ_prob[bracket_t > 0.5].detach()
                            eta_cur = bisect_eta_for_volume(rho_env, vol_target, beta_cur,
                                                              max_iter=eta_bisect_iter)
                        occ_proj = heaviside_projection(occ_prob, eta_cur, beta_cur)
                        cur_V = (occ_proj * bracket_t).sum() / n_bracket
                    else:
                        cur_V = (occ_prob * bracket_t).sum() / n_bracket
                    # Inequality g = V - V*. Penalty only when g > 0 (V > V*).
                    g_vol = cur_V - vol_target
                    violation = F.relu(g_vol)
                    l_vol = _dense_aug_lag_lambda[0] * violation + 0.5 * mu_aug_lag * violation ** 2
                    loss = loss + l_vol
                    _last_vol_loss = float(l_vol.item())
                    _g_vol_last = float(g_vol.detach().item())   # saved for outer lambda update
                elif vw > 0:
                    occ_prob = torch.sigmoid(occ_logits)
                    cur_fill = (occ_prob * design_t).sum() / n_des
                    _vol_loss = (cur_fill - vol_target) ** 2
                    loss = loss + vw * _vol_loss
                    _last_vol_loss = float(_vol_loss.item())
                if sw > 0:
                    occ_mirror = torch.flip(occ_logits, dims=[symmetry_axis])
                    sym_diff = ((occ_logits - occ_mirror) ** 2) * bracket_t
                    loss = loss + sw * sym_diff.sum() / bracket_t.sum().clamp_min(1.0)
                if pw > 0 and lp_t is not None:
                    loss = loss + pw * (bce_solid * lp_t).sum() / n_lp
                # Warm-up: connectivity/thinness only after first half (x̂_0 reliable)
                step_frac = i / max(1, len(timesteps) - 1)  # 0 → 1 over denoising
                if image_projection is not None and step_frac >= image_proj_warmup:
                    _image_projection_loss = image_projection.loss(occ_logits, bracket_t)
                    loss = loss + image_proj_w * _image_projection_loss
                    if inner == 0 and i % max(1, len(timesteps)//5) == 0:
                        print(f"    image projection @ step {i}: loss={_image_projection_loss.item():.4f}",
                              flush=True)
                if shape_qd is not None and step_frac >= shape_qd_warmup:
                    _shape_loss, _shape_z = shape_qd.loss(occ_logits, bc_t)
                    loss = loss + shape_qd_w * _shape_loss
                    if i % max(1, len(timesteps)//5) == 0:
                        print(f"    Shape-QD @ step {i}: loss={_shape_loss.item():.4f} target={shape_qd.target_niche}")
                if shape_anchor is not None and step_frac >= shape_qd_warmup:
                    _anchor_loss = shape_anchor.loss(occ_logits, bc_t)
                    loss = loss + shape_anchor_w * _anchor_loss
                    if i % max(1, len(timesteps)//5) == 0:
                        print(f"    Shape anchor @ step {i}: loss={_anchor_loss.item():.4f}")
                if shape_scaffold is not None and step_frac >= shape_qd_warmup:
                    _scaffold_loss = shape_scaffold.loss(occ_logits, bc_t)
                    loss = loss + shape_scaffold_w * _scaffold_loss
                    if i % max(1, len(timesteps)//5) == 0:
                        print(f"    Shape scaffold @ step {i}: loss={_scaffold_loss.item():.4f}")
                if shape_residual is not None and step_frac >= shape_qd_warmup:
                    if shape_residual.reference is None:
                        _cells = shape_residual.prepare(occ_logits, bc_t)
                        print(f"    Shape residual reference @ step {i}: "
                              f"add={_cells['add_cells']} remove={_cells['remove_cells']} "
                              f"neutral={_cells['neutral_cells']}")
                    _residual_loss = shape_residual.loss(occ_logits, bc_t)
                    loss = loss + shape_residual_w * _residual_loss
                    if i % max(1, len(timesteps)//5) == 0:
                        print(f"    Shape residual @ step {i}: loss={_residual_loss.item():.4f}")
                if (shape_transport is not None and step_frac >= shape_qd_warmup
                        and shape_transport.updates < shape_transport_max_updates
                        and (i - int(np.ceil(shape_qd_warmup * max(1, len(timesteps) - 1)))) % shape_transport_every == 0):
                    _transport_z, _transport_info = shape_transport.step(
                        z_in, shape_transport.embedding(torch.sigmoid(occ_logits), bc_t))
                    with torch.no_grad():
                        if z_param is None:
                            latents = latents + _transport_z.to(latents.dtype)
                        else:
                            z_param.add_(_transport_z.to(z_param.dtype))
                            latents = z_param.detach().to(pipe.dtype)
                    shape_transport.updates += 1
                    print(f"    Shape transport {shape_transport.updates}/{shape_transport_max_updates} @ step {i}: "
                          f"requested={_transport_info['requested_norm']:.4f} "
                          f"predicted={_transport_info['predicted_norm']:.4f} "
                          f"latent_step={_transport_info['latent_norm']:.4f} "
                          f"clipped={_transport_info['clipped']}")
                _last_thin = _last_thick_hard = _last_thick_soft = _last_rmin = _last_reach = None
                _last_bce = float(loss.item())   # core BCE total (out_w+bc_w+dw, before extras)
                if cw > 0 and step_frac > cw_warmup:
                    _l = reachability_loss(occ_logits, bc_t, envelope_mask=bracket_t,
                                           reach_threshold=reach_threshold, reach_kernel=reach_kernel,
                                           n_iters=reach_iters)
                    loss = loss + cw * _l
                    _last_reach = float(_l.item())
                if tw > 0 and step_frac > cw_warmup:
                    _l = thinness_loss(occ_logits, min_thick=min_thick)
                    loss = loss + tw * _l
                    _last_thin = float(_l.item())
                # opening-based thickness — kill 1-vox features + soft 3+ preference
                if tw_hard > 0 and step_frac > cw_warmup:
                    _l = thickness_kill_loss(occ_logits, kill_size=1)
                    loss = loss + tw_hard * _l
                    _last_thick_hard = float(_l.item())
                if tw_soft > 0 and step_frac > cw_warmup:
                    _l = thickness_kill_loss(occ_logits, kill_size=2)
                    loss = loss + tw_soft * _l
                    _last_thick_soft = float(_l.item())
                if tw_rmin > 0 and step_frac > cw_warmup:
                    _l = dense_rmin_filter_loss(occ_logits, r_min=rmin_radius, threshold=rmin_thresh,
                                                normalize=rmin_normalize)
                    loss = loss + tw_rmin * _l
                    _last_rmin = float(_l.item())
                # FEA compliance loss (subprocess to fenics env)
                fea_comp_this_step = None
                if fea_w > 0 and step_frac > fea_warmup and (i % fea_every_n == 0):
                    try:
                        from fea_compliance_loss import fea_compliance_loss
                        # 2-stage schedule: coarse during exploration, fine for refinement
                        if fea_mesh_size_fine is not None and step_frac > fea_fine_warmup:
                            m_size = fea_mesh_size_fine
                            m_cache = fea_mesh_cache_fine
                            stage = 'FINE'
                        else:
                            m_size = fea_mesh_size
                            m_cache = fea_mesh_cache
                            stage = 'coarse'
                        comp, seat_comp, back_comp = paired_fea_compliance(
                            occ_logits, bracket_mask, fea_nodes,
                            fea_domain_dir, fea_back_domain_dir, m_cache,
                            mesh_size=m_size, penal=fea_penal,
                            reference=dual_fea_reference,
                            back_mode=fea_back_load_mode,
                            back_magnitude=fea_back_load_magnitude,
                            verbose=(i % (fea_every_n*2) == 0))
                        # normalize: magnitude=1, sign+direction only (raw comp 1e10+ causes NaN cascade)
                        # FEA_NORMALIZE env var: '1' (default) = normalize on, '0' = raw
                        import os as _os
                        if _os.environ.get('FEA_NORMALIZE', '1') == '0':
                            loss = loss + fea_w * comp
                        else:
                            comp_normed = comp / (comp.detach().abs() + 1e-12)
                            loss = loss + fea_w * comp_normed
                        fea_comp_this_step = float(comp.item())
                        if back_comp is not None:
                            print(f"    [dense dual FEA step {i}] "
                                  f"seat={seat_comp.item():.4e} back={back_comp.item():.4e} "
                                  f"balanced={comp.item():.4e}", flush=True)
                        elif os.environ.get('FEA_SECOND_LOAD_STL'):
                            print(f"    [dense simultaneous FEA step {i}] C={comp.item():.4e}", flush=True)
                        if i % (fea_every_n*2) == 0:
                            print(f"    FEA compliance @ step {i} [{stage} mesh_size={m_size*1000:.1f}mm]: C={comp.item():.4e}  (fea_w={fea_w})")
                    except Exception as e:
                        print(f"    [WARN] FEA call failed at step {i}: {e}")

                if z_param is None:
                    # FlowDPS: manual normalized step (stateless)
                    grad = torch.autograd.grad(loss, z_in)[0]
            if z_param is None:
                with torch.no_grad():
                    if grad_normalize:
                        grad_norm = grad.flatten().norm() + 1e-8
                        if oc_flow_active:
                            # Kept for velocity construction below; no direct z edit.
                            _oc_grad = grad.detach()
                        else:
                            latents = latents - (eta / num_inference_steps / inner_steps) * grad / grad_norm
                    else:
                        # Magnitude-aware step (no normalize): step ∝ ‖grad‖. May explode if loss big.
                        if oc_flow_active:
                            _oc_grad = grad.detach()
                        else:
                            latents = latents - (eta / num_inference_steps / inner_steps) * grad
            else:
                # GuideFlow-style: Adam/SGD update on z_param (persistent state)
                loss.backward()
                z_opt.step()
            if inner == inner_steps - 1:
                pass  # logged below

        # Per-step loss CSV row (dense stage)
        if loss_csv is not None and not _vanilla:
            def _f(x): return '' if x is None else f'{x:.6e}'
            _fc = _f(fea_comp_this_step)
            _vol_str = _f(_last_vol_loss)
            _rmin = _f(_last_rmin)
            # Combine thinness / thick_hard / thick_soft for the single `l_thick` CSV column
            _thick_sum = None
            for v in (_last_thin, _last_thick_hard, _last_thick_soft):
                if v is not None:
                    _thick_sum = v if _thick_sum is None else _thick_sum + v
            _thick = _f(_thick_sum)
            _bce = _f(_last_bce)
            _reach = _f(_last_reach)
            # cols: stage,step,t,total,fea_comp,l_rmin,l_thick,l_hole,l_normal,l_overhang,
            #       l_normalfd,l_lap,l_coh,l_aniso,l_odw,l_dfodw,l_edt,l_snap,l_vol,l_bce,l_reach
            # cols: stage,step,t,total,fea_comp,l_rmin,l_thick,l_hole,l_interior,l_normalfd,l_lap,l_aniso,l_vol,l_bce,l_reach
            loss_csv.write(f'dense,{i},{t_now:.4f},{float(loss.item()):.6e},{_fc},'
                           f'{_rmin},{_thick},,,,,,{_vol_str},{_bce},{_reach}\n')
            loss_csv.flush()

        # Snapshot the Tweedie x0_hat occupancy at this step (for visualization).
        # Saves npz (occ_p) AND a marching-cubes mesh.obj so renderers don't need to MC.
        # Uses SAME convention as final mesh_dense.obj: envelope-masked + world frame.
        if snapshot_every > 0 and snapshot_dir is not None and \
                (i % snapshot_every == 0 or i == len(timesteps) - 1):
            try:
                from pathlib import Path as _P
                _sd = _P(snapshot_dir); _sd.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    occ_p_snap = torch.sigmoid(occ_logits).detach().cpu().numpy().astype('float32')
                np.savez(_sd / f'dense_step_{i:02d}.npz', occ_p=occ_p_snap, step=i,
                         t=t_now, loss=float(loss.item()))
                # MC → obj (NO envelope mask — raw sigmoid; world frame transform kept).
                from skimage import measure as _meas
                _vol = occ_p_snap[0, 0] if occ_p_snap.ndim == 5 else occ_p_snap
                if _vol.max() > 0.5 > _vol.min():
                    _v, _f, _, _ = _meas.marching_cubes(_vol, level=0.5, method='lewiner')
                    if env_origin is not None and env_pitch is not None:
                        _v = np.asarray(env_origin) + (_v + 0.5) * env_pitch
                    _mesh = trimesh.Trimesh(vertices=_v, faces=_f, process=False)
                    _mesh.export(str(_sd / f'dense_step_{i:02d}.obj'))
            except Exception as _e:
                print(f"    [WARN] snapshot at step {i} failed: {_e}")

        # Standard flow step (Euler), with an optional OC-Flow correction applied
        # to the velocity itself.  This makes the structural objective part of the
        # sampler dynamics, unlike post-hoc latent transport that the remaining
        # flow can immediately undo.
        flow_velocity = noise_pred
        _oc_step_frac = i / max(1, len(timesteps) - 1)
        if oc_flow_active and _oc_step_frac >= oc_flow_warmup and _oc_grad is not None:
            with torch.no_grad():
                _g = _oc_grad.to(noise_pred.dtype)
                _g_norm = _g.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                _g_unit = _g / _g_norm.reshape((-1,) + (1,) * (_g.ndim - 1))
                _corr = float(oc_flow_w) * _g_unit
                _v_norm = noise_pred.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                _corr_norm = _corr.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8)
                _cap = float(oc_flow_max_rel) * _v_norm
                _scale = torch.minimum(torch.ones_like(_corr_norm), _cap / _corr_norm)
                _corr = _corr * _scale.reshape((-1,) + (1,) * (_corr.ndim - 1))
                flow_velocity = noise_pred + _corr
                if i % max(1, len(timesteps)//5) == 0:
                    print(f"    OC-Flow @ step {i}: |corr|/|v|="
                          f"{float((_corr.flatten(1).norm(dim=1) / _v_norm.squeeze(1)).mean()):.3f}")
        if z_param is None:
            latents = scheduler.step(flow_velocity, t, latents).prev_sample
        else:
            with torch.no_grad():
                new_latents = scheduler.step(flow_velocity, t, z_param.detach().to(pipe.dtype)).prev_sample
            z_param.data = new_latents.to(torch.float32)
            latents = z_param.detach().to(pipe.dtype)

        # Aggressive memory cleanup (FlowDPS gradient graphs leave residuals)
        if _vanilla:
            del noise_pred_c, noise_pred_u, noise_pred
        else:
            del noise_pred_c, noise_pred_u, noise_pred, x0, x0_unscaled, occ_logits
        if z_param is None and not _vanilla:
            del grad, z_in   # grad/z_in only exist in flowdps guidance mode
        import gc; gc.collect(); torch.cuda.empty_cache()

        # Dense Aug Lag outer update: lambda ← max(0, lambda + alpha * mu * g)  (one-sided)
        if aug_lag:
            try:
                _g_val = float(_g_vol_last)
            except NameError:
                _g_val = 0.0
            _dense_aug_lag_lambda[0] = max(0.0,
                _dense_aug_lag_lambda[0] + lambda_alpha * mu_aug_lag * _g_val)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [dense aug-lag] step {i+1}: V={_g_val + vol_target:.4f}  V*-V={-_g_val:+.4f}  lambda={_dense_aug_lag_lambda[0]:.2f}", flush=True)

        if (i+1) % 10 == 0 or i == 0:
            if _vanilla:
                print(f"  step {i+1}/{num_inference_steps}: t={t_now:.3f} (vanilla)  cuda {torch.cuda.memory_allocated()/1e9:.1f}GB")
            else:
                _vol_print = '' if _last_vol_loss is None else f' vol={_last_vol_loss:.5f}'
                _env_print = '' if _env_ratio_last is None else f' env_excess={_env_ratio_last:.5f}'
                print(f"  step {i+1}/{num_inference_steps}: t={t_now:.3f} loss={loss.item():.5f}{_vol_print}{_env_print}  cuda {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Unscale latents for decoding
    latents = 1. / latents_scale * latents + latents_shift
    # Decode to dense occupancy index
    decoded = vae.decode_mesh(latents, mc_threshold=mc_threshold, return_index=True)
    index = decoded[0]
    print(f"dense FlowDPS index (pre-filter): {index.shape}")

    # Filter by bracket envelope (keep only coords inside bracket)
    # index has shape (N, 4) — (B, Z, Y, X)
    idx_np = index.cpu().numpy()
    # stash the PRE-FILTER token set so main can save mesh_dense_raw.obj
    dense_flowdps._prefilter_idx = idx_np.copy()
    if dense_token_policy == 'raw':
        print(f"dense index (raw policy — no envelope token filter or BC injection): {index.shape}")
        return index
    if os.environ.get('VANILLA', '0') == '1':
        # Pure Direct3D-S2 baseline: image conditioning only — no BC token
        # injection, no bracket filter. Set env VANILLA=1.
        print(f"dense index (VANILLA — backbone tokens as-is): {index.shape}")
        return index
    if os.environ.get('NO_BRACKET_FILTER', '0') == '1':
        # U-Net fully unconstrained: keep ALL dense-active tokens (+BC), envelope is
        # enforced ONLY at the final MC clip (force-envelope-clip). Set env
        # NO_BRACKET_FILTER=1 to enable.
        bc_coords = np.argwhere(bc_mask)
        bc_index = np.concatenate([np.zeros((len(bc_coords), 1), dtype=int), bc_coords], axis=1)
        combined = np.unique(np.concatenate([idx_np, bc_index], axis=0), axis=0)
        # CANVAS MARGIN (TOKEN_MARGIN=N): dilate the active set by N tokens in all
        # directions — pure canvas (no value forcing), so the refiner SDF can roll off
        # smoothly instead of truncating at the token frontier (64-block staircase on
        # the raw mesh). The final clip/hybrid boolean still bounds the result.
        _tm = int(os.environ.get('TOKEN_MARGIN', '0'))
        if _tm > 0:
            from scipy.ndimage import binary_dilation as _bd_tm
            occ = np.zeros((64, 64, 64), dtype=bool)
            occ[combined[:, 1], combined[:, 2], combined[:, 3]] = True
            occ = _bd_tm(occ, iterations=_tm)
            cc = np.argwhere(occ)
            combined = np.concatenate([np.zeros((len(cc), 1), dtype=int), cc], axis=1)
            print(f"dense FlowDPS index: token canvas margin +{_tm} → {len(combined)}")
        index = torch.from_numpy(combined).to(index.device).to(index.dtype)
        print(f"dense FlowDPS index (NO bracket filter, +BC): {index.shape}")
        return index
    in_bracket = bracket_mask[idx_np[:, 1], idx_np[:, 2], idx_np[:, 3]]
    # also force BC region included
    bc_coords = np.argwhere(bc_mask)
    bc_index = np.concatenate([np.zeros((len(bc_coords), 1), dtype=int), bc_coords], axis=1)
    # union
    filtered = idx_np[in_bracket]
    combined = np.unique(np.concatenate([filtered, bc_index], axis=0), axis=0)
    index = torch.from_numpy(combined).to(index.device).to(index.dtype)
    print(f"dense FlowDPS index (after bracket filter + BC): {index.shape}")
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="run config JSON; CLI overrides")
    ap.add_argument("--image", default=None, help="single cond image")
    ap.add_argument("--target-dir", default=None, help="dir of multi-view cond PNGs")
    ap.add_argument("--n-views", type=int, default=8, help="number of views to use from target-dir")
    ap.add_argument("--views", default=None,
                    help="comma-separated view stems loaded in this exact order from target-dir; "
                         "when omitted, use lexicographic PNG order")
    ap.add_argument("--bracket-occ", default="data_real/bracket/voxel.npz")
    ap.add_argument("--bc-proper", default="data_real/bracket/voxel.npz",
                    help="proper BC mask (fix + load STLs) — if exists, override bc_layers erosion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sdf-resolution", type=int, default=512)
    ap.add_argument("--mc-threshold", type=float, default=0.2,
                    help="sparse marching-cubes threshold; lower = inflated surface (less tear)")
    ap.add_argument("--sp-sdf-guidance-threshold", type=float, default=None,
                    help="iso value used by sparse SDF/occupancy/FEA guidance; default keeps legacy mc_threshold. "
                         "Set to 2*mc_threshold to compare against the pre-refiner extraction iso.")
    ap.add_argument("--sp-sdf-inside-low", action="store_true",
                    help="treat sparse SDF values below the guidance iso as solid, matching "
                         "sparse2mesh/refiner marching cubes; default preserves legacy guidance")
    ap.add_argument("--sp-dense-core-mm", type=float, default=0.0,
                    help="protect material more than this distance inside the dense 3D mesh "
                         "throughout sparse guidance and at sparse decoding; 0 disables")
    ap.add_argument("--sp-dense-core-w", type=float, default=0.0,
                    help="sparse guidance loss weight for violating the dense 3D material core")
    ap.add_argument("--sp-dense-core-tail-frac", type=float, default=0.0,
                    help="if >0, optimize the worst fraction of dense-core violations "
                         "(CVaR) instead of their mean; 0 keeps mean loss")
    ap.add_argument("--sp-dense-core-project", action="store_true",
                    help="diagnostic hard projection of the dense core on the final sparse SDF; "
                         "can introduce discontinuities, so keep disabled for loss-only guidance")
    ap.add_argument("--sp-wall-ray-w", type=float, default=0.0,
                    help="guide sparse SDF to retain the first visible material on both X side views")
    ap.add_argument("--sp-wall-ray-band", type=int, default=4,
                    help="512-grid samples supervised inward from each dense-support side face")
    ap.add_argument("--sp-wall-ray-min-mass", type=float, default=2.0,
                    help="minimum soft solid count in the side-view front band")
    ap.add_argument("--sp-wall-ray-tail-frac", type=float, default=0.0,
                    help="if positive, blend mean front-ray loss with worst fraction of rays")
    ap.add_argument("--sp-side-depth-w", type=float, default=0.0,
                    help="sparse guidance loss for excessive side-view first-hit recession vs dense mesh")
    ap.add_argument("--sp-side-depth-tolerance-mm", type=float, default=4.0,
                    help="allowed side-surface recession from dense reference before penalty")
    ap.add_argument("--sp-side-depth-tail-frac", type=float, default=0.0,
                    help="if positive, blend mean side-depth loss with worst fraction of rays")
    ap.add_argument("--sp-dense-trust-mm", type=float, default=0.0,
                    help="continuous 3D field constraint before sparse/refiner marching cubes: "
                         "the generated field must remain solid farther than this distance "
                         "inside the dense mesh; 0 disables")
    ap.add_argument("--sp-dense-trust-scale", type=float, default=40.0,
                    help="SDF-field slope per meter for the dense 3D trust constraint")
    ap.add_argument("--bce-boundary", type=float, default=None,
                    help="implicit decision boundary of the dense 3-region BCE EMPTY term (logit "
                         "shift by logit(τ); solid terms stay standard). Default None = AUTO: "
                         "follows --dense-index-threshold so the loss boundary and the active-token "
                         "extraction cut are ALWAYS equal (invariant). Pass 0.5 for legacy BCE.")
    ap.add_argument("--bce-mode", choices=["inout", "recentered", "shift"], default="inout",
                    help="non-0.5 boundary implementation: recentered = probability remap "
                         "(-0.8<-0.1->1 spec); shift = legacy symmetric logit-shift.")
    ap.add_argument("--dense-index-threshold", type=float, default=0.1,
                    help="dense→sparse ACTIVE-token threshold: sigmoid(occ_logit) >= T becomes an "
                         "active voxel for the sparse U-Net. NB: the dense BCE's implicit decision "
                         "boundary is 0.5 — the 0.1..0.5 'gray zone' tokens are BCE-suppressed but "
                         "still active, providing boundary growth headroom (thickness recovery). "
                         "Raise toward 0.5 for a tighter token set, lower for more headroom.")
    ap.add_argument("--thin-expand-vox", type=int, default=0,
                    help="thin-region selective index expansion before sparse stage (N voxel dilation)")
    ap.add_argument("--sp-support-halo-vox", type=int, default=0,
                    help="add this many 64-grid exterior token layers for sparse decoding only; "
                         "keep the original dense support as the topology target (0 disables)")
    ap.add_argument("--thin-thresh-vox", type=int, default=1,
                    help="EDT half-thickness threshold (vox) for identifying 'thin' solid")
    ap.add_argument("--fea-stress-expand", action='store_true',
                    help="Add high-stress region (top stress-percentile%% |∂C/∂ρ|) to sparse active voxel set (gradient-free)")
    ap.add_argument("--stress-percentile", type=float, default=80.0,
                    help="top-(100-p)%% |∂C/∂ρ| considered high-stress")
    ap.add_argument("--stress-expand-vox", type=int, default=1,
                    help="N-voxel dilation of high-stress region")
    ap.add_argument("--bc-layers", type=int, default=2)
    ap.add_argument("--dense-steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--eta", type=float, default=300.0)
    ap.add_argument("--bc-w", type=float, default=3.0)
    ap.add_argument("--out-w", type=float, default=10.0)
    ap.add_argument("--env-excess-w", type=float, default=0.0,
                    help="one-sided dense envelope excess loss; zero for already-feasible occupancy")
    ap.add_argument("--env-excess-tol", type=float, default=0.002,
                    help="allowed fraction of above-threshold occupancy outside the envelope")
    ap.add_argument("--dense-token-policy", choices=["legacy", "raw"], default="legacy",
                    help="raw keeps decoded dense tokens without envelope filtering or BC injection")
    ap.add_argument("--dw", type=float, default=0.0,
                    help="design region 'force solid' weight. 0=cond decides freely (hole/lattice possible), "
                         "0.3+=strongly push the design region to solid (porous hard to express)")
    ap.add_argument("--shape-qd-archive", default=None,
                    help="frozen_shape_space.npz made by run_shape_qd_representation_pilot.py")
    ap.add_argument("--shape-qd-target", type=int, default=-1,
                    help="CVT niche index in --shape-qd-archive; activates direct shape targeting with --shape-qd-w")
    ap.add_argument("--shape-qd-w", type=float, default=0.0,
                    help="dense-stage frozen shape-PCA target loss weight")
    ap.add_argument("--image-proj-target", default=None,
                    help="64x64 BC-registered top-image target NPZ with target and weight arrays")
    ap.add_argument("--image-proj-w", type=float, default=0.0,
                    help="training-free dense top-projection image loss weight; 0=off")
    ap.add_argument("--image-proj-warmup", type=float, default=0.35,
                    help="fraction of dense denoising before image projection guidance starts")
    ap.add_argument("--shape-anchor-bank", default=None,
                    help="valid_shape_prototypes.npz for direct multi-scale geometry anchoring")
    ap.add_argument("--shape-anchor-w", type=float, default=0.0,
                    help="dense-stage multi-scale geometry anchor weight")
    ap.add_argument("--shape-scaffold-w", type=float, default=0.0,
                    help="dense-stage coarse morphology scaffold BCE weight")
    ap.add_argument("--shape-scaffold-pool", type=int, default=4,
                    help="64³-to-macro pool factor for scaffold guidance")
    ap.add_argument("--shape-residual-w", type=float, default=0.0,
                    help="dense-stage contrastive macro morphology weight; acts only where target differs from the warm-up dense prior")
    ap.add_argument("--shape-residual-pool", type=int, default=4,
                    help="64³-to-macro pooling for contrastive residual morphology")
    ap.add_argument("--shape-residual-delta", type=float, default=0.15,
                    help="minimum target-minus-prior macro occupancy gap that defines a shape addition/removal")
    ap.add_argument("--shape-residual-neutral-w", type=float, default=0.05,
                    help="weak dense-prior retention weight outside contrastive target cells")
    ap.add_argument("--shape-transport-radius", type=float, default=0.0,
                    help="one-shot dense local-PCA phenotype displacement at shape warm-up; 0 disables")
    ap.add_argument("--shape-transport-dims", type=int, default=2,
                    help="number of frozen PCA descriptor coordinates used by local transport")
    ap.add_argument("--shape-transport-ridge", type=float, default=1e-3,
                    help="Tikhonov ridge for the local descriptor Jacobian pseudo-inverse")
    ap.add_argument("--shape-transport-max-rel", type=float, default=0.05,
                    help="maximum one-shot transport norm relative to current dense latent norm")
    ap.add_argument("--shape-transport-every", type=int, default=5,
                    help="dense denoising-step interval between local transport relinearizations")
    ap.add_argument("--shape-transport-max-updates", type=int, default=1,
                    help="maximum number of local transport updates in the dense stage")
    ap.add_argument("--oc-flow-w", type=float, default=0.0,
                    help="training-free OC-Flow dense velocity-control strength; 0=off. Uses the existing "
                         "differentiable structural loss before the Euler update instead of editing z directly")
    ap.add_argument("--oc-flow-warmup", type=float, default=0.35,
                    help="fraction of dense denoising before OC-Flow control activates (0..1)")
    ap.add_argument("--oc-flow-max-rel", type=float, default=0.5,
                    help="cap OC-Flow correction norm relative to model velocity norm per sample")
    ap.add_argument("--sp-shape-anchor-w", type=float, default=0.0,
                    help="sparse-stage multi-scale geometry anchor weight")
    ap.add_argument("--sp-image-proj-w", type=float, default=0.0,
                    help="sparse-stage camera-aligned image silhouette guidance on the decoded 64³ occupancy; "
                         "uses --image-proj-target, 0 disables")
    ap.add_argument("--sp-negative-space-mask", default=None,
                    help="optional NPZ containing a 64³ boolean mask of image-derived empty-space rays; "
                         "sparse decoder SDF is penalized directly inside this mask")
    ap.add_argument("--sp-negative-space-w", type=float, default=0.0,
                    help="direct sparse-stage empty-space BCE weight; 0 disables")
    ap.add_argument("--shape-anchor-expand-vox", type=int, default=0,
                    help="dilate target prototype support by this many 64³ voxels before sparse refinement")
    ap.add_argument("--shape-anchor-max-added", type=int, default=1500,
                    help="cap target-support voxels added to sparse interface to keep refiner memory bounded")
    ap.add_argument("--shape-qd-warmup", type=float, default=0.35,
                    help="fraction of dense denoising before shape-QD loss begins")
    ap.add_argument("--sp-shape-qd-w", type=float, default=0.0,
                    help="sparse-stage frozen shape-PCA target loss weight")
    ap.add_argument("--sp-shape-qd-warmup", type=float, default=0.0,
                    help="fraction of sparse denoising before shape-QD loss begins")
    ap.add_argument("--inner-steps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action='store_true',
                    help="enable torch deterministic mode (use_deterministic_algorithms, cudnn.deterministic). "
                         "Requires CUBLAS_WORKSPACE_CONFIG env var set externally. May slow some kernels.")
    ap.add_argument("--save-raw-sdf", default=None,
                    help="if set, intercept the final MC call and dump its input volume "
                         "as <path>.npz (sdf, grid_size, mc_threshold, env_origin, env_pitch). "
                         "Lets downstream tooling run SDF-level boolean union without re-MC.")
    ap.add_argument("--sp-sdf-smooth-sigma", type=float, default=0.0,
                    help="Gaussian smooth the 512-grid sparse SDF before final marching cubes; "
                         "sigma is in 512-grid voxels (0 disables)")
    ap.add_argument("--sp-sdf-smooth-volume-match", action="store_true",
                    help="after sparse SDF smoothing, shift its iso field to preserve the "
                         "pre-smoothing occupied-grid fraction")
    ap.add_argument("--posthoc-fea-steps", type=int, default=0,
                    help="PhysiOpt-style post-hoc baseline: AFTER sparse sampling finishes, run N "
                         "ADAM steps on the final sparse latent with FEM compliance + volume "
                         "regularization only (flow frozen). Implements post-hoc latent search "
                         "on our backbone/FEM for a timing-controlled comparison.")
    ap.add_argument("--posthoc-fea-lr", type=float, default=2e-3)
    ap.add_argument("--feaoff-out", default=None,
                    help="when set with posthoc: ALSO save the PRE-optimization (feaoff) mesh here "
                         "(one posthoc run yields both feaoff and posthoc, sharing generation).")
    ap.add_argument("--posthoc-vol-lambda", type=float, default=10.0,
                    help="lambda * |V - V0| volume regularization (V = mean rho in envelope; "
                         "V0 captured at step 0 — PhysiOpt Eq.(1) R term)")
    ap.add_argument("--sp-fea-mode", choices=['manual', 'adamw'], default='manual',
                    help="sparse FEA integration: 'manual' = separate scheduler hook with SGD lr=0.01 "
                         "(default, legacy). 'adamw' = integrate fea_loss into sparse guide loop, "
                         "so sp_opt (AdamW) handles fea gradient with momentum + cumulative effect.")
    # Engineering loss extensions
    ap.add_argument("--vol-projection", action='store_true',
                    help="hard volume projection: BCE push to exact target_frac (overrides vw)")
    ap.add_argument("--vol-proj-w", type=float, default=5.0,
                    help="weight for vol projection BCE loss")
    ap.add_argument("--vw", type=float, default=0.0,
                    help="volume target loss weight (sigmoid(occ).mean() → vol_target)")
    # --- TO-style Augmented Lagrangian on DENSE vol constraint (inequality V ≤ V*) ---
    ap.add_argument("--aug-lag", action='store_true',
                    help="DENSE stage Aug Lag inequality constraint V ≤ vol_target. "
                         "Measures occ_prob mean over design_t. lambda adaptive, only penalizes V > V*.")
    ap.add_argument("--lambda-init", type=float, default=0.0,
                    help="initial lambda for Aug Lag (default 0 = no initial penalty)")
    ap.add_argument("--mu-aug-lag", type=float, default=50.0,
                    help="quadratic penalty coef mu for Aug Lag")
    ap.add_argument("--lambda-alpha", type=float, default=1.0,
                    help="outer lambda update rate")
    ap.add_argument("--heaviside-proj", action='store_true',
                    help="apply Heaviside projection with eta bisection (TO standard)")
    ap.add_argument("--beta-init", type=float, default=2.0)
    ap.add_argument("--beta-max", type=float, default=20.0)
    ap.add_argument("--eta-bisect-iter", type=int, default=15)
    ap.add_argument("--vol-target", type=float, default=0.4,
                    help="target fill fraction for design region (0=empty, 1=fully solid)")
    ap.add_argument("--sw", type=float, default=0.0, help="bilateral symmetry loss weight")
    ap.add_argument("--symmetry-axis", type=int, default=0,
                    help="symmetry mirror axis (0=X, 1=Y, 2=Z); bracket default X")
    ap.add_argument("--pw", type=float, default=0.0, help="load-path corridor loss weight")
    ap.add_argument("--load-path-mask", default=None,
                    help="path to 64³ load-path mask .npz (key 'mask') for fix↔load corridor")
    ap.add_argument("--cw", type=float, default=0.0,
                    help="connectivity (reachability from BC) loss weight — penalizes disconnected lumps")
    ap.add_argument("--tw", type=float, default=0.0,
                    help="thinness loss weight — penalizes features < min_thick voxels")
    ap.add_argument("--min-thick", type=int, default=3,
                    help="minimum feature thickness in voxels (used with --tw)")
    ap.add_argument("--tw-hard", type=float, default=0.0,
                    help="HARD thickness penalty — opening with kernel=3 kills 1-vox features (2+ survives)")
    ap.add_argument("--tw-soft", type=float, default=0.0,
                    help="SOFT thickness preference — opening with kernel=5 kills ≤2-vox (3+ survives)")
    ap.add_argument("--tw-rmin", type=float, default=0.0,
                    help="TopOpt r_min density-filter loss weight (avg_pool radius). Smooth gradient growth.")
    ap.add_argument("--rmin-radius", type=int, default=2,
                    help="r_min radius in voxels for dense_rmin_filter_loss")
    ap.add_argument("--skip-sparse", action="store_true",
                    help="dense-only run: exit after dense stage (mesh.obj = mesh_dense.obj copy)")
    ap.add_argument("--dense-keep-bc-components", action="store_true",
                    help="hard dense-stage cleanup: keep only 6-connected active-token components that "
                         "touch fixed/load BC. Removes detached dense debris before sparse refinement; "
                         "does not bridge disconnected BCs, so use with --cw for an actual path constraint.")
    ap.add_argument("--dense-keep-largest-mesh-component", action="store_true",
                    help="after dense marching cubes, export only the largest connected mesh component. "
                         "Use with --dense-keep-bc-components and --cw when MC creates tiny detached "
                         "surface debris inside an otherwise BC-anchored token component.")
    ap.add_argument("--no-grad-normalize", action="store_true",
                    help="disable FlowDPS gradient normalization (magnitude-aware step). May explode.")
    ap.add_argument("--save-dense-cache", type=str, default=None,
                    help="path (.npz) to save dense stage state (latent_index, dense_active_64, dense_normal_64) after dense stage finishes")
    ap.add_argument("--load-dense-cache", type=str, default=None,
                    help="path (.npz) to load dense state from. SKIPS dense stage if cache exists.")
    ap.add_argument("--rmin-normalize", action="store_true",
                    help="make the dense r_min term a MEAN instead of a SUM over the 64^3 grid. The "
                         "sum form reached 407%% of the dense total loss on the caliper, saturating "
                         "the term so r_min radius had no effect on delivered thickness.")
    ap.add_argument("--rmin-thresh", type=float, default=0.5,
                    help="local-density threshold below which solid voxel is penalized")
    ap.add_argument("--cw-warmup", type=float, default=0.5,
                    help="fraction of denoising before cw/tw activate (0=always, 0.5=half-way)")
    ap.add_argument("--reach-kernel", type=int, default=41,
                    help="reachability_loss max_pool3d kernel (odd int). default=41 (NEW HEAD); "
                         "OLD 1c27bc9 paper-claim setup used 21. Smaller kernel → more localized "
                         "gap detection → sparser mesh topology → stronger FEA gradient differential.")
    ap.add_argument("--reach-iters", type=int, default=100,
                    help="reachability flood-fill propagation steps; each step grows the reached "
                         "set by one voxel, so this is the maximum reachable path length in voxels. "
                         "The 64^3 grid diagonal is 111, so 100 is only just enough.")
    ap.add_argument("--reach-threshold", type=float, default=0.1,
                    help="reachability 'solid for reach' occupancy threshold. default 0.1 (lenient: "
                         "faint occ 0.1-0.5 ghost thread counts as connected → may vanish in final "
                         "solid mesh). Raise toward 0.5 to only count genuinely-solid connections, "
                         "forcing a robust BC↔body strut that survives 512³ refinement.")
    # FEA compliance (FEniCS subprocess)
    ap.add_argument("--fea-w", type=float, default=0.0,
                    help="compliance loss weight (FEniCS-based TO); 0 = off")
    ap.add_argument("--fea-every-n", type=int, default=5,
                    help="call FEA every N denoising steps")
    ap.add_argument("--fea-warmup", type=float, default=0.3,
                    help="fraction of denoising before FEA activates (0=always)")
    ap.add_argument("--fea-domain-dir", default=None,
                    help="dir with original_DesignSpace.stl, fixed.stl, load.stl")
    ap.add_argument("--fea-back-domain-dir", default=None,
                    help="optional second FEA domain for backrest load; enables balanced seat/back FEA in dense and sparse")
    ap.add_argument("--fea-back-load-mode", default='y',
                    help="direction for the optional backrest FEA load")
    ap.add_argument("--fea-back-load-magnitude", type=float, default=200.0,
                    help="magnitude in N for the optional backrest FEA load")
    ap.add_argument("--fea-combine-loads", action='store_true',
                    help="apply seat and backrest forces simultaneously in one FEA solve")
    ap.add_argument("--fea-bracket-stl", default=None,
                    help="STL for voxel center origin (defaults to fea_domain_dir/original_DesignSpace.stl)")
    ap.add_argument("--fea-node-alignment", default=None,
                    help="alignment.json whose inverse affine maps generation-grid nodes into "
                         "the physical --fea-domain-dir frame; required when the generation "
                         "grid uses a legacy dense-aligned frame")
    ap.add_argument("--load-mode", default=None,
                    help="in-loop FEM load direction, exported to the FEA subprocess as LOAD_MODE: "
                         "x|-x|y|-y|z|-z (axis-aligned), or diag ((1,-1,-1)/sqrt(3), the solver's "
                         "own fallback when unset). Set it from the config's "
                         "stages.mesh.load_mode and keep it equal to stages.fea.force_dir, so the "
                         "in-loop objective is the load case the a-posteriori FEA verifies.")
    ap.add_argument("--fix-stl", default="data_real/bracket/fixed_remesh.stl",
                    help="fixture BC peg STL (default: main bracket, for backward compat). Set per-domain for cross-domain.")
    ap.add_argument("--load-stl", default="data_real/bracket/load_remesh.stl",
                    help="load BC peg STL (default: main bracket, for backward compat). Set per-domain for cross-domain.")
    ap.add_argument("--fea-mesh-cache", default='/tmp/bracket_fea.msh',
                    help="cached gmsh tet mesh (built once, reused)")
    ap.add_argument("--fea-mesh-size", type=float, default=0.0,
                    help="coarse FEA tet mesh size (raw STL units). "
                         "0/negative = auto (envelope longest_ext / --fea-mesh-cells-per-edge).")
    ap.add_argument("--fea-mesh-cells-per-edge", type=float, default=30.0,
                    help="auto mesh_size target: cells along longest envelope edge (when --fea-mesh-size<=0). "
                         "Default 30 → ~3000 tets for bracket-scale envelope.")
    ap.add_argument("--fea-mesh-size-fine", type=float, default=None,
                    help="fine FEA tet mesh size (m) for refinement phase (None = single-stage)")
    ap.add_argument("--fea-mesh-cache-fine", default='/tmp/bracket_fea_fine.msh',
                    help="cached fine gmsh tet mesh")
    ap.add_argument("--fea-fine-warmup", type=float, default=0.8,
                    help="fraction of denoising before switching coarse → fine FEA (0.8 = step 40+/50)")
    ap.add_argument("--fea-penal", type=float, default=2.0,  # best recipe (from ablation)
                    help="SIMP penalization p (default 3.0)")
    # Sparse FEA (ported from sparse_flowdps_localattn.py)
    ap.add_argument("--sp-fea-w", type=float, default=0.0,
                    help="sparse-stage FEA compliance weight (0=disable)")
    ap.add_argument("--sp-fea-every", type=int, default=2)
    ap.add_argument("--sp-fea-warmup", type=float, default=0.0)
    ap.add_argument("--sp-fea-steepness", type=float, default=4.0)  # best recipe (from ablation)
    ap.add_argument("--sp-fea-step-size", type=float, default=0.01,
                    help="manual sparse-FEA latent update size; 0.01 reproduces the legacy hardcoded value")
    ap.add_argument("--sp-fea-normalize", action='store_true')
    ap.add_argument("--sp-fea-reduce", type=str, default='amax', choices=['amax', 'mean', 'soft_frac'])
    ap.add_argument("--sp-fea-volume-neutral", action='store_true',
                    help="project sparse FEA sensitivity to zero first-order material change; "
                         "removes uniform mass addition and emphasizes load-path redistribution")
    # Step-by-step snapshots
    # --- Dense-stage optimizer mode (GuideFlow-on-dense option) ---
    ap.add_argument("--dense-opt", choices=['flowdps','adam','sgd'], default='flowdps',
                    help="'flowdps' = current stateless normalized step; "
                         "'adam'/'sgd' = persistent optimizer on z (Tweedie x_0 kept).")
    ap.add_argument("--dense-lr", type=float, default=5e-3,
                    help="learning rate for --dense-opt adam/sgd")

    # --- Dense→Sparse interface TO prune (FEA sensitivity-based) ---
    ap.add_argument("--to-prune-percentile", type=float, default=0.0,
                    help="After dense FlowDPS: prune active voxels with bottom-N%% |∂C/∂ρ| sensitivity. "
                         "BC voxels preserved; disconnected components dropped. 0=off, 20=drop lowest 20%%.")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="dense: save occ + mesh every N step (0 = off). → {out}/snapshots/dense_step_NN.{npz,obj}")
    ap.add_argument("--sp-snapshot-every", type=int, default=0,
                    help="sparse: save mesh every N step (0 = off). → {out}/snapshots/sparse_step_NN.obj")
    # === Sparse GuideFlow3D guidance args ===
    ap.add_argument("--sp-guide-w", type=float, default=30.0,
                    help="sparse GuideFlow3D base weight")
    ap.add_argument("--sp-guide-w-peak", type=float, default=80.0,
                    help="weight at late phase (annealed via cosine ramp)")
    ap.add_argument("--sp-anneal-start", type=float, default=0.66,
                    help="when to start ramp (fraction of total denoise steps)")
    ap.add_argument("--sp-n-inner-late", type=int, default=3,
                    help="inner SGD steps at late phase (1 at early)")
    ap.add_argument("--sp-guide-lr", type=float, default=5e-3,
                    help="SGD lr for sparse latent guidance")
    ap.add_argument("--sp-r-min-voxels", type=int, default=3,
                    help="TopOpt r_min in 128³-grid voxels")
    ap.add_argument("--sp-rmin-void-weight", type=float, default=0.3,
                    help="weight of the void-fill term in topopt_rmin_loss. The void term fills "
                         "thin voids around solids; with a large r_min it over-grows bumpy outgrowths. "
                         "Set 0 for solid-only thickening (cleaner surface).")
    ap.add_argument("--sp-thick-target", type=float, default=0.06,
                    help="per-voxel thickness target in normalized SDF units")
    ap.add_argument("--sp-cfg", type=float, default=7.0,
                    help="sparse stage CFG (lower = guidance dominant)")
    ap.add_argument("--sp-opt", default='sgd', choices=['sgd', 'adamw', 'adam', 'rmsprop'],
                    help="sparse guidance optimizer (persistent state)")
    ap.add_argument("--sp-hole-w", type=float, default=2.0,
                    help="no-hole loss weight (interior of dense-active must stay solid)")
    ap.add_argument("--sp-cone-cos", type=float, default=0.7071,
                    help="cos(cone half-angle). 0.7071=45°, 0.866=30°, 0.5=60°")
    ap.add_argument("--sp-no-diagonal", action='store_true',
                    help="disable per-cell diagonal cone (axis cones only)")
    ap.add_argument("--sp-normal-w", type=float, default=0.0,
                    help="Lipschitz-style normal-bound loss weight (per-cell ∇s vs dense MC normal). "
                         "0=off. Companion to sp-hole-w; targets thin-spoke fragmentation.")
    ap.add_argument("--sp-normal-cos", type=float, default=0.5,
                    help="cos(allowed angle). 0.5=60° (loose), 0.7071=45°, 0.866=30° (strict)")
    ap.add_argument("--sp-normal-p", type=float, default=3.0,
                    help="punitive hinge exponent. 2=quadratic, 3=cubic, 4=quartic. "
                         "Higher = more punitive on violations outside the cone.")
    ap.add_argument("--sp-overhang-w", type=float, default=0.0,
                    help="no-overhang / locally-convex-from-outside loss weight. "
                         "Enforces ∂²s/∂t² ≥ 0 in dense-normal tangent plane per shell cell.")
    ap.add_argument("--sp-overhang-p", type=float, default=3.0,
                    help="punitive hinge exponent for overhang violation")
    ap.add_argument("--sp-normal-fd-w", type=float, default=0.0,
                    help="**per-voxel local-FD** ∇s vs dense MC normal angle loss weight. "
                         "Computes ∇s by 6-neighbor central differences and constrains its angle "
                         "to dense normal per shell-cell voxel. Direct fragmentation detector.")
    ap.add_argument("--sp-normal-fd-cos", type=float, default=0.5,
                    help="cos(allowed angle) for FD ∇s vs n_d. 0.5=60°, 0.7071=45°, 0.866=30°.")
    ap.add_argument("--sp-normal-fd-p", type=float, default=3.0,
                    help="punitive hinge exponent for FD normal angle violation")
    ap.add_argument("--sp-lap-coarse-schedule", type=str, default="",
                    help="multi-res lap coarsening schedule: 'step_frac:K,...'. e.g., '0.66:4,0.85:2,1.0:1' (early K=4 coarse → late K=1 fine). Empty = use sparse lap (no coarsening).")
    ap.add_argument("--sp-pool-anchor-w", type=float, default=0.0,
                    help="weight for avg-pool anchor loss: push each sparse voxel toward K×K×K super-voxel mean")
    ap.add_argument("--sp-pool-anchor-schedule", type=str, default="",
                    help="multi-res pool anchor schedule: 'frac:K,...' e.g. '0.66:4,0.85:2,1.0:1'")
    ap.add_argument("--sp-param-pool-schedule", type=str, default="",
                    help="REPRESENTATION pooling: force sp_param to be coarse (K×K×K super-voxel = same value) after each step. 'frac:K,...' e.g. '0.66:4,0.85:2,1.0:1'")
    ap.add_argument("--sp-param-pool-blend", action='store_true',
                    help="Soft blend instead of hard pool. sp_param ← α*pooled + (1-α)*original where α = 1 - step_i/N (decays linearly to 0)")
    ap.add_argument("--sparse-steps", type=int, default=30,
                    help="Number of sparse denoising steps (default 30). All schedules (pool, weight anneal) auto-scale to this value.")
    ap.add_argument("--sp-lap-w", type=float, default=0.0,
                    help="edge-aware tangential Laplacian smoothness loss weight. "
                         "Penalizes ∇²s_tan (Laplace-Beltrami) on shell cells, weighted by "
                         "edge mask from dense normal variation (sharp edges exempted).")
    ap.add_argument("--sp-lap-sigma", type=float, default=0.3,
                    help="edge mask sigma: w = exp(-(1-cos(n_d, n_d_nbr))²/σ²). "
                         "Lower σ → more edges exempted, higher σ → more uniform smoothing.")
    ap.add_argument("--sp-coh-w", type=float, default=0.0,
                    help="Normal-coherence loss weight (R = mean resultant length). "
                         "Penalizes voxels whose neighborhood ∇s are scattered (= noise). "
                         "Curved smooth surface ≈ flat surface from R perspective.")
    ap.add_argument("--sp-coh-r", type=int, default=1,
                    help="Neighborhood radius for coherence computation. "
                         "1 → 3³=27 voxels, 2 → 5³=125 voxels (heavier).")
    ap.add_argument("--sp-coh-threshold", type=float, default=0.7,
                    help="R threshold below which voxel is considered noisy. "
                         "0.7 = mild, 0.5 = lenient, 0.85 = strict.")
    ap.add_argument("--sp-coh-p", type=float, default=3.0,
                    help="cubic punitive hinge exponent for coherence violation")
    # ── Refined coh via orientation-tensor λ_3 (surface-band, multi-cluster aware) ────
    ap.add_argument("--sp-aniso-w", type=float, default=0.0,
                    help="Refined noise loss via orientation-tensor λ_3. "
                         "Surface-band filtered + multi-cluster tolerant (flat/edge/corner OK; noise penalized).")
    ap.add_argument("--sp-aniso-r", type=int, default=1,
                    help="neighborhood radius (1→27 voxels)")
    ap.add_argument("--sp-aniso-band", type=float, default=0.2,
                    help="surface band: only voxels and neighbors with |s|<band are counted")
    ap.add_argument("--sp-aniso-lam3", type=float, default=0.10,
                    help="λ_3 threshold above which voxel considered noise (0.0=strict, 0.33=full random)")
    ap.add_argument("--sp-aniso-p", type=float, default=3.0,
                    help="punitive hinge exponent")
    # --- Off-diagonal Weingarten loss (FlatCAD-inspired, simplified axis-aligned FD) ---
    ap.add_argument("--sp-odw-w", type=float, default=0.0,
                    help="off-diagonal Hessian (|H_xy|+|H_yz|+|H_xz|) loss weight — CAD developability prior")
    ap.add_argument("--sp-odw-band", type=float, default=0.2,
                    help="surface band: only voxels with |s| < band counted")
    ap.add_argument("--sp-odw-p", type=float, default=1.0,
                    help="penalty exponent (1=L1, 2=L2)")
    # --- DF-ODW (dense-frame ODW) — rotation-invariant FlatCAD using dense MC normal frame ---
    ap.add_argument("--sp-dfodw-w", type=float, default=0.0,
                    help="DF-ODW loss weight — |e1^T H e2| projected to dense MC tangent frame")
    ap.add_argument("--sp-dfodw-band", type=float, default=0.2,
                    help="surface band for DF-ODW")
    ap.add_argument("--sp-dfodw-p", type=float, default=1.0,
                    help="DF-ODW penalty exponent")
    # --- A: EDT-anchored deviation ---
    ap.add_argument("--sp-edt-w", type=float, default=0.0,
                    help="EDT-anchored deviation loss weight")
    ap.add_argument("--sp-edt-tol", type=float, default=0.05,
                    help="tolerance: |sdf - ref_edt| < tol → penalty 0")
    ap.add_argument("--sp-edt-p", type=float, default=1.0,
                    help="EDT penalty exponent")
    ap.add_argument("--sp-edt-scale", type=float, default=1.0,
                    help="scale factor for EDT (voxel units → sparse sdf units)")
    # --- C: Snapshot anchor (graded freedom) ---
    ap.add_argument("--sp-snap-w", type=float, default=0.0,
                    help="snapshot anchor weight — pull deep voxels to step-0 sdf, surface free")
    ap.add_argument("--sp-snap-d-half", type=float, default=1.5,
                    help="depth (voxel units) where anchor_strength = 0.5")
    ap.add_argument("--sp-snap-sigma", type=float, default=0.5,
                    help="smoothness of freedom transition")
    ap.add_argument("--sp-shift", type=float, default=None,
                    help="Override scheduler.config.shift before set_timesteps. "
                         ">1 → dense at high t (default), <1 → dense at low t (slow at end).")
    ap.add_argument("--sp-reverse-t", type=float, default=0.0,
                    help="Monkey-patch sparse timesteps with inverse-shifted σ. "
                         "0 = off, 0.3 = aggressive reverse (dense at low t). "
                         "Takes precedence over default scheduler t.")
    ap.add_argument("--sp-interior-w", type=float, default=0.0,
                    help="deep interior solid-keep loss weight (penalize s>0 in eroded dense). 0=off")
    ap.add_argument("--sp-topology-sdf-fix", action="store_true",
                    help="convert positive-inside decoder values to mc_threshold-sdf for the "
                         "negative-inside hole/interior preservation loss; opt-in for comparison "
                         "with historical runs")
    ap.add_argument("--sp-interior-erode", type=int, default=2,
                    help="erode iter for deep interior. 1=2+ layer features, 2=3+ layer features")
    ap.add_argument("--sp-interior-depth-graded", action='store_true',
                    help="interior loss uses depth-graded weight (shell=0, depth k = slope*k) instead of binary mask")
    ap.add_argument("--sp-interior-depth-slope", type=float, default=2.0,
                    help="depth_w slope per layer for depth-graded interior (default 2.0 → depth 1=2, 2=4, 3=6)")
    ap.add_argument("--sp-hole-interior-thresh", type=float, default=2.0,
                    help="depth-into-solid threshold (voxels) defining interior vs surface")
    ap.add_argument("--sp-hole-exterior-thresh", type=float, default=2.0,
                    help="depth-out-of-solid threshold (voxels) defining exterior vs surface")
    # --- Sparse loss term weights (for ablation; previously hard-coded as 1.0 / 0.5) ---
    ap.add_argument("--sp-rmin-w",  type=float, default=1.0,
                    help="topopt r_min loss weight in sparse stage")
    ap.add_argument("--sp-thick-w", type=float, default=0.5,
                    help="per-voxel thickness loss weight in sparse stage")
    # --- Sparse region BCE (mirror of dense 3-region BCE — keep BC solid, outside empty) ---
    ap.add_argument("--sp-bc-w",     type=float, default=0.0,
                    help="sparse BCE: active voxels in BC region → sdf>0 (solid). Counters BC-as-hole drift.")
    ap.add_argument("--sp-out-w",    type=float, default=0.0,
                    help="sparse BCE: active voxels outside envelope → sdf<0 (empty).")
    ap.add_argument("--sp-design-w", type=float, default=0.0,
                    help="sparse BCE: active voxels in design region → sdf>0. Off by default (sparse should be free to carve).")
    ap.add_argument("--sp-bce-steepness", type=float, default=10.0,
                    help="logit gain: occ_logit = (sdf - mc_threshold) * steepness")
    # --- BC-buffer flesh (force solid in a dilated ring around BC) ---
    ap.add_argument("--sp-bc-buffer-w", type=float, default=0.0,
                    help="sparse BCE on a dilated ring AROUND bc_64 (force flesh near BC pegs)")
    ap.add_argument("--sp-bc-buffer-dilate", type=int, default=2,
                    help="voxel dilation iterations defining the buffer ring (default 2)")
    ap.add_argument("--force-envelope-clip", action="store_true",
                    help="HARD constraint: force envelope-OUTSIDE voxels empty on every refiner/sparse "
                         "SDF→MC (≥256³). The refiner CANNOT produce material beyond the envelope STL "
                         "(∪ pegs) — defined at the SDF source, not a post-clip. Symmetric to --force-bc-solid.")
    ap.add_argument("--restrict-active-to-envelope", action="store_true",
                    help="filter sparse latent_index to envelope-inside voxels only, so the sparse "
                         "refiner cannot generate material outside the envelope at all (no post-clip "
                         "of protruding content). BC (fix∪load) voxels are always kept.")
    ap.add_argument("--restrict-dilate-vox", type=float, default=0.0,
                    help="dilate the restrict-to-envelope keep region by N voxels (dense-side gate). "
                         "The sparse force-envelope-clip still trims to the TRUE envelope, so this only "
                         "lets the refiner form thicker boundary features (fatter dense → true-clip sparse). "
                         "1.0 ≈ one 64³ voxel (~3.1mm). 0 = strict inside (default).")
    ap.add_argument("--restrict-dilate-thin", type=float, default=0.0,
                    help="SELECTIVE dense-gate dilation: dilate the envelope gate by N voxels ONLY near "
                         "thin dense features (thickness < --thin-target-vox512), detected via "
                         "morphological opening of the 64³ active mask. Thick regions keep the true "
                         "envelope → less outside-envelope bloat than uniform --restrict-dilate-vox. "
                         "Combine: keep = true_env ∪ (dilated_env ∩ near_thin).")
    ap.add_argument("--thin-target-vox512", type=float, default=8.0,
                    help="thin-feature threshold for --restrict-dilate-thin, in 512³-grid voxels "
                         "(8 ≈ 3.1mm). Dense features thinner than this get the dilated gate.")
    ap.add_argument("--restrict-dilate-peg", type=float, default=-1.0,
                    help="gate dilation (voxels) near the fix/load pegs, separate from the thin "
                         "gate. -1 = same as --restrict-dilate-thin. Keep larger (e.g. 3) for load "
                         "attachment while the thin gate stays small (e.g. 2 → thin 1-2 + 2 ≤ 4-cell cap).")
    ap.add_argument("--bc-inloop-only", action="store_true",
                    help="with --force-bc-solid: apply the in-loop sparse SDF clamp (BC region "
                         "DEFINED as interior during refinement) but SKIP the final-MC peg "
                         "min-union bake — the mesh keeps the refiner's own surface.")
    ap.add_argument("--force-bc-solid", action="store_true",
                    help="hard-force BC (fix∪load) region solid on every refiner/sparse SDF→MC "
                         "(≥256³) AND in the sparse guidance/FEA loop — BC region always filled, "
                         "as a default (not an optimization loss). Uses fix/load STL SDFs (smooth).")
    ap.add_argument("--force-bc-fix-only", action="store_true",
                    help="with --force-bc-solid, hard-force only the fixed STL; retain load "
                         "interfaces through sparse soft guidance and the unmodified refiner surface")
    ap.add_argument("--force-bc-sdf-scale", type=float, default=40.0,
                    help="SDF→refiner mapping slope K (refiner_sdf = level - K·d). Higher = sharper "
                         "peg/envelope boundary. Continuous blend (no voxel staircase).")
    ap.add_argument("--force-bc-dilate-mm", type=float, default=6.0,
                    help="inflate the BC-solid region outward by this many mm (smooth SDF offset). "
                         "0 = raw peg STL only (thin). 6mm ≈ old voxel dilate=4 thickness (70k mm³).")
    ap.add_argument("--force-load-dilate-mm", type=float, default=None,
                    help="optional separate load-STL BC margin in mm; when unset, use "
                         "force-bc-dilate-mm for both fix and load STLs")
    ap.add_argument("--bc-proper-bce-highres", default=None,
                    help="optional separate high-res bc_proper.npz (e.g. 128³, 256³) used ONLY for "
                         "sparse BCE — reduces peg-centroid aliasing without affecting dense flow.")
    # --- Sparse volume constraint (counters FEA mass-inflation) ---
    ap.add_argument("--sp-vw", type=float, default=0.0,
                    help="sparse volume target loss weight: penalize mean(occ_prob) ≠ sp_vol_target")
    ap.add_argument("--sp-vol-target", type=float, default=0.3,
                    help="target mean occupancy probability over sparse active voxels")
    # --- TO-style Augmented Lagrangian on vol constraint ---
    ap.add_argument("--sp-aug-lag", action='store_true',
                    help="enable Augmented Lagrangian for sparse vol constraint: adaptive lambda + quadratic penalty. "
                         "loss = lambda*(V-V*) + (mu/2)*(V-V*)^2. lambda updated every step.")
    ap.add_argument("--sp-lambda-init", type=float, default=10.0,
                    help="initial Lagrange multiplier lambda for sp_aug_lag")
    ap.add_argument("--sp-mu-aug-lag", type=float, default=50.0,
                    help="quadratic penalty coefficient mu for sp_aug_lag")
    ap.add_argument("--sp-lambda-alpha", type=float, default=1.0,
                    help="outer-loop update rate for lambda: lambda += alpha * mu * (V-V*)")
    # --- TO-style Heaviside projection + beta-continuation ---
    ap.add_argument("--sp-heaviside-proj", action='store_true',
                    help="apply Heaviside projection to sparse occupancy with bisection on eta for exact volume. "
                         "Differentiable w.r.t. sdf, eta found by no-grad bisection.")
    ap.add_argument("--sp-beta-init", type=float, default=2.0,
                    help="initial sharpness beta for Heaviside projection")
    ap.add_argument("--sp-beta-max", type=float, default=20.0,
                    help="max sharpness beta (continuation: beta scales exp(step_frac) from init to max)")
    ap.add_argument("--sp-eta-bisect-iter", type=int, default=15,
                    help="bisection iterations for finding eta s.t. mean(heaviside)=vol_target")
    # --- Sparse interior freeze (option D) — detach deep_int voxel sdf in forward ---
    ap.add_argument("--sp-interior-freeze", action='store_true',
                    help="freeze sparse SDF on deep_int voxels (no gradient flow into latent for those positions); "
                         "uses the same di_t mask as sp_interior_w. Default off.")
    # --- Sparse interior LATENT freeze (option H) — restore deep_int sp_param to step-0 value ---
    ap.add_argument("--sp-interior-latent-freeze", action='store_true',
                    help="restore sp_param[deep_int] to step-0 snapshot after every scheduler/opt update. "
                         "This freezes the LATENT (not just the decoded sdf), so the model itself outputs "
                         "the same sdf at deep_int voxels every step. dense-derived first estimate preserved.")
    # --- Sparse shell-only mode — remove deep_int voxels from latent_index entirely ---
    ap.add_argument("--sp-shell-only", action='store_true',
                    help="prune deep_int voxels from latent_index before sparse stage. "
                         "deep_int defined by erode(dense_active, sp_interior_erode). "
                         "BC region voxels are always preserved. "
                         "sparse2mesh fills the holes with background = +1.0 (MC inside).")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from run_config import apply_config
    apply_config(ap, stage='mesh')
    args = ap.parse_args()
    if args.fea_combine_loads:
        if not args.fea_back_domain_dir:
            ap.error('--fea-combine-loads requires --fea-back-domain-dir')
        second_stl = Path(args.fea_back_domain_dir) / 'load.stl'
        if not second_stl.exists():
            ap.error(f'second load STL missing: {second_stl}')
        os.environ['FEA_SECOND_LOAD_STL'] = str(second_stl.resolve())
        os.environ['FEA_SECOND_LOAD_MAGNITUDE'] = str(args.fea_back_load_magnitude)
        os.environ['FEA_SECOND_LOAD_MODE'] = str(args.fea_back_load_mode)
        print(f'[FEA simultaneous loads] seat={args.load_mode} '
              f'back={args.fea_back_load_mode} {args.fea_back_load_magnitude:g} N '
              f'STL={second_stl}', flush=True)
    sp_iso = (float(args.sp_sdf_guidance_threshold)
              if args.sp_sdf_guidance_threshold is not None else float(args.mc_threshold))
    if not -1.0 < sp_iso < 1.0:
        ap.error('--sp-sdf-guidance-threshold must be between -1 and 1')
    if args.sp_dense_core_mm < 0 or args.sp_dense_core_w < 0:
        ap.error('--sp-dense-core-mm and --sp-dense-core-w must be nonnegative')
    if args.sp_image_proj_w < 0:
        ap.error('--sp-image-proj-w must be nonnegative')
    if args.sp_negative_space_w < 0:
        ap.error('--sp-negative-space-w must be nonnegative')
    if args.sp_negative_space_w > 0 and not args.sp_negative_space_mask:
        ap.error('--sp-negative-space-w requires --sp-negative-space-mask')
    if args.force_load_dilate_mm is not None and args.force_load_dilate_mm < 0:
        ap.error('--force-load-dilate-mm must be nonnegative')
    if args.force_bc_fix_only and not args.force_bc_solid:
        ap.error('--force-bc-fix-only requires --force-bc-solid')
    if args.sp_support_halo_vox < 0:
        ap.error('--sp-support-halo-vox must be nonnegative')
    if not 0 <= args.sp_dense_core_tail_frac <= 1:
        ap.error('--sp-dense-core-tail-frac must be within [0, 1]')
    if args.sp_dense_trust_mm < 0 or args.sp_dense_trust_scale <= 0:
        ap.error('--sp-dense-trust-mm must be nonnegative and --sp-dense-trust-scale positive')
    if args.sp_dense_core_mm > 0 and not args.sp_sdf_inside_low:
        ap.error('--sp-dense-core-mm requires --sp-sdf-inside-low')
    if args.sp_wall_ray_w < 0 or not 1 <= args.sp_wall_ray_band <= 8 or not 0 < args.sp_wall_ray_min_mass <= args.sp_wall_ray_band:
        ap.error('invalid --sp-wall-ray-* settings')
    if not 0 <= args.sp_wall_ray_tail_frac <= 1:
        ap.error('--sp-wall-ray-tail-frac must be within [0, 1]')
    if args.sp_wall_ray_w > 0 and not args.sp_sdf_inside_low:
        ap.error('--sp-wall-ray-w requires --sp-sdf-inside-low')
    if args.sp_side_depth_w < 0 or args.sp_side_depth_tolerance_mm < 0 or not 0 <= args.sp_side_depth_tail_frac <= 1:
        ap.error('invalid --sp-side-depth-* settings')
    if args.sp_side_depth_w > 0 and not args.sp_sdf_inside_low:
        ap.error('--sp-side-depth-w requires --sp-sdf-inside-low')
    def sp_occupancy_logit(sdf_values, steepness):
        delta = sp_iso - sdf_values if args.sp_sdf_inside_low else sdf_values - sp_iso
        return delta * steepness

    # In-loop FEM load direction. The FEA solver (fenics_fea_bracket.py) reads it from the
    # LOAD_MODE env var and defaults to 'diag' when unset, so export it here — this makes
    # stages.mesh.load_mode effective for a plain `--config` run too, not only when an
    # orchestrator sets the environment.
    if getattr(args, 'load_mode', None):
        os.environ['LOAD_MODE'] = str(args.load_mode)
        print(f"[load_mode] in-loop FEM load direction = {args.load_mode} (LOAD_MODE)")

    # ── BC-solid forcing + optional --save-raw-sdf: single MC monkey-patch ────
    # Force BC (fix ∪ load) region solid on every refiner/sparse SDF→MC (≥256³),
    # so the BC region is ALWAYS filled — in-loop snapshots + final output mesh.
    # Hard default, not an optimization loss.
    # BC region defined by the ACTUAL peg STL SDFs (fix.stl, load.stl), evaluated
    # at each grid's world-frame voxel centers — smooth peg shape, NO voxel
    # staircase. The mask is built lazily on the first ≥256³ MC call, once the
    # env frame (_bc_hi['env_origin'], ['env_pitch']) has been populated below.
    # Convention: refiner SDF has inside < level (verified: raw_sdf inside<0.4).
    _fec = bool(getattr(args, 'force_envelope_clip', False))
    _bc_hi = {'active': bool(getattr(args, 'force_bc_solid', False)) or _fec,
              'force_bc': bool(getattr(args, 'force_bc_solid', False)),
              'fix_only': bool(getattr(args, 'force_bc_fix_only', False)),
              'force_env': _fec,
              'mask': None, 'env_mask': None, 'env_origin': None, 'env_pitch': None,
              'fix_sdf': None, 'load_sdf': None, 'env_sdf': None}
    _sdf_mesh_cache = {}
    def _load_sdf_mesh(path):
        """Load triangle-soup STLs with trimesh cleanup before constructing a pysdf BVH.

        The bracket STLs store every triangle with separate vertices.  Passing that raw soup
        (process=False) to pysdf can return finite-looking sentinel distances around ±1.84e19,
        which overflow the float16 refiner field and create tiny MC components near BC seams.
        Trimesh processing welds duplicate vertices without changing the coordinate frame.
        """
        key = str(Path(path).expanduser().resolve())
        if key in _sdf_mesh_cache:
            return _sdf_mesh_cache[key]
        mesh = trimesh.load(key, force='mesh', process=True)
        mesh.remove_unreferenced_vertices()
        _sdf_mesh_cache[key] = mesh
        return mesh

    def _check_sdf_range(name, values, R):
        """Reject corrupt pysdf sentinel values before they reach a float16 MC field."""
        lo = float(np.nanmin(values)); hi = float(np.nanmax(values))
        frame_diag = float(np.linalg.norm(np.asarray(_bc_hi['env_pitch']) * 64.0))
        limit = max(1.0, frame_diag * 4.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo < -limit or hi > limit:
            raise RuntimeError(f'corrupt {name} SDF at R={R}: range=[{lo:.6g}, {hi:.6g}], '
                               f'expected within ±{limit:.6g}m; check STL topology')
        return values

    if _bc_hi['active']:
        from pysdf import SDF as _SDF_bc
        _fix_m = _load_sdf_mesh(args.fix_stl)
        _load_m = _load_sdf_mesh(args.load_stl)
        _bc_hi['fix_sdf'] = _SDF_bc(_fix_m.vertices.astype(np.float32), _fix_m.faces.astype(np.uint32))
        _bc_hi['load_sdf'] = _SDF_bc(_load_m.vertices.astype(np.float32), _load_m.faces.astype(np.uint32))
        if _fec:
            _env_m = _load_sdf_mesh(args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl')
            _bc_hi['env_sdf'] = _SDF_bc(_env_m.vertices.astype(np.float32), _env_m.faces.astype(np.uint32))
        print(f"[force_bc] BC-solid={_bc_hi['force_bc']} envelope-clip={_fec} "
              f"(STL-SDF hard constraint on refiner MC)", flush=True)

    def _build_bc_mask(R):
        """Build R³ BC-solid bool mask from peg STL SDFs at world voxel centers.
        R³ grid voxel center at index I maps to world = env_origin + (I + 0.5)*pitch_R,
        where pitch_R = pitch64*(64/R). NB: the half-voxel must scale with R — using a
        fixed +0.5*pitch64 (the old bug) shifts a 512³ grid by +3.5*pitch512 ≈ +1.355mm
        per axis vs the true STL frame."""
        eo = _bc_hi['env_origin']; ep = _bc_hi['env_pitch']
        if eo is None or ep is None:
            return None
        R64 = 64
        ax = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[0] + eo[0]
        ay = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[1] + eo[1]
        az = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[2] + eo[2]
        X, Y, Z = np.meshgrid(ax, ay, az, indexing='ij')
        q = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
        # pysdf: inside > 0.  Dilated peg field = max(sdf_fix, sdf_load) + margin
        # ( >0 inside dilated peg ).  Return the CONTINUOUS signed field mapped to the
        # refiner SDF convention (inside < level) so MC gets a smooth iso-crossing at the
        # dilated peg surface — NO voxel staircase (unlike a hard bool clamp).
        fix_margin_m = float(args.force_bc_dilate_mm) / 1000.0
        load_margin_m = float(args.force_load_dilate_mm if args.force_load_dilate_mm is not None
                              else args.force_bc_dilate_mm) / 1000.0
        d_fix = _check_sdf_range('fix', _bc_hi['fix_sdf'](q), R)
        d_load = _check_sdf_range('load', _bc_hi['load_sdf'](q), R)
        d_peg = (d_fix + fix_margin_m if _bc_hi['fix_only'] else
                 np.maximum(d_fix + fix_margin_m, d_load + load_margin_m))  # >0 inside
        m = d_peg.reshape(R, R, R).astype(np.float32)
        print(f"[force_bc_solid] built {R}³ BC-solid SDF field: inside(d>0)={int((m>0).sum()):,} voxels "
              f"(fix dilate={fix_margin_m*1000:.1f}mm, load dilate={load_margin_m*1000:.1f}mm, "
              f"continuous blend)", flush=True)
        return m

    def _build_env_mask(R):
        """R³ bool mask of voxels OUTSIDE the envelope (∪ pegs) — to force empty.
        envelope-outside = env_sdf<0 AND not inside(fix)/inside(load) (keep peg protrusions)."""
        eo = _bc_hi['env_origin']; ep = _bc_hi['env_pitch']
        if eo is None or ep is None or _bc_hi['env_sdf'] is None:
            return None
        R64 = 64
        ax = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[0] + eo[0]
        ay = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[1] + eo[1]
        az = (np.arange(R, dtype=np.float64) + 0.5) * (R64 / R) * ep[2] + eo[2]
        X, Y, Z = np.meshgrid(ax, ay, az, indexing='ij')
        q = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
        # Continuous "allowed region" signed field = max(d_env, d_peg_dilated) (>0 inside allowed).
        # envelope ∪ dilated-pegs.  Return continuous field for smooth clamp (no staircase).
        fix_margin_m = float(args.force_bc_dilate_mm) / 1000.0
        load_margin_m = float(args.force_load_dilate_mm if args.force_load_dilate_mm is not None
                              else args.force_bc_dilate_mm) / 1000.0
        d_env = _check_sdf_range('envelope', _bc_hi['env_sdf'](q), R)   # >0 inside envelope
        d_fix = _check_sdf_range('fix', _bc_hi['fix_sdf'](q), R)
        d_load = _check_sdf_range('load', _bc_hi['load_sdf'](q), R)
        d_peg = (d_fix + fix_margin_m if _bc_hi['fix_only'] else
                 np.maximum(d_fix + fix_margin_m, d_load + load_margin_m))
        d_allowed = np.maximum(d_env, d_peg)                            # >0 inside (env ∪ peg)
        m = d_allowed.reshape(R, R, R).astype(np.float32)
        print(f"[force_envelope_clip] built {R}³ allowed-region SDF field: inside={int((m>0).sum()):,} voxels "
              f"(fix dilate={fix_margin_m*1000:.1f}mm, load dilate={load_margin_m*1000:.1f}mm, "
              f"continuous)", flush=True)
        return m

    _dense_trust_cache = {}
    def _dense_trust_field(R):
        """Sample the dense *3D* mesh SDF only in its bounding box, once per grid size."""
        if R in _dense_trust_cache:
            return _dense_trust_cache[R]
        from pysdf import SDF as _TrustSDF
        path = Path(args.out) / 'mesh_dense.obj'
        if not path.exists():
            raise FileNotFoundError(f'dense trust surface missing: {path}')
        mesh = _load_sdf_mesh(path)
        if not mesh.is_watertight:
            raise ValueError(f'dense trust surface must be watertight: {path}')
        sdf = _TrustSDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        origin = np.asarray(_bc_hi['env_origin'], dtype=np.float32)
        pitch = np.asarray(_bc_hi['env_pitch'], dtype=np.float32) * (64.0 / R)
        pad = max(args.sp_dense_trust_mm / 1000.0, float(pitch.max()) * 2)
        lo = np.maximum(0, np.floor((mesh.bounds[0] - pad - origin) / pitch - 0.5).astype(int))
        hi = np.minimum(R, np.ceil((mesh.bounds[1] + pad - origin) / pitch - 0.5).astype(int) + 1)
        ax = origin[0] + (np.arange(lo[0], hi[0], dtype=np.float32) + 0.5) * pitch[0]
        ay = origin[1] + (np.arange(lo[1], hi[1], dtype=np.float32) + 0.5) * pitch[1]
        az = origin[2] + (np.arange(lo[2], hi[2], dtype=np.float32) + 0.5) * pitch[2]
        dist = np.empty((len(ax), len(ay), len(az)), dtype=np.float16)
        for start in range(0, len(ax), 8):
            stop = min(start + 8, len(ax))
            X, Y, Z = np.meshgrid(ax[start:stop], ay, az, indexing='ij')
            query = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
            dist[start:stop] = sdf(query).reshape(stop-start, len(ay), len(az)).astype(np.float16)
        sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(3))
        result = (sl, dist)
        _dense_trust_cache[R] = result
        print(f"[dense trust] sampled {dist.size:,} voxels @R={R}, "
              f"interior core={(dist > args.sp_dense_trust_mm/1000).sum():,}, "
              f"band={args.sp_dense_trust_mm:.3f}mm", flush=True)
        return result

    if args.sp_sdf_smooth_sigma < 0:
        ap.error('--sp-sdf-smooth-sigma must be nonnegative')
    if args.save_raw_sdf or _bc_hi['active'] or args.sp_dense_trust_mm > 0 or args.sp_sdf_smooth_sigma > 0:
        import skimage.measure as _meas
        _orig_mc = _meas.marching_cubes
        _captured_sdf = []
        def _mc_capture(volume, level=0, **kw):
            if args.sp_sdf_smooth_sigma > 0 and volume.ndim == 3 and max(volume.shape) >= 512:
                from scipy.ndimage import gaussian_filter as _gaussian_filter
                sigma = float(args.sp_sdf_smooth_sigma)
                inside = np.nonzero(volume < level)  # sparse refiner: lower SDF is solid
                if inside[0].size:
                    pad = int(np.ceil(4 * sigma)) + 2
                    lo = np.maximum(0, np.array([axis.min() for axis in inside]) - pad)
                    hi = np.minimum(volume.shape, np.array([axis.max() for axis in inside]) + pad + 1)
                    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
                    source = np.asarray(volume[sl], dtype=np.float32)
                    occupied_fraction = float((source < level).mean())
                    smoothed = _gaussian_filter(source, sigma=sigma, mode='nearest')
                    matched_level = (float(np.quantile(smoothed, occupied_fraction))
                                     if args.sp_sdf_smooth_volume_match else float(level))
                    volume = volume.copy()
                    volume[sl] = (smoothed + float(level) - matched_level).astype(volume.dtype)
                    print(f"[sparse SDF smooth] sigma={sigma:g} vox, volume_match="
                          f"{args.sp_sdf_smooth_volume_match}, iso_before={float(level):.5f}, "
                          f"iso_matched={matched_level:.5f}, crop={tuple((hi-lo).tolist())}",
                          flush=True)
            if (_bc_hi['active'] or args.sp_dense_trust_mm > 0) and volume.ndim == 3 and max(volume.shape) >= 256:
                R = volume.shape[0]
                volume = volume.copy()
                lv = float(level)
                if args.sp_dense_trust_mm > 0:
                    sl, dist = _dense_trust_field(R)
                    margin = args.sp_dense_trust_mm / 1000.0
                    for start in range(0, dist.shape[0], 8):
                        stop = min(start + 8, dist.shape[0])
                        d = dist[start:stop].astype(np.float32)
                        target = (lv - args.sp_dense_trust_scale * (d - margin)).astype(volume.dtype)
                        view = volume[sl[0].start+start:sl[0].start+stop, sl[1], sl[2]]
                        np.minimum(view, target, out=view, where=d > margin)
                # Map a pysdf signed field d (meters, inside>0) to the refiner SDF
                # convention (inside<level) as: refiner_sdf = level - K*d.
                # d=0 (surface) → level (iso-crossing) → smooth boundary, no staircase.
                K = float(getattr(args, 'force_bc_sdf_scale', 40.0))
                # BC solid: UNION (both solid) → elementwise MIN in refiner convention
                if _bc_hi['force_bc'] and not getattr(args, 'bc_inloop_only', False):
                    if _bc_hi['mask'] is None or _bc_hi['mask'].shape != volume.shape:
                        _bc_hi['mask'] = _build_bc_mask(R)
                    if _bc_hi['mask'] is not None and _bc_hi['mask'].shape == volume.shape:
                        peg_sdf = (lv - K * _bc_hi['mask']).astype(volume.dtype)
                        np.minimum(volume, peg_sdf, out=volume)   # smooth union
                # envelope-outside empty: INTERSECT with allowed region → elementwise MAX
                if _bc_hi['force_env']:
                    if _bc_hi['env_mask'] is None or _bc_hi['env_mask'].shape != volume.shape:
                        _bc_hi['env_mask'] = _build_env_mask(R)
                    if _bc_hi['env_mask'] is not None and _bc_hi['env_mask'].shape == volume.shape:
                        allow_sdf = (lv - K * _bc_hi['env_mask']).astype(volume.dtype)
                        np.maximum(volume, allow_sdf, out=volume)   # smooth clip to allowed region
            if args.save_raw_sdf and volume.ndim == 3:
                _captured_sdf.append({'vol': volume.copy(), 'level': float(level)})
            return _orig_mc(volume, level=level, **kw)
        _meas.marching_cubes = _mc_capture
        args._captured_sdf = _captured_sdf
        if args.save_raw_sdf:
            print(f"[save_raw_sdf] MC interception armed → {args.save_raw_sdf}", flush=True)

    # ── Deterministic mode (--deterministic or DETERMINISTIC=1 env) ───────────
    # guarantees the same result for the same seed (removes CUDA non-determinism).
    # CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set in the external env.
    if args.deterministic or os.environ.get('DETERMINISTIC', '0') == '1':
        import random as _random
        _random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        print(f"[deterministic] seed={args.seed}, cudnn.deterministic=True, "
              f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG','<unset>')}", flush=True)

    # Auto mesh_size from envelope STL bbox (if user left fea_mesh_size <= 0)
    if args.fea_w > 0 and args.fea_mesh_size <= 0 and args.fea_bracket_stl:
        try:
            import trimesh as _tm
            _stl = _tm.load(args.fea_bracket_stl, force='mesh')
            _ext_max = float((_stl.bounds[1] - _stl.bounds[0]).max())
            args.fea_mesh_size = _ext_max / float(args.fea_mesh_cells_per_edge)
            print(f"[FEA] auto mesh_size = {args.fea_mesh_size:.4f}  "
                  f"(envelope max ext = {_ext_max:.3f} / {args.fea_mesh_cells_per_edge:.0f} cells)")
        except Exception as e:
            print(f"[FEA] auto mesh_size FAILED ({e}) — falling back to 0.006")
            args.fea_mesh_size = 0.006

    # Per-step loss CSV — opened here, passed to dense_flowdps + closed at end.
    Path(args.out).mkdir(parents=True, exist_ok=True)
    loss_csv = open(Path(args.out) / 'loss_log.csv', 'w')
    loss_csv.write('stage,step,t,total,fea_comp,l_rmin,l_thick,l_hole,l_interior,l_normalfd,l_lap,l_aniso,l_vol,l_bce,l_reach\n')
    loss_csv.flush()

    print("Loading Direct3D-S2 pipeline...")
    pipe = Direct3DS2Pipeline.from_pretrained('wushuang98/Direct3D-S2', subfolder='direct3d-s2-v-1-1')
    pipe.to('cuda')

    # ── optional sidecar attach (ControlNet-style, env-controlled) ─────
    sidecar_ckpt = os.environ.get('SIDECAR_CKPT')
    if sidecar_ckpt and Path(sidecar_ckpt).exists():
        print(f"Attaching sidecar from {sidecar_ckpt}")
        import torch.nn as _nn
        class _Sidecar(_nn.Module):
            def __init__(self, in_ch=8, hidden=64):
                super().__init__(); self.hidden=hidden
                self.t_mlp = _nn.Sequential(_nn.Linear(1, hidden), _nn.SiLU(), _nn.Linear(hidden, hidden))
                self.in_conv = _nn.Conv3d(in_ch, hidden, 3, padding=1)
                self.block1 = _nn.Conv3d(hidden, hidden, 3, padding=1)
                self.block2 = _nn.Conv3d(hidden, hidden, 3, padding=1)
                self.out_conv = _nn.Conv3d(hidden, in_ch, 1)
                self.silu = _nn.SiLU()
            def forward(self, x, t):
                t_emb = self.t_mlp(t.view(-1, 1).float() / 1000.0).view(-1, self.hidden, 1, 1, 1)
                h = self.silu(self.in_conv(x.float()))
                h = self.silu(self.block1(h) + t_emb)
                h = self.silu(self.block2(h))
                return self.out_conv(h)
        class _Wrap(_nn.Module):
            def __init__(self, dd, sc):
                super().__init__(); self.dense_dit=dd; self.sidecar=sc
            def forward(self, x, t, cond):
                pred = self.dense_dit(x, t, cond)
                delta = self.sidecar(x, t)
                return pred + delta.to(pred.dtype)
            def __getattr__(self, name):
                try: return super().__getattr__(name)
                except AttributeError: return getattr(self._modules['dense_dit'], name)
        sc = _Sidecar(in_ch=8, hidden=64).cuda()
        sc.load_state_dict(torch.load(sidecar_ckpt, weights_only=True)['state'], strict=True)
        sc.eval()
        pipe.dense_dit = _Wrap(pipe.dense_dit, sc).cuda()
        print(f"  sidecar attached.")

    print("Loading bracket envelope...")
    br_pitch_xyz = None
    if args.bc_proper and Path(args.bc_proper).exists():
        d = np.load(args.bc_proper)
        br = d['bracket']
        bc = d['bc'].astype(bool)
        # NOTE: do NOT drop the load region from `bc`/`br` when the load interface is a void
        # (a caliper's piston pockets). It was tried and it is wrong on both counts:
        #  * It is not a waste. The pocket is in `bc`, and the aug-lag volume budget is imposed on
        #    design_t only ("inequality V <= 0.4 on design_t"), so filling it costs the design
        #    region nothing. post_hybrid_union_clip.py --load-cavity then subtracts the pocket, and
        #    what is left is exactly the bore WALL — the surface the FEA applies traction to.
        #    Measured on this caliper: mesh.obj held 12.96 cm3 inside the pockets, final.obj 0.24,
        #    and the a-posteriori FEA found a healthy 26,458 load / 7,814 fix facets.
        #  * Dropping it removes the guarantee that any material touches the pocket at all. Driving
        #    the pocket empty via out_t (out_w=30) leaves bc = fix only, 1.7% of the domain, and the
        #    run measured load facets collapsing 26,458 -> 9,522 and fix 7,814 -> 448, i.e. a part
        #    that is barely constrained and not the same physical problem.
        # Forcing a one-voxel wall band solid instead (pocket empty, band in bc) is the textbook
        # treatment but this grid cannot express it: at 64^3 one voxel is 3.181 mm, so the band comes
        # to 54.54 cm3 against a 30.33 cm3 pocket, bc grows to 13.4% of the domain (vs 6.9% for
        # plain fill), and the bands around the two bores in a bank merge (their gap is 4.62 mm).
        design = d['design'].astype(bool) if 'design' in d.files else br & ~bc
        br_pitch = float(d['pitch']) if 'pitch' in d else 0.0037535791666666665
        if 'pitch_xyz' in d.files:
            br_pitch_xyz = np.asarray(d['pitch_xyz'])
        print(f"  using PROPER BC (load+fix STLs): bracket {br.sum():,}, "
              f"BC {bc.sum():,} ({bc.sum()/br.sum()*100:.1f}%), design {design.sum():,}, pitch={br_pitch*1000:.3f}mm")
        if br_pitch_xyz is not None:
            print(f"    ANISOTROPIC pitch (mm): X={br_pitch_xyz[0]*1000:.3f} Y={br_pitch_xyz[1]*1000:.3f} Z={br_pitch_xyz[2]*1000:.3f}")
    else:
        ld = np.load(args.bracket_occ)
        br = ld['occupancy'] if 'occupancy' in ld else ld['bracket']
        br_pitch = float(ld['pitch']) if 'pitch' in ld else 0.0037535791666666665
        if 'pitch_xyz' in ld.files:
            br_pitch_xyz = np.asarray(ld['pitch_xyz'])
        inner = binary_erosion(br, iterations=args.bc_layers)
        design = inner & br
        bc = br & ~design
        print(f"  using EROSION-based BC: bracket {br.sum():,}, BC {bc.sum():,}, design {design.sum():,}, pitch={br_pitch*1000:.3f}mm")

    # Envelope world-frame origin — same logic as mesh_dense save, computed once for snapshot reuse.
    if args.bc_proper and Path(args.bc_proper).exists():
        _bb_npz = np.load(args.bc_proper)
    else:
        _bb_npz = np.load(args.bracket_occ)
    if 'origin' in _bb_npz.files:
        env_origin = np.asarray(_bb_npz['origin'])
    else:
        _stl_path = args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl'
        _env = trimesh.load(_stl_path, force='mesh')
        _ext = (_env.bounds[1] - _env.bounds[0]).max()
        env_origin = _env.bounds[0] - 0.05 * _ext
    # Use anisotropic pitch if available (npz array axis order [X,Y,Z]); else broadcast scalar.
    if br_pitch_xyz is not None:
        env_pitch = np.asarray(br_pitch_xyz, dtype=np.float64)
    else:
        env_pitch = np.array([br_pitch, br_pitch, br_pitch], dtype=np.float64)
    print(f"[env_frame] origin={env_origin*1000} mm  pitch_xyz={env_pitch*1000} mm")
    # Populate BC-solid env frame so the MC monkey-patch can build the STL-SDF mask.
    if _bc_hi.get('active'):
        _bc_hi['env_origin'] = np.asarray(env_origin, dtype=np.float64)
        _bc_hi['env_pitch']  = np.asarray(env_pitch,  dtype=np.float64)

    def _load_keep_alpha(p):
        im = Image.open(p)
        return im.convert('RGBA') if im.mode in ('RGBA', 'LA', 'PA') or 'transparency' in im.info else im.convert('RGB')

    use_mv = args.target_dir is not None
    if use_mv:
        target_dir = Path(args.target_dir)
        if args.views:
            view_names = [v.strip() for v in args.views.split(',') if v.strip()]
            if len(view_names) != args.n_views:
                raise ValueError(f"--views specifies {len(view_names)} views but --n-views={args.n_views}: "
                                 f"{view_names}")
            paths = [target_dir / f"{v}.png" for v in view_names]
            missing = [str(p) for p in paths if not p.is_file()]
            if missing:
                raise FileNotFoundError("configured conditioning views are missing: " + ", ".join(missing))
        else:
            paths = sorted(target_dir.glob("*.png"))[:args.n_views]
        imgs = [_load_keep_alpha(p) for p in paths]
        modes = {im.mode for im in imgs}
        print(f"input: {len(imgs)} views from {args.target_dir} modes={modes}")
        print("  view order: " + ", ".join(p.stem for p in paths))
        img_prep = pipe.prepare_image(imgs)
    else:
        img = _load_keep_alpha(args.image)
        img_prep = pipe.prepare_image(img)
        print(f"input: {args.image} {img.size} mode={img.mode}")

    # Load engineering load-path mask if specified
    load_path_mask = None
    if args.pw > 0 and args.load_path_mask:
        lp = np.load(args.load_path_mask)
        load_path_mask = lp['mask'].astype(bool)
        assert load_path_mask.shape == br.shape, \
            f"load_path_mask shape {load_path_mask.shape} != bracket {br.shape}"
        print(f"  loaded load-path mask: {int(load_path_mask.sum()):,} corridor voxels")

    # FEA: precompute voxel centers inside envelope (for subprocess fenics).
    # These are WORLD coordinates: fenics_fea_bracket.py matches them against the tet mesh of
    # the design domain with a cKDTree, so they must sit in the SAME frame the envelope mask
    # `br` was rasterized in. Use the env frame resolved above (npz['origin'] / ['pitch_xyz']
    # when present, STL-bbox fallback otherwise) — the convention voxel_nodes_from_stl() calls
    # "preferred". Deriving the origin from the STL bbox unconditionally (as this block used to)
    # only agrees with a uniscale npz on its longest axis: for the shipped grids it displaced the
    # node cloud by up to 19-24 voxels, so the FEM sensitivity was computed on a translated
    # density field.
    fea_nodes = None
    if args.fea_w > 0:
        if not args.fea_domain_dir:
            raise ValueError("--fea-w > 0 requires --fea-domain-dir")
        _ii = np.argwhere(br)                       # br is the 64³ envelope mask
        fea_nodes = env_origin + (_ii + 0.5) * env_pitch
        fea_nodes = fea_nodes_in_domain_frame(fea_nodes, args.fea_node_alignment,
                                               "dense grid → FEA domain")
        print(f"  FEA setup: w={args.fea_w} every {args.fea_every_n} step, warmup={args.fea_warmup}, "
              f"{len(fea_nodes)} envelope voxels, domain={args.fea_domain_dir}")
        print(f"    node frame: origin={env_origin*1000} mm  pitch_xyz={env_pitch*1000} mm")

    # === Stage 1: dense FlowDPS (or load from cache) ===
    _dense_cached = bool(args.load_dense_cache) and Path(args.load_dense_cache).exists()
    if _dense_cached:
        print(f"\n=== Stage 1: LOAD dense cache from {args.load_dense_cache} ===")
        _c = np.load(args.load_dense_cache)
        latent_index = torch.from_numpy(_c["latent_index"]).to("cuda")
        dense_normal_64 = torch.from_numpy(_c["dense_normal_64"]).to("cuda")
        # Free dense modules immediately
        pipe.dense_vae.cpu(); pipe.dense_dit.cpu(); pipe.dense_image_encoder.cpu()
        import gc; gc.collect(); torch.cuda.empty_cache()
        # Copy cached mesh_dense.obj if present (sibling .obj file)
        _src_mesh = str(Path(args.load_dense_cache).with_suffix("")) + "_mesh.obj"
        if Path(_src_mesh).exists():
            import shutil; shutil.copyfile(_src_mesh, str(Path(args.out)/"mesh_dense.obj"))
            print(f"  copied mesh_dense.obj from cache")
        print(f"  latent_index={tuple(latent_index.shape)}, dense_normal_64 valid cells={(dense_normal_64.norm(dim=-1)>1e-4).sum().item()}")
        # The sparse support expansion below the dense-generation branch is skipped
        # when a dense cache is loaded. Apply the same prototype expansion here so
        # sparse ablations have the support requested by their config.
        if args.shape_anchor_expand_vox > 0:
            if not args.shape_anchor_bank or args.shape_qd_target < 0:
                raise ValueError('--shape-anchor-expand-vox needs --shape-anchor-bank and --shape-qd-target')
            from scipy.ndimage import binary_dilation, distance_transform_edt
            anchor = np.load(args.shape_anchor_bank)['prototypes'][args.shape_qd_target] > .5
            anchor_core = anchor.copy()
            anchor = binary_dilation(anchor, iterations=args.shape_anchor_expand_vox)
            br_np = br.cpu().numpy() if hasattr(br, 'cpu') else np.asarray(br)
            bc_np = bc.cpu().numpy() if hasattr(bc, 'cpu') else np.asarray(bc)
            anchor = (anchor & br_np.astype(bool)) | bc_np.astype(bool)
            index_np = latent_index.cpu().numpy()
            current = np.zeros((64, 64, 64), dtype=bool)
            valid = ((index_np[:, 1:] >= 0) & (index_np[:, 1:] < 64)).all(axis=1)
            current[index_np[valid, 1], index_np[valid, 2], index_np[valid, 3]] = True
            additions = anchor & ~current
            if args.shape_anchor_max_added > 0 and additions.sum() > args.shape_anchor_max_added:
                distance = distance_transform_edt(~current)
                # Preserve requested chair members before spending the cap on a halo.
                core_candidates = np.flatnonzero((additions & anchor_core).ravel())
                halo_candidates = np.flatnonzero((additions & ~anchor_core).ravel())
                core_order = core_candidates[np.argsort(distance.ravel()[core_candidates])]
                halo_order = halo_candidates[np.argsort(distance.ravel()[halo_candidates])]
                chosen = np.concatenate((core_order, halo_order))[:args.shape_anchor_max_added]
                additions = np.zeros_like(additions)
                additions.ravel()[chosen] = True
            expanded = current | additions
            coords = np.argwhere(expanded)
            batch = np.zeros((len(coords), 1), dtype=index_np.dtype)
            latent_index = torch.from_numpy(np.concatenate([batch, coords], axis=1)).to(latent_index.device).to(latent_index.dtype)
            print(f"  [shape-anchor expand/cache] target={args.shape_qd_target} "
                  f"dilation={args.shape_anchor_expand_vox}: +{len(coords)-int(current.sum())} "
                  f"active voxels (total {len(coords)})", flush=True)
    else:
        # Stage 1: dense FlowDPS
        print(f"\n=== Stage 1: dense FlowDPS (steps={args.dense_steps}, eta={args.eta}) ===")
        latent_index = dense_flowdps(pipe, img_prep, br, design, bc,
                                      num_inference_steps=args.dense_steps,
                                      guidance_scale=args.cfg,
                                      eta=args.eta, bc_w=args.bc_w, out_w=args.out_w, dw=args.dw,
                                      env_excess_w=args.env_excess_w,
                                      env_excess_tol=args.env_excess_tol,
                                      dense_token_policy=args.dense_token_policy,
                                      inner_steps=args.inner_steps, seed=args.seed,
                                      mc_threshold=args.dense_index_threshold,
                                      bce_boundary=(args.bce_boundary if args.bce_boundary is not None else args.dense_index_threshold),
                                      bce_mode=args.bce_mode,
                                      use_multi_view=use_mv,
                                      vw=args.vw, vol_target=args.vol_target,
                                      vol_projection=args.vol_projection, vol_proj_w=args.vol_proj_w,
                                      aug_lag=args.aug_lag, lambda_init=args.lambda_init,
                                      mu_aug_lag=args.mu_aug_lag, lambda_alpha=args.lambda_alpha,
                                      heaviside_proj=args.heaviside_proj, beta_init=args.beta_init,
                                      beta_max=args.beta_max, eta_bisect_iter=args.eta_bisect_iter,
                                      sw=args.sw, symmetry_axis=args.symmetry_axis,
                                      pw=args.pw, load_path_mask=load_path_mask,
                                      cw=args.cw, tw=args.tw, min_thick=args.min_thick,
                                      cw_warmup=args.cw_warmup, reach_kernel=args.reach_kernel,
                                      reach_threshold=args.reach_threshold,
                                      reach_iters=args.reach_iters,
                                      rmin_normalize=args.rmin_normalize,
                                      tw_hard=args.tw_hard, tw_soft=args.tw_soft,
                                      tw_rmin=args.tw_rmin, rmin_radius=args.rmin_radius,
                                      rmin_thresh=args.rmin_thresh,
                                      fea_w=args.fea_w, fea_every_n=args.fea_every_n,
                                      fea_warmup=args.fea_warmup,
                                      fea_domain_dir=args.fea_domain_dir,
                                      fea_back_domain_dir=args.fea_back_domain_dir,
                                      fea_back_load_mode=args.fea_back_load_mode,
                                      fea_back_load_magnitude=args.fea_back_load_magnitude,
                                      fea_nodes=fea_nodes,
                                      fea_mesh_cache=args.fea_mesh_cache,
                                      fea_mesh_size=args.fea_mesh_size,
                                      fea_mesh_size_fine=args.fea_mesh_size_fine,
                                      fea_mesh_cache_fine=args.fea_mesh_cache_fine,
                                      fea_fine_warmup=args.fea_fine_warmup,
                                      fea_penal=args.fea_penal,
                                      snapshot_every=args.snapshot_every,
                                      snapshot_dir=f"{args.out}/snapshots" if args.snapshot_every > 0 else None,
                                      env_origin=env_origin, env_pitch=env_pitch,
                                      loss_csv=loss_csv,
                                      dense_opt=args.dense_opt,
                                      dense_lr=args.dense_lr,
                                      grad_normalize=not args.no_grad_normalize,
                                      shape_qd_archive=args.shape_qd_archive,
                                      shape_qd_target=args.shape_qd_target,
                                      shape_qd_w=args.shape_qd_w,
                                      shape_qd_warmup=args.shape_qd_warmup,
                                      image_proj_target=args.image_proj_target,
                                      image_proj_w=args.image_proj_w,
                                      image_proj_warmup=args.image_proj_warmup,
                                      shape_anchor_bank=args.shape_anchor_bank,
                                      shape_anchor_w=args.shape_anchor_w,
                                      shape_scaffold_w=args.shape_scaffold_w,
                                      shape_scaffold_pool=args.shape_scaffold_pool,
                                      shape_residual_w=args.shape_residual_w,
                                      shape_residual_pool=args.shape_residual_pool,
                                      shape_residual_delta=args.shape_residual_delta,
                                      shape_residual_neutral_w=args.shape_residual_neutral_w,
                                      shape_transport_radius=args.shape_transport_radius,
                                      shape_transport_dims=args.shape_transport_dims,
                                      shape_transport_ridge=args.shape_transport_ridge,
                                      shape_transport_max_rel=args.shape_transport_max_rel,
                                      shape_transport_every=args.shape_transport_every,
                                      shape_transport_max_updates=args.shape_transport_max_updates,
                                      oc_flow_w=args.oc_flow_w,
                                      oc_flow_warmup=args.oc_flow_warmup,
                                      oc_flow_max_rel=args.oc_flow_max_rel)
    
        # === Stress-aware index expansion (also activate the high-stress region of FEA ∂C/∂ρ) ===
        if args.fea_stress_expand:
            from fea_compliance_loss import LAST_FEA_DC_VOXEL
            if LAST_FEA_DC_VOXEL is not None:
                from scipy import ndimage as _ndi
                D = 64
                stress_abs = np.abs(LAST_FEA_DC_VOXEL)
                # inside the bracket envelope mask (br), excluding the area around the BC peg
                br_np = br.cpu().numpy() if hasattr(br, 'cpu') else np.asarray(br)
                bc_np = bc.cpu().numpy() if hasattr(bc, 'cpu') else np.asarray(bc)
                bc_dilated = _ndi.binary_dilation(bc_np.astype(bool), iterations=5)
                valid_region = br_np.astype(bool) & ~bc_dilated
                # percentile threshold (top (100-p)%)
                valid_stress = stress_abs[valid_region]
                if valid_stress.size:
                    thresh = np.percentile(valid_stress, args.stress_percentile)
                    high_stress_mask = (stress_abs > thresh) & valid_region
                    stress_expanded = _ndi.binary_dilation(high_stress_mask, iterations=args.stress_expand_vox)
                    # union with current latent_index
                    coords_np = latent_index.cpu().numpy()
                    in_range = ((coords_np[:, 1:] >= 0) & (coords_np[:, 1:] < D)).all(axis=1)
                    ci = coords_np[in_range]
                    cur_occ = np.zeros((D, D, D), dtype=bool)
                    cur_occ[ci[:, 1], ci[:, 2], ci[:, 3]] = True
                    new_occ = cur_occ | stress_expanded
                    new_coords = np.argwhere(new_occ)
                    batch_col = np.zeros((len(new_coords), 1), dtype=coords_np.dtype)
                    new_idx_np = np.concatenate([batch_col, new_coords], axis=1)
                    added = len(new_coords) - len(ci)
                    print(f"  [stress-expand] high-stress voxels (top {100-args.stress_percentile:.0f}% |∂C/∂ρ|): {int(high_stress_mask.sum())}, +{added} new active voxels (total {len(new_coords)})")
                    latent_index = torch.from_numpy(new_idx_np).to(latent_index.device).to(latent_index.dtype)
            else:
                print("  [stress-expand] SKIP — no cached FEA gradient (fea_w=0?)")
    
        # === Thin-region selective index expansion (sparse stage refines over a broader region) ===
        if args.thin_expand_vox > 0:
            from scipy import ndimage as _ndi
            coords_np = latent_index.cpu().numpy()
            D = 64  # dense grid resolution
            occ_dense = np.zeros((D, D, D), dtype=bool)
            # latent_index columns: (B, Z, Y, X) — keep only those in [0, D)
            in_range = ((coords_np[:, 1:] >= 0) & (coords_np[:, 1:] < D)).all(axis=1)
            ci = coords_np[in_range]
            occ_dense[ci[:, 1], ci[:, 2], ci[:, 3]] = True
            # EDT: distance from each solid voxel to nearest empty (voxel units = half-thickness)
            edt_vox = _ndi.distance_transform_edt(occ_dense)
            thin_solid = occ_dense & (edt_vox < (args.thin_thresh_vox + 0.5))
            thin_expanded = _ndi.binary_dilation(thin_solid, iterations=args.thin_expand_vox)
            new_occ = occ_dense | thin_expanded
            new_coords = np.argwhere(new_occ)
            batch_col = np.zeros((len(new_coords), 1), dtype=coords_np.dtype)
            new_idx_np = np.concatenate([batch_col, new_coords], axis=1)
            added = len(new_coords) - len(ci)
            print(f"  [thin-expand] thin solid: {int(thin_solid.sum())}/{int(occ_dense.sum())} voxels, +{added} new voxels (total {len(new_coords)})")
            latent_index = torch.from_numpy(new_idx_np).to(latent_index.device).to(latent_index.dtype)

        # Shape-anchor support expansion: a sparse decoder cannot grow a branch in a region
        # absent from latent_index.  Seed the target prototype's coarse support before sparse.
        if args.shape_anchor_expand_vox > 0:
            if not args.shape_anchor_bank or args.shape_qd_target < 0:
                raise ValueError('--shape-anchor-expand-vox needs --shape-anchor-bank and --shape-qd-target')
            from scipy.ndimage import binary_dilation as _anchor_dilate
            _anchor_np = np.load(args.shape_anchor_bank)['prototypes'][args.shape_qd_target] > .5
            _anchor_core = _anchor_np.copy()
            _anchor_np = _anchor_dilate(_anchor_np, iterations=args.shape_anchor_expand_vox)
            _br_np_anchor = br.cpu().numpy() if hasattr(br, 'cpu') else np.asarray(br)
            _bc_np_anchor = bc.cpu().numpy() if hasattr(bc, 'cpu') else np.asarray(bc)
            _anchor_np = (_anchor_np & _br_np_anchor.astype(bool)) | _bc_np_anchor.astype(bool)
            _idx_anchor = latent_index.cpu().numpy(); _cur_anchor = np.zeros((64,64,64), dtype=bool)
            _ok_anchor = ((_idx_anchor[:,1:] >= 0) & (_idx_anchor[:,1:] < 64)).all(axis=1)
            _cur_anchor[_idx_anchor[_ok_anchor,1], _idx_anchor[_ok_anchor,2], _idx_anchor[_ok_anchor,3]] = True
            _add_anchor = _anchor_np & ~_cur_anchor
            if args.shape_anchor_max_added > 0 and _add_anchor.sum() > args.shape_anchor_max_added:
                from scipy.ndimage import distance_transform_edt as _anchor_edt
                _dist_anchor = _anchor_edt(~_cur_anchor)
                _core_candidates = np.flatnonzero((_add_anchor & _anchor_core).ravel())
                _halo_candidates = np.flatnonzero((_add_anchor & ~_anchor_core).ravel())
                _core_order = _core_candidates[np.argsort(_dist_anchor.ravel()[_core_candidates])]
                _halo_order = _halo_candidates[np.argsort(_dist_anchor.ravel()[_halo_candidates])]
                _take_anchor = np.concatenate((_core_order, _halo_order))[:args.shape_anchor_max_added]
                _add_anchor = np.zeros_like(_add_anchor); _add_anchor.ravel()[_take_anchor] = True
            _union_anchor = _cur_anchor | _add_anchor
            _coords_anchor = np.argwhere(_union_anchor)
            _bat_anchor = np.zeros((len(_coords_anchor),1), dtype=_idx_anchor.dtype)
            latent_index = torch.from_numpy(np.concatenate([_bat_anchor,_coords_anchor],axis=1)).to(latent_index.device).to(latent_index.dtype)
            print(f"  [shape-anchor expand] target={args.shape_qd_target} dilation={args.shape_anchor_expand_vox}: "
                  f"+{len(_coords_anchor)-int(_cur_anchor.sum())} active voxels (total {len(_coords_anchor)})", flush=True)
    
        # === Dense→Sparse interface TO pruning (FEA sensitivity, Option B) ===
        # For each active voxel, compute |∂C/∂ρ|; voxels in the bottom percentile (= low load
        # contribution) are pruned. BC voxels are always preserved. Cleans dead-end tentacles
        # before sparse stage so topology_preservation can't re-grow them.
        if args.to_prune_percentile > 0 and args.fea_w > 0:
            try:
                from fea_compliance_loss import fea_compliance_loss
                from scipy import ndimage as _ndi
                coords_np = latent_index.cpu().numpy()
                occ_bin = np.zeros((64, 64, 64), dtype=np.float32)
                occ_bin[coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]] = 1.0
                br_np = br.cpu().numpy() if hasattr(br, 'cpu') else np.asarray(br)
                bc_np = bc.cpu().numpy() if hasattr(bc, 'cpu') else np.asarray(bc)
                occ_bin = occ_bin * br_np.astype(np.float32)  # clip to envelope
    
                # Build "sharp" logits so sigmoid → near-binary (solid≈0.99, empty≈0.01)
                logits = torch.from_numpy((occ_bin * 2 - 1) * 5.0).to('cuda').float().requires_grad_(True)
                comp = fea_compliance_loss(logits, br_np, fea_nodes,
                                            args.fea_domain_dir or '/tmp/fea_dom_prune',
                                            args.fea_mesh_cache,
                                            mesh_size=args.fea_mesh_size, penal=args.fea_penal)
                comp.backward()
                sens_mag = logits.grad.detach().abs().cpu().numpy()  # (64,64,64)
    
                n_before = int((occ_bin > 0.5).sum())
                active_non_bc = (occ_bin > 0.5) & ~bc_np.astype(bool)
                if active_non_bc.sum() > 0:
                    thresh = float(np.percentile(sens_mag[active_non_bc], args.to_prune_percentile))
                    # Keep BC always + active voxels with |∂C/∂ρ| ≥ threshold
                    keep = bc_np.astype(bool) | ((occ_bin > 0.5) & (sens_mag >= thresh))
                    # Optional: drop tiny floaters (component not touching BC)
                    lbl, _ = _ndi.label(keep)
                    bc_labels = np.unique(lbl[bc_np.astype(bool)]); bc_labels = bc_labels[bc_labels > 0]
                    keep_main = np.isin(lbl, bc_labels)
                    n_after = int(keep_main.sum())
                    new_coords = np.argwhere(keep_main)
                    batch_col = np.zeros((len(new_coords), 1), dtype=coords_np.dtype)
                    latent_index = torch.from_numpy(
                        np.concatenate([batch_col, new_coords], axis=1)
                    ).to(latent_index.device).to(latent_index.dtype)
                    print(f"  [TO-prune] FEA sensitivity at {args.to_prune_percentile}th pct → "
                          f"thr={thresh:.3e}  voxels {n_before:,} → {n_after:,} "
                          f"(removed {n_before-n_after:,}: low-sens + disconnected)", flush=True)
                else:
                    print(f"  [TO-prune] no non-BC active voxels — skipping", flush=True)
            except Exception as e:
                import traceback as _tb
                print(f"  [TO-prune] FAIL: {type(e).__name__}: {e}\n{_tb.format_exc()[:500]}", flush=True)
    
        # === Restrict active set to envelope (sparse can't generate outside) ===
        # Keep only latent_index voxels whose CENTER is inside the actual envelope
        # STL (pysdf test at 64³ voxel centers — NOT a pre-rasterized voxel blob),
        # UNION with the fix/load peg STL interiors so peg attachment survives.
        # The sparse refiner then physically cannot place material beyond the envelope.
        if getattr(args, 'restrict_active_to_envelope', False):
            from pysdf import SDF as _SDF_env
            D = 64
            # 64³ voxel-center world coords (same frame as env_origin/env_pitch)
            _ax = (np.arange(D, dtype=np.float64) + 0.5) * env_pitch[0] + env_origin[0]
            _ay = (np.arange(D, dtype=np.float64) + 0.5) * env_pitch[1] + env_origin[1]
            _az = (np.arange(D, dtype=np.float64) + 0.5) * env_pitch[2] + env_origin[2]
            _X, _Y, _Z = np.meshgrid(_ax, _ay, _az, indexing='ij')
            _q = np.stack([_X, _Y, _Z], axis=-1).reshape(-1, 3).astype(np.float32)
            _env_stl = _load_sdf_mesh(args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl')
            # dense-side dilation: keep voxels inside envelope OR up to N voxels outside it,
            # so the refiner can form thicker boundary features. pysdf: inside>0, so the
            # threshold is -(dilate_vox * mean pitch). The sparse force-envelope-clip still
            # trims the final MC to the TRUE envelope (env_sdf>0).
            _dil_m = float(getattr(args, 'restrict_dilate_vox', 0.0)) * float(np.mean(env_pitch))
            _d_env = _check_sdf_range(
                'envelope-active-gate',
                _SDF_env(_env_stl.vertices.astype(np.float32),
                         _env_stl.faces.astype(np.uint32))(_q), D).reshape(D, D, D)
            _keep_mask = (_d_env > -_dil_m)
            # SELECTIVE dilation: additionally open the gate near THIN dense features only.
            # Thin = active voxels erased by morphological opening (features thinner than
            # ~2k+1 dense voxels, k from --thin-target-vox512). Near-thin voxels get the
            # dilated gate; thick regions keep the true envelope → less uniform bloat.
            _dil_thin = float(getattr(args, 'restrict_dilate_thin', 0.0))
            if _dil_thin > 0:
                from scipy.ndimage import binary_dilation as _bd
                _occ64 = np.zeros((D, D, D), dtype=bool)
                _ci0 = latent_index.cpu().numpy()
                _inr0 = ((_ci0[:, 1:] >= 0) & (_ci0[:, 1:] < D)).all(axis=1)
                _in0 = _ci0[_inr0]
                _occ64[_in0[:, 1], _in0[:, 2], _in0[:, 3]] = True
                # thin detection on the REAL structure only: envelope-inside occupancy.
                # Raw occupancy includes the dense BC-forced blob (bc_proper peg dilated
                # ×4 → ~16-vox fake-thick region) — mask to envelope first.
                # LOCAL-THICKNESS criterion: a voxel is thin
                # iff no deep core (inscribed radius ≥ thr) exists within its 5³ window.
                # Unlike morphological opening this does NOT flag edges/corners of thick
                # bodies — only genuine 1-2-voxel plates/struts.
                from scipy.ndimage import distance_transform_edt as _edtf, maximum_filter as _mff
                _occ_in = _occ64 & (_d_env > 0)
                _t_dense = float(getattr(args, 'thin_target_vox512', 16.0)) / 8.0   # 512-vox → 64-vox
                _core_thr = _t_dense / 2.0 + 0.5      # target 16vox512(2 cells) → core ≤ 1.5
                _edt = _edtf(_occ_in)
                _core = _mff(_edt, size=5)
                _thin = _occ_in & (_core <= _core_thr)
                _near_thin = _bd(_thin, iterations=int(max(1, round(_dil_thin))))
                _dil_thin_m = _dil_thin * float(np.mean(env_pitch))
                # peg proximity: ALWAYS open the gate near fix/load — the attachment
                # region must be growable regardless of the thin detector. Separate
                # (typically larger) dilation than the thin gate.
                _dil_peg = float(getattr(args, 'restrict_dilate_peg', -1.0))
                if _dil_peg < 0: _dil_peg = _dil_thin
                _dil_peg_m = _dil_peg * float(np.mean(env_pitch))
                _d_fix_g = _check_sdf_range('fix-active-gate', _SDF_env(*(lambda m: (
                    m.vertices.astype(np.float32), m.faces.astype(np.uint32)))(
                    _load_sdf_mesh(args.fix_stl)))(_q), D).reshape(D, D, D)
                _d_load_g = _check_sdf_range('load-active-gate', _SDF_env(*(lambda m: (
                    m.vertices.astype(np.float32), m.faces.astype(np.uint32)))(
                    _load_sdf_mesh(args.load_stl)))(_q), D).reshape(D, D, D)
                _near_peg = np.maximum(_d_fix_g, _d_load_g) > -_dil_peg_m
                _add = ((_d_env > -_dil_thin_m) & (~_keep_mask) & _near_thin) \
                     | ((_d_env > -_dil_peg_m) & (~_keep_mask) & _near_peg)
                _keep_mask = _keep_mask | _add
                print(f"  [restrict-thin] target={getattr(args,'thin_target_vox512',16.0):.0f}vox512 "
                      f"(local-thickness core≤{_core_thr:.1f}, env-masked occ): thin={int(_thin.sum()):,} "
                      f"near={int(_near_thin.sum()):,} near_peg={int(_near_peg.sum()):,} "
                      f"gate+={int(_add.sum()):,} voxels (dilate {_dil_thin:.1f}vox)", flush=True)
            # union peg STL interiors (fix, load) so BC voxels survive
            for _pn in (args.fix_stl, args.load_stl):
                _pm = _load_sdf_mesh(_pn)
                _keep_mask |= (_SDF_env(_pm.vertices.astype(np.float32),
                                        _pm.faces.astype(np.uint32))(_q) > 0).reshape(D, D, D)
            _ci = latent_index.cpu().numpy()
            _inr = ((_ci[:, 1:] >= 0) & (_ci[:, 1:] < D)).all(axis=1)
            _in = _ci[_inr]
            _keep = _keep_mask[_in[:, 1], _in[:, 2], _in[:, 3]]
            _n0 = len(_in); _n1 = int(_keep.sum())
            latent_index = torch.from_numpy(_in[_keep]).to(latent_index.device).to(latent_index.dtype)
            print(f"  [restrict-envelope] STL-SDF inside test (dilate={getattr(args,'restrict_dilate_vox',0.0):.1f}vox="
                  f"{_dil_m*1000:.2f}mm): active voxels {_n0:,} → {_n1:,} "
                  f"(dropped {_n0-_n1:,} outside envelope∪pegs)", flush=True)

        # The dense token field can contain tiny, detached components even when
        # fixed/load BC are mutually reachable.  They are not a valid structural
        # alternative and become sparse-stage debris if passed onward.  Keep every
        # 6-connected component anchored to either physical BC; a true BC bridge
        # remains the responsibility of reachability_loss (--cw), not this filter.
        if getattr(args, 'dense_keep_bc_components', False):
            from scipy.ndimage import label as _cc_label
            _tok = latent_index.detach().cpu().numpy()
            _active = np.zeros((64, 64, 64), dtype=bool)
            _valid = ((_tok[:, 1:] >= 0) & (_tok[:, 1:] < 64)).all(axis=1)
            _tok = _tok[_valid]
            _active[_tok[:, 1], _tok[:, 2], _tok[:, 3]] = True
            _bc_np = (bc.detach().cpu().numpy() if hasattr(bc, 'detach') else np.asarray(bc)).astype(bool)
            _structure = np.zeros((3, 3, 3), dtype=np.uint8)
            _structure[1, 1, :] = 1; _structure[1, :, 1] = 1; _structure[:, 1, 1] = 1
            _labels, _n_comp = _cc_label(_active, structure=_structure)
            _anchored = np.unique(_labels[_bc_np & (_labels > 0)])
            _keep_active = np.isin(_labels, _anchored)
            _kept = _tok[_keep_active[_tok[:, 1], _tok[:, 2], _tok[:, 3]]]
            if len(_anchored) > 0 and len(_kept) > 0:
                print(f"  [dense-bc-components] {_n_comp} active components → {len(_anchored)} BC-anchored; "
                      f"tokens {len(_tok):,} → {len(_kept):,}", flush=True)
                latent_index = torch.from_numpy(_kept).to(latent_index.device).to(latent_index.dtype)
            else:
                print("  [dense-bc-components] no BC-anchored component found; leaving tokens unchanged", flush=True)

        latent_index = sort_block(latent_index, pipe.sparse_dit_512.selection_block_size)

        # Save dense-stage MC mesh (envelope-masked, in envelope world frame)
        dense_normal_64 = None
        try:
            from skimage import measure as _measure
            coords64 = latent_index.cpu().numpy()
            occ64 = np.zeros((64, 64, 64), dtype=np.float32)
            occ64[coords64[:, 1], coords64[:, 2], coords64[:, 3]] = 1.0
            bracket_np = br.cpu().numpy() if hasattr(br, 'cpu') else br
            # mesh_dense_raw.obj — PRE-filter tokens, NO bracket mask (raw dense as-is)
            try:
                _pre = getattr(dense_flowdps, '_prefilter_idx', None)
                if _pre is not None:
                    _occ_raw = np.zeros((64, 64, 64), dtype=np.float32)
                    _pin = _pre[((_pre[:, 1:] >= 0) & (_pre[:, 1:] < 64)).all(axis=1)]
                    _occ_raw[_pin[:, 1], _pin[:, 2], _pin[:, 3]] = 1.0
                    _vr, _fr, _, _ = _measure.marching_cubes(_occ_raw, level=0.5, method='lewiner')
                    _vr_w = env_origin + (_vr + 0.5) * env_pitch
                    trimesh.Trimesh(_vr_w, _fr).export(str(Path(args.out) / 'mesh_dense_raw.obj'))
                    print(f"  [dense MC] mesh_dense_raw.obj saved (pre-filter tokens={len(_pin):,}, no mask)", flush=True)
            except Exception as _e:
                print(f"  [dense MC] mesh_dense_raw skipped: {_e}", flush=True)
            occ_masked = occ64 * bracket_np.astype(np.float32)
            v_d, f_d, n_d_verts, _ = _measure.marching_cubes(occ_masked, level=0.5, method='lewiner')
            # Per-cell outward normal: face-area-weighted mean over MC faces, bucketed by face centroid → 64³ cell.
            # n_d_verts (skimage 'descent' default) points away from higher density = OUTWARD from solid.
            v_d_face = v_d[f_d]                                          # (F, 3, 3)
            face_cent = v_d_face.mean(axis=1)                            # (F, 3) — in 64³ (Z, Y, X)
            n_face = n_d_verts[f_d].mean(axis=1)                         # (F, 3)
            n_face = n_face / np.linalg.norm(n_face, axis=-1, keepdims=True).clip(1e-9)
            e1 = v_d_face[:, 1] - v_d_face[:, 0]
            e2 = v_d_face[:, 2] - v_d_face[:, 0]
            face_area = np.linalg.norm(np.cross(e1, e2), axis=-1) / 2.0  # (F,)
            ci_z = np.clip(face_cent[:, 0].astype(int), 0, 63)
            ci_y = np.clip(face_cent[:, 1].astype(int), 0, 63)
            ci_x = np.clip(face_cent[:, 2].astype(int), 0, 63)
            flat = (ci_z * 64 + ci_y) * 64 + ci_x
            accum = np.zeros((64*64*64, 3), dtype=np.float32)
            np.add.at(accum, flat, n_face * face_area[:, None])
            dense_normal_64 = torch.from_numpy(accum.reshape(64, 64, 64, 3)).to('cuda')
            _n_cells = int((np.linalg.norm(accum, axis=-1) > 1e-4).sum())
            print(f"  [dense MC] dense_normal_64 built: {_n_cells} cells with valid normal", flush=True)
            # Reuse env_origin / env_pitch computed before dense_flowdps_inference (same convention as snapshots).
            v_d_world = env_origin + (v_d + 0.5) * env_pitch
            mesh_dense = trimesh.Trimesh(v_d_world, f_d)
            if getattr(args, 'dense_keep_largest_mesh_component', False):
                _parts = mesh_dense.split(only_watertight=False)
                if len(_parts) > 1:
                    _areas = [float(part.area) for part in _parts]
                    _keep_i = int(np.argmax(_areas))
                    mesh_dense = _parts[_keep_i]
                    print(f"  [dense-mesh-components] {len(_parts)} MC components → largest BC-bearing "
                          f"surface (area {mesh_dense.area:.6g} m²; removed {len(_parts)-1} tiny debris components)", flush=True)
            out_d = Path(args.out); out_d.mkdir(parents=True, exist_ok=True)
            mesh_dense.export(str(out_d / 'mesh_dense.obj'))
            print(f"  [dense MC] saved mesh_dense.obj V={len(mesh_dense.vertices):,} F={len(mesh_dense.faces):,} "
                  f"(envelope-masked, world frame)", flush=True)
        except Exception as e:
            import traceback as _tb
            print(f"  mesh_dense save FAIL: {e}\n{_tb.format_exc()[:500]}", flush=True)
    
        # Aggressively free dense stage memory
        pipe.dense_vae.cpu(); pipe.dense_dit.cpu(); pipe.dense_image_encoder.cpu()
        import gc; gc.collect(); torch.cuda.empty_cache()
        print(f"after dense cleanup: cuda allocated {torch.cuda.memory_allocated()/1e9:.2f} GB")
        # Save dense cache for future fast restart
        if args.save_dense_cache:
            Path(args.save_dense_cache).parent.mkdir(parents=True, exist_ok=True)
            _dn64_np = dense_normal_64.cpu().numpy() if dense_normal_64 is not None else np.zeros((64,64,64,3), dtype=np.float32)
            np.savez(args.save_dense_cache,
                     latent_index=latent_index.cpu().numpy(),
                     dense_normal_64=_dn64_np)
            # Also copy mesh_dense.obj alongside cache for reuse
            _dst_mesh = str(Path(args.save_dense_cache).with_suffix("")) + "_mesh.obj"
            try:
                import shutil; shutil.copyfile(str(Path(args.out)/"mesh_dense.obj"), _dst_mesh)
            except Exception: pass
            print(f"  [dense-cache] saved → {args.save_dense_cache}")

    if args.skip_sparse:
        # Copy dense MC mesh as the final output so downstream tooling sees mesh.obj
        try:
            import shutil
            src = Path(args.out) / 'mesh_dense.obj'
            dst = Path(args.out) / 'mesh.obj'
            if src.exists():
                shutil.copyfile(str(src), str(dst))
                print(f"[skip_sparse] dense-only run: copied mesh_dense.obj → mesh.obj")
        except Exception as e:
            print(f"[skip_sparse] copy failed: {e}")
        return

    # Give the sparse decoder degrees of freedom just outside the coarse dense
    # surface, without changing the dense mesh or its topology target.  This
    # placement also applies when Stage 1 was loaded from a dense cache.
    _sparse_reference_index = latent_index
    if args.sp_support_halo_vox:
        from scipy.ndimage import binary_dilation as _halo_dilate
        _old_idx = latent_index.detach().cpu().numpy()
        _old_occ = np.zeros((64, 64, 64), dtype=bool)
        _old_occ[_old_idx[:, 1], _old_idx[:, 2], _old_idx[:, 3]] = True
        _new_occ = _halo_dilate(_old_occ, iterations=args.sp_support_halo_vox)
        _new_coords = np.argwhere(_new_occ)
        _new_idx = np.column_stack((np.zeros(len(_new_coords), dtype=_old_idx.dtype), _new_coords))
        latent_index = sort_block(torch.from_numpy(_new_idx).to(_sparse_reference_index.device),
                                  pipe.sparse_dit_512.selection_block_size)
        print(f"  [sparse-support halo] {int(_old_occ.sum()):,} -> {len(_new_coords):,} "
              f"active tokens; dense topology target unchanged", flush=True)

    # ========== Sparse FEA hook setup (ported from sparse_flowdps_localattn.py) ==========
    sp_fea_cache = None
    sp_fea_reference = {}
    sp_fea_hook_counter = [0]
    # Common setup (always needed when sp_fea_w > 0, regardless of mode):
    if args.sp_fea_w > 0 or args.posthoc_fea_steps > 0:
        if args.sp_fea_volume_neutral:
            os.environ['SP_FEA_VOLUME_NEUTRAL'] = '1'
            print('  [sparse FEA] volume-neutral sensitivity projection ACTIVE', flush=True)
        else:
            os.environ.pop('SP_FEA_VOLUME_NEUTRAL', None)
        from fea_compliance_loss import voxel_nodes_from_stl, fea_compliance_loss
        bracket_mask_np, nodes_inside_np, pitch_m = voxel_nodes_from_stl(
            args.bracket_occ, args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl')
        nodes_inside_np = fea_nodes_in_domain_frame(nodes_inside_np, args.fea_node_alignment,
                                                     "sparse grid → FEA domain")
        from pathlib import Path as _P
        import shutil as _sh
        _dom_dir = args.fea_domain_dir or '/tmp/fea_dom_sparse'
        _P(_dom_dir).mkdir(parents=True, exist_ok=True)
        _srcmap = {'original_DesignSpace.stl': (args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl'),
                   'fixed.stl': args.fix_stl, 'load.stl': args.load_stl}
        for _f in ['original_DesignSpace.stl', 'fixed.stl', 'load.stl']:
            _src = _srcmap[_f]
            _dst = f'{_dom_dir}/{_f}'
            if not _P(_dst).exists() and _P(_src).exists():
                _sh.copy(_src, _dst)
        _bc_d = np.load(args.bc_proper)
        bc_mask_64 = (_bc_d['fix'].astype(bool) | _bc_d['load'].astype(bool))
        if args.fea_back_domain_dir:
            if 'back_load' not in _bc_d.files:
                raise ValueError('dual FEA requires back_load in --bc-proper NPZ')
            bc_mask_64 |= _bc_d['back_load'].astype(bool)
        # force-bc-solid: use the SAME STL-SDF BC definition as generation (dilate margin),
        # built at 64³ (FEA grid). Consistent meshgrid convention with make_bc_proper_pysdf.
        if _bc_hi.get('active'):
            _bc64_stl = _build_bc_mask(64)
            if _bc64_stl is not None:
                bc_mask_64 = (_bc64_stl > 0)   # SDF field → inside bool for FEA
                print(f"  [sparse FEA] BC from STL-SDF (dilate={getattr(args,'force_bc_dilate_mm',0):.1f}mm): "
                      f"{int(bc_mask_64.sum())} of 64³ voxels", flush=True)
        _bb_d = np.load(args.bracket_occ)
        if 'origin' in _bb_d.files:
            _env_center = np.zeros(3, dtype=np.float64); _env_longest_half = 1.0
            _env_origin = _bb_d['origin']
        else:
            _env = trimesh.load(args.fea_bracket_stl or 'data_real/bracket/original_DesignSpace.stl', force='mesh')
            _env_center = (_env.bounds[0] + _env.bounds[1]) / 2
            _env_longest_half = float((_env.bounds[1] - _env.bounds[0]).max() / 2)
            _env_origin = _env.bounds[0] - 0.05 * (_env.bounds[1] - _env.bounds[0]).max()
        spatial_cache = {
            'env_center': torch.tensor(_env_center, dtype=torch.float32),
            'longest_half': torch.tensor(_env_longest_half, dtype=torch.float32),
            'env_origin': torch.tensor(_env_origin, dtype=torch.float32),
            'env_pitch': float(pitch_m), 'R64': 64, 'sparse_res': 512,
        }
        # When sp_shell_only, build deep_int mask so the FEA aggregator can fill those
        # cells with a strong-inside logit (else they have cnts=0 → wrong occupancy).
        # Note: dense_active_64 is only defined later (line ~2006). At this point
        # latent_index already exists, so derive the active mask directly from it.
        if args.sp_shell_only:
            from scipy.ndimage import binary_erosion as _be_fea
            _da_for_fea = np.zeros((64, 64, 64), dtype=bool)
            _li_np = latent_index.cpu().numpy()
            _da_for_fea[_li_np[:, 1], _li_np[:, 2], _li_np[:, 3]] = True
            _deep_np_fea = _be_fea(_da_for_fea,
                                    iterations=max(1, args.sp_interior_erode)).astype(np.float32)
        else:
            _deep_np_fea = np.zeros_like(bracket_mask_np, dtype=np.float32)
        sp_fea_cache = (bracket_mask_np, nodes_inside_np,
                         args.fea_domain_dir or '/tmp/fea_dom_sparse', bc_mask_64, spatial_cache,
                         _deep_np_fea)
        print(f'  [sparse FEA] bracket_mask={int(bracket_mask_np.sum())}, '
              f'nodes={len(nodes_inside_np)}, BC={int(bc_mask_64.sum())}'
              f'{", deep_int_force=" + str(int(_deep_np_fea.sum())) if args.sp_shell_only else ""}',
              flush=True)

        # monkey-patch sparse scheduler step
        vae512 = pipe.sparse_vae_512
        decoder = vae512.decoder
        sched = pipe.sparse_scheduler_512
        orig_step = sched.step
        step_holder = [0]
        # The scheduler is called once per configured sparse denoising step.
        # This used to be hard-coded to 30, so 50-step runs reached a progress
        # fraction of 1.67 and used the wrong warmup/decay schedule.
        SPARSE_TOTAL = max(1, int(args.sparse_steps))
        latent_index_for_hook = [latent_index]   # captured

        def patched_step(noise_pred, t, latents, **kwargs):
            res = orig_step(noise_pred, t, latents, **kwargs)
            step_holder[0] += 1
            step_frac = step_holder[0] / SPARSE_TOTAL
            if step_frac < args.sp_fea_warmup: return res
            if step_holder[0] % args.sp_fea_every != 0: return res
            try:
                with torch.enable_grad():
                    # train() so torchsparse builds backward kmap metadata (eval() leaves it None,
                    # which makes conv_forward_implicit_gemm backward kernel fail at runtime).
                    decoder.train()
                    # fp16 chain — fea_compliance_loss handles fp16↔numpy cast internally
                    lat = res.prev_sample.detach().clone().requires_grad_(True)
                    unscaled = (1.0 / vae512.latents_scale * lat + vae512.latents_shift).to(lat.dtype)
                    sparse_lat = sp.SparseTensor(unscaled, latent_index_for_hook[0].int())
                    output = decoder(sparse_lat, factor=None, return_feat=False)
                    sdf = output.feats
                    out_coords = output.coords[:, 1:]
                    R64 = 64
                    br_np, nd_np, dom, bc64, smap, _deep_np_fea = sp_fea_cache
                    br_t = torch.from_numpy(br_np).to(sdf.device)
                    bc_t = torch.from_numpy(bc64).to(sdf.device)
                    _deep_t = torch.from_numpy(_deep_np_fea).bool().to(sdf.device)
                    # sparse decoder output is on 512^3 grid; envelope mask br_t is 64^3 in the
                    # same normalized cube → direct downsample: out_coords / 8 → xyz64.
                    xyz64 = (out_coords.long() // (int(smap['sparse_res']) // R64)).clamp(0, R64-1)
                    inside_act = br_t[xyz64[:,0], xyz64[:,1], xyz64[:,2]]
                    xyz64_in = xyz64[inside_act]
                    sdf_in = sdf.squeeze(-1)[inside_act]
                    flat_idx = xyz64_in[:,0]*R64*R64 + xyz64_in[:,1]*R64 + xyz64_in[:,2]
                    # Grad-preserving aggregation: out-of-place scatter_add.
                    # in-place scatter_add_/index_add_ strip grad from src; torch.sparse.mm doesn't
                    # backprop into the dense rhs reliably for this shape.
                    N_pts = sdf_in.shape[0]
                    if N_pts > 0:
                        src_vals = (torch.sigmoid(sp_occupancy_logit(sdf_in, args.sp_fea_steepness))
                                    if args.sp_fea_reduce == 'soft_frac' else sdf_in)
                        sums = torch.zeros(R64**3, device=sdf.device, dtype=src_vals.dtype
                                           ).scatter_add(0, flat_idx, src_vals)
                        with torch.no_grad():
                            ones_val = torch.ones(N_pts, device=sdf.device, dtype=src_vals.dtype)
                            cnts = torch.zeros(R64**3, device=sdf.device, dtype=src_vals.dtype
                                               ).scatter_add(0, flat_idx, ones_val)
                    else:
                        sums = torch.zeros(R64**3, device=sdf.device, dtype=sdf.dtype)
                        cnts = torch.zeros(R64**3, device=sdf.device, dtype=sdf.dtype)
                    dense_mean = sums / cnts.clamp(min=1.0)
                    if args.sp_fea_reduce == 'soft_frac':
                        dense = dense_mean.view(R64, R64, R64)
                        occ_logits = torch.logit(dense.clamp(1e-6, 1-1e-6))
                    else:
                        _missing_sdf = 1.0 if args.sp_sdf_inside_low else -1.0
                        dense = torch.where(cnts > 0, dense_mean, torch.tensor(_missing_sdf, device=sdf.device, dtype=sdf.dtype)).view(R64, R64, R64)
                        # paper ref: main text, SIMP-style density rho = sigma(k * sdf), penalization
                        #            exponent p (Supplementary penalization ablation, Table tab:penal).
                        occ_logits = sp_occupancy_logit(dense, args.sp_fea_steepness)
                    # Clamp envelope-inside logits → ρ ∈ [~0.007, ~0.993] so the fenics K matrix
                    # condition number stays bounded (else ρ=1 vs ρ=1e-3 → cond ~1e9 → solver fails).
                    occ_logits = occ_logits.clamp(-5.0, 5.0)
                    occ_logits = torch.where(br_t, occ_logits, torch.tensor(-5.0, device=occ_logits.device, dtype=occ_logits.dtype))
                    # shell-only: force deep_int region as strong inside (sparse didn't decode there)
                    if args.sp_shell_only:
                        occ_logits = torch.where(_deep_t, torch.tensor(5.0, device=occ_logits.device, dtype=occ_logits.dtype), occ_logits)
                    occ_logits = torch.where(bc_t, torch.tensor(5.0, device=occ_logits.device, dtype=occ_logits.dtype), occ_logits)
                    # fp32 cast so backward grad (~1/2e5) doesn't underflow in fp16;
                    # ToFloat backward auto-casts grad back to fp16 for upstream (dense_mean → sdf).
                    occ_logits = occ_logits.float()
                    if os.environ.get('SP_FEA_DEBUG', '0') == '1' and step_holder[0] <= 30:
                        with torch.no_grad():
                            rho_in_br = torch.sigmoid(occ_logits)[br_t]
                            print(f"    [SP_FEA_DEBUG] step {step_holder[0]}: "
                                  f"dense_mean=[{dense_mean.min():.3f},{dense_mean.max():.3f}] "
                                  f"rho_in_br>0.5={int((rho_in_br>0.5).sum())}/{rho_in_br.numel()} "
                                  f"mean={rho_in_br.mean():.4f}", flush=True)
                    # force-bc-solid: pass BC mask so FEA sets ρ=1 at BC and zeros its gradient
                    # (BC = always-solid peg, not a learning target). bc64 is [Z,Y,X] @64³.
                    _fea_bc_mask = bc64.astype(bool) if getattr(args, 'force_bc_solid', False) else None
                    fea_loss, seat_loss, back_loss = paired_fea_compliance(
                        occ_logits, br_np, nd_np, dom, args.fea_back_domain_dir,
                        args.fea_mesh_cache, args.fea_mesh_size, args.fea_penal,
                        sp_fea_reference, args.fea_back_load_mode,
                        args.fea_back_load_magnitude, bc_mask_np=_fea_bc_mask)
                    if not torch.isfinite(fea_loss).item():
                        print(f"    [sp FEA step {step_holder[0]}] SKIP comp={fea_loss.item()} (non-finite)", flush=True)
                    else:
                        decay = max(0.3, 1.0 - 0.7 * step_frac)
                        fea_term = fea_loss / (fea_loss.detach().abs() + 1e-12) if args.sp_fea_normalize else fea_loss
                        loss = args.sp_fea_w * decay * fea_term
                        loss.backward()
                        # apply gradient push, but only if grad is also finite
                        if lat.grad is not None and torch.isfinite(lat.grad).all().item():
                            try: res.prev_sample = res.prev_sample - lat.grad * args.sp_fea_step_size
                            except: pass
                            print(f"    [sp FEA step {step_holder[0]}] comp={fea_loss.item():.3e} decay={decay:.2f} grad_norm={lat.grad.norm().item():.3e}", flush=True)
                            if back_loss is not None:
                                print(f"    [sp dual FEA step {step_holder[0]}] "
                                      f"seat={seat_loss.item():.4e} "
                                      f"back={back_loss.item():.4e} "
                                      f"balanced={fea_loss.item():.4e}", flush=True)
                            elif args.fea_combine_loads:
                                print(f"    [sp simultaneous FEA step {step_holder[0]}] "
                                      f"C={fea_loss.item():.4e}", flush=True)
                        else:
                            print(f"    [sp FEA step {step_holder[0]}] SKIP grad non-finite (comp={fea_loss.item():.3e})", flush=True)
                        # log to loss CSV regardless of grad apply success
                        # cols: stage,step,t,total,fea_comp,l_rmin,l_thick,l_hole,l_interior,l_normalfd,l_lap,l_aniso,l_vol,l_bce,l_reach
                        loss_csv.write(f'sparse_fea,{step_holder[0]},,{float(loss.item()):.6e},'
                                       f'{float(fea_loss.item()):.6e},,,,,,,,,,\n')
                        loss_csv.flush()
            except Exception as e:
                print(f'    [sp FEA] FAIL: {type(e).__name__}: {str(e)[:200]}', flush=True)
                if os.environ.get('SP_FEA_DEBUG', '0') == '1':
                    import traceback as _tb
                    for line in _tb.format_exc().split('\n'):
                        if line.lstrip().startswith('File ') or 'TypeError' in line or 'RuntimeError' in line:
                            print('    [TB] ' + line[:200], flush=True)
            finally:
                decoder.eval()
            return res

        if args.sp_fea_mode == 'manual' and args.sp_fea_w > 0:
            sched.step = patched_step
            print(f'  [sparse FEA] hook installed (every={args.sp_fea_every}, warmup={args.sp_fea_warmup}, '
                  f'mode=manual, step_size={args.sp_fea_step_size})', flush=True)
        elif args.sp_fea_w <= 0:
            print('  [sparse FEA] hook NOT installed (sp_fea_w=0; cache built for posthoc only)', flush=True)
        else:
            print(f'  [sparse FEA] adamw mode — integrated into sparse guide loop (no scheduler hook)', flush=True)

    # ========== End sparse FEA hook ==========

    # =====================================================================
    # Stage 2: sparse512 with GuideFlow3D-style guidance (r_min + thickness)
    # =====================================================================
    # Replace vanilla pipe.inference with custom loop that:
    #   1. Compute base velocity via sparse_dit_512 (frozen) — no_grad CFG
    #   2. Standard scheduler.step → updated latents
    #   3. AdamW... wait actually SGD-based gradient step on latents (persistent
    #      optimizer state across denoise steps so guidance_weight scales linearly)
    #   4. sparse_vae_512.train() so torchsparse builds backward kmap (key fix)
    # Final: refiner + MC
    print(f"\n=== Stage 2: sparse512 with GuideFlow3D guidance ===")
    print(f"  active voxels: {len(latent_index)}")
    # A shape-independent feasibility constraint: preserve the interior of the
    # dense 3D surface when refining it. The dense result carries the macro
    # topology selected by the image/QD stage; sparse sampling may alter only
    # its boundary band. This is a signed-distance constraint over the entire
    # part, not a hand-drawn repair region around particular bosses.
    _dense_core_sdf = None
    _dense_core_cache = {}
    if args.sp_dense_core_mm > 0:
        from pysdf import SDF as _CoreSDF
        _dense_core_path = Path(args.out) / 'mesh_dense.obj'
        if not _dense_core_path.exists():
            raise FileNotFoundError(f'dense core anchor missing: {_dense_core_path}')
        _dense_core_mesh = trimesh.load(str(_dense_core_path), force='mesh', process=False)
        if not _dense_core_mesh.is_watertight:
            raise ValueError(f'dense core anchor must be watertight: {_dense_core_path}')
        _dense_core_sdf = _CoreSDF(_dense_core_mesh.vertices.astype(np.float32),
                                   _dense_core_mesh.faces.astype(np.uint32))
        print(f"  [dense core] anchor={_dense_core_path} margin={args.sp_dense_core_mm:.3f}mm "
              f"weight={args.sp_dense_core_w:.3g}", flush=True)

    def _dense_core_mask(coords):
        if _dense_core_sdf is None:
            return None
        cached = _dense_core_cache.get('coords')
        if cached is not None and cached.shape == coords.shape and torch.equal(cached, coords):
            return _dense_core_cache['mask']
        ijk = coords[:, 1:].detach().cpu().numpy().astype(np.float32)
        pitch512 = np.asarray(env_pitch, dtype=np.float32) * (64.0 / 512.0)
        world = np.asarray(env_origin, dtype=np.float32) + (ijk + 0.5) * pitch512
        distance = _dense_core_sdf(world)
        inside = distance >= args.sp_dense_core_mm / 1000.0
        mask = torch.from_numpy(inside).to(device=coords.device)
        _dense_core_cache['coords'] = coords.detach()
        _dense_core_cache['mask'] = mask
        print(f"  [dense core] protected {int(inside.sum()):,}/{len(inside):,} "
              f"sparse SDF samples ({100*inside.mean():.1f}%)", flush=True)
        return mask
    # === Memory hygiene before sparse stage ===
    # 1) Force eval mode on sparse modules (no dropout, no train-time bookkeeping)
    try: pipe.sparse_dit_512.eval()
    except Exception: pass
    try: pipe.sparse_image_encoder.eval()
    except Exception: pass
    # 2) Pre-warm caching allocator with a large dummy alloc → forces driver to reserve
    #    a big contiguous pool. Free immediately; subsequent small allocs in sparse step 1
    #    reuse this reserved pool → no fragmentation OOM.
    #    Needed because dense peak on wheel (n_views=1) is ~10 GB vs bracket (n_views=6) ~22 GB
    #    → wheel's caching pool is too small entering sparse stage.
    try:
        _free, _total = torch.cuda.mem_get_info()
        # reserve ~70 % of free memory as a single chunk, then free
        _bytes = int(_free * 0.70)
        _dummy = torch.empty(_bytes // 4, dtype=torch.float32, device='cuda')
        del _dummy
        torch.cuda.empty_cache()
        print(f"  [sparse-prewarm] reserved+freed {_bytes/1e9:.1f} GB to expand caching pool")
    except Exception as _e:
        print(f"  [sparse-prewarm] skipped: {type(_e).__name__}: {str(_e)[:80]}")

    # Pre-compute dense active mask at 64³ from latent_index — used by topology_no_hole_loss
    dense_active_64 = torch.zeros(64, 64, 64, dtype=torch.bool, device='cuda')
    coords_np_64 = _sparse_reference_index.cpu().numpy()
    dense_active_64[coords_np_64[:, 1], coords_np_64[:, 2], coords_np_64[:, 3]] = True
    print(f"  dense active mask: {int(dense_active_64.sum())} / {64**3} voxels solid", flush=True)
    _wall_fronts = prepare_side_fronts(dense_active_64) if args.sp_wall_ray_w > 0 else None
    _side_depth_refs = None
    if args.sp_side_depth_w > 0:
        _side_depth_refs = dense_side_depth_reference(
            Path(args.out) / 'mesh_dense.obj', dense_active_64, env_origin, env_pitch,
            tolerance_mm=args.sp_side_depth_tolerance_mm)
        print(f"  [side-depth ref] rays={[r['valid_rays'] for r in _side_depth_refs]} "
              f"mean surface offsets={[round(r['mean_reference_offset'], 2) for r in _side_depth_refs]} "
              f"tolerance={args.sp_side_depth_tolerance_mm:.2f}mm", flush=True)

    # === sp_shell_only — drop deep_int voxels from sparse stage latent_index ===
    # dense_active_64 still keeps all voxels (BG of mesh extraction will fill deep_int as +1.0)
    if args.sp_shell_only:
        from scipy.ndimage import binary_erosion as _be_sh
        _da_np = dense_active_64.cpu().numpy()
        _deep_np = _be_sh(_da_np, iterations=max(1, args.sp_interior_erode))
        _shell_np = _da_np & ~_deep_np
        # Always preserve BC region (fix + load) voxels even if they'd be in deep_int
        _bc_np = bc.cpu().numpy().astype(bool) if hasattr(bc, 'cpu') else np.asarray(bc).astype(bool)
        _keep_64 = _shell_np | _bc_np
        # Filter latent_index by _keep_64
        before = len(latent_index)
        _idx_np = latent_index.cpu().numpy()
        _keep_mask = _keep_64[_idx_np[:, 1], _idx_np[:, 2], _idx_np[:, 3]]
        _filtered = _idx_np[_keep_mask]
        latent_index = torch.from_numpy(_filtered).to(latent_index.device).to(latent_index.dtype)
        # Re-sort blocks after filtering
        latent_index = sort_block(latent_index, pipe.sparse_dit_512.selection_block_size)
        after = len(latent_index)
        print(f"  [shell-only] erode={args.sp_interior_erode}: dropped {before-after}/{before} "
              f"({100*(before-after)/before:.1f}%) deep_int voxels from sparse stage "
              f"(shell={int(_shell_np.sum())}, deep_int={int(_deep_np.sum())}, BC preserved={int(_bc_np.sum())})",
              flush=True)
        # Note: dense_active_64 itself is NOT modified — topology_preservation_loss
        # still uses the full active region for shell/exterior cone definitions.

    # Sparse region BCE masks (mirror of dense 3-region BCE — only if any weight > 0)
    sp_bce_active = (args.sp_bc_w > 0 or args.sp_out_w > 0 or args.sp_design_w > 0
                     or args.sp_bc_buffer_w > 0)
    if sp_bce_active:
        _br_64 = torch.from_numpy(br.cpu().numpy().astype(np.float32)
                                   if hasattr(br, 'cpu') else np.asarray(br).astype(np.float32)).to('cuda')
        _bc_64 = torch.from_numpy(bc.cpu().numpy().astype(np.float32)
                                   if hasattr(bc, 'cpu') else np.asarray(bc).astype(np.float32)).to('cuda')
        _design_64 = (_br_64 - _bc_64).clamp(0, 1)
        _out_64 = 1.0 - _br_64
        # BC buffer ring — dilation(bc, k) AND NOT bc, intersected with bracket envelope
        if args.sp_bc_buffer_w > 0:
            from scipy.ndimage import binary_dilation as _bdil
            _bc_np = bc.astype(bool) if hasattr(bc, 'astype') else np.asarray(bc).astype(bool)
            _br_np = br.astype(bool) if hasattr(br, 'astype') else np.asarray(br).astype(bool)
            _ring_np = _bdil(_bc_np, iterations=int(args.sp_bc_buffer_dilate)) & ~_bc_np & _br_np
            _bc_buffer_64 = torch.from_numpy(_ring_np.astype(np.float32)).to('cuda')
            print(f"  sp BC-buffer: dilate×{args.sp_bc_buffer_dilate} → ring {int(_ring_np.sum())} voxels", flush=True)
        else:
            _bc_buffer_64 = None
        print(f"  sp region BCE: bc_w={args.sp_bc_w} buffer_w={args.sp_bc_buffer_w} "
              f"out_w={args.sp_out_w} design_w={args.sp_design_w} steepness={args.sp_bce_steepness}", flush=True)
        # ── Optional high-res BC masks for sparse BCE only (de-aliases peg centroid) ──
        if args.bc_proper_bce_highres and Path(args.bc_proper_bce_highres).exists():
            from scipy.ndimage import binary_dilation as _bdil_hr
            _hr = np.load(args.bc_proper_bce_highres)
            _br_HR = torch.from_numpy(_hr['bracket'].astype(np.float32)).to('cuda')
            _bc_HR = torch.from_numpy(_hr['bc'].astype(np.float32)).to('cuda')
            _design_HR = (_br_HR - _bc_HR).clamp(0, 1)
            _out_HR = 1.0 - _br_HR
            if args.sp_bc_buffer_w > 0:
                _bc_np_hr = _hr['bc'].astype(bool)
                _br_np_hr = _hr['bracket'].astype(bool)
                _ring_np_hr = _bdil_hr(_bc_np_hr, iterations=int(args.sp_bc_buffer_dilate)) & ~_bc_np_hr & _br_np_hr
                _bc_buffer_HR = torch.from_numpy(_ring_np_hr.astype(np.float32)).to('cuda')
            else:
                _bc_buffer_HR = None
            print(f"  sp BCE high-res mask: {args.bc_proper_bce_highres}  "
                  f"shape={_br_HR.shape}  bc_voxels={int(_bc_HR.sum().item()):,}", flush=True)
        else:
            _br_HR = None; _bc_HR = None; _design_HR = None; _out_HR = None; _bc_buffer_HR = None
    print(f"  sp_guide: lr={args.sp_guide_lr}, w={args.sp_guide_w}, r_min={args.sp_r_min_voxels}, "
          f"thick_target={args.sp_thick_target}", flush=True)
    _shape_qd_sparse = None
    _shape_anchor_sparse = None
    _sp_image_projection = None
    _sp_projection_env = None
    _sp_negative_space = None
    _shape_bc_64 = torch.from_numpy(bc.cpu().numpy().astype(np.float32)
                                       if hasattr(bc, 'cpu') else np.asarray(bc).astype(np.float32)).to('cuda')
    if args.sp_image_proj_w > 0:
        if not args.image_proj_target:
            raise ValueError('--sp-image-proj-w requires --image-proj-target')
        from image_projection_loss import ImageProjectionLoss
        _sp_image_projection = ImageProjectionLoss(args.image_proj_target, 'cuda')
        _sp_projection_env = torch.from_numpy(np.asarray(br, dtype=np.float32)).to('cuda')
        print(f"  sparse image projection: target={args.image_proj_target} "
              f"w={args.sp_image_proj_w}", flush=True)
    if args.sp_negative_space_w > 0:
        _neg_npz = np.load(args.sp_negative_space_mask)
        _neg_np = np.asarray(_neg_npz['mask'], dtype=bool)
        if _neg_np.shape != (64, 64, 64):
            raise ValueError('sp-negative-space-mask must contain mask[64,64,64]')
        _bc_np = np.asarray(bc.cpu().numpy() if hasattr(bc, 'cpu') else bc, dtype=bool)
        if np.any(_neg_np & _bc_np):
            raise ValueError('sp-negative-space-mask overlaps mandatory BC voxels')
        _sp_negative_space = torch.from_numpy(_neg_np).to('cuda')
        print(f"  sparse negative space: mask={args.sp_negative_space_mask} "
              f"voxels={int(_neg_np.sum()):,} w={args.sp_negative_space_w}", flush=True)
    if args.sp_shape_qd_w > 0:
        if not args.shape_qd_archive or args.shape_qd_target < 0:
            raise ValueError('--sp-shape-qd-w needs --shape-qd-archive and --shape-qd-target')
        from shape_qd_loss import FrozenShapePCA
        _shape_qd_sparse = FrozenShapePCA(args.shape_qd_archive, args.shape_qd_target, 'cuda')
        print(f"  sparse Shape-QD: target niche={args.shape_qd_target} w={args.sp_shape_qd_w}", flush=True)
    if args.sp_shape_anchor_w > 0:
        if not args.shape_anchor_bank or args.shape_qd_target < 0:
            raise ValueError('--sp-shape-anchor-w needs --shape-anchor-bank and --shape-qd-target')
        from shape_qd_loss import PrototypeShapeAnchor
        _shape_anchor_sparse = PrototypeShapeAnchor(args.shape_anchor_bank, args.shape_qd_target, 'cuda')
        print(f"  sparse shape anchor: target niche={args.shape_qd_target} w={args.sp_shape_anchor_w}", flush=True)
    sparse_img = img_prep

    # Encode multi-view for sparse stage cond
    with torch.no_grad():
        if use_mv:
            cond_s, uncond_s = encode_multi_view(pipe, sparse_img, pipe.sparse_image_encoder,
                                                  do_classifier_free_guidance=True)
        else:
            cond_s, uncond_s = pipe.encode_image(sparse_img, pipe.sparse_image_encoder,
                                                  do_classifier_free_guidance=True)

    # Optionally override scheduler shift: <1 → dense at LOW t (late, fine-grained),
    # >1 → dense at HIGH t (early, default for D3D-S2). None = keep default.
    if args.sp_shift is not None:
        try:
            pipe.sparse_scheduler_512.config.shift = float(args.sp_shift)
            print(f"  [sparse sched] override shift = {float(args.sp_shift):.3f}", flush=True)
        except Exception as e:
            print(f"  [sparse sched] WARN: shift override failed: {e}", flush=True)
    pipe.sparse_scheduler_512.set_timesteps(args.sparse_steps, device='cuda')
    # Optional explicit reverse: if --sp-reverse-t is set, monkey-patch timesteps
    # via inverse-shift transform on sigmas → dense at low σ (= low t = late steps).
    if args.sp_reverse_t > 0:
        shift_rev = float(args.sp_reverse_t)   # interpretation: lower=more aggressive
        try:
            sigmas_uniform = torch.linspace(1.0, 0.0, 31, device='cuda')[:-1]
            sigmas_shifted = shift_rev * sigmas_uniform / (1.0 + (shift_rev - 1.0) * sigmas_uniform)
            new_t = sigmas_shifted * 1000.0
            pipe.sparse_scheduler_512.timesteps = new_t
            sigmas_full = torch.cat([sigmas_shifted, torch.tensor([0.0], device='cuda')])
            pipe.sparse_scheduler_512.sigmas = sigmas_full
            print(f"  [sparse sched] reverse-t shift={shift_rev}, t schedule: "
                  f"{[round(x.item()) for x in new_t[::5].tolist()]}", flush=True)
        except Exception as e:
            print(f"  [sparse sched] WARN: reverse-t failed: {e}", flush=True)
    if args.sp_guide_w > 0:
        # Train mode so torchsparse builds backward kmap (else conv_forward_implicit_gemm
        # receives None for out_in_map_bwd → TypeError)
        pipe.sparse_vae_512.train()

    # Augmented Lagrangian state (single-element list for mutability across inner loop)
    _aug_lag_lambda = [float(args.sp_lambda_init)] if args.sp_aug_lag else [0.0]
    if args.sp_aug_lag:
        print(f"  [sp aug-lag] lambda_init={_aug_lag_lambda[0]:.2f}  mu={args.sp_mu_aug_lag:.1f}  "
              f"alpha={args.sp_lambda_alpha:.2f}  vol_target={args.sp_vol_target:.3f}", flush=True)
    if args.sp_heaviside_proj:
        print(f"  [sp heaviside] beta_init={args.sp_beta_init:.1f}  beta_max={args.sp_beta_max:.1f}  "
              f"bisect_iter={args.sp_eta_bisect_iter}  vol_target={args.sp_vol_target:.3f}", flush=True)

    latent_shape = (len(latent_index), pipe.sparse_dit_512.out_channels)
    g = torch.Generator(device='cuda').manual_seed(args.seed)
    sparse_latents = torch.randn(latent_shape, dtype=pipe.dtype, device='cuda', generator=g)
    sp_param = nn.Parameter(sparse_latents.float())
    if args.sp_opt == 'sgd':
        sp_opt = torch.optim.SGD([sp_param], lr=args.sp_guide_lr, momentum=0.9)
    elif args.sp_opt == 'adamw':
        sp_opt = torch.optim.AdamW([sp_param], lr=args.sp_guide_lr, betas=(0.9, 0.99), weight_decay=0)
    elif args.sp_opt == 'adam':
        sp_opt = torch.optim.Adam([sp_param], lr=args.sp_guide_lr, betas=(0.9, 0.99))
    elif args.sp_opt == 'rmsprop':
        sp_opt = torch.optim.RMSprop([sp_param], lr=args.sp_guide_lr, momentum=0.9, alpha=0.99)
    else:
        raise ValueError(args.sp_opt)
    print(f"  sp_opt: {args.sp_opt}, lr={args.sp_guide_lr}", flush=True)

    # === Option C — snapshot anchor (graded freedom) precompute ===
    _snap_sdf = None
    _snap_anchor = None
    if args.sp_snap_w > 0:
        from scipy.ndimage import distance_transform_edt as _edt_sn
        _da_np_sn = dense_active_64.cpu().numpy().astype(bool)
        _depth_in_sn = _edt_sn(_da_np_sn).astype('float32')   # (64,64,64) depth into solid
        _depth_t = torch.from_numpy(_depth_in_sn).to('cuda')
        # freedom: sigmoid(-(depth - d_half) / sigma) — 1 at surface, 0 deep
        # anchor_strength = 1 - freedom; we compute per-voxel after first decode
        _snap_depth_t = _depth_t   # used in loop
        print(f"  [snap-anchor] precomputed depth, d_half={args.sp_snap_d_half}, sigma={args.sp_snap_sigma}", flush=True)

    # === Option H — latent freeze on deep_int voxels ===
    # Pre-compute the sparse-index mask of deep_int voxels (so sp_param[mask] can be restored).
    # Snapshot is captured at the end of step 0 (= dense→sparse first-estimate of the latent).
    _sp_lf_mask = None       # bool tensor (M,)
    _sp_lf_snap = None       # snapshot of sp_param at end of step 0
    if args.sp_interior_latent_freeze:
        from scipy.ndimage import binary_erosion as _be_lf
        _da_np = dense_active_64.cpu().numpy()
        _di_np = _be_lf(_da_np, iterations=max(1, args.sp_interior_erode))
        _idx_np = latent_index.cpu().numpy()
        _is_di_lf = _di_np[_idx_np[:, 1], _idx_np[:, 2], _idx_np[:, 3]]
        _sp_lf_mask = torch.from_numpy(_is_di_lf).bool().to(sp_param.device)
        print(f"  [latent-freeze] erode={args.sp_interior_erode}: "
              f"{int(_sp_lf_mask.sum())}/{len(_sp_lf_mask)} ({100*_sp_lf_mask.sum().item()/len(_sp_lf_mask):.1f}%) "
              f"latent vectors will be restored after each update", flush=True)

    # force-bc-solid: build 512³ STL-SDF BC mask for in-loop clamping (sparse coords are @512).
    _bc_solid_t = None
    if _bc_hi.get('force_bc'):
        _bc512 = _bc_hi['mask'] if (_bc_hi.get('mask') is not None and _bc_hi['mask'].shape[0] == 512) else _build_bc_mask(512)
        if _bc512 is not None:
            if _bc_hi.get('mask') is None:
                _bc_hi['mask'] = _bc512   # reuse for the final refiner MC (same 512³)
            # _bc512 is a continuous SDF field (inside>0); in-loop uses inside bool
            _bc_solid_t = torch.from_numpy((_bc512 > 0).astype('float32')).to('cuda')
            print(f"  [force_bc_solid] in-loop BC clamp active ({int((_bc512>0).sum()):,} voxels @512³ STL-SDF)", flush=True)

    import gc as _gc
    for step_i, t in enumerate(pipe.sparse_scheduler_512.timesteps):
        ts = torch.tensor([t], dtype=pipe.dtype, device='cuda')
        # ① base velocity (no grad)
        with torch.no_grad():
            x_input = sp.SparseTensor(sp_param.detach().to(pipe.dtype), latent_index.int())
            nc = pipe.sparse_dit_512(x=x_input, t=ts, cond=cond_s).feats
            nu = pipe.sparse_dit_512(x=x_input, t=ts, cond=uncond_s).feats
            pred = nu + args.sp_cfg * (nc - nu)
            new_latents = pipe.sparse_scheduler_512.step(
                pred, t, sp_param.detach(), generator=g).prev_sample
        sp_param.data = new_latents.float()
        # Restore deep_int latents after outer scheduler step (option H)
        if _sp_lf_mask is not None and _sp_lf_snap is not None:
            with torch.no_grad():
                sp_param.data[_sp_lf_mask] = _sp_lf_snap[_sp_lf_mask]

        # ② GuideFlow3D guidance with annealing
        if args.sp_guide_w > 0:
            w_now = weight_anneal(step_i, args.sparse_steps, args.sp_guide_w, args.sp_guide_w_peak,
                                    ramp_start=args.sp_anneal_start)
            n_inner = n_inner_anneal(step_i, args.sparse_steps, n_base=1, n_late=args.sp_n_inner_late,
                                       ramp_start=args.sp_anneal_start)
            for inner_i in range(n_inner):
                sp_opt.zero_grad()
                # Ensure train mode each iter — empty_cache + sparse_dit no_grad calls
                # can invalidate torchsparse backward kmap → conv_forward_implicit_gemm fails.
                pipe.sparse_vae_512.train()
                ld = (1.0 / pipe.sparse_vae_512.latents_scale * sp_param
                       + pipe.sparse_vae_512.latents_shift)
                ldsp = sp.SparseTensor(ld.to(pipe.dtype).contiguous(),
                                        latent_index.int().contiguous())
                try:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        reconst_x, feat = pipe.sparse_vae_512.decode_mesh(
                            latents=ldsp, mc_threshold=0.0, return_feat=True)
                    sdf_sp = reconst_x.feats.float()
                    cd_sp = reconst_x.coords
                    # === (option D-improved) sp_interior_freeze: hard-clamp deep_int voxel sdf
                    # to a strong inside value (+1.0). Constant value → no gradient flow into
                    # latent (no detach() needed) AND forward output is itself frozen, so
                    # interior shape cannot drift via latent supervision elsewhere either.
                    # Shell = 2-depth means deep_int = erode(active, 2) by default.
                    if args.sp_interior_freeze:
                        from scipy.ndimage import binary_erosion as _be_di
                        _da = dense_active_64.detach().cpu().numpy().astype(bool) if hasattr(dense_active_64, 'detach') else dense_active_64.astype(bool)
                        _di_np = _be_di(_da, iterations=max(1, args.sp_interior_erode))
                        _di_t = torch.from_numpy(_di_np.astype('float32')).to(sdf_sp.device)
                        _cd64 = (cd_sp[:, 1:].long() // 8).clamp(0, 63)
                        _is_di_sp = _di_t[_cd64[:, 0], _cd64[:, 1], _cd64[:, 2]]   # (M,)
                        _di_mask = _is_di_sp.bool().unsqueeze(-1)                  # (M, 1)
                        # Use mc_threshold + 1.0 → guaranteed inside for marching cubes
                        _inside_target = float(sp_iso - 1.0 if args.sp_sdf_inside_low else sp_iso + 1.0)
                        if step_i == 0 and inner_i == 0:
                            _n_di = int(_is_di_sp.sum().item()); _n_tot = sdf_sp.shape[0]
                            print(f"  [interior-freeze D-improved] erode={args.sp_interior_erode}, mc_thr={args.mc_threshold}, clamp_to={_inside_target}, step 0: {_n_di}/{_n_tot} ({100*_n_di/_n_tot:.1f}%) voxels clamped", flush=True)
                        _sdf_solid = torch.full_like(sdf_sp, fill_value=_inside_target)
                        sdf_sp = torch.where(_di_mask, _sdf_solid, sdf_sp)
                    # === force-bc-solid (in-loop): hard-clamp BC (fix∪load) voxel sdf to a
                    # strong inside value so guidance/thickness/FEA all see BC as solid.
                    # Sparse VAE convention here: inside > mc_threshold (solid = higher).
                    # BC membership from the 512³ STL-SDF mask, indexed by sparse coords directly. ===
                    if _bc_hi.get('active') and _bc_solid_t is not None:
                        _cd512 = cd_sp[:, 1:].long().clamp(0, 511)
                        _is_bc_sp = _bc_solid_t[_cd512[:, 0], _cd512[:, 1], _cd512[:, 2]]
                        _bc_mask_sp = _is_bc_sp.bool().unsqueeze(-1)
                        _bc_solid_val = torch.full_like(sdf_sp, fill_value=float(sp_iso - 1.0 if args.sp_sdf_inside_low else sp_iso + 1.0))
                        sdf_sp = torch.where(_bc_mask_sp, _bc_solid_val, sdf_sp)
                    # The losses below expect negative-inside SDF. The decoder/MC path
                    # initializes unobserved voxels to +1 and unions solid BC with MIN,
                    # so its actual material convention is sdf < iso. The opt-in corrected
                    # mode uses sdf - iso; legacy mode remains for reproducibility.
                    _sdf_sp_in = sdf_sp - sp_iso if args.sp_sdf_inside_low else sp_iso - sdf_sp
                    l_rmin = topopt_rmin_loss(_sdf_sp_in, cd_sp,
                                              grid_res=128, r_min_voxels=args.sp_r_min_voxels,
                                              void_weight=args.sp_rmin_void_weight,
                                              mc_threshold=args.mc_threshold)
                    l_thick = thickness_per_voxel_loss(_sdf_sp_in, target_norm=args.sp_thick_target,
                                                       mc_threshold=args.mc_threshold)
                    # The topology helper uses negative-inside SDF, whereas the decoder
                    # and marching cubes use positive-inside values. Keep historical behavior
                    # available for controlled comparisons, with an explicit corrected mode.
                    topology_sdf = _sdf_sp_in if args.sp_topology_sdf_fix else sdf_sp
                    l_hole, l_interior = topology_preservation_loss(topology_sdf, cd_sp, dense_active_64,
                                                         interior_thresh=args.sp_hole_interior_thresh,
                                                         exterior_thresh=args.sp_hole_exterior_thresh,
                                                         cone_cos=args.sp_cone_cos,
                                                         no_diagonal=args.sp_no_diagonal,
                                                         interior_w=args.sp_interior_w,
                                                         interior_erode=args.sp_interior_erode,
                                                         interior_depth_graded=args.sp_interior_depth_graded,
                                                         interior_depth_slope=args.sp_interior_depth_slope)
                    if args.sp_normal_w > 0 and dense_normal_64 is not None:
                        l_normal = topology_normal_lipschitz_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            cone_cos=args.sp_normal_cos, p=args.sp_normal_p)
                    else:
                        l_normal = torch.zeros((), device=sdf_sp.device)
                    if args.sp_overhang_w > 0 and dense_normal_64 is not None:
                        l_overhang = topology_no_overhang_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            p=args.sp_overhang_p)
                    else:
                        l_overhang = torch.zeros((), device=sdf_sp.device)
                    if args.sp_normal_fd_w > 0 and dense_normal_64 is not None:
                        l_normalfd = topology_normal_fd_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            cone_cos=args.sp_normal_fd_cos, p=args.sp_normal_fd_p)
                    else:
                        l_normalfd = torch.zeros((), device=sdf_sp.device)
                    if args.sp_lap_w > 0 and dense_normal_64 is not None:
                        # Multi-res schedule: 'frac:K,...' — K is super-voxel pool size
                        # E.g., '0.66:4,0.85:2,1.0:1' = step_frac<0.66 → 4×4×4 pool → lap on 128³
                        sched_K = 1
                        if args.sp_lap_coarse_schedule:
                            _sf = step_i / max(1.0, float(args.sparse_steps))
                            for tok in args.sp_lap_coarse_schedule.split(','):
                                t_thr, K_str = tok.split(':')
                                if _sf <= float(t_thr):
                                    sched_K = int(K_str); break
                        if sched_K > 1:
                            # Option A: actual avg-pool sparse SDF to (512/K)³ dense, then Laplacian
                            dense_c, mask_c = coarsen_sparse_to_dense(sdf_sp, cd_sp, sched_K, R=512, default=1.0)
                            l_lap = dense_lap_loss(dense_c, mask=mask_c)
                            del dense_c, mask_c
                        else:
                            # K=1: original sparse edge-aware lap
                            l_lap = topology_lap_smooth_loss(
                                sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                                edge_sigma=args.sp_lap_sigma)
                    else:
                        l_lap = torch.zeros((), device=sdf_sp.device)
                    if args.sp_coh_w > 0 and dense_normal_64 is not None:
                        l_coh = topology_normal_coherence_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            nb_radius=int(args.sp_coh_r),
                            R_threshold=args.sp_coh_threshold,
                            p=args.sp_coh_p)
                    else:
                        l_coh = torch.zeros((), device=sdf_sp.device)
                    if args.sp_aniso_w > 0 and dense_normal_64 is not None:
                        l_aniso = topology_normal_anisotropy_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            nb_radius=int(args.sp_aniso_r),
                            surface_band=args.sp_aniso_band,
                            lam3_threshold=args.sp_aniso_lam3,
                            p=args.sp_aniso_p)
                    else:
                        l_aniso = torch.zeros((), device=sdf_sp.device)
                    if args.sp_odw_w > 0:
                        l_odw = topology_off_diagonal_hessian_loss(
                            sdf_sp, cd_sp, dense_active_64,
                            surface_band=args.sp_odw_band, p=args.sp_odw_p)
                    else:
                        l_odw = torch.zeros((), device=sdf_sp.device)
                    if args.sp_dfodw_w > 0 and dense_normal_64 is not None:
                        l_dfodw = topology_dense_frame_odw_loss(
                            sdf_sp, cd_sp, dense_active_64, dense_normal_64,
                            surface_band=args.sp_dfodw_band, p=args.sp_dfodw_p)
                    else:
                        l_dfodw = torch.zeros((), device=sdf_sp.device)
                    if args.sp_edt_w > 0:
                        l_edt = topology_edt_anchored_loss(
                            sdf_sp, cd_sp, dense_active_64,
                            tol=args.sp_edt_tol, p=args.sp_edt_p, scale=args.sp_edt_scale)
                    else:
                        l_edt = torch.zeros((), device=sdf_sp.device)
                    if args.sp_snap_w > 0:
                        # Take snapshot at first decode of step 0
                        if _snap_sdf is None and step_i == 0 and inner_i == 0:
                            _snap_sdf = sdf_sp.detach().clone()
                            # Compute anchor weight per sparse voxel from depth
                            _cd64 = (cd_sp[:, 1:].long() // 8).clamp(0, 63)
                            _depth_at = _snap_depth_t[_cd64[:, 0], _cd64[:, 1], _cd64[:, 2]]
                            _freedom = torch.sigmoid(-(_depth_at - args.sp_snap_d_half) / args.sp_snap_sigma)
                            _snap_anchor = (1.0 - _freedom).unsqueeze(-1)   # (M, 1) — strong at deep
                            print(f"  [snap-anchor] snapshot taken: M={sdf_sp.shape[0]}, "
                                  f"mean_anchor={_snap_anchor.mean().item():.3f}", flush=True)
                        if _snap_sdf is not None:
                            l_snap = topology_snapshot_anchor_loss(sdf_sp, _snap_sdf, _snap_anchor)
                        else:
                            l_snap = torch.zeros((), device=sdf_sp.device)
                    else:
                        l_snap = torch.zeros((), device=sdf_sp.device)
                    if args.sp_vw > 0 or args.sp_aug_lag:
                        # Mean occupancy over sparse active voxels → target.
                        _occ_p = torch.sigmoid(sp_occupancy_logit(sdf_sp.squeeze(-1), args.sp_bce_steepness))
                        # (4) Heaviside projection + beta-continuation + eta bisection (TO standard)
                        if args.sp_heaviside_proj:
                            _sf_h = float(step_i) / max(1.0, float(args.sparse_steps))
                            # exponential continuation: beta = beta_init * (beta_max/beta_init)^step_frac
                            _beta = args.sp_beta_init * (args.sp_beta_max / max(args.sp_beta_init, 1e-3)) ** _sf_h
                            with torch.no_grad():
                                _eta = bisect_eta_for_volume(
                                    _occ_p.detach().flatten(), args.sp_vol_target, _beta,
                                    max_iter=args.sp_eta_bisect_iter)
                            _occ_proj = heaviside_projection(_occ_p, _eta, _beta)
                            _g_vol = _occ_proj.mean() - args.sp_vol_target
                        else:
                            _g_vol = _occ_p.mean() - args.sp_vol_target
                        # (1) Augmented Lagrangian: lambda*(V-V*) + (mu/2)*(V-V*)^2
                        if args.sp_aug_lag:
                            l_vol = (_aug_lag_lambda[0] * _g_vol
                                     + 0.5 * args.sp_mu_aug_lag * _g_vol ** 2)
                        else:
                            l_vol = _g_vol ** 2  # standard quadratic (vw fixed)
                    else:
                        l_vol = torch.zeros((), device=sdf_sp.device)
                        _g_vol = None
                    if args.sp_dense_core_mm > 0 and args.sp_dense_core_w > 0:
                        _core = _dense_core_mask(cd_sp)
                        _core_target = sp_iso - 0.1
                        _core_pen = F.softplus((sdf_sp.squeeze(-1) - _core_target) * 10.0) / 10.0
                        if args.sp_dense_core_tail_frac > 0:
                            _violations = _core_pen[_core]
                            _k = max(1, int(_violations.numel() * args.sp_dense_core_tail_frac))
                            l_dense_core = _violations.topk(_k, sorted=False).values.mean()
                        else:
                            l_dense_core = (_core_pen * _core).sum() / _core.sum().clamp(min=1)
                    else:
                        l_dense_core = torch.zeros((), device=sdf_sp.device)
                    if _wall_fronts is not None:
                        l_wall_ray, _wall_ray_stats = sparse_side_front_loss(
                            sdf_sp, cd_sp, _wall_fronts, sp_iso,
                            band_voxels=args.sp_wall_ray_band,
                            min_solid_mass=args.sp_wall_ray_min_mass,
                            tail_fraction=args.sp_wall_ray_tail_frac,
                            grid_size=512)
                        if step_i == 0 and inner_i == 0:
                            print(f"  [wall-ray] {_wall_ray_stats} band={args.sp_wall_ray_band} "
                                  f"min_mass={args.sp_wall_ray_min_mass} "
                                  f"tail_frac={args.sp_wall_ray_tail_frac}", flush=True)
                    else:
                        l_wall_ray = torch.zeros((), device=sdf_sp.device)
                    if _side_depth_refs is not None:
                        l_side_depth, _side_depth_stats = sparse_side_depth_loss(
                            sdf_sp, cd_sp, _side_depth_refs, sp_iso,
                            tail_fraction=args.sp_side_depth_tail_frac)
                        if step_i == 0 and inner_i == 0:
                            print(f"  [side-depth] {_side_depth_stats} "
                                  f"tail_frac={args.sp_side_depth_tail_frac}", flush=True)
                    else:
                        l_side_depth = torch.zeros((), device=sdf_sp.device)
                    loss = (args.sp_rmin_w * l_rmin
                            + args.sp_thick_w * l_thick
                            + args.sp_dense_core_w * l_dense_core
                            + args.sp_wall_ray_w * l_wall_ray
                            + args.sp_side_depth_w * l_side_depth
                            + args.sp_hole_w * l_hole
                            + args.sp_interior_w * l_interior
                            + args.sp_normal_w * l_normal
                            + args.sp_overhang_w * l_overhang
                            + args.sp_normal_fd_w * l_normalfd
                            + args.sp_lap_w * l_lap
                            + args.sp_coh_w * l_coh
                            + args.sp_aniso_w * l_aniso
                            + args.sp_odw_w * l_odw
                            + args.sp_dfodw_w * l_dfodw
                            + args.sp_edt_w * l_edt
                            + args.sp_snap_w * l_snap
                            + args.sp_vw * l_vol) * w_now
                    # === multi-res avg-pool anchor (each sparse voxel ← K×K×K mean) ===
                    if args.sp_pool_anchor_w > 0:
                        pool_K = 4
                        if args.sp_pool_anchor_schedule:
                            _sf = step_i / max(1.0, float(args.sparse_steps))
                            for tok in args.sp_pool_anchor_schedule.split(','):
                                t_thr, K_str = tok.split(':')
                                if _sf <= float(t_thr):
                                    pool_K = int(K_str); break
                        if pool_K > 1:
                            l_pool = avg_pool_anchor_loss(sdf_sp, cd_sp, pool_K, R=512)
                            loss = loss + args.sp_pool_anchor_w * l_pool
                    # Sparse region BCE (mirror dense 3-region BCE)
                    if sp_bce_active:
                        # Auto-detect mask resolution. Sparse coords are at 512;
                        # mask may be 64³ (default), 128³, 256³, or 512³ — adjust downsample ratio.
                        _R_mask = int(_bc_64.shape[0])
                        _ratio  = max(1, 512 // _R_mask)
                        _xyz = (cd_sp[:, 1:].long() // _ratio).clamp(0, _R_mask - 1)
                        _occ_logit = sp_occupancy_logit(sdf_sp.squeeze(-1), args.sp_bce_steepness)
                        _ones = torch.ones_like(_occ_logit); _zeros = torch.zeros_like(_occ_logit)
                        _bce_solid = F.binary_cross_entropy_with_logits(_occ_logit, _ones, reduction='none')
                        _bce_empty = F.binary_cross_entropy_with_logits(_occ_logit, _zeros, reduction='none')
                        _l_bce = 0.0
                        if args.sp_bc_w > 0:
                            _m = _bc_64[_xyz[:,0], _xyz[:,1], _xyz[:,2]]
                            _l_bce = _l_bce + args.sp_bc_w * (_bce_solid * _m).sum() / _m.sum().clamp_min(1.0)
                        if args.sp_bc_buffer_w > 0 and _bc_buffer_64 is not None:
                            _m = _bc_buffer_64[_xyz[:,0], _xyz[:,1], _xyz[:,2]]
                            _l_bce = _l_bce + args.sp_bc_buffer_w * (_bce_solid * _m).sum() / _m.sum().clamp_min(1.0)
                        if args.sp_design_w > 0:
                            _m = _design_64[_xyz[:,0], _xyz[:,1], _xyz[:,2]]
                            _l_bce = _l_bce + args.sp_design_w * (_bce_solid * _m).sum() / _m.sum().clamp_min(1.0)
                        if args.sp_out_w > 0:
                            _m = _out_64[_xyz[:,0], _xyz[:,1], _xyz[:,2]]
                            _l_bce = _l_bce + args.sp_out_w * (_bce_empty * _m).sum() / _m.sum().clamp_min(1.0)
                        loss = loss + _l_bce * w_now
                    # Sparse shape-QD: aggregate the active 512³ sparse field into the exact
                    # 64³ proxy used by dense targeting, then retain the target morphology
                    # through the refiner.  Missing sparse cells stay empty.
                    if ((_shape_qd_sparse is not None and step_i / max(1.0, float(args.sparse_steps)) >= args.sp_shape_qd_warmup)
                            or _shape_anchor_sparse is not None or _sp_image_projection is not None):
                        _xyz_qd = (cd_sp[:, 1:].long() // 8).clamp(0, 63)
                        _flat_qd = _xyz_qd[:, 0] * 4096 + _xyz_qd[:, 1] * 64 + _xyz_qd[:, 2]
                        _occ_qd = torch.sigmoid(sp_occupancy_logit(sdf_sp.squeeze(-1), args.sp_bce_steepness))
                        _sum_qd = torch.zeros(64**3, device=sdf_sp.device, dtype=_occ_qd.dtype)
                        _cnt_qd = torch.zeros(64**3, device=sdf_sp.device, dtype=_occ_qd.dtype)
                        _sum_qd.scatter_add_(0, _flat_qd, _occ_qd)
                        _cnt_qd.scatter_add_(0, _flat_qd, torch.ones_like(_occ_qd))
                        _logit_qd = torch.logit((_sum_qd / _cnt_qd.clamp_min(1)).clamp(1e-5, 1-1e-5)).view(64, 64, 64)
                        if _shape_qd_sparse is not None:
                            _l_shape_qd, _ = _shape_qd_sparse.loss(_logit_qd, _shape_bc_64)
                            loss = loss + args.sp_shape_qd_w * _l_shape_qd
                            if step_i % max(1, args.sparse_steps // 5) == 0 and inner_i == 0:
                                print(f"  [sp Shape-QD] step {step_i}: loss={_l_shape_qd.item():.4f}", flush=True)
                        if _sp_image_projection is not None:
                            _l_sp_image = _sp_image_projection.loss(_logit_qd, _sp_projection_env)
                            loss = loss + args.sp_image_proj_w * _l_sp_image
                            if step_i % max(1, args.sparse_steps // 5) == 0 and inner_i == 0:
                                print(f"  [sp image projection] step {step_i}: loss={_l_sp_image.item():.4f}", flush=True)
                    if _shape_anchor_sparse is not None:
                        _l_anchor_sp = _shape_anchor_sparse.loss(_logit_qd, _shape_bc_64)
                        loss = loss + args.sp_shape_anchor_w * _l_anchor_sp
                        if step_i % max(1, args.sparse_steps // 5) == 0 and inner_i == 0:
                            print(f"  [sp shape anchor] step {step_i}: loss={_l_anchor_sp.item():.4f}", flush=True)
                    if _sp_negative_space is not None:
                        _neg_xyz = (cd_sp[:, 1:].long() // 8).clamp(0, 63)
                        _neg_select = _sp_negative_space[
                            _neg_xyz[:, 0], _neg_xyz[:, 1], _neg_xyz[:, 2]]
                        if _neg_select.any():
                            # The sparse decoder/refiner convention is negative-inside;
                            # use the high-res decoded SDF directly, not the 64³ mean proxy.
                            _neg_solid_logit = (sp_iso - sdf_sp.squeeze(-1)) * args.sp_bce_steepness
                            _l_neg = F.softplus(_neg_solid_logit[_neg_select]).mean()
                            loss = loss + args.sp_negative_space_w * _l_neg
                            if step_i % max(1, args.sparse_steps // 5) == 0 and inner_i == 0:
                                print(f"  [sp negative space] step {step_i}: loss={_l_neg.item():.4f} "
                                      f"active={int(_neg_select.sum())}", flush=True)
                    # ── ADAMW-MODE SPARSE FEA INTEGRATION ────────────────────
                    # adamw mode: build the FEA term from the SAME decoder graph and add it
                    # to the regular sparse guidance loss.  A previous implementation called
                    # autograd.grad(FEA) first and then loss.backward(); autograd.grad freed
                    # the shared decoder graph, so every regular guidance backward failed.
                    # One combined backward is both the correct optimizer objective and avoids
                    # that double-backward failure.
                    _last_fea_adamw = None
                    _scaled_fea = None
                    if args.sp_fea_w > 0 and args.sp_fea_mode == 'adamw':
                        _sp_sf = step_i / max(1.0, float(args.sparse_steps))
                        if _sp_sf >= args.sp_fea_warmup and step_i % args.sp_fea_every == 0:
                            try:
                                _br_np_a, _nd_np_a, _dom_a, _bc64_a, _smap_a, _deep_a = sp_fea_cache
                                _br_t_a = torch.from_numpy(_br_np_a).to(sdf_sp.device)
                                _bc_t_a = torch.from_numpy(_bc64_a).to(sdf_sp.device)
                                _R64 = 64
                                _xyz_a = (cd_sp[:, 1:].long() // 8).clamp(0, 63)
                                _flat_a = _xyz_a[:,0]*_R64*_R64 + _xyz_a[:,1]*_R64 + _xyz_a[:,2]
                                _src_a = (torch.sigmoid(sp_occupancy_logit(
                                    sdf_sp.squeeze(-1), args.sp_fea_steepness))
                                          if args.sp_fea_reduce == 'soft_frac'
                                          else sdf_sp.squeeze(-1))
                                _sums = torch.zeros(_R64**3, device=sdf_sp.device, dtype=_src_a.dtype)
                                _sums.scatter_add_(0, _flat_a, _src_a)
                                _ones_a = torch.ones(cd_sp.shape[0], device=sdf_sp.device, dtype=_src_a.dtype)
                                _cnts = torch.zeros(_R64**3, device=sdf_sp.device, dtype=_src_a.dtype)
                                _cnts.scatter_add_(0, _flat_a, _ones_a)
                                _dmean = _sums / _cnts.clamp(min=1.0)
                                if args.sp_fea_reduce == 'soft_frac':
                                    _occ_l = torch.logit(_dmean.clamp(1e-6, 1-1e-6))
                                else:
                                    _occ_l = sp_occupancy_logit(_dmean, args.sp_fea_steepness)
                                _occ_l = _occ_l.clamp(-5.0, 5.0).view(_R64, _R64, _R64)
                                _occ_l = torch.where(_br_t_a.bool(), _occ_l,
                                                      torch.tensor(-5.0, device=_occ_l.device, dtype=_occ_l.dtype))
                                _occ_l = torch.where(_bc_t_a.bool(),
                                                      torch.tensor(5.0, device=_occ_l.device, dtype=_occ_l.dtype),
                                                      _occ_l)
                                _occ_l = _occ_l.float()
                                _fea_a, _seat_a, _back_a = paired_fea_compliance(
                                    _occ_l, _br_np_a, _nd_np_a, _dom_a,
                                    args.fea_back_domain_dir, args.fea_mesh_cache,
                                    args.fea_mesh_size, args.fea_penal, sp_fea_reference,
                                    args.fea_back_load_mode, args.fea_back_load_magnitude)
                                if torch.isfinite(_fea_a).item():
                                    _decay_a = max(0.3, 1.0 - 0.7 * _sp_sf)
                                    _fea_term_a = _fea_a / (_fea_a.detach().abs() + 1e-12) if args.sp_fea_normalize else _fea_a
                                    _scaled_fea = args.sp_fea_w * _decay_a * _fea_term_a
                                    _last_fea_adamw = float(_fea_a.item())
                                    print(f"    [sp FEA adamw step {step_i}] comp={_fea_a.item():.3e} decay={_decay_a:.2f} (combined-bw)", flush=True)
                                    if _back_a is not None:
                                        print(f"    [sp dual FEA adamw step {step_i}] "
                                              f"seat={_seat_a.item():.4e} "
                                              f"back={_back_a.item():.4e}", flush=True)
                                    elif args.fea_combine_loads:
                                        print(f"    [sp simultaneous FEA adamw step {step_i}] "
                                              f"C={_fea_a.item():.4e}", flush=True)
                                    del _fea_a, _fea_term_a, _occ_l, _sums, _cnts, _dmean, _src_a, _ones_a, _flat_a, _xyz_a, _br_t_a, _bc_t_a
                            except Exception as _e:
                                print(f"    [sp FEA adamw step {step_i}] FAIL: {type(_e).__name__}: {str(_e)[:120]}", flush=True)
                    (loss + _scaled_fea if _scaled_fea is not None else loss).backward()
                    if _scaled_fea is not None:
                        print(f"    [sp FEA combined grad step {step_i}] total_norm={sp_param.grad.norm().item():.3e}", flush=True)
                        _scaled_fea = None
                    _grad_debug = (os.environ.get('D3DS2_SPARSE_GRAD_DEBUG', '0') == '1'
                                   and inner_i == n_inner - 1
                                   and (step_i == 0 or step_i == args.sparse_steps - 1))
                    if _grad_debug:
                        _before_step = sp_param.detach().clone()
                        _grad = sp_param.grad
                        print(f"  [sparse grad diagnostic] step={step_i} "
                              f"grad_norm={_grad.norm().item() if _grad is not None else float('nan'):.6e} "
                              f"grad_nonzero={int(torch.count_nonzero(_grad).item()) if _grad is not None else 0}",
                              flush=True)
                    sp_opt.step()
                    if _grad_debug:
                        _delta = sp_param.detach() - _before_step
                        print(f"  [sparse grad diagnostic] step={step_i} "
                              f"param_delta_norm={_delta.norm().item():.6e} "
                              f"param_delta_max={_delta.abs().max().item():.6e}", flush=True)
                        del _before_step, _delta
                    sp_opt.zero_grad(set_to_none=True)
                    # === multi-res REPRESENTATION pool: force sp_param to be coarse ===
                    if args.sp_param_pool_schedule:
                        param_K = 1
                        _sf = step_i / max(1.0, float(args.sparse_steps))
                        for tok in args.sp_param_pool_schedule.split(','):
                            t_thr, K_str = tok.split(':')
                            if _sf <= float(t_thr):
                                param_K = int(K_str); break
                        if param_K > 1:
                            with torch.no_grad():
                                Rc = 512 // param_K
                                _coords_c = (latent_index[:, 1:].long() // param_K).clamp(0, Rc - 1)
                                _flat = (_coords_c[:, 0] * Rc + _coords_c[:, 1]) * Rc + _coords_c[:, 2]
                                _cnt = torch.zeros(Rc**3, device=sp_param.device, dtype=sp_param.dtype)
                                _cnt.scatter_add_(0, _flat, torch.ones_like(_flat, dtype=sp_param.dtype))
                                _cnt_safe = _cnt.clamp(min=1.0)
                                # Per-channel pool: scatter sum + divide → mean → gather back
                                C = sp_param.shape[-1]
                                _alpha = max(0.0, 1.0 - step_i / max(1.0, float(args.sparse_steps))) if args.sp_param_pool_blend else 1.0
                                for c in range(C):
                                    _sum_c = torch.zeros(Rc**3, device=sp_param.device, dtype=sp_param.dtype)
                                    _sum_c.scatter_add_(0, _flat, sp_param.data[:, c])
                                    _mean_c = _sum_c / _cnt_safe
                                    if _alpha >= 1.0:
                                        sp_param.data[:, c] = _mean_c[_flat]
                                    else:
                                        sp_param.data[:, c] = _alpha * _mean_c[_flat] + (1 - _alpha) * sp_param.data[:, c]
                                del _coords_c, _flat, _cnt, _cnt_safe
                    # Aggressive memory cleanup every inner iteration (multi-res helpers may leak)
                    import gc as _gc_step; _gc_step.collect(); torch.cuda.empty_cache()
                    # Option H — restore deep_int latents after inner opt step too.
                    # Take snapshot at the end of step 0 (= dense→sparse first-estimate).
                    if _sp_lf_mask is not None:
                        if _sp_lf_snap is None and step_i == 0 and inner_i == n_inner - 1:
                            _sp_lf_snap = sp_param.data.detach().clone()
                            print(f"  [latent-freeze] snapshot taken at end of step 0 "
                                  f"(mean |sp_param|={sp_param.data.abs().mean().item():.4f})", flush=True)
                        elif _sp_lf_snap is not None:
                            with torch.no_grad():
                                sp_param.data[_sp_lf_mask] = _sp_lf_snap[_sp_lf_mask]
                    if (step_i + 1) % 5 == 0 or step_i == 0:
                        _norm_str = f" normal={l_normal.item():.5f}" if args.sp_normal_w > 0 else ""
                        _over_str = f" overhang={l_overhang.item():.5f}" if args.sp_overhang_w > 0 else ""
                        _nfd_str = f" nfd={l_normalfd.item():.5f}" if args.sp_normal_fd_w > 0 else ""
                        _lap_str = f" lap={l_lap.item():.5f}" if args.sp_lap_w > 0 else ""
                        _coh_str = f" coh={l_coh.item():.5f}" if args.sp_coh_w > 0 else ""
                        _aniso_str = f" aniso={l_aniso.item():.5f}" if args.sp_aniso_w > 0 else ""
                        _odw_str = f" odw={l_odw.item():.5f}" if args.sp_odw_w > 0 else ""
                        _dfodw_str = f" dfodw={l_dfodw.item():.5f}" if args.sp_dfodw_w > 0 else ""
                        _edt_str = f" edt={l_edt.item():.5f}" if args.sp_edt_w > 0 else ""
                        _snap_str = f" snap={l_snap.item():.5f}" if args.sp_snap_w > 0 else ""
                        _vol_str = f" vol={l_vol.item():.5f}" if args.sp_vw > 0 else ""
                        _core_str = f" core={l_dense_core.item():.5f}" if args.sp_dense_core_w > 0 else ""
                        _wall_str = f" wallray={l_wall_ray.item():.5f}" if args.sp_wall_ray_w > 0 else ""
                        _side_depth_str = f" sidedepth={l_side_depth.item():.5f}" if args.sp_side_depth_w > 0 else ""
                        print(f"  [sp guide] step {step_i+1:2d}/{args.sparse_steps} t={float(t):.0f}  "
                              f"inner={inner_i+1}/{n_inner} w={w_now:.1f}  "
                              f"total={loss.item():.5f} rmin={l_rmin.item():.5f} "
                              f"thick={l_thick.item():.5f} hole={l_hole.item():.5f}{_core_str}{_wall_str}{_side_depth_str}{_norm_str}{_over_str}{_nfd_str}{_lap_str}{_coh_str}{_aniso_str}{_odw_str}{_dfodw_str}{_edt_str}{_snap_str}{_vol_str} "
                              f"M={sdf_sp.shape[0]}", flush=True)
                    # log only the last inner iteration per outer step (avoid n_inner duplicates)
                    if inner_i == n_inner - 1:
                        # cols: stage,step,t,total,fea_comp,l_rmin,l_thick,l_hole,l_interior,l_normalfd,l_lap,l_aniso,l_vol,l_bce,l_reach
                        # fea_comp filled when adamw mode integrated FEA fired this step
                        _fea_csv = f'{_last_fea_adamw:.6e}' if _last_fea_adamw is not None else ''
                        loss_csv.write(f'sparse,{step_i},{float(t):.4f},{float(loss.item()):.6e},{_fea_csv},'
                                       f'{float(l_rmin.item()):.6e},{float(l_thick.item()):.6e},'
                                       f'{float(l_hole.item()):.6e},{float(l_interior.item()):.6e},'
                                       f'{float(l_normalfd.item()):.6e},{float(l_lap.item()):.6e},'
                                       f'{float(l_aniso.item()):.6e},{float(l_vol.item()):.6e},,\n')
                        loss_csv.flush()
                    del loss, l_rmin, l_thick, l_hole, l_interior, l_dense_core, l_wall_ray, l_side_depth, l_normal, l_overhang, l_normalfd, l_lap, l_coh, l_aniso, l_odw, l_dfodw, l_edt, l_snap, l_vol, sdf_sp, cd_sp, reconst_x, feat, ld, ldsp
                    _gc.collect(); torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  [sp guide] step {step_i+1} inner {inner_i} FAIL {type(e).__name__}: {str(e)[:80]}",
                          flush=True)
                    _gc.collect(); torch.cuda.empty_cache()
                    break

        # Augmented Lagrangian outer-loop update: lambda ← lambda + alpha * mu * (V - V*)
        if args.sp_aug_lag and _g_vol is not None:
            with torch.no_grad():
                _g_val = float(_g_vol.detach().item())
                _aug_lag_lambda[0] = max(0.0,
                    _aug_lag_lambda[0] + args.sp_lambda_alpha * args.sp_mu_aug_lag * _g_val)
            if (step_i + 1) % 5 == 0 or step_i == 0:
                print(f"  [sp aug-lag] step {step_i+1}: V={_g_val + args.sp_vol_target:.4f}  "
                      f"V*-V={-_g_val:+.4f}  lambda={_aug_lag_lambda[0]:.2f}", flush=True)

        # Sparse snapshot — decode mesh at this step (after inner loop)
        if args.sp_snapshot_every > 0 and (
                step_i % args.sp_snapshot_every == 0
                or step_i == len(pipe.sparse_scheduler_512.timesteps) - 1):
            try:
                from pathlib import Path as _P
                _sd = _P(args.out) / 'snapshots'; _sd.mkdir(parents=True, exist_ok=True)
                pipe.sparse_vae_512.eval()
                with torch.no_grad():
                    _ld = (1.0 / pipe.sparse_vae_512.latents_scale * sp_param.detach().to(pipe.dtype)
                            + pipe.sparse_vae_512.latents_shift)
                    _ldsp = sp.SparseTensor(_ld, latent_index.int())
                    _meshes = pipe.sparse_vae_512.decode_mesh(
                        latents=_ldsp, mc_threshold=args.mc_threshold)
                    if _meshes and len(_meshes[0].vertices) > 0:
                        _meshes[0].export(str(_sd / f'sparse_step_{step_i:02d}.obj'))
                pipe.sparse_vae_512.train()  # restore for next inner step
                _gc.collect(); torch.cuda.empty_cache()
            except Exception as _e:
                print(f"    [WARN] sparse snapshot step {step_i} failed: {type(_e).__name__}: {str(_e)[:80]}")
                pipe.sparse_vae_512.train()

    # === feaoff mesh stash: pre-optimization sparse latent (= feaoff shape) ===
    # one posthoc run can emit BOTH feaoff (here) and posthoc (final), sharing generation.
    _latents_feaoff = None
    if args.posthoc_fea_steps > 0 and getattr(args, 'feaoff_out', None):
        with torch.no_grad():
            _latents_feaoff = (1.0 / pipe.sparse_vae_512.latents_scale
                               * sp_param.detach().clone().to(pipe.dtype)
                               + pipe.sparse_vae_512.latents_shift)
        print(f"  [feaoff-stash] captured pre-posthoc latent for {args.feaoff_out}", flush=True)

    # === PhysiOpt-style post-hoc latent optimization (FEM-only, flow frozen) ===
    if args.posthoc_fea_steps > 0:
        assert sp_fea_cache is not None, "posthoc-fea requires sp_fea cache"
        from fea_compliance_loss import fea_compliance_loss as _fcl_ph
        _vae_ph = pipe.sparse_vae_512
        _dec_ph = _vae_ph.decoder
        _dec_ph.train()   # torchsparse backward kmap
        _br_np_ph, _nd_np_ph, _dom_ph, _bc64_ph, _smap_ph, _deep_np_ph = sp_fea_cache
        _br_t_ph = torch.from_numpy(_br_np_ph).to('cuda')
        _bc_t_ph = torch.from_numpy(_bc64_ph).to('cuda')
        _deep_t_ph = torch.from_numpy(_deep_np_ph).bool().to('cuda')
        _lat_ph = sp_param.detach().clone().requires_grad_(True)
        _opt_ph = torch.optim.Adam([_lat_ph], lr=args.posthoc_fea_lr)
        _V0_ph = None
        print(f"  [posthoc-fea] {args.posthoc_fea_steps} ADAM steps (lr={args.posthoc_fea_lr}, "
              f"vol_lambda={args.posthoc_vol_lambda}) — PhysiOpt-style post-hoc latent search", flush=True)
        for _k_ph in range(args.posthoc_fea_steps):
            try:
                _opt_ph.zero_grad()
                with torch.enable_grad():
                    _uns = (1.0 / _vae_ph.latents_scale * _lat_ph.to(pipe.dtype) + _vae_ph.latents_shift)
                    _out = _dec_ph(sp.SparseTensor(_uns, latent_index.int()), factor=None, return_feat=False)
                    _sdf = _out.feats
                    _oc = _out.coords[:, 1:]
                    R64 = 64
                    _xyz64 = (_oc.long() // (int(_smap_ph['sparse_res']) // R64)).clamp(0, R64-1)
                    _ins = _br_t_ph[_xyz64[:,0], _xyz64[:,1], _xyz64[:,2]]
                    _xyz_in = _xyz64[_ins]
                    _sdf_in = _sdf.squeeze(-1)[_ins]
                    _fi = _xyz_in[:,0]*R64*R64 + _xyz_in[:,1]*R64 + _xyz_in[:,2]
                    _Np = _sdf_in.shape[0]
                    if _Np > 0:
                        _src = (torch.sigmoid(sp_occupancy_logit(_sdf_in, args.sp_fea_steepness))
                                if args.sp_fea_reduce == 'soft_frac' else _sdf_in)
                        _sums = torch.zeros(R64**3, device=_sdf.device, dtype=_src.dtype
                                            ).scatter_add(0, _fi, _src)
                        with torch.no_grad():
                            _ones = torch.ones(_Np, device=_sdf.device, dtype=_src.dtype)
                            _cnts = torch.zeros(R64**3, device=_sdf.device, dtype=_src.dtype
                                                ).scatter_add(0, _fi, _ones)
                    else:
                        raise RuntimeError("no active points inside envelope")
                    _dm = _sums / _cnts.clamp(min=1.0)
                    if args.sp_fea_reduce == 'soft_frac':
                        _dense = _dm.view(R64, R64, R64)
                        _ol = torch.logit(_dense.clamp(1e-6, 1-1e-6))
                    else:
                        _missing_sdf = 1.0 if args.sp_sdf_inside_low else -1.0
                        _dense = torch.where(_cnts > 0, _dm, torch.tensor(_missing_sdf, device=_sdf.device, dtype=_sdf.dtype)).view(R64, R64, R64)
                        _ol = sp_occupancy_logit(_dense, args.sp_fea_steepness)
                    _ol = _ol.clamp(-5.0, 5.0)
                    _ol = torch.where(_br_t_ph, _ol, torch.tensor(-5.0, device=_ol.device, dtype=_ol.dtype))
                    if args.sp_shell_only:
                        _ol = torch.where(_deep_t_ph, torch.tensor(5.0, device=_ol.device, dtype=_ol.dtype), _ol)
                    _ol = torch.where(_bc_t_ph, torch.tensor(5.0, device=_ol.device, dtype=_ol.dtype), _ol)
                    _ol = _ol.float()
                    _rho = torch.sigmoid(_ol)
                    _V = _rho[_br_t_ph].mean()
                    if _V0_ph is None: _V0_ph = _V.detach()
                    _fea = _fcl_ph(_ol, _br_np_ph, _nd_np_ph, _dom_ph, args.fea_mesh_cache,
                                   mesh_size=args.fea_mesh_size, penal=3.0,
                                   bc_mask_np=_bc64_ph.astype(bool))
                    if not torch.isfinite(_fea).item():
                        print(f"    [posthoc {_k_ph+1}] SKIP non-finite comp", flush=True); continue
                    _loss = _fea / (_fea.detach().abs() + 1e-12)                             + args.posthoc_vol_lambda * (_V - _V0_ph).abs()
                    _loss.backward()
                if torch.isfinite(_lat_ph.grad).all().item():
                    _opt_ph.step()
                if (_k_ph+1) % 5 == 0 or _k_ph == 0:
                    print(f"    [posthoc {_k_ph+1}/{args.posthoc_fea_steps}] C={_fea.item():.4e} "
                          f"V={_V.item():.4f} (V0={_V0_ph.item():.4f}) "
                          f"gnorm={_lat_ph.grad.norm().item():.3e}", flush=True)
                _gc.collect(); torch.cuda.empty_cache()
            except Exception as _e_ph:
                print(f"    [posthoc {_k_ph+1}] FAIL {type(_e_ph).__name__}: {str(_e_ph)[:80]}", flush=True)
                _gc.collect(); torch.cuda.empty_cache()
        with torch.no_grad():
            sp_param.data.copy_(_lat_ph.detach().to(sp_param.dtype))
        # aggressive cleanup: ADAM state + grad graphs + LAST-iteration loop locals
        # must all be gone before the refiner's fixed 8GB dense-grid allocation.
        _dec_ph.eval()
        _opt_ph.zero_grad(set_to_none=True)
        del _opt_ph, _lat_ph, _br_t_ph, _bc_t_ph, _deep_t_ph
        _out = _sdf = _oc = _uns = _ol = _rho = _dense = _dm = _sums = _cnts = None
        _fea = _loss = _V = _V0_ph = _src = _ones = _fi = _xyz64 = _xyz_in = None
        _sdf_in = _ins = None
        _gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        print(f"  [posthoc-fea] done — latent updated (cuda {torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)

    # Decode + refiner + MC
    print(f"  decoding final + refiner...")
    pipe.sparse_vae_512.eval()
    pipe.sparse_dit_512.to('cpu')
    del cond_s, uncond_s, x_input, nc, nu, pred
    _gc.collect(); torch.cuda.empty_cache()

    # Compute latents_final FIRST (uses sp_param), then free everything before refiner
    latents_final = (1.0 / pipe.sparse_vae_512.latents_scale * sp_param.detach().to(pipe.dtype)
                     + pipe.sparse_vae_512.latents_shift)
    # ── NUCLEAR cleanup before refiner: AdamW state explicit + triple gc + ipc_collect ──
    # 1. AdamW state tensors (m, v) — del sp_opt alone does not free immediately, explicitly clear the state dict
    try:
        if sp_opt is not None and hasattr(sp_opt, 'state'):
            for _st in sp_opt.state.values():
                for _k in list(_st.keys()):
                    _t = _st.pop(_k, None)
                    if isinstance(_t, torch.Tensor): del _t
            sp_opt.state.clear()
    except Exception: pass
    try:
        del sp_param, sp_opt
    except Exception: pass
    # 2. Offload ALL non-essential modules — only 'refiner' + 'sparse_vae_512' kept on GPU.
    for _mod_name in ('dense_vae', 'dense_dit',
                      'sparse_dit_512',
                      'sparse_vae_1024', 'sparse_dit_1024', 'refiner_1024',
                      'sparse_image_encoder', 'dense_image_encoder'):
        _m = getattr(pipe, _mod_name, None)
        if _m is not None:
            try: _m.to('cpu')
            except Exception: pass
    # 3. Triple gc + synchronize + empty_cache + ipc_collect + reset stats
    import gc as _gc
    for _ in range(3):
        _gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    try: torch.cuda.ipc_collect()
    except Exception: pass
    try: torch.cuda.reset_peak_memory_stats()
    except Exception: pass
    print(f"  [refiner pre-cleanup] cuda allocated {torch.cuda.memory_allocated()/1e9:.2f} GB  "
          f"reserved {torch.cuda.memory_reserved()/1e9:.2f} GB")
    latents_sp = sp.SparseTensor(latents_final, latent_index.int())
    with torch.no_grad():
        decoded = pipe.sparse_vae_512.decode_mesh(latents=latents_sp,
                                                   mc_threshold=args.mc_threshold,
                                                   return_feat=True)
        if args.sp_dense_core_mm > 0 and args.sp_dense_core_project:
            _core = _dense_core_mask(decoded[0].coords)
            _sdf_before = decoded[0].feats
            _core_target = min(sp_iso, args.mc_threshold * 2.0) - 0.1
            _violating = int((_core & (_sdf_before.squeeze(-1) > _core_target)).sum().item())
            decoded[0].feats = torch.where(
                _core[:, None], torch.minimum(_sdf_before, torch.full_like(_sdf_before, _core_target)),
                _sdf_before)
            print(f"  [dense core] sparse decoder projected {_violating:,} "
                  f"core SDF samples to <= {_core_target:.3f}", flush=True)
        # Diagnostic only: compare the sparse decoder's surface with the refined one.
        # This does not change either tensor or the generation path.
        if os.environ.get('D3DS2_SAVE_PRE_REFINER', '0') == '1':
            _pre = pipe.sparse_vae_512.sparse2mesh(
                decoded[0], mc_threshold=args.mc_threshold * 2.0)[0]
            _pre.export(str(Path(args.out) / 'mesh_pre_refiner.obj'))
            print(f"  [diagnostic] pre-refiner mesh: V={len(_pre.vertices):,} "
                  f"F={len(_pre.faces):,}")
            del _pre
        # decode done — offload sparse_vae_512 too so refiner has max headroom
        try: pipe.sparse_vae_512.to('cpu')
        except Exception: pass
        _gc.collect(); torch.cuda.empty_cache()
        print(f"  [refiner pre-cleanup-2 after decode_mesh] cuda allocated {torch.cuda.memory_allocated()/1e9:.2f} GB")
        meshes = pipe.refiner.run(*decoded, mc_threshold=args.mc_threshold * 2.0)
    mesh = meshes[0]
    print(f"  Stage 2 mesh: V={len(mesh.vertices):,} F={len(mesh.faces):,}")

    if args.sdf_resolution == 1024:
        print("\n=== Stage 3: sparse1024 ===")
        torch.cuda.empty_cache()
        mesh_norm = normalize_mesh(mesh)
        latent_index_1024 = mesh2index(mesh_norm, size=1024, factor=8)
        latent_index_1024 = sort_block(latent_index_1024, pipe.sparse_dit_1024.selection_block_size)
        with torch.no_grad():
            mesh = pipe.inference(img_prep, pipe.sparse_vae_1024, pipe.sparse_dit_1024,
                                  pipe.sparse_image_encoder, pipe.sparse_scheduler_1024,
                                  generator=None, mode='sparse1024', mc_threshold=args.mc_threshold,
                                  latent_index=latent_index_1024, remove_interior=True,
                                  num_inference_steps=15, guidance_scale=7.0)[0]

    print(f"\nfinal mesh: V={len(mesh.vertices):,} F={len(mesh.faces):,}")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    # Inverse anisotropic transform: refiner output in normalized [-1, 1] mapping
    # to dense voxel [0, R] → world frame via env_origin + voxel_idx * pitch_xyz.
    try:
        # Refiner MC runs on the R³ (512³) grid; normalize idx/R*2-1 → v_norm∈[-1,1].
        # Invert at the ACTUAL refiner resolution R_ with a resolution-correct half-voxel:
        # world = origin + (idx_R + 0.5)*pitch_R,  pitch_R = pitch64*(64/R_).
        # (Old bug: R_=64 + (idx64+0.5)*pitch64 → +3.5*pitch512 ≈ +1.355mm shift/axis.)
        R_ = 512
        v_norm = mesh.vertices
        voxel_idx = (v_norm + 1.0) * R_ / 2.0
        pitch_R = np.asarray(env_pitch) * (64.0 / R_)
        world_v = np.asarray(env_origin) + (voxel_idx + 0.5) * pitch_R
        mesh.vertices = world_v.astype(mesh.vertices.dtype)
        print(f"  [final] inverse aniso transform → world frame: bbox extent "
              f"{(mesh.bounds[1]-mesh.bounds[0])*1000} mm")
    except Exception as _e:
        print(f"  [final] inverse aniso skipped: {_e}")
    mesh.export(str(out_dir/'mesh.obj'))
    mesh.export(str(out_dir/'mesh.glb'))

    # ── feaoff 2nd pass: mesh the stashed pre-posthoc latent → feaoff-out ───────
    # runs AFTER the primary (posthoc) mesh is saved; guarded so a failure here
    # never loses the posthoc result. reuses refiner (kept on GPU); reloads vae_512.
    if _latents_feaoff is not None:
        try:
            import gc as _gc2
            _gc2.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            pipe.sparse_vae_512.to('cuda'); pipe.sparse_vae_512.eval()
            print(f"  [feaoff-2nd] decoding pre-posthoc latent → refiner "
                  f"(cuda {torch.cuda.memory_allocated()/1e9:.2f}GB)", flush=True)
            _lat_sp_fo = sp.SparseTensor(_latents_feaoff, latent_index.int())
            with torch.no_grad():
                _decoded_fo = pipe.sparse_vae_512.decode_mesh(latents=_lat_sp_fo,
                                                              mc_threshold=args.mc_threshold,
                                                              return_feat=True)
                try: pipe.sparse_vae_512.to('cpu')
                except Exception: pass
                _gc2.collect(); torch.cuda.empty_cache()
                _meshes_fo = pipe.refiner.run(*_decoded_fo, mc_threshold=args.mc_threshold * 2.0)
            _mesh_fo = _meshes_fo[0]
            # same inverse aniso transform as primary
            try:
                _R = 512
                _vi = (_mesh_fo.vertices + 1.0) * _R / 2.0
                _pitchR = np.asarray(env_pitch) * (64.0 / _R)
                _mesh_fo.vertices = (np.asarray(env_origin) + (_vi + 0.5) * _pitchR).astype(_mesh_fo.vertices.dtype)
            except Exception as _e_t:
                print(f"  [feaoff-2nd] inverse aniso skipped: {_e_t}")
            _fo_dir = Path(args.feaoff_out); _fo_dir.mkdir(parents=True, exist_ok=True)
            _mesh_fo.export(str(_fo_dir/'mesh.obj'))
            _mesh_fo.export(str(_fo_dir/'mesh.glb'))
            print(f"  [feaoff-2nd] saved feaoff mesh → {_fo_dir} "
                  f"V={len(_mesh_fo.vertices):,} F={len(_mesh_fo.faces):,}", flush=True)
        except Exception as _e_fo:
            print(f"  [feaoff-2nd] FAILED ({type(_e_fo).__name__}: {str(_e_fo)[:120]}) — "
                  f"posthoc mesh is safe; run feaoff separately for {args.feaoff_out}", flush=True)

    # ── Save raw SDF (if requested) ────────────────────────────────────────────
    if getattr(args, 'save_raw_sdf', None) and getattr(args, '_captured_sdf', None):
        _cap = args._captured_sdf
        print(f"[save_raw_sdf] {len(_cap)} MC calls intercepted", flush=True)
        for _k, _c in enumerate(_cap):
            print(f"  [{_k}] shape={_c['vol'].shape}  level={_c['level']}", flush=True)
        _tgt = None
        for _c in reversed(_cap):
            if max(_c['vol'].shape) >= 512:
                _tgt = _c; break
        if _tgt is not None:
            _sdf = _tgt['vol'].astype(np.float32)
            _R   = int(_sdf.shape[0])
            _pitch = np.asarray(env_pitch, dtype=np.float64) * (64.0 / _R)
            _path = Path(args.save_raw_sdf)
            _path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(_path),
                sdf=_sdf,
                grid_size=np.int32(_R),
                mc_threshold=np.float32(_tgt['level']),
                env_origin=np.asarray(env_origin, dtype=np.float64),
                env_pitch=_pitch,
                env_pitch_64=np.asarray(env_pitch, dtype=np.float64),
            )
            print(f"[save_raw_sdf] saved → {_path}  shape={_sdf.shape}  "
                  f"range=[{_sdf.min():.3f}, {_sdf.max():.3f}]  "
                  f"size={Path(_path).stat().st_size/1e6:.1f} MB", flush=True)
        else:
            print(f"[save_raw_sdf] WARN: no MC call >=512 found, nothing saved", flush=True)

    print(f"saved → {out_dir}")
    try: loss_csv.close()
    except Exception: pass


if __name__ == "__main__":
    main()
