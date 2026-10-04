"""Differentiable compliance loss via FEniCS subprocess call.

The FEA solver runs in a separate conda env (fenics) due to dolfinx
dependency. We bridge it to torch via:

    forward(occ_logits, bracket_mask, voxel_centers, ...):
        ρ_grid = sigmoid(occ_logits)
        ρ_inside = ρ_grid[bracket_mask]  (1D, ~10K values)
        save ρ_inside.npy + nodes.npy
        subprocess: fenics_fea_bracket.py → dc.npy + compliance.json
        return compliance (torch scalar)

    backward(grad_compliance):
        ∂L/∂ρ_inside = grad_compliance * dc[inside]  (from FEA adjoint)
        scatter back to 64³ grid
        ∂L/∂occ_logits = ∂L/∂ρ_grid * σ'(occ_logits)   (sigmoid derivative)
        return that

Usage in d3ds2_flowdps.py:
    from fea_compliance_loss import fea_compliance_loss
    if fea_w > 0 and step_frac > fea_warmup and (i % fea_every_n == 0):
        comp = fea_compliance_loss(occ_logits, bracket_mask, voxel_centers,
                                     fenics_bin, fea_script, domain_dir)
        loss = loss + fea_w * comp

paper ref: main text, Physics Guidance: differentiable FEM compliance C = f^T u
backpropagated into the latent (with the sensitivity filter, main text);
Supplementary Sparse-Stage Loss Terms term (6) Sparse-FEM compliance for the sparse stage.
"""
from __future__ import annotations
import subprocess, json, os, re
from pathlib import Path
import numpy as np
import torch


FENICS_BIN_DEFAULT  = os.environ.get('FEA_FENICS_BIN', 'python')
FEA_SCRIPT_DEFAULT  = os.environ.get('FEA_FENICS_SCRIPT', './code/fenics_fea_bracket.py')
# Per-process unique work_dir so concurrent MESH jobs don't clobber each
# other's rho.npy / nodes.npy / dc.npy / dc_info.json. Override via FEA_WORK_DIR env.
WORK_DIR_DEFAULT    = os.environ.get('FEA_WORK_DIR', f'/tmp/fea_loss_{os.getpid()}')


def call_fenics_fea(rho_inside: np.ndarray, nodes_inside: np.ndarray,
                    domain_dir: str, mesh_cache: str,
                    fenics_bin: str = FENICS_BIN_DEFAULT,
                    fea_script: str = FEA_SCRIPT_DEFAULT,
                    work_dir: str = WORK_DIR_DEFAULT,
                    mesh_size: float = 0.006, penal: float = 3.0,
                    verbose: bool = False,
                    vtk_out: str = None) -> tuple[float, np.ndarray]:
    """Run a single FEA call. Returns (compliance, sensitivity_on_nodes).

    vtk_out: if set, also write displacement + E + ‖u‖ to this .pvd path
             (gets overwritten each call — last call wins).
    """
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    rho_path = f'{work_dir}/rho.npy'
    nodes_path = f'{work_dir}/nodes.npy'
    dc_path = f'{work_dir}/dc.npy'
    info_path = f'{work_dir}/dc_info.json'
    np.save(rho_path, rho_inside.astype(np.float64))
    np.save(nodes_path, nodes_inside.astype(np.float64))

    import os as _os
    cmd = [fenics_bin, fea_script,
           '--domain-dir', domain_dir,
           '--density', rho_path, '--nodes', nodes_path,
           '--output', dc_path,
           '--mesh-cache', mesh_cache,
           '--mesh-size', str(mesh_size),
           '--penal', str(penal),
           '--load-magnitude', _os.environ.get('FEA_LOAD_MAGNITUDE', '42300.0')]
    second_stl = _os.environ.get('FEA_SECOND_LOAD_STL')
    if second_stl:
        cmd += ['--second-load-stl', second_stl,
                '--second-load-magnitude', _os.environ.get('FEA_SECOND_LOAD_MAGNITUDE', '200'),
                '--second-load-mode', _os.environ.get('FEA_SECOND_LOAD_MODE', 'y')]
    if _os.environ.get('FEA_DIAG', '0') == '1':
        print(f'  [FEA-CMD] load={_os.environ.get("FEA_LOAD_MAGNITUDE", "42300.0")} '
              f'rho_in: min={rho_inside.min():.3f} max={rho_inside.max():.3f} mean={rho_inside.mean():.3f} n={len(rho_inside)}', flush=True)
    # vtk_out via param OR env (FEA_VTK_OUT)
    vtk = vtk_out or _os.environ.get('FEA_VTK_OUT', None)
    if vtk:
        Path(vtk).parent.mkdir(parents=True, exist_ok=True)
        cmd += ['--vtk-out', vtk]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        if verbose:
            print('FEA STDERR:', res.stderr[-500:])
        detail = (res.stderr or res.stdout).strip().splitlines()[-1:]
        raise RuntimeError(f'fenics_fea_bracket.py failed (exit {res.returncode}): '
                           f'{detail[0][:300] if detail else "no solver output"}')
    fixed = re.search(r'Dirichlet:\s*(\d+)\s+nodes', res.stdout)
    loaded = re.search(r'Load:\s*(\d+)\s+nodes', res.stdout)
    if fixed is None or loaded is None or int(fixed.group(1)) == 0 or int(loaded.group(1)) == 0:
        raise RuntimeError('FEniCS returned no valid fixed/load nodes; refusing a zero-load '
                           f'compliance gradient. Solver tail: {res.stdout[-300:]}')
    if second_stl:
        second_loaded = re.search(r'Load2:\s*(\d+)\s+nodes', res.stdout)
        if second_loaded is None or int(second_loaded.group(1)) == 0 or \
                'Load combination: simultaneous' not in res.stdout:
            raise RuntimeError('FEniCS did not apply both loads in one solve')
    if _os.environ.get('FEA_DIAG', '0') == '1':
        # fenics succeeded — print last few stdout lines to see compliance value reported
        for ln in res.stdout.strip().split('\n')[-3:]:
            print('  [FENICS-OUT] ' + ln, flush=True)
    if verbose:
        for line in res.stdout.strip().split('\n')[-3:]:
            print('  ' + line)

    with open(info_path) as f: info = json.load(f)
    dc = np.load(dc_path)
    return float(info['compliance']), dc


# Module-level cache: last call's ∂C/∂ρ_voxel (64³). For stress-aware
# active-voxel-set expansion (gradient-free, set operation only).
LAST_FEA_DC_VOXEL: np.ndarray | None = None
LAST_FEA_RHO_VOXEL: np.ndarray | None = None


class FEAComplianceLoss(torch.autograd.Function):
    """Subprocess-based compliance, manual backward via stored sensitivity."""

    @staticmethod
    def forward(ctx, occ_logits: torch.Tensor,
                bracket_mask_np: np.ndarray,
                nodes_inside_np: np.ndarray,
                domain_dir: str, mesh_cache: str,
                mesh_size: float = 0.006, penal: float = 3.0,
                verbose: bool = False, vtk_out: str = None,
                bc_mask_np: np.ndarray = None):
        # ρ = sigmoid(occ_logits) at inside voxels only.
        # If bc_mask_np provided: force ρ=1 at BC voxels (peg region always solid,
        # not a learning target). Gradient at BC also zeroed in backward.
        with torch.no_grad():
            rho_grid = torch.sigmoid(occ_logits)
            mask_t = torch.from_numpy(bracket_mask_np).to(occ_logits.device).bool()
            if bc_mask_np is not None:
                bc_mask_t = torch.from_numpy(bc_mask_np).to(occ_logits.device).bool()
                rho_grid = torch.where(bc_mask_t, torch.ones_like(rho_grid), rho_grid)
            else:
                bc_mask_t = None
            rho_inside_t = rho_grid[mask_t]
            # cast to fp32 for numpy (fenics needs fp32+); occ_logits dtype preserved via dc_t below
            rho_inside_np = rho_inside_t.detach().cpu().float().numpy()

        comp_val, dc_np = call_fenics_fea(rho_inside_np, nodes_inside_np,
                                            domain_dir, mesh_cache,
                                            mesh_size=mesh_size, penal=penal,
                                            verbose=verbose, vtk_out=vtk_out)
        # Save dc for backward — keep fp32 to avoid fp16 overflow on sensitivity values
        # (dc can be 1e10~1e14; fp16 max=65504 → overflow without normalize)
        dc_t = torch.from_numpy(dc_np).to(occ_logits.device, dtype=torch.float32)
        # Save bc_mask_t (or dummy) for backward: force gradient=0 at BC voxels.
        if bc_mask_t is None:
            bc_mask_t = torch.zeros_like(mask_t)
        ctx.save_for_backward(occ_logits, mask_t, dc_t, bc_mask_t)
        # === stash full-grid dc + rho for stress-aware index expansion ===
        global LAST_FEA_DC_VOXEL, LAST_FEA_RHO_VOXEL
        dc_grid = np.zeros(bracket_mask_np.shape, dtype=np.float32)
        dc_grid[bracket_mask_np] = dc_np
        rho_grid_np = np.zeros(bracket_mask_np.shape, dtype=np.float32)
        rho_grid_np[bracket_mask_np] = rho_inside_np
        LAST_FEA_DC_VOXEL = dc_grid
        LAST_FEA_RHO_VOXEL = rho_grid_np
        # Return fp32 to avoid fp16 overflow (real compliance values often exceed fp16 max=65504).
        # The hook normalizes by |fea_loss.detach()| so absolute magnitude is fine; backward
        # downcasts grad to occ_logits.dtype as needed.
        return torch.tensor(comp_val, device=occ_logits.device, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        # paper ref: main text, Physics Guidance: backprop of FEM compliance C = f^T u into the
        #            latent via ∂C/∂ρ, followed by the sensitivity filter (main text).
        occ_logits, mask_t, dc_t, bc_mask_t = ctx.saved_tensors
        # ∂L/∂ρ_inside = grad_output * dc_inside  (dc is ∂C/∂ρ)
        # CRITICAL: (grad_output × dc) in fp32 to avoid overflow — dc ~1e10-1e14, fp16 max=65504.
        # But result (after × small fea_w) is small ≪ 1, so safe to cast to fp16 immediately after.
        scaled_fp32 = grad_output.float() * dc_t.float()             # fp32, fits in fp16 after scale
        d_rho_grid = torch.zeros_like(occ_logits)                    # match occ_logits dtype (fp16)
        d_rho_grid[mask_t] = scaled_fp32.to(occ_logits.dtype)
        # ---- topology-optimization sensitivity filter (Sigmund 1997) ----
        import os
        sig = float(os.environ.get('FEA_SENS_FILTER_SIGMA', '1.5'))
        if sig > 0:
            from scipy.ndimage import gaussian_filter as _gf
            arr = d_rho_grid.detach().cpu().float().numpy()
            arr = _gf(arr, sigma=sig)
            d_rho_grid = torch.from_numpy(arr).to(d_rho_grid.device, dtype=occ_logits.dtype)
        # σ'(logits) = ρ · (1 − ρ)  — all fp16 (rho ∈ [0,1] safe)
        rho_grid = torch.sigmoid(occ_logits)
        if os.environ.get('SP_FEA_VOLUME_NEUTRAL', '0') == '1':
            # Project the compliance sensitivity onto the tangent of constant
            # first-order material amount.  This removes the trivial uniform
            # "add material everywhere" direction while retaining spatial
            # sensitivity contrast, so guidance expresses a load path even
            # when the outer experiment does not impose a volume target.
            # For a logit update, d(rho)/d(logit)=rho', hence the first-order
            # density change is weighted by rho'^2.
            free_t = mask_t & ~bc_mask_t
            rho_prime = rho_grid * (1 - rho_grid)
            weights = rho_prime.square()
            denom = weights[free_t].sum().clamp_min(1e-12)
            shift = -(d_rho_grid[free_t] * weights[free_t]).sum() / denom
            d_rho_grid = torch.where(free_t, d_rho_grid + shift, d_rho_grid)
        d_logits = d_rho_grid * rho_grid * (1 - rho_grid)
        # force BC ρ=1: gradient=0 at BC voxels (not a learning subject)
        if bc_mask_t is not None:
            d_logits = torch.where(bc_mask_t, torch.zeros_like(d_logits), d_logits)
        # Other inputs are non-tensor → no grad.
        # 10 args in forward (ctx excl): occ_logits, bracket_mask_np, nodes_inside_np,
        #   domain_dir, mesh_cache, mesh_size, penal, verbose, vtk_out, bc_mask_np
        return d_logits, None, None, None, None, None, None, None, None, None


def fea_compliance_loss(occ_logits: torch.Tensor,
                         bracket_mask_np: np.ndarray,
                         nodes_inside_np: np.ndarray,
                         domain_dir: str, mesh_cache: str,
                         mesh_size: float = 0.006, penal: float = 3.0,
                         verbose: bool = False,
                         mc_aligned: bool = False,
                         mc_threshold: float = 0.1,
                         sharpness: float = 8.0,
                         bc_mask_np: np.ndarray = None) -> torch.Tensor:
    """Compliance loss via fenics subprocess.

    mc_aligned=True shifts/sharpens sigmoid so that raw rho > mc_threshold
    voxel ≈ 1 and raw rho < mc_threshold voxel ≈ 0 — i.e. FEA "solid"
    definition matches the mesh extracted at mc_threshold. autograd handles
    the chain rule through the linear (sharp_logits = (occ - logit_mc) * s)
    transform, so no FEAComplianceLoss internals need to change.

    Default (mc_aligned=False) preserves legacy behavior (raw sigmoid).
    """
    if mc_aligned:
        import math
        logit_mc = math.log(mc_threshold / (1 - mc_threshold))
        sharp_logits = (occ_logits - logit_mc) * sharpness
    else:
        sharp_logits = occ_logits
    return FEAComplianceLoss.apply(
        sharp_logits, bracket_mask_np, nodes_inside_np,
        domain_dir, mesh_cache, mesh_size, penal, verbose, None, bc_mask_np)


def voxel_centers_inside(bracket_npz_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Load voxel grid, return (mask 64³ bool, nodes Nx3 m, pitch m).

    nodes are voxel centers in world coords (matching how the npz was generated).
    """
    import trimesh
    d = np.load(bracket_npz_path)
    mask = d['bracket'].astype(bool)
    pitch = float(d['pitch'])
    # Origin is implicit (from voxelize_caliper/voxelize_bracket).
    # We recompute it from the STL — caller can override.
    raise NotImplementedError("supply origin from caller")


def voxel_nodes_from_stl(bracket_npz_path: str, stl_path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute (mask, nodes Nx3, pitch). Uses npz['origin'] if present (preferred), else STL bbox margin.
    Accepts npz with either 'bracket' (bc_proper.npz) or 'occupancy' (bracket_voxel_stl.npz) key."""
    import trimesh
    d = np.load(bracket_npz_path)
    if 'bracket' in d.files:
        mask = d['bracket'].astype(bool)
    elif 'occupancy' in d.files:
        mask = d['occupancy'].astype(bool)
    else:
        raise KeyError(f"npz {bracket_npz_path} has no 'bracket' or 'occupancy' key: {list(d.files)}")
    pitch = float(d['pitch'])
    if 'origin' in d.files:
        origin = d['origin']
    else:
        ds = trimesh.load(stl_path, force='mesh')
        bbox = ds.bounds.copy()
        ext = bbox[1] - bbox[0]
        margin = 0.05 * ext.max()
        bbox[0] -= margin; bbox[1] += margin
        origin = bbox[0]
    ii = np.argwhere(mask)
    nodes = origin + (ii + 0.5) * pitch
    return mask, nodes, pitch


if __name__ == '__main__':
    # smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bracket-npz', default='data_real/bracket/voxel.npz')
    ap.add_argument('--stl', default='data_real/bracket/original_DesignSpace.stl')
    ap.add_argument('--domain-dir', default='data_real/bracket/fea_domain')
    ap.add_argument('--mesh-cache', default='/tmp/bracket_fea.msh')
    args = ap.parse_args()

    mask, nodes, pitch = voxel_nodes_from_stl(args.bracket_npz, args.stl)
    print(f'mask: {mask.sum()} inside voxels, pitch {pitch*1000:.3f}mm')

    # Build a dummy occ_logits that gives ρ=0.5 everywhere
    occ = torch.zeros(mask.shape, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu', requires_grad=True)
    comp = fea_compliance_loss(occ, mask, nodes, args.domain_dir, args.mesh_cache, verbose=True)
    print(f'compliance(0) = {comp.item():.4e}')

    # Test backward
    comp.backward()
    print(f'occ.grad — sum={occ.grad.sum().item():.3e}  abs.mean={occ.grad.abs().mean().item():.3e}')
    print(f'  nonzero locations: {(occ.grad != 0).sum().item()} (= inside voxel count)')
