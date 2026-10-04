"""Pose-calibrated side-depth supervision for the sparse bracket SDF."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def dense_side_depth_reference(
    mesh_path: str | Path,
    dense_active: torch.Tensor,
    origin: np.ndarray,
    pitch64: np.ndarray,
    *,
    grid_size: int = 512,
    depth_samples: int = 16,
    tolerance_mm: float = 4.0,
) -> list[dict[str, torch.Tensor | int]]:
    """Sample the dense mesh's first surface on calibrated X-side orthographic rays.

    The sparse decoder has coordinates in the same 512-grid frame. A reference
    ray is retained only if the dense support is nonempty at that YZ location.
    Positive pysdf values denote material. The reference first-hit offset is
    interpolated between adjacent samples; a tolerance permits variation from
    the coarse dense surface without copying its staircase exactly.
    """
    from pysdf import SDF
    import trimesh

    active = dense_active.detach().cpu().numpy().astype(bool)
    coarse = active.shape[0]
    if grid_size % coarse:
        raise ValueError("grid_size must be divisible by the dense grid size")
    scale = grid_size // coarse
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not mesh.is_watertight:
        raise ValueError(f"dense mesh is not watertight: {mesh_path}")
    field = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    origin = np.asarray(origin, dtype=np.float32)
    pitch = np.asarray(pitch64, dtype=np.float32) * (coarse / grid_size)
    yy, zz = np.indices((grid_size, grid_size), dtype=np.int32)
    yc, zc = yy // scale, zz // scale
    coarse_x = np.arange(coarse, dtype=np.int32)[:, None, None]
    first = np.where(active, coarse_x, coarse).min(axis=0)
    last = np.where(active, coarse_x, -1).max(axis=0)
    references = []
    for front, direction in ((first, 1), (last, -1)):
        has_dense = (front < coarse) if direction == 1 else (front >= 0)
        ray_id = np.flatnonzero(has_dense[yc, zc].ravel())
        y = (ray_id // grid_size).astype(np.int32)
        z = (ray_id % grid_size).astype(np.int32)
        near = front[yc[y, z], zc[y, z]] * scale
        if direction == -1:
            near = (front[yc[y, z], zc[y, z]] + 1) * scale - 1
        # Start outside the 64-grid cell and march inward through its first
        # 512-grid samples. It is enough to identify the dense front surface.
        offsets = np.arange(-scale, depth_samples - scale, dtype=np.int32)
        x = near[:, None] + direction * offsets[None, :]
        q = np.empty((len(ray_id), len(offsets), 3), dtype=np.float32)
        q[:, :, 0] = origin[0] + (x + 0.5) * pitch[0]
        q[:, :, 1] = origin[1] + (y[:, None] + 0.5) * pitch[1]
        q[:, :, 2] = origin[2] + (z[:, None] + 0.5) * pitch[2]
        signed = field(q.reshape(-1, 3)).reshape(len(ray_id), len(offsets))
        inside = signed > 0
        hit = inside.any(axis=1)
        first_hit = inside.argmax(axis=1)
        selected = np.flatnonzero(hit)
        hit_i = first_hit[selected]
        ref_offset = offsets[hit_i].astype(np.float32)
        has_previous = hit_i > 0
        if has_previous.any():
            row = selected[has_previous]
            now = hit_i[has_previous]
            before = signed[row, now - 1]
            after = signed[row, now]
            frac = np.clip(-before / np.maximum(after - before, 1e-9), 0.0, 1.0)
            ref_offset[has_previous] = offsets[now - 1] + frac
        allowed = np.clip(ref_offset + tolerance_mm / (pitch[0] * 1000.0), 0.0,
                          float(depth_samples)).astype(np.float32)
        keep_ray = ray_id[selected]
        lut = np.full(grid_size * grid_size, -1, dtype=np.int32)
        lut[keep_ray] = np.arange(len(keep_ray), dtype=np.int32)
        near_flat = np.zeros(grid_size * grid_size, dtype=np.int32)
        near_flat[keep_ray] = near[selected]
        device = dense_active.device
        references.append({
            "lut": torch.from_numpy(lut).to(device),
            "near": torch.from_numpy(near_flat).to(device),
            "allowed": torch.from_numpy(allowed).to(device),
            "direction": direction,
            "valid_rays": len(keep_ray),
            "mean_reference_offset": float(ref_offset.mean()),
        })
    return references


def sparse_side_depth_loss(
    sdf: torch.Tensor,
    coords: torch.Tensor,
    references: list[dict[str, torch.Tensor | int]],
    iso: float,
    *,
    depth_samples: int = 16,
    grid_size: int = 512,
    steepness: float = 10.0,
    tail_fraction: float = 0.0,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Soft first-hit depth: sum of empty-prefix transmittances along each ray."""
    if not 0 <= tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in [0,1]")
    xyz = coords[:, 1:].long()
    x, y, z = xyz.unbind(dim=1)
    flat_ray = y * grid_size + z
    occupancy = torch.sigmoid((iso - sdf.reshape(-1)) * steepness)
    losses = []
    stats = {}
    for side, ref in enumerate(references):
        compact = ref["lut"][flat_ray].long()
        depth = ref["direction"] * (x - ref["near"][flat_ray].long())
        use = (compact >= 0) & (depth >= 0) & (depth < depth_samples)
        n_rays = ref["allowed"].numel()
        index = compact[use] * depth_samples + depth[use]
        occ = torch.zeros(n_rays * depth_samples, device=sdf.device, dtype=occupancy.dtype)
        occ = occ.scatter_add(0, index, occupancy[use]).reshape(n_rays, depth_samples)
        occ = occ.clamp(0.0, 1.0 - 1e-6)
        with torch.no_grad():
            counts = torch.zeros(n_rays * depth_samples, device=sdf.device, dtype=torch.int32)
            counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.int32))
            has_any = counts.reshape(n_rays, depth_samples).sum(dim=1) > 0
            stats[f"side{side}_rays"] = int(has_any.sum().item())
            stats[f"side{side}_missing"] = int((~has_any).sum().item())
        transmittance = torch.cumprod(1.0 - occ, dim=1)
        first_depth = transmittance.sum(dim=1)
        per_ray = F.softplus((first_depth - ref["allowed"]) * 2.0) / 2.0
        per_ray = per_ray[has_any]
        if per_ray.numel():
            if tail_fraction:
                k = max(1, int(per_ray.numel() * tail_fraction))
                losses.append(0.5 * per_ray.mean()
                              + 0.5 * per_ray.topk(k, sorted=False).values.mean())
            else:
                losses.append(per_ray.mean())
    if not losses:
        return sdf.sum() * 0.0, stats
    return torch.stack(losses).mean(), stats
