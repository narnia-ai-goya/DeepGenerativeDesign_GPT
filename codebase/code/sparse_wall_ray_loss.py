"""Differentiable support of the first visible sparse surface on two side views."""

import torch
import torch.nn.functional as F


def prepare_side_fronts(dense_active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """First/last active X cell for each YZ ray; -1 means intentional empty ray."""
    if dense_active.ndim != 3 or dense_active.shape[0] != dense_active.shape[1] or dense_active.shape[1] != dense_active.shape[2]:
        raise ValueError("dense_active must be a cubic [X,Y,Z] array")
    size = dense_active.shape[0]
    xx = torch.arange(size, device=dense_active.device)[:, None, None]
    first = torch.where(dense_active, xx, size).amin(dim=0)
    last = torch.where(dense_active, xx, -1).amax(dim=0)
    return torch.where(first == size, -1, first), last


def sparse_side_front_loss(
    sdf: torch.Tensor,
    coords: torch.Tensor,
    fronts: tuple[torch.Tensor, torch.Tensor],
    iso: float,
    *,
    band_voxels: int = 4,
    min_solid_mass: float = 2.0,
    tail_fraction: float = 0.0,
    occupancy_steepness: float = 10.0,
    grid_size: int = 512,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Penalize side rays whose first dense-support cell is empty in sparse SDF.

    The dense 64-grid support selects which side rays ought to hit material. Only
    the first ``band_voxels`` decoded voxels from each side contribute. Rays that
    are empty in dense stay unpenalized, preserving intended openings. This is a
    one-sided, local front-surface loss; it does not fill the whole dense volume.
    ``sdf < iso`` is solid, matching sparse2mesh/MC in corrected-sign runs.
    """
    first, last = fronts
    coarse_size = first.shape[0]
    if grid_size % coarse_size or band_voxels < 1 or band_voxels > grid_size // coarse_size:
        raise ValueError("band_voxels must fit within one coarse support cell")
    if not 0 < min_solid_mass <= band_voxels:
        raise ValueError("min_solid_mass must lie in (0, band_voxels]")
    if not 0 <= tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in [0, 1]")
    if coords.ndim != 2 or coords.shape[1] != 4 or sdf.shape[0] != coords.shape[0]:
        raise ValueError("expected sparse coords [N,4] and matching SDF [N] or [N,1]")

    scale = grid_size // coarse_size
    xyz = coords[:, 1:].long()
    x, y, z = xyz.unbind(dim=1)
    yc, zc = y // scale, z // scale
    ray = y * grid_size + z
    solid = torch.sigmoid((float(iso) - sdf.reshape(-1)) * occupancy_steepness)
    losses = []
    stats: dict[str, int] = {}
    for name, front, sign in (("minus", first, 1), ("plus", last, -1)):
        coarse_front = front[yc, zc]
        near_index = (coarse_front * scale if sign == 1
                      else (coarse_front + 1) * scale - 1)
        depth = sign * (x - near_index)
        use = (coarse_front >= 0) & (depth >= 0) & (depth < band_voxels)
        ray_mass = torch.zeros(grid_size * grid_size, device=sdf.device, dtype=solid.dtype)
        ray_mass = ray_mass.scatter_add(0, ray[use], solid[use])
        with torch.no_grad():
            counts = torch.zeros(grid_size * grid_size, device=sdf.device, dtype=torch.int32)
            counts.scatter_add_(0, ray[use], torch.ones_like(ray[use], dtype=torch.int32))
            valid = counts >= band_voxels
            stats[f"{name}_valid_rays"] = int(valid.sum().item())
            stats[f"{name}_partial_rays"] = int(((counts > 0) & ~valid).sum().item())
        if valid.any():
            # Soft hinge: almost zero once sufficient solid is present at the
            # front, with gradient on every missing sample in the front band.
            per_ray = F.softplus((min_solid_mass - ray_mass[valid]) * 4.0) / 4.0
            if tail_fraction:
                top_n = max(1, int(per_ray.numel() * tail_fraction))
                losses.append(0.5 * per_ray.mean()
                              + 0.5 * per_ray.topk(top_n, sorted=False).values.mean())
            else:
                losses.append(per_ray.mean())
    if not losses:
        return sdf.reshape(-1).sum() * 0.0, stats
    return torch.stack(losses).mean(), stats
