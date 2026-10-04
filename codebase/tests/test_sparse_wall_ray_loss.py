"""The wall-ray objective must detect a side recess without filling empty rays."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from sparse_wall_ray_loss import prepare_side_fronts, sparse_side_front_loss


def test_side_front_loss_penalizes_recess_and_backpropagates() -> None:
    dense = torch.zeros((4, 4, 4), dtype=torch.bool)
    dense[1:3, 1, 1] = True
    fronts = prepare_side_fronts(dense)
    # The single occupied YZ coarse cell covers 4x4 fine rays, with two
    # samples on the leading face of each side. One ray is recessed.
    xyz = [(x, y, z) for y in range(4, 8) for z in range(4, 8)
           for x in (4, 5, 10, 11)]
    coords = torch.tensor([[0, *v] for v in xyz], dtype=torch.int32)
    sdf = torch.full((len(xyz), 1), -0.5, requires_grad=True)
    with torch.no_grad():
        for j, (x, y, z) in enumerate(xyz):
            if x in (4, 5) and (y, z) == (4, 4):
                sdf[j] = 0.8
    loss, stats = sparse_side_front_loss(
        sdf, coords, fronts, 0.0, band_voxels=2, min_solid_mass=1.0,
        grid_size=16)
    assert stats == {"minus_valid_rays": 16, "minus_partial_rays": 0,
                     "plus_valid_rays": 16, "plus_partial_rays": 0}
    loss.backward()
    bad = [j for j, (x, y, z) in enumerate(xyz)
           if x in (4, 5) and (y, z) == (4, 4)]
    assert all(sdf.grad[j].item() > 0 for j in bad)
    assert torch.isfinite(loss)
    tail_loss, _ = sparse_side_front_loss(
        sdf.detach(), coords, fronts, 0.0, band_voxels=2,
        min_solid_mass=1.0, tail_fraction=0.125, grid_size=16)
    assert tail_loss.item() > loss.item()


def test_empty_reference_ray_has_no_loss() -> None:
    dense = torch.zeros((4, 4, 4), dtype=torch.bool)
    fronts = prepare_side_fronts(dense)
    coords = torch.tensor([[0, 4, 4, 4]], dtype=torch.int32)
    sdf = torch.tensor([[0.9]], requires_grad=True)
    loss, stats = sparse_side_front_loss(sdf, coords, fronts, 0.0,
                                        band_voxels=2, min_solid_mass=1.0,
                                        grid_size=16)
    assert loss.item() == 0.0
    assert stats["minus_valid_rays"] == stats["plus_valid_rays"] == 0
