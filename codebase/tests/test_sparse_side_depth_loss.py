"""Sparse side-depth gradients should act on recessed front samples."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from sparse_side_depth_loss import sparse_side_depth_loss


def test_recessed_side_depth_has_gradient() -> None:
    lookup = torch.full((16 * 16,), -1, dtype=torch.int32)
    lookup[1 * 16 + 1] = 0
    near = torch.zeros_like(lookup)
    near[1 * 16 + 1] = 4
    reference = [{"lut": lookup, "near": near,
                  "allowed": torch.tensor([1.5]), "direction": 1}]
    coords = torch.tensor([[0, x, 1, 1] for x in range(4, 8)], dtype=torch.int32)
    sdf = torch.tensor([[1.0], [1.0], [-1.0], [-1.0]], requires_grad=True)
    loss, stats = sparse_side_depth_loss(sdf, coords, reference, 0.0,
                                         depth_samples=4, grid_size=16)
    assert stats == {"side0_rays": 1, "side0_missing": 0}
    loss.backward()
    assert sdf.grad[0].item() > 0 and sdf.grad[1].item() > 0
    filled, _ = sparse_side_depth_loss(torch.full((4, 1), -1.0), coords,
                                       reference, 0.0, depth_samples=4,
                                       grid_size=16)
    assert filled.item() < loss.item()
