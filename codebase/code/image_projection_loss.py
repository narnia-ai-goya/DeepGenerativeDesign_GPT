"""Training-free image-to-dense top-projection guidance."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class ImageProjectionLoss:
    """Match dense active-token silhouettes to registered orthographic image masks.

    Legacy NPZ files contain `target`/`weight` for the top (X,Y) projection.
    Multi-view NPZ files may instead contain target_front/weight_front (X,Z),
    target_right/weight_right (Y,Z), target_top/weight_top (X,Y), and optional
    target_rear/weight_rear (X,Z).
    A camera_grid_<view> array replaces an axis projection with orthographic
    ray samples in the exact image camera frame (grid_sample order Z,Y,X).
    """

    def __init__(self, path: str, device: torch.device, active_threshold: float = 0.1,
                 temperature: float = 0.1):
        data = np.load(path)
        if 'active_threshold' in data.files:
            active_threshold = float(data['active_threshold'])
        if not 0 < active_threshold < 1 or temperature <= 0:
            raise ValueError('active_threshold must be in (0,1) and temperature must be positive')
        specifications = [('top', 'target', 'weight', 2)] \
            if 'target' in data.files else [
                ('front', 'target_front', 'weight_front', 1),
                ('right', 'target_right', 'weight_right', 0),
                ('top', 'target_top', 'weight_top', 2),
                ('rear', 'target_rear', 'weight_rear', 1)]
        self.projections = []
        for name, target_key, weight_key, axis in specifications:
            if target_key not in data.files:
                continue
            if weight_key not in data.files:
                raise ValueError(f'{target_key} lacks {weight_key}')
            target = np.asarray(data[target_key], dtype=np.float32)
            weight = np.asarray(data[weight_key], dtype=np.float32)
            if target.shape != (64, 64) or weight.shape != (64, 64):
                raise ValueError(f'{name} projection arrays must be 64x64')
            if not np.isfinite(target).all() or not np.isfinite(weight).all():
                raise ValueError(f'{name} projection target contains non-finite values')
            target_t = torch.as_tensor(np.clip(target, 0, 1), device=device)
            weight_t = torch.as_tensor(np.clip(weight, 0, 1), device=device)
            if weight_t.sum() < 1:
                raise ValueError(f'{name} projection target contains no supervised pixels')
            camera_key = f'camera_grid_{name}'
            if camera_key in data.files:
                camera_grid = np.asarray(data[camera_key], dtype=np.float32)
                if (camera_grid.ndim != 4 or camera_grid.shape[1:] != (64, 64, 3)
                        or not np.isfinite(camera_grid).all()):
                    raise ValueError(f'{camera_key} must be finite [rays,64,64,3]')
                projection = torch.as_tensor(camera_grid, device=device)
            else:
                projection = axis
            self.projections.append((name, projection, target_t, weight_t))
        if not self.projections:
            raise ValueError('projection file contains no target')
        self.temperature = temperature
        self.logit_threshold = float(np.log(active_threshold / (1 - active_threshold)))

    def loss(self, occ_logits: torch.Tensor, envelope_mask: torch.Tensor) -> torch.Tensor:
        if occ_logits.shape != (64, 64, 64):
            raise ValueError(f'expected 64x64x64 occupancy logits, got {tuple(occ_logits.shape)}')
        inside_logits = occ_logits.masked_fill(envelope_mask <= 0.5, -20.0)
        losses = []
        for _, projection, target, weight in self.projections:
            if isinstance(projection, torch.Tensor):
                sampled = F.grid_sample(
                    inside_logits[None, None], projection[None],
                    mode='bilinear', padding_mode='zeros', align_corners=True)[0, 0]
                valid = (projection.abs() <= 1).all(dim=-1)
                sampled = sampled.masked_fill(~valid, -20.0)
                projected_logits = self.temperature * torch.logsumexp(
                    sampled / self.temperature, dim=0)
            else:
                projected_logits = self.temperature * torch.logsumexp(
                    inside_logits / self.temperature, dim=projection)
            projected_logits = projected_logits - self.logit_threshold
            per_pixel = F.binary_cross_entropy_with_logits(
                projected_logits, target, reduction='none')
            losses.append((per_pixel * weight).sum() / weight.sum().clamp_min(1.0))
        return torch.stack(losses).mean()
