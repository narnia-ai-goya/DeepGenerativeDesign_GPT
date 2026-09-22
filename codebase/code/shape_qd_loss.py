"""Differentiable frozen-PCA geometry descriptor for the direct Shape-QD pilot."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class FrozenShapePCA:
    def __init__(self, archive_path, target_niche, device):
        data = np.load(archive_path)
        self.mean = torch.as_tensor(data['feature_mean'], dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(data['feature_scale'], dtype=torch.float32, device=device).clamp_min(1e-6)
        self.components = torch.as_tensor(data['pca_components'], dtype=torch.float32, device=device)
        self.archive_scale = torch.as_tensor(data['archive_scale'], dtype=torch.float32, device=device).clamp_min(1e-6)
        centers = torch.as_tensor(data['cvt_centroids'], dtype=torch.float32, device=device)
        if not 0 <= target_niche < len(centers):
            raise ValueError(f'target niche must be in [0,{len(centers)-1}]')
        self.target = centers[target_niche]
        self.target_niche = target_niche

    def embedding(self, occ_prob, bc_mask):
        # Match the archive feature: three 48x48 silhouette/depth maps. Soft OR is stable and
        # differentiable, while depth is a ray-wise material centroid.
        free = occ_prob * (1.0 - bc_mask.float())
        maps = []
        for axis in (2, 1, 0):  # top, front, right: same ordering as representation pilot
            silhouette = 1.0 - torch.prod((1.0 - free).clamp(1e-5, 1.0), dim=axis)
            n = free.shape[axis]
            coords = torch.arange(n, device=free.device, dtype=free.dtype)
            shape = [1, 1, 1]; shape[axis] = n
            depth = (free * coords.reshape(shape)).sum(dim=axis) / free.sum(dim=axis).clamp_min(1e-5)
            depth = (depth / max(1, n - 1)) * silhouette
            pair = torch.stack((silhouette, depth), dim=0)[None]
            maps.append(F.interpolate(pair, size=(48, 48), mode='bilinear', align_corners=False).reshape(-1))
        feature = torch.cat(maps)
        standardized = (feature - self.mean) / self.scale
        z = self.components @ standardized
        return z / self.archive_scale

    def loss(self, occ_logits, bc_mask):
        z = self.embedding(torch.sigmoid(occ_logits), bc_mask)
        return ((z - self.target) ** 2).mean(), z


class PrototypeShapeAnchor:
    """Multi-scale occupancy anchor; evaluation and dense control share the 64³ field."""
    def __init__(self, bank_path, target_niche, device):
        data = np.load(bank_path)
        prototypes = torch.as_tensor(data['prototypes'], dtype=torch.float32, device=device)
        if not 0 <= target_niche < len(prototypes):
            raise ValueError(f'target niche must be in [0,{len(prototypes)-1}]')
        self.target = prototypes[target_niche]
        self.target_niche = target_niche

    def loss(self, occ_logits, bc_mask):
        free = torch.sigmoid(occ_logits) * (1.0 - bc_mask.float())
        target = self.target * (1.0 - bc_mask.float())
        total = torch.zeros((), device=free.device)
        for kernel, weight in ((1, 0.5), (2, 1.0), (4, 2.0), (8, 3.0)):
            if kernel == 1:
                cur, ref = free, target
            else:
                cur = F.avg_pool3d(free[None,None], kernel, kernel).squeeze()
                ref = F.avg_pool3d(target[None,None], kernel, kernel).squeeze()
            total = total + weight * F.smooth_l1_loss(cur, ref)
        return total


class MacroShapeScaffold:
    """Strong low-frequency morphology condition, leaving fine sparse detail free."""
    def __init__(self, bank_path, target_niche, device, pool=4):
        data = np.load(bank_path)
        proto = torch.as_tensor(data['prototypes'], dtype=torch.float32, device=device)
        if not 0 <= target_niche < len(proto):
            raise ValueError(f'target niche must be in [0,{len(proto)-1}]')
        self.target = proto[target_niche]
        self.pool = int(pool)

    def loss(self, occ_logits, bc_mask):
        rho = torch.sigmoid(occ_logits) * (1.0 - bc_mask.float())
        target = self.target * (1.0 - bc_mask.float())
        # Macro cells are probability targets, so BCE supplies a non-saturating local gradient.
        rho = F.avg_pool3d(rho[None,None], self.pool, self.pool).squeeze().clamp(1e-4, 1-1e-4)
        target = F.avg_pool3d(target[None,None], self.pool, self.pool).squeeze()
        return F.binary_cross_entropy(rho, target)
