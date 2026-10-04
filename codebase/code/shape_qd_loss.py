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


class ContrastiveMacroMorphology:
    """Steer only macro cells where a target differs from the sampled dense prior.

    A full prototype BCE is dominated by the large region shared by the current
    sample and the prototype.  This controller snapshots the unguided trajectory
    at warm-up, then applies separate solid/empty likelihoods only where the
    target asks for a material addition or removal at the requested macro scale.
    It is deliberately dense-only: sparse preservation is a separate problem.
    """
    def __init__(self, bank_path, target_niche, device, pool=4, delta=0.15,
                 neutral_weight=0.05):
        data = np.load(bank_path)
        prototypes = torch.as_tensor(data['prototypes'], dtype=torch.float32, device=device)
        if not 0 <= target_niche < len(prototypes):
            raise ValueError(f'target niche must be in [0,{len(prototypes)-1}]')
        if pool < 1 or 64 % int(pool):
            raise ValueError('pool must be a positive divisor of 64')
        self.target = prototypes[target_niche]
        self.target_niche = target_niche
        self.pool = int(pool)
        self.delta = float(delta)
        self.neutral_weight = float(neutral_weight)
        self.reference = None
        self.add_mask = None
        self.remove_mask = None
        self.neutral_mask = None

    def _macro(self, field):
        return F.avg_pool3d(field[None, None], self.pool, self.pool).squeeze()

    def prepare(self, occ_logits, bc_mask):
        """Freeze the first reliable dense prediction as the local posterior reference."""
        free = torch.sigmoid(occ_logits).detach() * (1.0 - bc_mask.float())
        reference = self._macro(free)
        target = self._macro(self.target * (1.0 - bc_mask.float()))
        macro_free = self._macro((1.0 - bc_mask.float())) > 0.999
        self.reference = reference
        self.add_mask = (target - reference >= self.delta) & macro_free
        self.remove_mask = (reference - target >= self.delta) & macro_free
        self.neutral_mask = ~(self.add_mask | self.remove_mask) & macro_free
        return {
            'add_cells': int(self.add_mask.sum().item()),
            'remove_cells': int(self.remove_mask.sum().item()),
            'neutral_cells': int(self.neutral_mask.sum().item()),
        }

    def loss(self, occ_logits, bc_mask):
        if self.reference is None:
            raise RuntimeError('Call prepare() before contrastive morphology loss')
        free = torch.sigmoid(occ_logits) * (1.0 - bc_mask.float())
        current = self._macro(free).clamp(1e-4, 1.0 - 1e-4)
        total = torch.zeros((), device=current.device)
        if self.add_mask.any():
            total = total + F.binary_cross_entropy(current[self.add_mask],
                                                   torch.ones_like(current[self.add_mask]))
        if self.remove_mask.any():
            total = total + F.binary_cross_entropy(current[self.remove_mask],
                                                   torch.zeros_like(current[self.remove_mask]))
        if self.neutral_weight > 0 and self.neutral_mask.any():
            total = total + self.neutral_weight * F.smooth_l1_loss(
                current[self.neutral_mask], self.reference[self.neutral_mask])
        return total


class LocalPcaTransport(FrozenShapePCA):
    """Small, repeated minimum-norm transport in frozen shape-PCA measure space.

    This is not a prototype reconstruction loss.  At a reliable dense posterior
    state it asks for a bounded displacement toward a chosen niche, then uses
    the local descriptor Jacobian to find the smallest VAE-latent move that
    realizes that displacement to first order.
    """
    def __init__(self, archive_path, target_niche, device, dims=2, radius=0.5,
                 ridge=1e-3, max_relative_latent_step=0.05):
        super().__init__(archive_path, target_niche, device)
        self.dims = int(dims)
        if not 1 <= self.dims <= len(self.target):
            raise ValueError(f'dims must be in [1,{len(self.target)}]')
        self.radius = float(radius)
        self.ridge = float(ridge)
        self.max_relative_latent_step = float(max_relative_latent_step)
        self.updates = 0

    def desired_displacement(self, embedding):
        delta = self.target[:self.dims] - embedding[:self.dims]
        norm = delta.norm()
        if norm <= 1e-8:
            return torch.zeros_like(delta)
        return delta * min(1.0, self.radius / float(norm.detach()))

    def step(self, z_in, embedding):
        """Return a detached latent move and diagnostics without mutating z_in."""
        desired = self.desired_displacement(embedding)
        if not bool(torch.any(desired)):
            return torch.zeros_like(z_in), {'requested_norm': 0.0, 'latent_norm': 0.0,
                                             'predicted_norm': 0.0, 'clipped': False}
        rows = []
        for dim in range(self.dims):
            grad = torch.autograd.grad(embedding[dim], z_in, retain_graph=True,
                                       create_graph=False)[0]
            rows.append(grad.reshape(-1))
        jacobian = torch.stack(rows)
        gram = jacobian @ jacobian.T
        eye = torch.eye(self.dims, dtype=gram.dtype, device=gram.device)
        coeff = torch.linalg.solve(gram + self.ridge * eye, desired)
        delta = (jacobian.T @ coeff).reshape_as(z_in)
        latent_norm = z_in.detach().norm()
        # A zero-valued synthetic or freshly initialized latent must still be
        # transportable; real diffusion latents are normally far above this floor.
        max_norm = self.max_relative_latent_step * latent_norm.clamp_min(1.0)
        delta_norm = delta.norm()
        clipped = bool(delta_norm > max_norm)
        if clipped:
            delta = delta * (max_norm / delta_norm.clamp_min(1e-12))
        predicted = jacobian @ delta.reshape(-1)
        return delta.detach(), {
            'requested_norm': float(desired.norm().detach().item()),
            'latent_norm': float(delta.norm().detach().item()),
            'predicted_norm': float(predicted.norm().detach().item()),
            'clipped': clipped,
        }
