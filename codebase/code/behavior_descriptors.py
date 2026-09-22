"""Physics-derived behavior descriptors for verified finite-element results."""
from __future__ import annotations

import math

import numpy as np


def strain_energy_concentration(cell_energy, cell_volume):
    """Return a mesh-size-independent concentration score in ``[0, 1)``.

    The score is ``1 - exp(-KL(p || q))``, where ``p`` is each element's
    fraction of total strain energy and ``q`` is its fraction of total volume.
    It is zero for uniform energy density and increases as load transfer is
    localized into a smaller part of the structure.  Scaling every energy by
    the same factor therefore does not change the descriptor.
    """
    energy = np.maximum(np.asarray(cell_energy, dtype=float), 0.0)
    volume = np.maximum(np.asarray(cell_volume, dtype=float), 0.0)
    if energy.shape != volume.shape:
        raise ValueError("cell_energy and cell_volume must have the same shape")
    energy_sum = float(energy.sum())
    volume_sum = float(volume.sum())
    if energy_sum <= 0 or volume_sum <= 0:
        return 0.0
    p = energy / energy_sum
    q = volume / volume_sum
    active = (p > 0) & (q > 0)
    divergence = float(np.sum(p[active] * np.log(p[active] / q[active])))
    return float(1.0 - math.exp(-max(divergence, 0.0)))


def linear_tet_strain_energy(points, tets, displacement, youngs_modulus, poisson_ratio):
    """Recover ``sigma:epsilon`` energy per linear tetrahedron from nodal displacement."""
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=int)
    displacement = np.asarray(displacement, dtype=float)
    vertices = points[tets]
    jacobian = np.stack((vertices[:, 1]-vertices[:, 0],
                         vertices[:, 2]-vertices[:, 0],
                         vertices[:, 3]-vertices[:, 0]), axis=2)
    inverse = np.linalg.inv(jacobian)
    gradients = np.empty((len(tets), 4, 3), dtype=float)
    gradients[:, 1:, :] = inverse
    gradients[:, 0, :] = -inverse.sum(axis=1)
    displacement_gradient = np.einsum("nic,nis->ncs", displacement[tets], gradients)
    strain = 0.5 * (displacement_gradient + displacement_gradient.transpose(0, 2, 1))
    lame_lambda = (youngs_modulus * poisson_ratio /
                   ((1+poisson_ratio) * (1-2*poisson_ratio)))
    lame_mu = youngs_modulus / (2*(1+poisson_ratio))
    trace = np.trace(strain, axis1=1, axis2=2)
    density = lame_lambda * trace**2 + 2*lame_mu * np.sum(strain*strain, axis=(1, 2))
    volume = np.abs(np.linalg.det(jacobian)) / 6.0
    return np.maximum(density, 0.0) * volume, volume
