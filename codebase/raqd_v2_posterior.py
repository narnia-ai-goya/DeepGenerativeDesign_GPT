"""Small exact GP for mixed categorical/continuous RA-QD genomes."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

from qd_archive import PARAMETERS
from raqd_posterior import STYLES, genome_to_unit


def encode(genomes):
    encoded = [genome_to_unit(g) for g in genomes]
    return (np.asarray([v[0] for v in encoded], dtype=int),
            np.asarray([v[1] for v in encoded], dtype=float))


def matern52_mixed(style_a, x_a, style_b, x_b, lengthscales, style_rho):
    delta = (x_a[:, None, :] - x_b[None, :, :]) / lengthscales
    radius = np.sqrt(np.sum(delta * delta, axis=2))
    root5r = np.sqrt(5.0) * radius
    continuous = (1 + root5r + 5 * radius * radius / 3) * np.exp(-root5r)
    categorical = np.where(style_a[:, None] == style_b[None, :], 1.0, style_rho)
    return continuous * categorical


class MixedExactGP:
    """Exact standardized GP with style-specific realization noise."""

    def __init__(self):
        self.fitted = False

    @staticmethod
    def unpack(theta):
        d = len(PARAMETERS)
        lengthscales = np.exp(theta[:d])
        style_rho = .001 + .948 * expit(theta[d])
        noise = np.exp(theta[d + 1:d + 1 + len(STYLES)])
        return lengthscales, style_rho, noise

    def fit(self, genomes, raw_y, theta=None, optimize=True):
        self.styles, self.x = encode(genomes)
        raw_y = np.asarray(raw_y, dtype=float)
        self.center = float(raw_y.mean())
        self.scale = max(float(raw_y.std()), 1e-10)
        self.y = (raw_y - self.center) / self.scale
        d = self.x.shape[1]
        initial = np.r_[np.log(np.full(d, .45)), 0., np.log(np.full(len(STYLES), .35))]
        bounds = [(np.log(.04), np.log(3.0))] * d + [(-7., 7.)] + [
            (np.log(.025), np.log(2.0))] * len(STYLES)

        def objective(candidate):
            lengthscales, rho, noise = self.unpack(candidate)
            k = matern52_mixed(self.styles, self.x, self.styles, self.x, lengthscales, rho)
            k.flat[::len(k) + 1] += noise[self.styles] ** 2 + 1e-7
            try:
                factor = cho_factor(k, lower=True, check_finite=False)
                alpha = cho_solve(factor, self.y, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e20
            return .5 * float(self.y @ alpha) + np.log(np.diag(factor[0])).sum()

        if theta is None:
            theta = initial
        if optimize:
            result = minimize(objective, theta, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 300, "ftol": 1e-10})
            theta = result.x
            self.optimization = {"success": bool(result.success), "nll": float(result.fun),
                                 "message": str(result.message), "iterations": int(result.nit)}
        else:
            self.optimization = {"success": True, "nll": float(objective(theta)),
                                 "message": "fixed hyperparameters", "iterations": 0}
        self.theta = np.asarray(theta)
        self.lengthscales, self.style_rho, self.noise = self.unpack(self.theta)
        k = matern52_mixed(self.styles, self.x, self.styles, self.x,
                           self.lengthscales, self.style_rho)
        k.flat[::len(k) + 1] += self.noise[self.styles] ** 2 + 1e-7
        self.factor = cho_factor(k, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, self.y, check_finite=False)
        self.fitted = True
        return self

    def predict(self, genomes, realization=True):
        if not self.fitted:
            raise RuntimeError("fit must be called before predict")
        styles, x = encode(genomes)
        cross = matern52_mixed(styles, x, self.styles, self.x,
                               self.lengthscales, self.style_rho)
        mean = cross @ self.alpha
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        variance = np.maximum(1e-10, 1 - np.sum(cross * solved.T, axis=1))
        if realization:
            variance += self.noise[styles] ** 2
        return self.center + self.scale * mean, self.scale * np.sqrt(variance)

    def interval_probability(self, genomes, lower, upper):
        mean, std = self.predict(genomes, realization=True)
        probability = norm.cdf((upper - mean) / std) - norm.cdf((lower - mean) / std)
        return mean, std, probability

    def diagnostics(self):
        return {
            "kernel": "equicorrelated categorical × Matern-5/2 ARD",
            "lengthscales": {name: float(value) for name, value in
                             zip(PARAMETERS, self.lengthscales)},
            "style_rho": float(self.style_rho),
            "realization_noise_standardized": {style: float(value) for style, value in
                                                zip(STYLES, self.noise)},
            "response_mean": self.center,
            "response_std": self.scale,
            "optimization": self.optimization,
        }
