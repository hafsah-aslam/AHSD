"""Competing detectability predictors (spec §6 S2, §7).

All are computed per held-out class c on the same LOACO fold as S2:
  wasserstein   per-feature 1-D Wasserstein distance between class-c flows and
                the fold's training-attack flows, averaged over features
                (Sarhan et al. 2021). Expected sign vs AUC: negative.
  mahalanobis   mean Mahalanobis distance of class-c window-mean features from
                benign training windows (Ledoit-Wolf covariance). Expected: +.
  stationarity  mean over class-c windows of mean_t ‖z_t − z_{t−1}‖₂ /
                sqrt(Σ_d var_t z_d), on the model's embeddings. Expected: +.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf

EXPECTED_SIGN = {"D_pred": 1, "mahalanobis": 1, "wasserstein": -1, "stationarity": 1}


def _flows(X: np.ndarray, cap: int, rng) -> np.ndarray:
    F = X.reshape(-1, X.shape[-1])
    return F if len(F) <= cap else F[rng.choice(len(F), cap, replace=False)]


def wasserstein_to_train_attacks(X_class: np.ndarray, X_train_attack: np.ndarray,
                                 cap: int = 50_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a, b = _flows(X_class, cap, rng), _flows(X_train_attack, cap, rng)
    return float(np.mean([wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])]))


class BenignMahalanobis:
    def __init__(self, X_benign_train: np.ndarray):
        M = X_benign_train.mean(1)
        self.mu = M.mean(0)
        self.prec = LedoitWolf().fit(M).precision_

    def __call__(self, X: np.ndarray) -> np.ndarray:
        d = X.mean(1) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec, d))


def stationarity_index(z: np.ndarray) -> np.ndarray:
    """z: (N, T, D) -> (N,)."""
    step = np.linalg.norm(np.diff(z, axis=1), axis=2).mean(1)
    spread = np.sqrt(z.var(axis=1).sum(1))
    return step / np.maximum(spread, 1e-12)
