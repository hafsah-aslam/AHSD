"""P1c detectors (AMENDMENT_03). Every score: higher = more anomalous; never flipped.

Post-hoc scores on the supervised AHSD backbone: MSP, Energy, Mahalanobis and kNN
on penultimate features. Benign-only detectors on flattened windows: Isolation
Forest, OCSVM, PCA reconstruction, Autoencoder. `fit(reference)` sets the benign
reference; re-anchoring = calling fit again on anchor windows.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

KNN_K = 10


def msp_score(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return 1.0 - p.max(1)


def energy_score(logits: np.ndarray) -> np.ndarray:
    m = logits.max(1, keepdims=True)
    return -(m[:, 0] + np.log(np.exp(logits - m).sum(1)))


class BenignMahalanobis:
    def fit(self, F: np.ndarray):
        self.mu = F.mean(0)
        self.prec = LedoitWolf().fit(F).precision_
        return self

    def score(self, F):
        d = F - self.mu
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec, d), 0))


class BenignKNN:
    def fit(self, F):
        self.nn = NearestNeighbors(n_neighbors=min(KNN_K, len(F))).fit(_l2(F))
        return self

    def score(self, F):
        return self.nn.kneighbors(_l2(F))[0][:, -1]


def _l2(F):
    return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)


def _flat(X):
    return X.reshape(len(X), -1).astype(np.float32)


class IF:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X):
        self.m = IsolationForest(n_estimators=200, random_state=self.seed).fit(_flat(X))
        return self

    def score(self, X):
        return -self.m.score_samples(_flat(X))


class OCSVM:
    def __init__(self, seed=None):
        pass

    def fit(self, X):
        self.m = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale").fit(_flat(X))
        return self

    def score(self, X):
        return -self.m.decision_function(_flat(X))


class PCARecon:
    def __init__(self, seed=None):
        pass

    def fit(self, X):
        F = _flat(X)
        n = min(len(F) - 1, F.shape[1])
        full = PCA(n_components=n, svd_solver="full").fit(F)
        k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), 0.95) + 1)
        self.m = PCA(n_components=min(k, n), svd_solver="full").fit(F)
        self.k = self.m.n_components_
        return self

    def score(self, X):
        F = _flat(X)
        R = self.m.inverse_transform(self.m.transform(F))
        return ((F - R) ** 2).mean(1)


class AE:
    def __init__(self, seed, epochs=30, batch=128, lr=1e-3):
        self.seed, self.epochs, self.batch, self.lr = seed, epochs, batch, lr

    def fit(self, X):
        torch.manual_seed(self.seed)
        F = torch.from_numpy(_flat(X))
        d = F.shape[1]
        self.m = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
                               nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, d))
        opt = torch.optim.Adam(self.m.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(F), generator=g).split(self.batch):
                loss = ((self.m(F[idx]) - F[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        F = torch.from_numpy(_flat(X))
        return ((self.m(F) - F) ** 2).mean(1).numpy()


BENIGN_ONLY = {"IF": IF, "OCSVM": OCSVM, "PCA": PCARecon, "AE": AE}
DETECTORS = ["AHSD_stress", "MSP", "Energy", "Mahalanobis", "kNN", "IF", "OCSVM", "PCA", "AE"]
NO_REFERENCE = {"MSP", "Energy"}  # re-anchoring does not apply (A3 operational choices)
