"""S2-dir (EXPLORATORY; AMENDMENT_02 A2.3). Never used in any gate.

    u = W (z − E₀),  W = Σ_s^(−1/2),  Σ_s = 0.9 Σ_benign + 0.1 (tr Σ_benign / D) I
    P_{c,d}(ω)      per-channel one-sided power of u (S2 normalisation, no channel averaging)
    raw(c)          = Σ_d Σ_ω |H(ω)|² (P_{c,d}(ω) − P_{benign,d}(ω))²
    Q_b             = the same quantity between two random halves of the benign windows
    S2dir(c)        = raw(c) / std over seeds of Q_b   (the normaliser is applied by the caller)
"""

from __future__ import annotations

import numpy as np

SHRINKAGE = 0.1


def whitener(z_benign_minus_E0: np.ndarray, shrinkage: float = SHRINKAGE) -> np.ndarray:
    """W = Σ_s^(−1/2) from benign (z − E₀) of shape (N, T, D), pooled over windows and time."""
    X = z_benign_minus_E0.reshape(-1, z_benign_minus_E0.shape[-1]).astype(np.float64)
    S = np.cov(X, rowvar=False)
    D = S.shape[0]
    S = (1 - shrinkage) * S + shrinkage * (np.trace(S) / D) * np.eye(D)
    w, V = np.linalg.eigh(S)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def channel_power(u: np.ndarray) -> np.ndarray:
    """(N, T, D) -> mean over windows of the one-sided per-channel power, shape (T//2+1, D)."""
    N, T, D = u.shape
    P = np.abs(np.fft.rfft(u, axis=1)) ** 2 / T ** 2
    w = np.full(P.shape[1], 2.0)
    w[0] = 1.0
    if T % 2 == 0:
        w[-1] = 1.0
    return (P * w[None, :, None]).mean(0)


def raw_distance(g2: np.ndarray, P_a: np.ndarray, P_b: np.ndarray) -> float:
    """Σ_d Σ_ω |H(ω)|² (P_a − P_b)²; g2 has shape (T//2+1,)."""
    return float((g2[:, None] * (P_a - P_b) ** 2).sum())


def s2dir_raw(z_class: np.ndarray, z_benign: np.ndarray, z_benign_train: np.ndarray, E0: np.ndarray,
              g2: np.ndarray, seed: int) -> dict:
    """Unnormalised S2-dir for one class on one trained model, plus the benign-halves quantity Q_b."""
    W = whitener(z_benign_train - E0)
    whiten = lambda z: (z - E0) @ W.T  # noqa: E731
    P_c = channel_power(whiten(z_class))
    u_b = whiten(z_benign)
    P_b = channel_power(u_b)
    perm = np.random.default_rng(seed).permutation(len(u_b))
    h = len(perm) // 2
    Q_b = raw_distance(g2, channel_power(u_b[perm[:h]]), channel_power(u_b[perm[h:2 * h]]))
    return {"raw": raw_distance(g2, P_c, P_b), "Q_benign_halves": Q_b,
            "whitener_condition_number": float(np.linalg.cond(W))}
