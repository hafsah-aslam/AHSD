"""S2: spectral detectability predictor (spec §6).

The trained recurrence is linearised with constant gates (mean α, β, γ over
benign training windows). With E₀ subtracted, input u_t = z_t − E₀ and output
y_t = h_t − E_t (whose magnitude is the stress):

    h_t = (1−β) h_{t−1} + (β−α) E_{t−1} + α u_t
    E_t = (1−γ) E_{t−1} + γ h_t
    y_t = h_t − E_t = (1−γ)(h_t − E_{t−1})

H(ω) is the rFFT of the first T = 32 impulse-response samples (17 bins incl.
DC). For γ = 0 the filter is low-pass with DC gain α/β; for γ > 0 its
infinite-horizon DC gain is 0 (high-pass): persistent offsets are absorbed.

Power is one-sided and normalised so that Σ_ω P(ω) = mean_t u_t² (Parseval),
which makes S = Σ_ω |H(ω)|² P(ω) the predicted mean-square stress signal
mean_t y_t² for inputs that are periodic in the window.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- filters
def ahsd_impulse_response(alpha, beta, gamma, T: int = 32) -> np.ndarray:
    """Impulse response of y = h − E to u. Scalars or per-channel arrays (D,) -> (T,) or (T, D)."""
    a, b, g = (np.asarray(v, dtype=np.float64) for v in (alpha, beta, gamma))
    shape = np.broadcast(a, b, g).shape
    h = np.zeros(shape)
    E = np.zeros(shape)
    out = np.zeros((T,) + shape)
    for t in range(T):
        u = 1.0 if t == 0 else 0.0
        h_new = h + a * (u - E) - b * (h - E)
        E_new = E + g * (h_new - E)
        out[t] = h_new - E_new
        h, E = h_new, E_new
    return out


def simulate_ahsd_linear(alpha, beta, gamma, u: np.ndarray) -> np.ndarray:
    """Run the constant-gate recurrence on u (..., T); returns y = h − E with zero initial state."""
    u = np.asarray(u, dtype=np.float64)
    h = np.zeros(u.shape[:-1])
    E = np.zeros(u.shape[:-1])
    y = np.zeros_like(u)
    for t in range(u.shape[-1]):
        h = h + alpha * (u[..., t] - E) - beta * (h - E)
        E = E + gamma * (h - E)
        y[..., t] = h - E
    return y


def lif_impulse_response(beta_leak, T: int = 32) -> np.ndarray:
    """Membrane v_t = β v_{t−1} + u_t (sub-threshold LIF)."""
    return np.asarray(beta_leak, dtype=np.float64)[..., None] ** np.arange(T) if np.ndim(beta_leak) \
        else float(beta_leak) ** np.arange(T, dtype=np.float64)


def ssm_impulse_response(a, b=1.0, c=1.0, T: int = 32) -> np.ndarray:
    """Diagonal SSM x_t = a x_{t−1} + b u_t, y = Re(c x); a may be complex (S4D)."""
    a = np.asarray(a)
    k = np.arange(T)
    ir = np.real(np.asarray(c)[..., None] * np.asarray(b)[..., None] * a[..., None] ** k) if a.ndim \
        else np.real(c * b * a ** k)
    return ir


def jacobian_impulse_response(step_fn, h_star: np.ndarray, u_star: np.ndarray, readout=None,
                              T: int = 32, eps: float = 1e-4) -> np.ndarray:
    """Linearise h_t = f(h_{t−1}, u_t) at (h*, u*) with numerical Jacobians (e.g. a GRU).

    Returns the (T, D_out, D_in) impulse-response matrices; per-channel |H|² is
    averaged downstream.
    """
    Dh, Du = len(h_star), len(u_star)
    f0 = step_fn(h_star, u_star)
    A = np.zeros((Dh, Dh))
    Bm = np.zeros((Dh, Du))
    for i in range(Dh):
        e = np.zeros(Dh)
        e[i] = eps
        A[:, i] = (step_fn(h_star + e, u_star) - f0) / eps
    for j in range(Du):
        e = np.zeros(Du)
        e[j] = eps
        Bm[:, j] = (step_fn(h_star, u_star + e) - f0) / eps
    C = np.eye(Dh) if readout is None else readout
    out = np.zeros((T, C.shape[0], Du))
    Ak = np.eye(Dh)
    for t in range(T):
        out[t] = C @ Ak @ Bm
        Ak = A @ Ak
    return out


def transfer_function(ir: np.ndarray, T: int = 32) -> np.ndarray:
    """H(ω) over the T//2+1 rFFT bins (17 for T = 32), along axis 0."""
    return np.fft.rfft(ir, n=T, axis=0)


def gain2(H: np.ndarray) -> np.ndarray:
    """|H(ω)|², averaged over any channel axes -> (n_bins,)."""
    g = np.abs(H) ** 2
    return g.reshape(g.shape[0], -1).mean(1) if g.ndim > 1 else g


def ir_tail_ratio(alpha, beta, gamma, T: int = 32, horizon: int = 4096) -> float:
    """Energy of the impulse response beyond T, relative to its total: truncation diagnostic."""
    ir = ahsd_impulse_response(alpha, beta, gamma, horizon)
    e = (ir ** 2).reshape(horizon, -1).sum(1)
    return float(e[T:].sum() / max(e.sum(), 1e-300))


# ------------------------------------------------------------------ power
def window_power(u: np.ndarray) -> np.ndarray:
    """One-sided per-window power, averaged over channels.

    u: (N, T, D) embedded windows minus E₀ -> (N, T//2+1), with Σ_ω P = mean_{t,d} u².
    """
    N, T, D = u.shape
    U = np.fft.rfft(u, axis=1)
    P = np.abs(U) ** 2 / T ** 2
    w = np.full(P.shape[1], 2.0)
    w[0] = 1.0
    if T % 2 == 0:
        w[-1] = 1.0
    return (P * w[None, :, None]).mean(2)


def in_band_energy(g2: np.ndarray, P: np.ndarray) -> np.ndarray:
    """S = Σ_ω |H(ω)|² P(ω) per window."""
    return P @ g2


def detectability(S_class: np.ndarray, S_benign: np.ndarray) -> float:
    """D_pred(c) = (mean S_c − mean S_benign) / std(S_benign)."""
    sd = float(np.std(S_benign, ddof=1)) if len(S_benign) > 1 else 0.0
    return float((np.mean(S_class) - np.mean(S_benign)) / max(sd, 1e-12))
