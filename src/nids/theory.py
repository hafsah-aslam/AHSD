"""S1 boil-rate bound (draft derivation; see docs/S1_BOUND.md).

S1 update, with s_t = mean_d |h_t − E_{t−1}| and γ_t = γ e^{−s_t/τ}:

    E_t − E_0 = (1 − κ)(E_{t−1} − E_0) + γ_t (h_t − E_{t−1})

Taking the mean absolute value over channels, m_t = mean_d |E_t − E_0|:

    m_t ≤ (1 − κ) m_{t−1} + γ s_t e^{−s_t/τ} ≤ (1 − κ) m_{t−1} + γ τ / e

because x e^{−x/τ} ≤ τ/e for x ≥ 0. With m_0 = 0:

    m_t ≤ (γ τ / e) · (1 − (1 − κ)^t) / κ   ≤ γ τ / (e κ)

so per step the equilibrium moves at most γτ/e (mean-abs), and in total at
most γτ/(eκ), whatever the attack does. The bound uses only the update rule
(no linearisation, no assumption on h).
"""

from __future__ import annotations

import math

import numpy as np


def max_drift(gamma: float, tau: float, kappa: float, steps) -> np.ndarray:
    """Upper bound on mean_d |E_t − E_0| after `steps` steps from E = E_0."""
    t = np.asarray(steps, dtype=np.float64)
    r = gamma * tau / math.e
    if kappa <= 0:
        return r * t
    return r * (1 - (1 - kappa) ** t) / kappa


def max_absorbed_fraction(gamma: float, tau: float, kappa: float, ramp_steps: int,
                          deviation: float, horizon: int) -> float:
    """Bound on the fraction of a deviation of mean-abs size `deviation` that E can absorb.

    For an attack ramp reaching `deviation` after `ramp_steps` and holding to
    `horizon` steps, absorption is at most max_drift(horizon) / deviation.
    """
    return float(min(1.0, max_drift(gamma, tau, kappa, horizon) / max(deviation, 1e-12)))


def min_residual_stress(gamma: float, tau: float, kappa: float, deviation: float, steps) -> np.ndarray:
    """Lower bound on mean_d |h_t − E_t| when h holds a deviation `deviation` from E_0 (triangle inequality)."""
    return np.maximum(0.0, deviation - max_drift(gamma, tau, kappa, steps))
