"""S2 math (spec §10): on synthetic sinusoids the predicted in-band energy must
match the simulated stress within 5%."""

import numpy as np
import pytest
import torch

from nids import spectral
from nids.models.ahsd import AHSD

T = 32
PARAMS = [(0.5, 0.3, 0.0), (0.3, 0.5, 0.0), (0.6, 0.4, 0.2), (0.8, 0.6, 0.1)]


def _predicted(alpha, beta, gamma, u_window):
    g2 = spectral.gain2(spectral.transfer_function(spectral.ahsd_impulse_response(alpha, beta, gamma, T)))
    return float(spectral.in_band_energy(g2, spectral.window_power(u_window[None, :, None]))[0])


@pytest.mark.parametrize("alpha,beta,gamma", PARAMS)
@pytest.mark.parametrize("k", [0, 1, 3, 8, 15, 16])
def test_sinusoid_energy_matches_simulation(alpha, beta, gamma, k):
    if spectral.ir_tail_ratio(alpha, beta, gamma, T) > 1e-3:
        pytest.skip("impulse response not contained in 32 steps")
    n = 64 * T
    t = np.arange(n)
    u = 1.3 * np.cos(2 * np.pi * k * t / T + 0.4) + (0.7 if k == 0 else 0.0)
    y = spectral.simulate_ahsd_linear(alpha, beta, gamma, u)
    sim = float(np.mean(y[-16 * T:] ** 2))  # steady state
    pred = _predicted(alpha, beta, gamma, u[-T:])
    # 5% relative; when the true steady state is ~0 (DC under gamma > 0, which the
    # 32-step truncated H cannot null exactly) allow 1e-3 of the input energy.
    assert pred == pytest.approx(sim, rel=0.05, abs=1e-3 * float(np.mean(u ** 2)))


def test_mixture_of_sinusoids():
    a, b, g = 0.5, 0.3, 0.0
    t = np.arange(64 * T)
    u = 0.5 + np.cos(2 * np.pi * 2 * t / T) + 0.3 * np.sin(2 * np.pi * 11 * t / T)
    y = spectral.simulate_ahsd_linear(a, b, g, u)
    assert _predicted(a, b, g, u[-T:]) == pytest.approx(np.mean(y[-16 * T:] ** 2), rel=0.05)


def test_power_parseval():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(5, T, 7))
    assert np.allclose(spectral.window_power(u).sum(1), (u ** 2).mean((1, 2)))
    assert spectral.window_power(u).shape == (5, T // 2 + 1)


def test_gamma_makes_filter_high_pass():
    """Frog-boiling mechanism: γ > 0 drives the infinite-horizon DC gain to 0."""
    H0 = spectral.transfer_function(spectral.ahsd_impulse_response(0.5, 0.3, 0.0, 4096), 4096)
    Hg = spectral.transfer_function(spectral.ahsd_impulse_response(0.5, 0.3, 0.05, 4096), 4096)
    assert abs(H0[0]) == pytest.approx(0.5 / 0.3, rel=1e-6)
    assert abs(Hg[0]) < 1e-6
    # but high frequencies still pass
    assert abs(Hg[len(Hg) // 2]) > 0.1 * abs(H0[len(H0) // 2])


def test_torch_module_matches_linear_filter():
    """With constant gates the AHSD module's h − E equals the linearised filter, and
    sqrt(predicted energy) matches the module's RMS stress signal within 5%."""
    D, alpha, beta = 4, 0.5, 0.3
    m = AHSD(n_features=D, D=D, variant="fixed").double()
    with torch.no_grad():
        m.gates.weight.zero_()
        m.gates.bias[:D] = float(np.log(alpha / (1 - alpha)))
        m.gates.bias[D:] = float(np.log(beta / (1 - beta)))
    t = np.arange(16 * T)
    u = np.stack([np.cos(2 * np.pi * k * t / T) for k in (0, 1, 4, 9)], 1)
    out = m.recur(torch.tensor(u[None]), return_states=True)
    y_mod = (out["h"] - out["E"])[0].detach().numpy()
    y_lin = np.stack([spectral.simulate_ahsd_linear(alpha, beta, 0.0, u[:, d]) for d in range(D)], 1)
    assert np.allclose(y_mod, y_lin, atol=1e-10)
    rms_sim = np.sqrt(np.mean(y_mod[-8 * T:] ** 2))
    pred = np.sqrt(_predicted_multi(alpha, beta, 0.0, u[-T:]))
    assert pred == pytest.approx(rms_sim, rel=0.05)


def _predicted_multi(alpha, beta, gamma, u):
    g2 = spectral.gain2(spectral.transfer_function(spectral.ahsd_impulse_response(alpha, beta, gamma, T)))
    return float(spectral.in_band_energy(g2, spectral.window_power(u[None]))[0])


def test_detectability_sign_for_beaconing():
    """A class with less in-band energy than benign gets D_pred < 0 (predicted inversion)."""
    rng = np.random.default_rng(1)
    g2 = spectral.gain2(spectral.transfer_function(spectral.ahsd_impulse_response(0.5, 0.3, 0.0, T)))
    benign = rng.normal(0, 1.0, (200, T, 8))
    beacon = 0.2 * np.cos(2 * np.pi * 4 * np.arange(T) / T)[None, :, None] + rng.normal(0, 0.1, (200, T, 8))
    burst = rng.normal(2.0, 1.0, (200, T, 8))
    Sb = spectral.in_band_energy(g2, spectral.window_power(benign))
    assert spectral.detectability(spectral.in_band_energy(g2, spectral.window_power(beacon)), Sb) < 0
    assert spectral.detectability(spectral.in_band_energy(g2, spectral.window_power(burst)), Sb) > 0


def test_lif_and_ssm_filters():
    assert np.allclose(spectral.lif_impulse_response(0.9, 5), 0.9 ** np.arange(5))
    assert np.allclose(spectral.ssm_impulse_response(0.5, T=4), [1, 0.5, 0.25, 0.125])


def test_jacobian_linearisation_of_linear_map():
    A = np.array([[0.7, 0.1], [0.0, 0.5]])
    Bm = np.array([[1.0], [0.5]])
    ir = spectral.jacobian_impulse_response(lambda h, u: A @ h + Bm @ u, np.zeros(2), np.zeros(1), T=3)
    assert np.allclose(ir[0], Bm, atol=1e-6)
    assert np.allclose(ir[2], A @ A @ Bm, atol=1e-6)
