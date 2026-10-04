import numpy as np
import pytest
import torch

from nids import metrics, scores, theory
from nids.models.ahsd import AHSD, VARIANTS


@pytest.mark.parametrize("variant", VARIANTS)
def test_ahsd_variants_forward_backward(variant):
    m = AHSD(n_features=6, D=16, variant=variant, gamma=0.05, kappa=0.05)
    x = torch.randn(8, 32, 6)
    out = m(x, return_states=True)
    assert out["logits"].shape == (8, 2) and out["stress"].shape == (8,)
    out["logits"].sum().backward()
    if variant in ("fixed", "no_stress"):
        assert float(m.gamma()) == 0.0
        assert torch.allclose(out["E"], m.E0.expand_as(out["E"]))  # E never moves at γ = 0
    if variant in ("adaptive", "S1"):
        assert 0 < float(m.gamma().detach()) <= 0.2 and m.gamma_logit.grad is not None


def test_s1_stop_gradient_on_equilibrium():
    m = AHSD(n_features=4, D=8, variant="S1", kappa=0.05)
    m.tau_ref.fill_(float("inf"))  # gate constant: E can then depend on h only through sg(h)
    out = m.recur(torch.randn(2, 16, 8, requires_grad=True), return_states=True)
    # spec: sg on h_t in the update term (the gate's own stress input is not stopped)
    out["E"].sum().backward()
    assert m.gates.weight.grad is None or torch.all(m.gates.weight.grad == 0)


@pytest.mark.parametrize("gamma,tau,kappa", [(0.2, 0.5, 0.01), (0.1, 1.0, 0.05), (0.2, 2.0, 0.05)])
@pytest.mark.parametrize("R", [2, 8, 32])
def test_s1_drift_never_exceeds_bound(gamma, tau, kappa, R):
    """Boil-rate bound: mean|E_t − E_0| ≤ (γτ/e)(1 − (1−κ)^t)/κ for any input sequence,
    including ramps of any length and the drift-maximising constant offset s = τ."""
    D, steps = 16, 400
    m = AHSD(n_features=D, D=D, variant="S1", kappa=kappa).double()
    with torch.no_grad():
        m.gamma_logit.fill_(50.0)  # gamma -> gamma_max
        m.gamma_max = gamma
        m.tau_ref.fill_(1.0)
        m.tau_mult = tau
    rng = np.random.default_rng(0)
    worst = 0.0
    for target in (0.5 * tau, tau, 3 * tau, 20 * tau):
        ramp = np.minimum(np.arange(steps) / R, 1.0)[:, None] * target * np.sign(rng.normal(size=D))[None]
        out = m.recur(torch.tensor(ramp[None] * 1.0), return_states=True)
        drift = (out["E"] - m.E0).abs().mean(-1)[0].detach().numpy()
        bound = theory.max_drift(gamma, tau, kappa, np.arange(1, steps + 1))
        assert np.all(drift <= bound + 1e-9)
        worst = max(worst, float((drift / bound).max()))
    assert worst > 0.2  # the bound is not vacuous for these inputs


def test_adaptive_equilibrium_absorbs_persistent_offset():
    """No attacker: with γ > 0 a constant offset's stress decays; at γ = 0 it does not."""
    stress = {}
    for g in (0.0, 0.1):
        m = AHSD(n_features=4, D=4, variant="frozen", gamma=g).double()
        z = torch.full((1, 200, 4), 2.0, dtype=torch.float64)
        stress[g] = m.recur(z, return_states=True)["stress_t"][0].detach().numpy()
    assert stress[0.1][-1] < 0.01 * stress[0.1][5]
    assert stress[0.0][-1] > 0.5 * stress[0.0][5]


def test_mean_ci_uses_t_not_196():
    r = metrics.mean_ci([1, 2, 3, 4, 5])
    assert r["ci95"] == pytest.approx(2.776 * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5))


def test_d_prime_orientation_fixed():
    y = np.r_[np.zeros(100), np.ones(100)]
    s = np.r_[np.random.default_rng(0).normal(0, 1, 100), np.random.default_rng(1).normal(2, 1, 100)]
    assert metrics.d_prime(y, s) > 0 and metrics.d_prime(y, -s) < 0


def test_delong_matches_sklearn_auc_and_detects_difference():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    good = y + rng.normal(0, 0.5, 600)
    bad = y + rng.normal(0, 3.0, 600)
    r = metrics.delong(y, good, bad)
    assert r["auc1"] == pytest.approx(metrics.auc(y, good))
    assert r["p"] < 1e-3
    assert metrics.delong(y, good, good)["p"] == pytest.approx(1.0)


def test_holm():
    adj = metrics.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_s3_flags_both_tails():
    rng = np.random.default_rng(0)
    vb_s, vb_d = rng.lognormal(0, 0.2, 500), rng.normal(1, 0.3, (500, 32))
    s = np.array([1.0, 5.0, 0.2])
    d = np.stack([rng.normal(1, 0.3, 32), rng.normal(1, 0.3, 32), np.full(32, 1.0)])
    sc = scores.s3_score(s, d, vb_s, vb_d)
    assert sc[1] > sc[0] and sc[2] > sc[0]
