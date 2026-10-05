import numpy as np
import torch

from nids import clad


def test_clad_loss_matches_eq7_by_hand():
    z = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [0.0, 1.0]]), dim=-1)
    y = torch.tensor([0, 0, 1, 0])
    d = lambda a, b: (1 - float(a @ b)) / 2  # noqa: E731
    ben, mal = [0, 1, 3], [2]
    tot = 0
    for i in ben:
        p = [d(z[i], z[j]) ** 2 for j in ben if j != i]
        n = [(1 - d(z[i], z[j])) ** 2 for j in mal]
        tot += np.mean(p) + np.mean(n)
    assert abs(float(clad.clad_loss(z, y)) - tot / len(ben)) < 1e-6


def test_clad_learns_and_scores_attack_higher():
    rng = np.random.default_rng(0)
    Xb = rng.normal(0, 1, (400, 4, 3)).astype(np.float32)
    Xa = rng.normal(2.5, 1, (400, 4, 3)).astype(np.float32)
    X = np.concatenate([Xb, Xa])
    y = np.r_[np.zeros(400), np.ones(400)].astype(int)
    m = clad.CLAD(seed=0, epochs=5, batch=64, d_model=32, f_out=8).fit(X, y, X, y)
    s = m.score(X)
    assert s[y == 1].mean() > s[y == 0].mean()
    assert np.isclose(np.linalg.norm(m.mu), 1.0)
    a = clad.CLAD(seed=0, epochs=1, batch=64, d_model=16, f_out=4).fit(X, y, X, y).score(X[:10])
    b = clad.CLAD(seed=0, epochs=1, batch=64, d_model=16, f_out=4).fit(X, y, X, y).score(X[:10])
    assert np.array_equal(a, b)  # seeded before construction


def test_lycos_heavy_tailed_rule():
    from nids import schema
    assert schema.is_heavy_tailed("fwd_pkt_len_max") and schema.is_heavy_tailed("flow_duration")
    assert not schema.is_heavy_tailed("flag_SYN".lower()) and not schema.is_heavy_tailed("ip_prot")
    assert not schema.is_heavy_tailed("down_up_ratio")
    assert schema.is_heavy_tailed("IN_BYTES") and not schema.is_heavy_tailed("PROTOCOL")
