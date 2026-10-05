"""AHSD: adaptive homeostatic sequence detector (spec §5–6).

    z_t   = LayerNorm(W x_t)                                  (D = 128)
    Δ_t   = z_t − E_{t−1}
    h_t   = h_{t−1} + α_t ⊙ Δ_t − β_t ⊙ (h_{t−1} − E_{t−1})
    E_t   = E_{t−1} + γ (h_t − E_{t−1})                        (fixed/adaptive/frozen)
    E_t   = E_{t−1} + γ_t (sg(h_t) − E_{t−1}) − κ (E_{t−1} − E_0),
            γ_t = γ · exp(−s_t / τ),  s_t = mean_d |h_t − E_{t−1}|   (S1)
    stress = mean_{t,d} |h_t − E_t|

α_t, β_t = sigmoid(W [z_t, h_{t−1}]) per channel. h_0 = E_0. E_0 is a buffer:
an EMA of benign embeddings during training, set exactly to the mean benign
training embedding afterwards (`set_E0`), and re-estimated on a target network
for re-anchoring (S4).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

VARIANTS = ("fixed", "adaptive", "frozen", "no_stress", "S1")


class AHSD(nn.Module):
    def __init__(self, n_features: int, D: int = 128, n_out: int = 2, variant: str = "fixed",
                 gamma: float = 0.0, gamma_max: float = 0.2, tau_mult: float = 1.0, kappa: float = 0.0,
                 E0_momentum: float = 0.01):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.D = variant, D
        self.gamma_max, self.tau_mult, self.kappa = gamma_max, tau_mult, kappa
        self.embed = nn.Sequential(nn.Linear(n_features, D), nn.LayerNorm(D))
        self.gates = nn.Linear(2 * D, 2 * D)
        self.learn_gamma = variant in ("adaptive", "S1")
        if self.learn_gamma:
            # start at gamma = 0.05 inside [0, gamma_max]
            g0 = 0.05 / gamma_max
            self.gamma_logit = nn.Parameter(torch.tensor(math.log(g0 / (1 - g0))))
        self.register_buffer("gamma_const", torch.tensor(float(0.0 if variant in ("fixed", "no_stress") else gamma)))
        self.register_buffer("E0", torch.zeros(D))
        self.register_buffer("E0_set", torch.tensor(False))
        self.register_buffer("tau_ref", torch.tensor(1.0))
        self.E0_momentum = E0_momentum
        self.use_stress = variant != "no_stress"
        self.head = nn.Sequential(nn.Linear(2 * D + int(self.use_stress), D), nn.GELU(), nn.Linear(D, n_out))

    # ------------------------------------------------------------------
    def gamma(self) -> torch.Tensor:
        if self.learn_gamma:
            return self.gamma_max * torch.sigmoid(self.gamma_logit)
        return self.gamma_const

    def tau(self) -> torch.Tensor:
        return self.tau_mult * self.tau_ref

    @torch.no_grad()
    def update_E0_ema(self, z_benign: torch.Tensor):
        if z_benign.numel() == 0:
            return
        m = z_benign.reshape(-1, self.D).mean(0)
        if not bool(self.E0_set):
            self.E0.copy_(m)
            self.E0_set.fill_(True)
        else:
            self.E0.mul_(1 - self.E0_momentum).add_(self.E0_momentum * m)

    @torch.no_grad()
    def set_E0(self, E0: torch.Tensor):
        self.E0.copy_(E0)
        self.E0_set.fill_(True)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, return_states: bool = False) -> dict:
        return self.recur(self.embed(x), return_states)

    def recur(self, z: torch.Tensor, return_states: bool = False) -> dict:
        """Run the recurrence on embeddings z (B, T, D)."""
        B, T, _ = z.shape
        E = self.E0.expand(B, self.D)
        h = E
        g = self.gamma()
        hs, Es, als, bes, gts, ds = [], [], [], [], [], []
        for t in range(T):
            zt = z[:, t]
            ab = torch.sigmoid(self.gates(torch.cat([zt, h], -1)))
            a, b = ab[:, :self.D], ab[:, self.D:]
            delta = zt - E
            h = h + a * delta - b * (h - E)
            if self.variant == "S1":
                s_t = (h - E).abs().mean(-1, keepdim=True)
                g_t = g * torch.exp(-s_t / self.tau())
                E = E + g_t * (h.detach() - E) - self.kappa * (E - self.E0)
            else:
                g_t = g.expand(B, 1)
                E = E + g_t * (h - E)
            hs.append(h)
            Es.append(E)
            if return_states:
                als.append(a)
                bes.append(b)
                gts.append(g_t)
                ds.append(delta)
        H = torch.stack(hs, 1)
        EE = torch.stack(Es, 1)
        stress_td = (H - EE).abs()
        stress = stress_td.mean((1, 2))
        feats = [H[:, -1], H.mean(1)]
        if self.use_stress:
            feats.append(stress[:, None])
        pen = self.head[1](self.head[0](torch.cat(feats, -1)))  # penultimate (Linear -> GELU)
        logits = self.head[2](pen)
        out = {"logits": logits, "stress": stress, "z": z, "penultimate": pen}
        if return_states:
            out.update(h=H, E=EE, alpha=torch.stack(als, 1), beta=torch.stack(bes, 1),
                       gamma_t=torch.stack(gts, 1), delta=torch.stack(ds, 1), stress_t=stress_td.mean(-1))
        return out
