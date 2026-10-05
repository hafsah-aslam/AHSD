"""CLAD (Wilkie et al., IEEE TNSM 2026, doi 10.1109/TNSM.2026.3652529; arXiv 2601.09902),
implemented from the paper. Choices the paper leaves unspecified are fixed in AMENDMENT_04.

Encoder φ: Linear(f → d_model), L × [Linear(d_model → d_model), ReLU], Linear(d_model → f_o), L2-normalise.
Loss (Eq. 7), over benign anchors i in the batch:
    mean_p d(z_i, z_p)^2  +  mean_n (1 − d(z_i, z_n))^2,   d(z, z') = (1 − z·z') / 2
    (p: other benign samples in the batch; n: malicious samples in the batch)
Score (Eqs. 8–9): s(x) = −z·μ, μ = normalised sum of benign training embeddings (higher = more anomalous).
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import auc

DEFAULTS = dict(d_model=256, depth=2, f_out=64, lr=1e-3, weight_decay=1e-4, batch=256, epochs=15, warmup_frac=0.1)


class Encoder(nn.Module):
    def __init__(self, f_in: int, d_model: int, depth: int, f_out: int):
        super().__init__()
        layers = [nn.Linear(f_in, d_model)]
        for _ in range(depth):
            layers += [nn.Linear(d_model, d_model), nn.ReLU()]
        layers.append(nn.Linear(d_model, f_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def clad_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Eq. 7. z: (B, f_o) unit vectors; y: (B,) 0 = benign, 1 = malicious."""
    d = (1 - z @ z.T) / 2
    ben = y == 0
    mal = ~ben
    if ben.sum() == 0:
        return z.sum() * 0
    db = d[ben]                      # anchors = benign rows
    pos = ben.clone().float()[None, :].expand_as(db).clone()
    idx = torch.nonzero(ben).squeeze(1)
    pos[torch.arange(len(idx)), idx] = 0  # exclude self
    neg = mal.float()[None, :].expand_as(db)
    term_p = (db ** 2 * pos).sum(1) / pos.sum(1).clamp(min=1)
    term_n = ((1 - db) ** 2 * neg).sum(1) / neg.sum(1).clamp(min=1)
    has_p, has_n = pos.sum(1) > 0, neg.sum(1) > 0
    return (term_p * has_p + term_n * has_n).mean()


class CLAD:
    def __init__(self, seed: int, **kw):
        self.seed, self.hp = seed, {**DEFAULTS, **kw}

    @staticmethod
    def _flat(X):
        return torch.from_numpy(np.ascontiguousarray(X.reshape(len(X), -1), dtype=np.float32))

    @torch.no_grad()
    def embed(self, X, bs=2048):
        self.enc.eval()
        F_ = self._flat(X)
        return torch.cat([self.enc(F_[a:a + bs]) for a in range(0, len(F_), bs)]).numpy()

    def fit(self, Xtr, ytr, Xva, yva):
        hp = self.hp
        torch.manual_seed(self.seed)  # before construction
        self.enc = Encoder(int(np.prod(Xtr.shape[1:])), hp["d_model"], hp["depth"], hp["f_out"])
        opt = torch.optim.AdamW(self.enc.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        Ftr, Ytr = self._flat(Xtr), torch.from_numpy(np.asarray(ytr, dtype=np.int64))
        steps_per = math.ceil(len(Ftr) / hp["batch"])
        total = hp["epochs"] * steps_per
        warm = max(1, int(hp["warmup_frac"] * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, hist = -1.0, None, []
        for ep in range(hp["epochs"]):
            self.enc.train()
            for idx in torch.randperm(len(Ftr), generator=g).split(hp["batch"]):
                loss = clad_loss(self.enc(Ftr[idx]), Ytr[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            self._set_centroid(Xtr[np.asarray(ytr) == 0])
            v = auc(np.asarray(yva), self.score(Xva))
            hist.append({"epoch": ep + 1, "val_auroc": v})
            if v > best:
                best, best_state = v, copy.deepcopy(self.enc.state_dict())
        self.enc.load_state_dict(best_state)
        self._set_centroid(Xtr[np.asarray(ytr) == 0])
        self.history, self.best_val_auroc = hist, best
        return self

    def _set_centroid(self, Xb):
        s = self.embed(Xb).sum(0)
        self.mu = s / max(np.linalg.norm(s), 1e-12)

    def score(self, X):
        return -(self.embed(X) @ self.mu)
