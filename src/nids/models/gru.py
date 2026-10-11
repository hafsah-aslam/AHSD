"""GRU backbone (AMENDMENT_07 A7.3): second neural backbone for the backbone replication.

    z_t = LayerNorm(W x_t)               (D = 128)
    h_t = GRU(z_t, h_{t-1})               single layer, hidden D
    logits = Linear(GELU(Linear([h_T, mean_t h_t])))

Same interface as AHSD's forward (logits, stress, z, penultimate) so train.fit / predict work
unchanged; it has no stress, so "stress" is all zeros and is never scored.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUBackbone(nn.Module):
    def __init__(self, n_features: int, D: int = 128, n_out: int = 2):
        super().__init__()
        self.D = D
        self.embed = nn.Sequential(nn.Linear(n_features, D), nn.LayerNorm(D))
        self.gru = nn.GRU(D, D, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, n_out))

    def forward(self, x: torch.Tensor, return_states: bool = False) -> dict:
        z = self.embed(x)
        H, _ = self.gru(z)
        pen = self.head[1](self.head[0](torch.cat([H[:, -1], H.mean(1)], -1)))
        out = {"logits": self.head[2](pen), "stress": torch.zeros(len(x)), "z": z, "penultimate": pen}
        if return_states:
            out["h"] = H
        return out
