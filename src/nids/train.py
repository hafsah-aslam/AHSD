"""Shared training loop (spec §5): AdamW lr 1e-3, cosine, wd 1e-4, clip 1.0,
batch 256, 15 epochs, checkpoint = best validation macro-F1. No per-model tuning.

The test split is never touched here.
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import classification

DEFAULTS = dict(lr=1e-3, weight_decay=1e-4, clip=1.0, batch=256, epochs=15)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n, bs, rng=None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for a in range(0, n, bs):
        yield order[a:a + bs]


@torch.no_grad()
def predict(model, X: np.ndarray, bs: int = 1024, states: bool = False) -> dict:
    model.eval()
    outs = {"prob": [], "stress": [], "z": [], "delta_norm": [], "alpha": [], "beta": [], "gamma_t": []}
    for idx in _batches(len(X), bs):
        o = model(torch.from_numpy(np.ascontiguousarray(X[idx])), return_states=states)
        outs["prob"].append(torch.softmax(o["logits"], -1)[:, 1].numpy())
        outs["stress"].append(o["stress"].numpy())
        if states:
            outs["z"].append(o["z"].numpy())
            outs["delta_norm"].append(o["delta"].norm(dim=-1).numpy())
            outs["alpha"].append(o["alpha"].mean((0, 1)).numpy()[None])
            outs["beta"].append(o["beta"].mean((0, 1)).numpy()[None])
            outs["gamma_t"].append(o["gamma_t"].mean().reshape(1).numpy())
    res = {k: np.concatenate(v) for k, v in outs.items() if v}
    if states:
        # batch means of per-channel gate means, weighted equally per batch (fine for a mean)
        for k in ("alpha", "beta"):
            res[k] = res[k].mean(0)
        res["gamma_t"] = float(res["gamma_t"].mean())
    return res


@torch.no_grad()
def benign_embedding_mean(model, X_benign: np.ndarray, bs: int = 1024) -> torch.Tensor:
    model.eval()
    s, n = None, 0
    for idx in _batches(len(X_benign), bs):
        z = model.embed(torch.from_numpy(np.ascontiguousarray(X_benign[idx])))
        zs = z.sum((0, 1))
        s = zs if s is None else s + zs
        n += z.shape[0] * z.shape[1]
    return s / max(n, 1)


@torch.no_grad()
def calibrate_tau(model, X_benign: np.ndarray, bs: int = 1024) -> float:
    """Median per-step stress mean_d|h_t − E_{t−1}| on benign windows (S1's τ unit)."""
    model.eval()
    vals = []
    old = model.tau_ref.clone()
    model.tau_ref.fill_(float("inf"))  # gate open while measuring
    for idx in _batches(len(X_benign), bs):
        o = model(torch.from_numpy(np.ascontiguousarray(X_benign[idx])), return_states=True)
        prev_E = torch.cat([model.E0.expand(o["E"].shape[0], 1, -1), o["E"][:, :-1]], 1)
        vals.append((o["h"] - prev_E).abs().mean(-1).flatten().numpy())
    model.tau_ref.copy_(old)
    return float(np.median(np.concatenate(vals)))


def fit(model, train: dict, val: dict, seed: int, epochs: int | None = None, log=print,
        checkpoint: str = "best", **kw) -> dict:
    """checkpoint="best": best validation macro-F1 (default); "final": last epoch (AMENDMENT_05 A5.3)."""
    if checkpoint not in ("best", "final"):
        raise ValueError(checkpoint)
    hp = {**DEFAULTS, **kw}
    if epochs is not None:
        hp["epochs"] = epochs
    set_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr = train["X"], train["y_bin"]
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    steps = hp["epochs"] * int(np.ceil(len(Xtr) / hp["batch"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    is_s1 = getattr(model, "variant", None) == "S1"
    # S1 τ unit (AMENDMENT_02 A2.6): median per-step benign *validation* stress, gate open;
    # measured before training, re-measured once after epoch 1, then frozen.
    Xvb = val["X"][val["y_bin"] == 0]
    tau_log = []
    if is_s1:
        tau_log.append(calibrate_tau(model, Xvb))
        model.tau_ref.fill_(tau_log[-1])
    best, best_state, history = -1.0, None, []
    t0 = time.time()
    for ep in range(hp["epochs"]):
        model.train()
        tot = 0.0
        for idx in _batches(len(Xtr), hp["batch"], rng):
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[idx]))
            yb = torch.from_numpy(ytr[idx])
            if hasattr(model, "update_E0_ema"):
                with torch.no_grad():
                    model.update_E0_ema(model.embed(xb[yb == 0]))
            out = model(xb)
            loss = F.cross_entropy(out["logits"], yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["clip"])
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if is_s1 and ep == 0:
            tau_log.append(calibrate_tau(model, Xvb))  # recalibrate once, then frozen
            model.tau_ref.fill_(tau_log[-1])
        pv = predict(model, val["X"])
        f1 = classification(val["y_bin"], (pv["prob"] >= 0.5).astype(int))["macro_f1"]
        rec = {"epoch": ep + 1, "train_loss": tot / len(Xtr), "val_macro_f1": f1}
        if hasattr(model, "gamma"):
            rec["gamma"] = float(model.gamma().detach())
        history.append(rec)
        log(f"  ep {ep + 1:2d} loss {rec['train_loss']:.4f} val-F1 {f1:.4f}"
            + (f" γ {rec['gamma']:.4f}" if "gamma" in rec else ""))
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
    if checkpoint == "best":
        model.load_state_dict(best_state)
    if hasattr(model, "set_E0"):
        model.set_E0(benign_embedding_mean(model, Xtr[ytr == 0]))
    out = {"history": history, "best_val_macro_f1": best, "seconds": time.time() - t0, "hparams": hp,
           "checkpoint": checkpoint, "final_val_macro_f1": history[-1]["val_macro_f1"]}
    if is_s1:
        out["tau_ref_log"] = tau_log
    return out
