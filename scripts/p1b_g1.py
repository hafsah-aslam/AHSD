#!/usr/bin/env python
"""P1b-G1 (AMENDMENT_02 A2.6): γ sweep, adaptive γ, and S1 grid; in-distribution on grouped_random.

One JSON per (dataset, model, seed) under results/p1b/g1/. Gate evaluation and
S1 selection happen in scripts/report_p1b.py, from these JSONs only.

  python scripts/p1b_g1.py [--resume] [--datasets D2 ...] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import metrics, predictors  # noqa: E402
from nids.models.ahsd import AHSD  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.train import fit, predict  # noqa: E402

MIN_CLASS_WINDOWS = 20


def model_grid(cfg: dict) -> list[dict]:
    g = [{"id": f"frozen_g{gam:g}", "variant": "frozen", "gamma": gam} for gam in cfg["gammas"]]
    g.append({"id": "adaptive", "variant": "adaptive"})
    g += [{"id": f"S1_t{t:g}_k{k:g}", "variant": "S1", "tau_mult": t, "kappa": k}
          for t in cfg["s1_tau_mult"] for k in cfg["s1_kappa"]]
    return g


def build(n_features: int, spec: dict, D: int) -> AHSD:
    kw = {k: spec[k] for k in ("gamma", "tau_mult", "kappa") if k in spec}
    return AHSD(n_features, D=D, variant=spec["variant"], **kw)


def run_one(arc: Archive, ds: str, lab: str, classes: list[str], spec: dict, seed: int, cfg: dict) -> dict:
    tr, va, te = (arc.windows(arc.sets[k], labelling=lab) for k in ("train", "val", "test"))
    torch.set_num_threads(cfg["threads"])
    m = build(arc.n_features, spec, cfg["D"])
    tinfo = fit(m, tr, va, seed=seed, epochs=cfg["epochs"], log=lambda s: print(s, flush=True))
    st = predict(m, te["X"], states=True)
    y = te["y_bin"]
    pred = (st["prob"] >= 0.5).astype(int)
    per_class = {}
    for c in np.unique(te["y_cls"]):
        if c == 0 or (te["y_cls"] == c).sum() < MIN_CLASS_WINDOWS:
            continue
        sel = (te["y_cls"] == 0) | (te["y_cls"] == c)
        per_class[classes[c]] = {
            "n": int((te["y_cls"] == c).sum()),
            "d_prime_stress": metrics.d_prime((te["y_cls"][sel] == c).astype(int), st["stress"][sel]),
            "stationarity": float(predictors.stationarity_index(st["z"][te["y_cls"] == c]).mean()),
        }
    return {
        "test": {
            "d_prime_stress": metrics.d_prime(y, st["stress"]),
            "auc_stress": metrics.auc(y, st["stress"]),
            "d_prime_prob": metrics.d_prime(y, st["prob"]),
            "auc_prob": metrics.auc(y, st["prob"]),
            "macro_f1_pct": 100 * metrics.classification(y, pred)["macro_f1"],
            "n": {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())},
        },
        "per_class": per_class,
        "best_val_macro_f1": tinfo["best_val_macro_f1"],
        "gamma_final": float(m.gamma().detach()),
        "gamma_by_epoch": [h.get("gamma") for h in tinfo["history"]],
        "training": tinfo,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1b_g1.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.smoke:
        cfg.update(cfg["smoke"])
    else:
        require_clean_tree("P1b-G1")
    out_root = ROOT / cfg["results_dir"]
    for ds in cfg["datasets"]:
        if a.datasets and ds not in a.datasets:
            continue
        lab = cfg["labelling"].get(ds, "file")
        arc = Archive(ROOT / cfg["processed_dir"] / ds / cfg["scheme"] / "archive")
        meta = json.loads((ROOT / cfg["processed_dir"] / ds / "index_meta.json").read_text())
        classes = meta["family"]["classes"] if lab == "family" else meta["classes"]
        data_sha = json.loads((ROOT / cfg["processed_dir"] / ds / "archive_meta.json").read_text())["schemes"][cfg["scheme"]]["sha256"]
        for spec in model_grid(cfg):
            for seed in cfg["seeds"]:
                path = out_root / ds / spec["id"] / f"seed{seed}.json"
                if a.resume and path.exists():
                    continue
                t0 = time.time()
                print(f"[G1 {ds}] {spec['id']} seed {seed}", flush=True)
                res = run_one(arc, ds, lab, classes, spec, seed, cfg)
                write_json(path, {"dataset": ds, "scheme": cfg["scheme"], "labelling": lab, "model": spec,
                                  "seed": seed, **res, "wall_seconds": time.time() - t0, "smoke": a.smoke,
                                  "provenance": provenance(config={k: v for k, v in cfg.items() if k != "smoke"},
                                                           data_sha256=data_sha)})


if __name__ == "__main__":
    main()
