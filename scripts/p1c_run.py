#!/usr/bin/env python
"""P1c (AMENDMENT_03 A3.3): 9 detectors on D1 + D3 natural_novelty, drift diagnostic, re-anchoring.

One JSON per (dataset, seed) under results/p1c/. Conditions: reference = benign training
windows ("none"), or anchor_100 / anchor_500 benign windows from the first test-period
groups ("N100", "N500"). The supervised backbone is trained once per seed and never
retrained; re-anchoring refits only benign references (no attack labels).

  python scripts/p1c_run.py [--resume] [--smoke]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_loaco as R  # noqa: E402
from nids import detectors as det  # noqa: E402
from nids import metrics  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.preflight import preflight  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.train import benign_embedding_mean  # noqa: E402

CFG = {"datasets": ["D1", "D3"], "seeds": [17, 23, 42, 101, 202], "D": 128, "epochs": 15, "threads": 4,
       "conditions": ["none", "N100", "N500"]}


class P1cArchive(Archive):
    def __init__(self, path):
        super().__init__(path)
        self.meta = json.loads((Path(path) / "p1c_meta.json").read_text())


@torch.no_grad()
def backbone_outputs(m, X, bs=1024) -> dict:
    m.eval()
    out = {"stress": [], "logits": [], "pen": []}
    for a in range(0, len(X), bs):
        o = m(torch.from_numpy(np.ascontiguousarray(X[a:a + bs])))
        out["stress"].append(o["stress"].numpy())
        out["logits"].append(o["logits"].numpy())
        out["pen"].append(o["penultimate"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def run(ds: str, seed: int, cfg: dict) -> dict:
    arc = P1cArchive(ROOT / cfg["processed_dir"] / ds / "temporal_gap" / "p1c")
    meta = arc.meta
    W = {k: arc.windows(v) for k, v in arc.sets.items()}
    tr, va = W["train"], W["val"]
    Xtrb = tr["X"][tr["y_bin"] == 0]
    Xvab = va["X"][va["y_bin"] == 0]
    classes = meta["natural_novelty"]
    eval_sets = {"val_benign": Xvab, "test_benign": W["test_benign"]["X"],
                 **{f"test/{c}": W[f"test/{c}"]["X"] for c in classes}}
    refs = {"none": Xtrb, "N100": W["anchor_100"]["X"], "N500": W["anchor_500"]["X"]}

    m, tinfo = R.train_model(arc, tr, va, seed, cfg, log=lambda s: None)
    scores = {c: {} for c in cfg["conditions"]}  # condition -> detector -> set -> scores
    base = {k: backbone_outputs(m, X) for k, X in eval_sets.items()}
    ref_pen = {k: backbone_outputs(m, X)["pen"] for k, X in refs.items()}
    for cond in cfg["conditions"]:
        s = scores[cond]
        # AHSD stress: benign reference = E0
        if cond == "none":
            s["AHSD_stress"] = {k: v["stress"] for k, v in base.items()}
        else:
            m2 = copy.deepcopy(m)
            m2.set_E0(benign_embedding_mean(m2, refs[cond]))
            s["AHSD_stress"] = {k: backbone_outputs(m2, X)["stress"] for k, X in eval_sets.items()}
        # MSP / Energy: no benign reference (re-anchoring not applicable): same scores in every condition
        s["MSP"] = {k: det.msp_score(v["logits"]) for k, v in base.items()}
        s["Energy"] = {k: det.energy_score(v["logits"]) for k, v in base.items()}
        maha = det.BenignMahalanobis().fit(ref_pen[cond])
        knn = det.BenignKNN().fit(ref_pen[cond])
        s["Mahalanobis"] = {k: maha.score(v["pen"]) for k, v in base.items()}
        s["kNN"] = {k: knn.score(v["pen"]) for k, v in base.items()}
        for name, cls in det.BENIGN_ONLY.items():
            d = cls(seed).fit(refs[cond])
            s[name] = {k: d.score(X) for k, X in eval_sets.items()}

    res = {"auc": {}, "d_prime_attack_vs_valbenign": {}, "d_prime_testbenign_vs_valbenign": {}}
    for cond in cfg["conditions"]:
        for d_name in det.DETECTORS:
            sc = scores[cond][d_name]
            tb, vb = sc["test_benign"], sc["val_benign"]
            key = f"{cond}/{d_name}"
            res["d_prime_testbenign_vs_valbenign"][key] = metrics.d_prime(np.r_[np.zeros(len(vb)), np.ones(len(tb))],
                                                                          np.r_[vb, tb])
            for c in classes:
                a = sc[f"test/{c}"]
                res["auc"].setdefault(key, {})[c] = metrics.auc(np.r_[np.zeros(len(tb)), np.ones(len(a))], np.r_[tb, a])
                res["d_prime_attack_vs_valbenign"].setdefault(key, {})[c] = metrics.d_prime(
                    np.r_[np.zeros(len(vb)), np.ones(len(a))], np.r_[vb, a])
    return {"dataset": ds, "seed": seed, "classes": classes, "detectors": det.DETECTORS,
            "no_reference_detectors": sorted(det.NO_REFERENCE), **res,
            "n": {k: int(len(v)) for k, v in eval_sets.items()} | {f"ref_{k}": int(len(v)) for k, v in refs.items()},
            "training": tinfo,
            "data_sha256": meta["archive"]["sha256"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, D3 only; written under smoke_out/")
    a = ap.parse_args()
    cfg = {**CFG, "processed_dir": "data/processed"}
    out = ROOT / "results/p1c"
    if a.smoke:
        cfg.update(seeds=[17], epochs=1, datasets=["D3"])
        out = ROOT / "smoke_out/results/p1c"
    else:
        require_clean_tree("P1c")
        pf = preflight("P1c")
        print(f"preflight: {pf['summary']}", flush=True)
    for ds in cfg["datasets"]:
        for seed in cfg["seeds"]:
            path = out / ds / f"seed{seed}.json"
            if a.resume and path.exists():
                continue
            t0 = time.time()
            print(f"[P1c {ds}] seed {seed}", flush=True)
            r = run(ds, seed, cfg)
            r.update(wall_seconds=time.time() - t0, smoke=a.smoke,
                     provenance=provenance(config={k: v for k, v in cfg.items()}, data_sha256=r.pop("data_sha256")))
            write_json(path, r)
            print(f"   done in {r['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
