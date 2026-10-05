#!/usr/bin/env python
"""P1b: S2-dir (EXPLORATORY, AMENDMENT_02 A2.3) on the P1 pooled-set LOACO runs.

P1 stored no checkpoints or embeddings, so every P1 run in scope is re-executed
through the same code path (scripts/run_loaco.py: train_model + score_class) with
its own seed, data and thread count. The re-executed run must reproduce the
recorded P1 stress AUC, probability AUC and D_pred within 1e-6; otherwise this
script stops (fail closed) and S2-dir is not reported. Measured AUCs stay the P1
values. One JSON per (evaluation, class, seed) under results/p1b/s2dir/.

  python scripts/p1b_s2dir.py [--resume] [--only D2_gr ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_loaco as R  # noqa: E402
from nids import s2dir  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.train import predict  # noqa: E402

SCOPE = ("D1_family_gr", "D2_gr", "D3_gr")
TOL = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--max-runs", type=int, default=None, help="for smoke tests")
    a = ap.parse_args()
    require_clean_tree("P1b S2-dir")
    cfg = yaml.safe_load(open(ROOT / "configs/p1_pilot.yaml"))
    evs = {e["id"]: e for e in cfg["evaluations"]}
    out_root = ROOT / "results/p1b/s2dir"
    n = 0
    for ev_id in SCOPE:
        if a.only and ev_id not in a.only:
            continue
        ev = evs[ev_id]
        arc = Archive(ROOT / cfg["processed_dir"] / ev["dataset"] / ev["scheme"] / "archive")
        meta = json.loads((ROOT / cfg["processed_dir"] / ev["dataset"] / "index_meta.json").read_text())
        lab = ev.get("labelling", "file")
        classes = meta["family"]["classes"] if lab == "family" else meta["classes"]
        folds = arc.folds_family if lab == "family" else arc.folds
        for p1 in sorted((ROOT / "results/p1" / ev_id).glob("*/seed*.json")):
            ref = json.loads(p1.read_text())
            cls, seed = ref["held_out"], ref["seed"]
            path = out_root / ev_id / p1.parent.name / p1.name
            if a.resume and path.exists():
                continue
            if a.max_runs is not None and n >= a.max_runs:
                return
            t0 = time.time()
            print(f"[{ev_id}] re-execute {cls} seed {seed}", flush=True)
            f = folds[cls]
            tr, va, te = (arc.windows(f[k], labelling=lab) for k in ("train", "val", "test"))
            m, _ = R.train_model(arc, tr, va, seed, cfg, log=lambda s: None)
            cache = {}
            res = R.score_class(m, arc, tr, va, te, classes.index(cls), seed, cfg, cache)
            repro = {
                "stress_auc": (ref["measured"]["stress"]["auc"], res["measured"]["stress"]["auc"]),
                "prob_auc": (ref["measured"]["prob"]["auc"], res["measured"]["prob"]["auc"]),
                "D_pred": (ref["predictors"]["D_pred"], res["predictors"]["D_pred"]),
            }
            dev = {k: abs(x - y) for k, (x, y) in repro.items()}
            if max(dev.values()) > TOL:
                raise SystemExit(f"{ev_id}/{cls}/seed{seed}: re-execution does not reproduce P1 {dev}; "
                                 "S2-dir not computed (fail closed)")
            st = predict(m, te["X"], states=True)
            is_c, is_b = te["y_cls"] == classes.index(cls), te["y_cls"] == 0
            E0 = m.E0.numpy()
            g2 = np.asarray(res["s2"]["H_gain2"])
            sd = s2dir.s2dir_raw(st["z"][is_c], st["z"][is_b], cache["trb"]["z"], E0, g2, seed)
            write_json(path, {
                "eval_id": ev_id, "dataset": ev["dataset"], "scheme": ev["scheme"], "labelling": lab,
                "rho_set": ev["rho_set"], "held_out": cls, "seed": seed, "exploratory": True,
                "s2dir": sd, "reproduction": {"p1_file": str(p1.relative_to(ROOT)), "values": repro,
                                              "abs_dev": dev, "tolerance": TOL, "ok": True},
                "p1_measured_stress_auc": ref["measured"]["stress"]["auc"],
                "p1_code_commit": ref["provenance"]["code_commit"],
                "wall_seconds": time.time() - t0,
                "provenance": provenance(config={**cfg, "evaluation": ev, "experiment": "P1b-S2dir"},
                                         data_sha256=ref["provenance"]["data_sha256"]),
            })
            n += 1


if __name__ == "__main__":
    main()
