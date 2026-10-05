#!/usr/bin/env python
"""P1d (AMENDMENT_04): 10 detectors on D5 natural_novelty, D5 LOACO, D3 LOACO; Wilkie on D3 natural_novelty.

Per (evaluation, unit, seed) one JSON under results/p1d/<eval>/. Unit = held-out class (LOACO)
or "all" (natural_novelty: one model per seed, every natural-novelty class scored).
Detectors and score directions as AMENDMENT_03; CLAD as AMENDMENT_04. Per-class AUC: held-out
class vs the evaluation's test benign windows.

  python scripts/p1d_run.py [--resume] [--only D5_nn ...] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_loaco as R  # noqa: E402
from p1c_run import P1cArchive, backbone_outputs  # noqa: E402
from nids import detectors as det  # noqa: E402
from nids import metrics  # noqa: E402
from nids.clad import CLAD  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.preflight import preflight  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402

SUPERVISED = ["AHSD_stress", "MSP", "Energy", "Mahalanobis", "kNN"]
BENIGN_ONLY = ["IF", "OCSVM", "PCA", "AE"]
ALL = SUPERVISED + BENIGN_ONLY + ["Wilkie_CLAD"]
CFG = {"seeds": [17, 23, 42, 101, 202], "D": 128, "epochs": 15, "threads": 4, "processed_dir": "data/processed"}
EVALS = [
    {"id": "D5_nn", "dataset": "D5", "kind": "natural_novelty", "package": "temporal_gap/p1d", "detectors": ALL},
    {"id": "D5_loaco", "dataset": "D5", "kind": "loaco", "scheme": "grouped_random", "detectors": ALL},
    {"id": "D3_loaco", "dataset": "D3", "kind": "loaco", "scheme": "grouped_random", "detectors": ALL},
    {"id": "D3_nn_wilkie", "dataset": "D3", "kind": "natural_novelty", "package": "temporal_gap/p1c",
     "detectors": ["Wilkie_CLAD"]},
]


def score_sets(arc, tr, va, eval_sets: dict, seed: int, detectors: list[str], cfg: dict) -> tuple[dict, dict]:
    """Scores of every requested detector on every eval set; benign references = training benign."""
    out, info = {}, {}
    Xtrb = tr["X"][tr["y_bin"] == 0]
    if any(d in SUPERVISED for d in detectors):
        m, tinfo = R.train_model(arc, tr, va, seed, cfg, log=lambda s: None)
        info["ahsd_best_val_macro_f1"] = tinfo["best_val_macro_f1"]
        base = {k: backbone_outputs(m, X) for k, X in eval_sets.items()}
        pen_tr = backbone_outputs(m, Xtrb)["pen"]
        maha, knn = det.BenignMahalanobis().fit(pen_tr), det.BenignKNN().fit(pen_tr)
        out["AHSD_stress"] = {k: v["stress"] for k, v in base.items()}
        out["MSP"] = {k: det.msp_score(v["logits"]) for k, v in base.items()}
        out["Energy"] = {k: det.energy_score(v["logits"]) for k, v in base.items()}
        out["Mahalanobis"] = {k: maha.score(v["pen"]) for k, v in base.items()}
        out["kNN"] = {k: knn.score(v["pen"]) for k, v in base.items()}
    for name in BENIGN_ONLY:
        if name in detectors:
            d = det.BENIGN_ONLY[name](seed).fit(Xtrb)
            out[name] = {k: d.score(X) for k, X in eval_sets.items()}
    if "Wilkie_CLAD" in detectors:
        c = CLAD(seed).fit(tr["X"], tr["y_bin"], va["X"], va["y_bin"])
        info["clad_best_val_auroc"] = c.best_val_auroc
        out["Wilkie_CLAD"] = {k: c.score(X) for k, X in eval_sets.items()}
    return out, info


def aucs(scores: dict, classes: list[str], benign_key: str) -> dict:
    res = {}
    for d, sc in scores.items():
        tb = sc[benign_key]
        res[d] = {c: metrics.auc(np.r_[np.zeros(len(tb)), np.ones(len(sc[f'test/{c}']))], np.r_[tb, sc[f"test/{c}"]])
                  for c in classes}
    return res


def run_nn(ev, seed, cfg):
    arc = P1cArchive(ROOT / cfg["processed_dir"] / ev["dataset"] / ev["package"])
    classes = arc.meta["natural_novelty"]
    W = {k: arc.windows(v) for k, v in arc.sets.items() if not k.startswith("anchor_")}
    eval_sets = {"test_benign": W["test_benign"]["X"], **{f"test/{c}": W[f"test/{c}"]["X"] for c in classes}}
    scores, info = score_sets(arc, W["train"], W["val"], eval_sets, seed, ev["detectors"], cfg)
    return {"unit": "all", "classes": classes, "auc": aucs(scores, classes, "test_benign"), "info": info,
            "n": {k: int(len(v)) for k, v in eval_sets.items()}, "data_sha256": arc.meta["archive"]["sha256"]}


def run_loaco(ev, arc, meta, cls, seed, cfg):
    f = arc.folds[cls]
    tr, va, te = (arc.windows(f[k]) for k in ("train", "val", "test"))
    c = meta["classes"].index(cls)
    eval_sets = {"test_benign": te["X"][te["y_cls"] == 0], f"test/{cls}": te["X"][te["y_cls"] == c]}
    if not np.all((te["y_cls"] == 0) | (te["y_cls"] == c)):
        raise SystemExit(f"{ev['id']}/{cls}: test holds other classes")
    scores, info = score_sets(arc, tr, va, eval_sets, seed, ev["detectors"], cfg)
    return {"unit": cls, "classes": [cls], "auc": aucs(scores, [cls], "test_benign"), "info": info,
            "n": {k: int(len(v)) for k, v in eval_sets.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, first fold; under smoke_out/")
    a = ap.parse_args()
    cfg = dict(CFG)
    out_root = ROOT / "results/p1d"
    if a.smoke:
        cfg.update(seeds=[17], epochs=1)
        out_root = ROOT / "smoke_out/results/p1d"
    else:
        require_clean_tree("P1d")
        print(f"preflight: {preflight('P1d')['summary']}", flush=True)
    for ev in EVALS:
        if a.only and ev["id"] not in a.only:
            continue
        if ev["kind"] == "natural_novelty":
            for seed in cfg["seeds"]:
                path = out_root / ev["id"] / "all" / f"seed{seed}.json"
                if a.resume and path.exists():
                    continue
                t0 = time.time()
                print(f"[{ev['id']}] seed {seed}", flush=True)
                r = run_nn(ev, seed, cfg)
                data_sha = r.pop("data_sha256")
                write_json(path, {"eval_id": ev["id"], "dataset": ev["dataset"], "kind": ev["kind"], "seed": seed, **r,
                                  "wall_seconds": time.time() - t0, "smoke": a.smoke,
                                  "provenance": provenance(config={**cfg, "evaluation": ev}, data_sha256=data_sha)})
        else:
            arc = Archive(ROOT / cfg["processed_dir"] / ev["dataset"] / ev["scheme"] / "archive")
            meta = json.loads((ROOT / cfg["processed_dir"] / ev["dataset"] / "index_meta.json").read_text())
            data_sha = json.loads((ROOT / cfg["processed_dir"] / ev["dataset"] / "archive_meta.json").read_text())["schemes"][ev["scheme"]]["sha256"]
            finfo = meta["schemes"][ev["scheme"]]["loaco"]
            feasible = [c for c, rec in finfo.items() if rec.get("eligible") and rec.get("feasible")]
            write_json(out_root / ev["id"] / "folds.json", {"feasible": feasible, "fold_info": finfo,
                                                            "provenance": provenance(config={**cfg, "evaluation": ev})})
            for cls in (feasible[:1] if a.smoke else feasible):
                for seed in cfg["seeds"]:
                    path = out_root / ev["id"] / R._safe(cls) / f"seed{seed}.json"
                    if a.resume and path.exists():
                        continue
                    t0 = time.time()
                    print(f"[{ev['id']}] {cls} seed {seed}", flush=True)
                    r = run_loaco(ev, arc, meta, cls, seed, cfg)
                    write_json(path, {"eval_id": ev["id"], "dataset": ev["dataset"], "kind": ev["kind"], "seed": seed,
                                      **r, "wall_seconds": time.time() - t0, "smoke": a.smoke,
                                      "provenance": provenance(config={**cfg, "evaluation": ev}, data_sha256=data_sha)})


if __name__ == "__main__":
    main()
