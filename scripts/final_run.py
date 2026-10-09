#!/usr/bin/env python
"""Final benchmark run (AMENDMENT_06). Descriptive only; no gate.

One code commit for every number in the paper. Seeds 17/23/42/101/202. Detectors:
  classifier-confidence  MSP, Energy, P_attack (P_attack added post-G3, A6.2)
  representation-distance AHSD_stress, Mahalanobis, kNN (penultimate features)
  benign-only            IF, OCSVM, PCA, AE
  Wilkie_CLAD            (AMENDMENT_04)
  XGBoost                supervised reference (spec §7), fixed parameters
Every score: higher = more anomalous / attack; never flipped.

Checkpoint sensitivity (A6, every evaluation): the AHSD backbone and CLAD are trained once
per unit; both the best-validation state (primary) and the final-epoch state are kept and
scored. Benign-only detectors and XGBoost have no checkpoint and are scored once.

Evaluations (one JSON per (evaluation, unit, seed) under results/final/<eval>/<unit>/):
  loaco_D1fam, loaco_D2, loaco_D3   grouped_random LOACO (D1: attack families)
  loaco_D5pb                         D5 purged_block, single feasible fold (dos_hulk; limitation)
  nn_D1, nn_D3, nn_D5               natural_novelty, train-tail validation packages
  indist_<ds>_<scheme>              in-distribution binary detection, D1/D2/D3/D5 x temporal_gap/grouped_random;
                                     for D1-D3 temporal_gap the same models are also scored zero-shot on
                                     every other NF-v3 dataset incl. D4 (transfer packages, source cleaner)

  python scripts/final_run.py [--resume] [--only loaco_D3 ...] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1c_run import P1cArchive, backbone_outputs  # noqa: E402
from nids import detectors as det  # noqa: E402
from nids import metrics  # noqa: E402
from nids.clad import CLAD  # noqa: E402
from nids.models.ahsd import AHSD  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.preflight import preflight  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.train import benign_embedding_mean, fit, set_seed  # noqa: E402

CONF = ["MSP", "Energy", "P_attack"]
REPR = ["AHSD_stress", "Mahalanobis", "kNN"]
BEN = ["IF", "OCSVM", "PCA", "AE"]
CLAD_ = "Wilkie_CLAD"
XGB = "XGBoost"
ALL = CONF + REPR + BEN + [CLAD_, XGB]
CK_DEP = CONF + REPR + [CLAD_]
CFG = {"seeds": [17, 23, 42, 101, 202], "D": 128, "epochs": 15, "threads": 4, "processed_dir": "data/processed"}
PROC = ROOT / CFG["processed_dir"]
TRANSFER_TARGETS = {"D1": ["D2", "D3", "D4"], "D2": ["D1", "D3", "D4"], "D3": ["D1", "D2", "D4"]}

EVALS = (
    [{"id": "loaco_D1fam", "kind": "loaco", "dataset": "D1", "scheme": "grouped_random", "labelling": "family"},
     {"id": "loaco_D2", "kind": "loaco", "dataset": "D2", "scheme": "grouped_random", "labelling": "file"},
     {"id": "loaco_D3", "kind": "loaco", "dataset": "D3", "scheme": "grouped_random", "labelling": "file"},
     {"id": "loaco_D5pb", "kind": "loaco", "dataset": "D5", "scheme": "purged_block", "labelling": "file"},
     {"id": "nn_D1", "kind": "natural_novelty", "dataset": "D1", "package": "temporal_gap/p1c"},
     {"id": "nn_D3", "kind": "natural_novelty", "dataset": "D3", "package": "temporal_gap/p1c"},
     {"id": "nn_D5", "kind": "natural_novelty", "dataset": "D5", "package": "temporal_gap/p1d"}]
    + [{"id": f"indist_{ds}_{sch}", "kind": "indist", "dataset": ds, "scheme": sch,
        "transfer_to": TRANSFER_TARGETS.get(ds, []) if sch == "temporal_gap" else []}
       for ds in ("D1", "D2", "D3", "D5") for sch in ("temporal_gap", "grouped_random")]
)


def score_all(n_features: int, tr: dict, va: dict, eval_sets: dict, seed: int, cfg: dict) -> tuple[dict, dict]:
    """scores[checkpoint][detector][set]; "best" holds every detector, "final" the checkpoint-dependent ones."""
    torch.set_num_threads(cfg["threads"])
    Xtrb = tr["X"][tr["y_bin"] == 0]
    S = {"best": {}, "final": {}}
    info = {}
    set_seed(seed)  # before construction: initial weights depend on this seed only
    m = AHSD(n_features, D=cfg["D"], variant="fixed")
    tinfo = fit(m, tr, va, seed=seed, epochs=cfg["epochs"], log=lambda s: None, keep_states=True)
    info["ahsd"] = {"best_val_macro_f1": tinfo["best_val_macro_f1"], "final_val_macro_f1": tinfo["final_val_macro_f1"],
                    "best_epoch": int(np.argmax([h["val_macro_f1"] for h in tinfo["history"]]) + 1),
                    "history": tinfo["history"], "seconds": tinfo["seconds"]}
    for ck in ("best", "final"):
        m.load_state_dict(tinfo["states"][ck])
        m.set_E0(benign_embedding_mean(m, Xtrb))
        out = {k: backbone_outputs(m, X) for k, X in eval_sets.items()}
        pen_tr = backbone_outputs(m, Xtrb)["pen"]
        maha, knn = det.BenignMahalanobis().fit(pen_tr), det.BenignKNN().fit(pen_tr)
        S[ck]["MSP"] = {k: det.msp_score(v["logits"]) for k, v in out.items()}
        S[ck]["Energy"] = {k: det.energy_score(v["logits"]) for k, v in out.items()}
        S[ck]["P_attack"] = {k: det.p_attack_score(v["logits"]) for k, v in out.items()}
        S[ck]["AHSD_stress"] = {k: v["stress"] for k, v in out.items()}
        S[ck]["Mahalanobis"] = {k: maha.score(v["pen"]) for k, v in out.items()}
        S[ck]["kNN"] = {k: knn.score(v["pen"]) for k, v in out.items()}
    t0 = time.time()
    for name in BEN:
        d = det.BENIGN_ONLY[name](seed).fit(Xtrb)
        S["best"][name] = {k: d.score(X) for k, X in eval_sets.items()}
    info["benign_only_seconds"] = time.time() - t0
    t0 = time.time()
    c = CLAD(seed).fit(tr["X"], tr["y_bin"], va["X"], va["y_bin"])
    info["clad"] = {"best_val_auroc": c.best_val_auroc, "final_val_auroc": c.history[-1]["val_auroc"],
                    "best_epoch": int(np.argmax([h["val_auroc"] for h in c.history]) + 1)}
    for ck in ("best", "final"):
        c.use(ck, Xtrb)
        S[ck][CLAD_] = {k: c.score(X) for k, X in eval_sets.items()}
    info["clad_seconds"] = time.time() - t0
    t0 = time.time()
    x = det.XGB(seed).fit(tr["X"], tr["y_bin"])
    S["best"][XGB] = {k: x.score(X) for k, X in eval_sets.items()}
    info["xgb_seconds"] = time.time() - t0
    return S, info


def per_class_auc(S: dict, classes: list[str], benign_key: str) -> dict:
    res = {}
    for ck, by_det in S.items():
        res[ck] = {}
        for d, sc in by_det.items():
            tb = sc[benign_key]
            res[ck][d] = {c: metrics.auc(np.r_[np.zeros(len(tb)), np.ones(len(sc[f"test/{c}"]))], np.r_[tb, sc[f"test/{c}"]])
                          for c in classes}
    return res


def binary_metrics(S: dict, y_by_set: dict) -> dict:
    """Detection metrics per (checkpoint, detector, set); classification at 0.5 for P_attack and XGBoost."""
    res = {}
    for ck, by_det in S.items():
        res[ck] = {}
        for d, sc in by_det.items():
            res[ck][d] = {}
            for k, y in y_by_set.items():
                r = metrics.detection(y, sc[k])
                if d in ("P_attack", XGB):
                    r.update(metrics.classification(y, (sc[k] >= 0.5).astype(int)))
                res[ck][d][k] = r
    return res


def run_nn(ev, seed, cfg):
    arc = P1cArchive(PROC / ev["dataset"] / ev["package"])
    classes = arc.meta["natural_novelty"]
    W = {k: arc.windows(v) for k, v in arc.sets.items() if not k.startswith("anchor_")}
    eval_sets = {"test_benign": W["test_benign"]["X"], **{f"test/{c}": W[f"test/{c}"]["X"] for c in classes}}
    S, info = score_all(arc.n_features, W["train"], W["val"], eval_sets, seed, cfg)
    return {"unit": "all", "classes": classes, "auc": per_class_auc(S, classes, "test_benign"), "info": info,
            "n": {k: int(len(v)) for k, v in eval_sets.items()}}


def loaco_context(ev):
    base = PROC / ev["dataset"]
    if ev["scheme"] == "purged_block":
        pm = json.loads((ROOT / "results/p1d/d5_purged_block_meta.json").read_text())
        arc = Archive(base / "purged_block" / "archive")
        return arc, pm["classes"], arc.folds, pm["feasible_folds"], pm["loaco"], pm["archive"]["sha256"]
    arc = Archive(base / ev["scheme"] / "archive")
    meta = json.loads((base / "index_meta.json").read_text())
    am = json.loads((base / "archive_meta.json").read_text())["schemes"][ev["scheme"]]
    if ev["labelling"] == "family":
        classes, finfo, folds = meta["family"]["classes"], meta["schemes"][ev["scheme"]]["loaco_family"], arc.folds_family
    else:
        classes, finfo, folds = meta["classes"], meta["schemes"][ev["scheme"]]["loaco"], arc.folds
    feasible = [c for c, r in finfo.items() if r.get("eligible") and r.get("feasible")]
    return arc, classes, folds, feasible, finfo, am["sha256"]


def run_loaco(ev, arc, classes, folds, cls, seed, cfg):
    f = folds[cls]
    lab = ev["labelling"]
    tr, va, te = (arc.windows(f[k], labelling=lab) for k in ("train", "val", "test"))
    c = classes.index(cls)
    if not np.all((te["y_cls"] == 0) | (te["y_cls"] == c)):
        raise SystemExit(f"{ev['id']}/{cls}: test holds other classes")
    eval_sets = {"test_benign": te["X"][te["y_cls"] == 0], f"test/{cls}": te["X"][te["y_cls"] == c]}
    S, info = score_all(arc.n_features, tr, va, eval_sets, seed, cfg)
    return {"unit": cls, "classes": [cls], "auc": per_class_auc(S, [cls], "test_benign"), "info": info,
            "n": {k: int(len(v)) for k, v in eval_sets.items()}}


def run_indist(ev, seed, cfg):
    ds, sch = ev["dataset"], ev["scheme"]
    arc = Archive(PROC / ds / sch / "archive")
    tr, va = arc.windows(arc.sets["train"]), arc.windows(arc.sets["val"])
    eval_sets, ys, sha = {}, {}, {}
    for k in ("test", "test_natural"):
        w = arc.windows(arc.sets[k])
        eval_sets[k], ys[k] = w["X"], w["y_bin"]
    for tgt in ev["transfer_to"]:
        tp = PROC / "transfer" / f"{ds}__to__{tgt}"
        ta = Archive(tp / sch)
        tm = json.loads((tp / "meta.json").read_text())
        if ta.n_features != arc.n_features or ta.cleaner != arc.cleaner:
            raise SystemExit(f"{ds}->{tgt}: transfer package not built with the source cleaner")
        w = ta.windows(ta.sets["test"])
        eval_sets[f"transfer/{tgt}"], ys[f"transfer/{tgt}"] = w["X"], w["y_bin"]
        sha[tgt] = tm["schemes"][sch]["sha256"]
    S, info = score_all(arc.n_features, tr, va, eval_sets, seed, cfg)
    return {"unit": "all", "sets": list(eval_sets), "metrics": binary_metrics(S, ys), "info": info,
            "n": {k: {"benign": int((y == 0).sum()), "attack": int((y == 1).sum())} for k, y in ys.items()},
            "transfer_sha256": sha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 epoch, first fold; under smoke_out/")
    a = ap.parse_args()
    cfg = dict(CFG)
    out_root = ROOT / "results/final"
    if a.smoke:
        cfg.update(seeds=[17], epochs=1)
        out_root = ROOT / "smoke_out/results/final"
    else:
        require_clean_tree("final run")
        print(f"preflight: {preflight('final')['summary']}", flush=True)

    def write(path, ev, seed, r, t0, data_sha):
        write_json(path, {"eval_id": ev["id"], "kind": ev["kind"], "dataset": ev["dataset"], "seed": seed, **r,
                          "wall_seconds": time.time() - t0, "smoke": a.smoke,
                          "provenance": provenance(config={**cfg, "evaluation": ev}, data_sha256=data_sha)})

    for ev in EVALS:
        if a.only and ev["id"] not in a.only:
            continue
        if ev["kind"] == "loaco":
            arc, classes, folds, feasible, finfo, data_sha = loaco_context(ev)
            if sorted(folds) != sorted(set(folds) | set(feasible)) or not set(feasible) <= set(folds):
                raise SystemExit(f"{ev['id']}: feasible folds missing from the archive")
            write_json(out_root / ev["id"] / "folds.json", {"classes": classes, "feasible": feasible, "fold_info": finfo,
                                                            "provenance": provenance(config={**cfg, "evaluation": ev})})
            units = [(cls, lambda s, cls=cls: run_loaco(ev, arc, classes, folds, cls, s, cfg))
                     for cls in (feasible[:1] if a.smoke else feasible)]
        elif ev["kind"] == "natural_novelty":
            data_sha = json.loads((PROC / ev["dataset"] / ev["package"] / "p1c_meta.json").read_text())["archive"]["sha256"]
            units = [("all", lambda s: run_nn(ev, s, cfg))]
        else:
            data_sha = json.loads((PROC / ev["dataset"] / "archive_meta.json").read_text())["schemes"][ev["scheme"]]["sha256"]
            units = [("all", lambda s: run_indist(ev, s, cfg))]
        for unit, fn in units:
            for seed in cfg["seeds"]:
                path = out_root / ev["id"] / unit.replace("/", "_").replace(" ", "_") / f"seed{seed}.json"
                if a.resume and path.exists():
                    continue
                t0 = time.time()
                print(f"[{ev['id']}] {unit} seed {seed}", flush=True)
                write(path, ev, seed, fn(seed), t0, data_sha)


if __name__ == "__main__":
    main()
