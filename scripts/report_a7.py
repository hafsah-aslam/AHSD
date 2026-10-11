"""AMENDMENT_07 additions to the final report, from results JSON only (called by report_final.py).

  GRU backbone replication (A7.3)   GRU-confidence / GRU-representation families, never pooled with AHSD's
  rolling cut-points (A7.2)         per cut, pooled over (dataset, cut, class) units, per dataset, per unique class
  XGBoost-s (A7.4)                  seed std and AUC next to the deterministic XGBoost
  operational metrics (A7.5)        TPR at 1% / 0.1% FPR and PR-AUC per family and evaluation
  ablations                         backbone, checkpoint, XGBoost seeding, protocol, cut fraction
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from final_run import BEN, CLAD_, CONF, EVALS, GRU_CONF, GRU_REPR, REPR, XGB, XGBS, rolling_evals
from nids.provenance import ROOT

SEEDS = [17, 23, 42, 101, 202]
FAM7 = {"confidence": CONF, "representation": REPR, "benign-only": BEN, "CLAD": [CLAD_], "XGBoost": [XGB],
        "GRU-confidence": GRU_CONF, "GRU-representation": GRU_REPR, "XGBoost-s": [XGBS]}
NOVELTY_AHSD = ["confidence", "representation", "benign-only", "CLAD"]
NOVELTY_GRU = ["GRU-confidence", "GRU-representation", "benign-only", "CLAD"]
DETS7 = [d for f in FAM7.values() for d in f]
CK_DEP7 = CONF + REPR + [CLAD_] + GRU_CONF + GRU_REPR
OPS = ["tpr_at_1pct_fpr", "tpr_at_0.1pct_fpr", "pr_auc"]
PRIMARY = [e["id"] for e in EVALS if e["kind"] in ("loaco", "natural_novelty")]
LOACO = [e["id"] for e in EVALS if e["kind"] == "loaco"]
NN = [e["id"] for e in EVALS if e["kind"] == "natural_novelty"]
N_BOOT, BOOT_SEED = 2000, 0


def boot(M, rng_seed=BOOT_SEED):
    rng = np.random.default_rng(rng_seed)
    C, S = M.shape
    b = [M[np.ix_(rng.integers(0, C, C), rng.integers(0, S, S))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(M.mean()), "ci95": [float(lo), float(hi)], "n_units": int(C)}


def tensors(runs):
    """A[eval][ck][det][class][seed] = auc; O[eval][det][class][metric][seed] (best checkpoint)."""
    A = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    O = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    classes = {}
    for r in runs:
        if r["kind"] not in ("loaco", "natural_novelty"):
            continue
        e = r["eval_id"]
        classes.setdefault(e, [])
        for c in r["classes"]:
            if c not in classes[e]:
                classes[e].append(c)
        for ck, by in r["auc"].items():
            for d, bc in by.items():
                for c, a in bc.items():
                    A[e][ck][d][c][r["seed"]] = a
        for d, bc in r["ops"]["best"].items():
            for c, mm in bc.items():
                for m, v in mm.items():
                    O[e][d][c][m][r["seed"]] = v
    return A, O, classes


def check(A, classes, evals):
    for e in evals:
        if e not in A:
            raise SystemExit(f"A7: {e} has no runs")
        for ck, dets in (("best", DETS7), ("final", CK_DEP7)):
            for d in dets:
                for c in classes[e]:
                    if sorted(A[e][ck][d][c]) != SEEDS:
                        raise SystemExit(f"A7: {e}/{ck}/{d}/{c}: seeds {sorted(A[e][ck][d][c])}")


def fmat(Ack, fam, units):
    """(U, S) family mean per unit and seed; units = [(eval, class)] with Ack[eval][det][class][seed]."""
    return np.array([[np.mean([Ack[e][d][c][s] for d in FAM7[fam]]) for s in SEEDS] for e, c in units])


def omat(O, fam, units, m):
    return np.array([[np.nanmean([O[e][d][c][m][s] for d in FAM7[fam]]) for s in SEEDS] for e, c in units])


def by_ck(A, ck):
    return {e: A[e][ck] for e in A}


def seed_std(A, d, units):
    return float(np.mean([np.std([A[e]["best"][d][c][s] for s in SEEDS], ddof=1) for e, c in units]))


def leader(fams, means):
    return max(fams, key=lambda f: means[f]["mean"])


def unit_stats(A, O, units):
    best = by_ck(A, "best")
    st = {"n_units": len(units), "families": {}, "ops": {}, "seed_std": {}, "cells": {}}
    for f in FAM7:
        st["families"][f] = boot(fmat(best, f, units))
        st["ops"][f] = {m: boot(omat(O, f, units, m)) for m in OPS}
    for d in DETS7:
        st["seed_std"][d] = seed_std(A, d, units)
        st["cells"][d] = {f"{e}|{c}": float(np.mean([A[e]["best"][d][c][s] for s in SEEDS])) for e, c in units}
    st["leader_ahsd"] = leader(NOVELTY_AHSD, st["families"])
    st["leader_gru"] = leader(NOVELTY_GRU, st["families"])
    for x in ("XGBoost", "XGBoost-s"):
        st[f"{x}_minus_leader_ahsd"] = boot(fmat(best, x, units) - fmat(best, st["leader_ahsd"], units))
    st["contrasts"] = {
        "benign-only − representation": boot(fmat(best, "benign-only", units) - fmat(best, "representation", units)),
        "benign-only − GRU-representation": boot(fmat(best, "benign-only", units) - fmat(best, "GRU-representation", units)),
        "representation − GRU-representation": boot(fmat(best, "representation", units)
                                                    - fmat(best, "GRU-representation", units)),
        "confidence − GRU-confidence": boot(fmat(best, "confidence", units) - fmat(best, "GRU-confidence", units)),
        "CLAD − benign-only": boot(fmat(best, "CLAD", units) - fmat(best, "benign-only", units))}
    fin = by_ck(A, "final")
    st["checkpoint"] = {f: boot(fmat(fin, f, units) - fmat(best, f, units))
                        for f in ("confidence", "representation", "CLAD", "GRU-confidence", "GRU-representation")}
    msp = {d: float(np.mean([np.mean([A[e]["best"][d][c][s] for s in SEEDS]) < 0.5 for e, c in units]))
           for d in ("MSP", "Energy", "GRU_MSP", "GRU_Energy", "P_attack", "GRU_P_attack")}
    st["frac_units_below_0.5"] = msp
    return st


def a7_stats(runs) -> dict:
    A, O, classes = tensors(runs)
    roll = [e["id"] for e in rolling_evals()]
    check(A, classes, PRIMARY + roll)
    feas = json.loads((ROOT / "results/final_v2/rolling_feasibility.json").read_text())
    R = {"evals": {}, "rolling": {}, "feasibility": {"n_units": feas["n_units"], "n_unique": feas["n_unique_classes"],
                                                      "by_cut": {ds: {c: {"feasible": r["feasible"],
                                                                          "classes": r["natural_novelty"]}
                                                                      for c, r in v.items()}
                                                                 for ds, v in feas["datasets"].items()}}}
    for e in PRIMARY + roll:
        R["evals"][e] = unit_stats(A, O, [(e, c) for c in classes[e]])
        R["evals"][e]["classes"] = classes[e]
    # rolling: pooled, per dataset, per cut fraction, per unique (dataset, class)
    units = [(e, c) for e in roll for c in classes[e]]
    R["rolling"]["pooled"] = unit_stats(A, O, units)
    ds_of = {e["id"]: e["dataset"] for e in rolling_evals()}
    cut_of = {e["id"]: e["cut"] for e in rolling_evals()}
    R["rolling"]["by_dataset"] = {ds: unit_stats(A, O, [u for u in units if ds_of[u[0]] == ds])
                                  for ds in sorted(set(ds_of.values()))}
    R["rolling"]["by_cut"] = {str(f): unit_stats(A, O, [u for u in units if cut_of[u[0]] == f])
                              for f in sorted(set(cut_of.values()))}
    uniq = defaultdict(list)
    for e, c in units:
        uniq[(ds_of[e], c)].append(e)
    best = by_ck(A, "best")
    R["rolling"]["unique"] = {}
    for f in FAM7:
        M = np.array([[np.mean([np.mean([best[e][d][c][s] for d in FAM7[f]]) for e in es]) for s in SEEDS]
                      for (ds, c), es in sorted(uniq.items())])
        R["rolling"]["unique"][f] = boot(M)
    R["rolling"]["unique_classes"] = {f"{ds}/{c}": sorted(cut_of[e] for e in es) for (ds, c), es in sorted(uniq.items())}
    # seed-std ratios: natural novelty (primary NN, rolling) vs LOACO, per detector
    lo_units = [(e, c) for e in LOACO for c in classes[e]]
    nn_units = [(e, c) for e in NN for c in classes[e]]
    R["seed_std_ratio"] = {}
    for d in DETS7:
        lo = seed_std(A, d, lo_units)
        R["seed_std_ratio"][d] = {"loaco": lo, "nn": seed_std(A, d, nn_units), "rolling": seed_std(A, d, units),
                                  "nn_ratio": seed_std(A, d, nn_units) / lo if lo > 0 else float("nan"),
                                  "rolling_ratio": seed_std(A, d, units) / lo if lo > 0 else float("nan"),
                                  "rolling_by_dataset": {ds: seed_std(A, d, [u for u in units if ds_of[u[0]] == ds])
                                                         for ds in sorted(set(ds_of.values()))}}
    # leader counts over the 7 primary evaluations
    R["leaders"] = {e: {"ahsd": R["evals"][e]["leader_ahsd"], "gru": R["evals"][e]["leader_gru"]} for e in PRIMARY}
    R["xgb_ge_leader"] = {x: [e for e in PRIMARY if R["evals"][e]["families"][x]["mean"]
                              >= R["evals"][e]["families"][R["evals"][e]["leader_ahsd"]]["mean"]]
                          for x in ("XGBoost", "XGBoost-s")}
    return R


def indist_a7(I) -> dict:
    """In-distribution GRU confidence scores and XGBoost-s, from report_final.indist_stats output."""
    out = {}
    for e, by in I.items():
        b = by["best"]
        out[e] = {d: {"auc": b[d]["test"]["auc"]["mean"], "std": b[d]["test"]["auc"]["std"]}
                  for d in ("MSP", "P_attack", "GRU_MSP", "GRU_P_attack", "XGBoost", "XGBoost_s") if d in b}
    return out
