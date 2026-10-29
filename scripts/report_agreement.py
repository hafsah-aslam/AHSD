"""Protocol agreement: does LOACO evidence agree with natural-novelty evidence? From results JSON only.

Post-hoc and descriptive: no new runs, no new models, no thresholds other than chance (AUC 0.5).
For each dataset with both protocols (D1, D3, D5), each backbone (AHSD, GRU) and each natural-novelty view
(the fixed pre-registered split, and the rolling cut-points pooled per dataset), compares the per-detector
mean AUC (over classes and the 5 seeds, best checkpoint) and TPR at 1% FPR between the two protocols:

  delta_auc            AUC_natural - AUC_LOACO per detector; mean, mean |.|, max |.|
  delta_tpr1           same for TPR at 1% FPR
  spearman / kendall   rank agreement of the detectors
  discordant_pairs     fraction of detector pairs whose order reverses
  winner               top detector under each protocol, and whether it changes
  chance_side_flips    detectors on opposite sides of AUC 0.5 under the two protocols
  family_leader        leading novelty family (report_a7 definition), and whether it changes

Class-matched check: D3 classes evaluated under both protocols (exact class names).

Writes results/final_v2/agreement.json and prints a summary.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_run import BEN, CLAD_, CONF, GRU_CONF, GRU_REPR, REPR, XGB, rolling_evals  # noqa: E402
from nids.provenance import ROOT, write_json  # noqa: E402
from report_a7 import SEEDS, tensors  # noqa: E402

RES = ROOT / "results/final_v2"
BACKBONES = {"AHSD": CONF + REPR + BEN + [CLAD_, XGB], "GRU": GRU_CONF + GRU_REPR + BEN + [CLAD_, XGB]}
FAMILIES = {"AHSD": {"confidence": CONF, "representation": REPR, "benign-only": BEN, "CLAD": [CLAD_]},
            "GRU": {"GRU-confidence": GRU_CONF, "GRU-representation": GRU_REPR, "benign-only": BEN,
                    "CLAD": [CLAD_]}}
LOACO = {"D1": "loaco_D1fam", "D2": "loaco_D2", "D3": "loaco_D3", "D5": "loaco_D5pb"}
NN = {"D1": "nn_D1", "D3": "nn_D3", "D5": "nn_D5"}
MATCHED = {"D3": ["Backdoor", "password", "xss"]}


def load():
    runs = [json.loads(p.read_text()) for p in sorted(RES.glob("*/*/seed*.json"))]
    commits = {r["provenance"]["code_commit"] for r in runs}
    if len(commits) != 1 or any(r["provenance"]["code_dirty"] or r.get("smoke") for r in runs):
        raise SystemExit(f"agreement: commits {sorted(commits)}, dirty or smoke runs present")
    return runs, commits.pop()


def det_means(A, O, units, dets):
    """Per-detector mean over (eval, class) units and seeds: AUC and TPR at 1% FPR."""
    auc = {d: float(np.mean([A[e]["best"][d][c][s] for e, c in units for s in SEEDS])) for d in dets}
    tpr = {d: float(np.nanmean([O[e][d][c]["tpr_at_1pct_fpr"][s] for e, c in units for s in SEEDS])) for d in dets}
    return auc, tpr


def compare(lo_auc, nn_auc, lo_tpr, nn_tpr, fams):
    dets = list(lo_auc)
    x, y = np.array([lo_auc[d] for d in dets]), np.array([nn_auc[d] for d in dets])
    dt = np.array([nn_tpr[d] - lo_tpr[d] for d in dets])
    pairs = list(combinations(range(len(dets)), 2))
    disc = sum((x[i] - x[j]) * (y[i] - y[j]) < 0 for i, j in pairs)
    rank = lambda v: {d: int(r) for d, r in zip(dets, (-v).argsort().argsort() + 1)}  # noqa: E731
    rl, rn = rank(x), rank(y)
    fam = lambda m: {f: float(np.mean([m[d] for d in ds])) for f, ds in fams.items()}  # noqa: E731
    fl, fn = fam(lo_auc), fam(nn_auc)
    lead_l, lead_n = max(fl, key=fl.get), max(fn, key=fn.get)
    win_l, win_n = dets[int(x.argmax())], dets[int(y.argmax())]
    return {
        "n_detectors": len(dets),
        "auc_loaco": lo_auc, "auc_natural": nn_auc, "tpr1_loaco": lo_tpr, "tpr1_natural": nn_tpr,
        "delta_auc": {d: float(y[i] - x[i]) for i, d in enumerate(dets)},
        "mean_delta_auc": float((y - x).mean()), "mean_abs_delta_auc": float(np.abs(y - x).mean()),
        "max_abs_delta_auc": float(np.abs(y - x).max()),
        "mean_delta_tpr1": float(dt.mean()), "mean_abs_delta_tpr1": float(np.abs(dt).mean()),
        "spearman": float(spearmanr(x, y).statistic), "kendall": float(kendalltau(x, y).statistic),
        "discordant_pairs": int(disc), "n_pairs": len(pairs), "discordant_frac": disc / len(pairs),
        "rank_loaco": rl, "rank_natural": rn,
        "mean_abs_rank_shift": float(np.mean([abs(rl[d] - rn[d]) for d in dets])),
        "max_abs_rank_shift": int(max(abs(rl[d] - rn[d]) for d in dets)),
        "winner_loaco": win_l, "winner_natural": win_n, "winner_flip": win_l != win_n,
        "chance_side_flips": [d for d in dets if (lo_auc[d] - 0.5) * (nn_auc[d] - 0.5) < 0],
        "family_auc_loaco": fl, "family_auc_natural": fn,
        "family_leader_loaco": lead_l, "family_leader_natural": lead_n, "family_leader_flip": lead_l != lead_n,
    }


def main():
    runs, commit = load()
    A, O, classes = tensors(runs)
    roll = {}
    for e in rolling_evals():
        roll.setdefault(e["dataset"], []).extend((e["id"], c) for c in classes[e["id"]])
    configs = {}
    for ds in NN:
        lo_units = [(LOACO[ds], c) for c in classes[LOACO[ds]]]
        views = {"fixed": [(NN[ds], c) for c in classes[NN[ds]]], "rolling": roll[ds]}
        for bb, dets in BACKBONES.items():
            la, lt = det_means(A, O, lo_units, dets)
            for v, units in views.items():
                na, nt = det_means(A, O, units, dets)
                r = compare(la, na, lt, nt, FAMILIES[bb])
                r.update(dataset=ds, backbone=bb, view=v, loaco_eval=LOACO[ds],
                         n_loaco_units=len(lo_units), n_natural_units=len(units))
                configs[f"{ds}|{bb}|{v}"] = r
    matched = {}
    for ds, cls in MATCHED.items():
        for bb, dets in BACKBONES.items():
            for c in cls:
                la, lt = det_means(A, O, [(LOACO[ds], c)], dets)
                na, nt = det_means(A, O, [(NN[ds], c)], dets)
                matched[f"{ds}|{bb}|{c}"] = compare(la, na, lt, nt, FAMILIES[bb])
    C = list(configs.values())
    allpairs = [(r["auc_loaco"][d], r["auc_natural"][d]) for r in C for d in r["auc_loaco"]]
    agg = {
        "n_configurations": len(C),
        "configs_natural_lower_mean_auc": sum(r["mean_delta_auc"] < 0 for r in C),
        "winner_flips": sum(r["winner_flip"] for r in C),
        "family_leader_flips": sum(r["family_leader_flip"] for r in C),
        "configs_with_chance_side_flip": sum(bool(r["chance_side_flips"]) for r in C),
        "chance_side_flips_total": sum(len(r["chance_side_flips"]) for r in C),
        "detector_config_cells": len(allpairs),
        "spearman_range": [min(r["spearman"] for r in C), max(r["spearman"] for r in C)],
        "spearman_median": float(np.median([r["spearman"] for r in C])),
        "discordant_frac_range": [min(r["discordant_frac"] for r in C), max(r["discordant_frac"] for r in C)],
        "discordant_frac_mean": float(np.mean([r["discordant_frac"] for r in C])),
        "mean_abs_delta_auc_range": [min(r["mean_abs_delta_auc"] for r in C), max(r["mean_abs_delta_auc"] for r in C)],
        "max_abs_delta_auc": max(r["max_abs_delta_auc"] for r in C),
        "mean_delta_auc_range": [min(r["mean_delta_auc"] for r in C), max(r["mean_delta_auc"] for r in C)],
        "mean_abs_delta_tpr1_range": [min(r["mean_abs_delta_tpr1"] for r in C), max(r["mean_abs_delta_tpr1"] for r in C)],
        "max_abs_rank_shift": max(r["max_abs_rank_shift"] for r in C),
        "matched_spearman_range": [min(r["spearman"] for r in matched.values()),
                                   max(r["spearman"] for r in matched.values())],
        "matched_mean_abs_delta_auc_range": [min(r["mean_abs_delta_auc"] for r in matched.values()),
                                             max(r["mean_abs_delta_auc"] for r in matched.values())],
        "matched_winner_flips": sum(r["winner_flip"] for r in matched.values()),
        "matched_n": len(matched),
        "natural_novelty_infeasible": [ds for ds in LOACO if ds not in NN],
    }
    out = {"descriptive": True, "post_hoc": True, "new_runs": 0, "code_commit": commit,
           "checkpoint": "best", "seeds": SEEDS, "configurations": configs, "matched_classes": matched,
           "aggregate": agg}
    write_json(RES / "agreement.json", out)
    print(f"{'config':<22}{'rho':>7}{'tau':>7}{'disc':>7}{'mΔ':>8}{'m|Δ|':>7}{'max|Δ|':>8}{'m|ΔTPR|':>9}"
          f"  winner L -> N   family L -> N   chance flips")
    for k, r in configs.items():
        print(f"{k:<22}{r['spearman']:7.2f}{r['kendall']:7.2f}{r['discordant_frac']:7.2f}{r['mean_delta_auc']:+8.3f}"
              f"{r['mean_abs_delta_auc']:7.3f}{r['max_abs_delta_auc']:8.3f}{r['mean_abs_delta_tpr1']:9.3f}  "
              f"{r['winner_loaco']}->{r['winner_natural']}  {r['family_leader_loaco']}->{r['family_leader_natural']}"
              f"  {r['chance_side_flips']}")
    for k, r in matched.items():
        print(f"matched {k:<22}{r['spearman']:7.2f}{r['mean_delta_auc']:+8.3f}{r['mean_abs_delta_auc']:7.3f}  "
              f"{r['winner_loaco']}->{r['winner_natural']}  {r['chance_side_flips']}")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
