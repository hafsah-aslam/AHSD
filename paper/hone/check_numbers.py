#!/usr/bin/env python
"""Check that numbers quoted in the HoNE paper match the results JSON (results/final_v2). Fails loudly."""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "results/final_v2/summary.json").read_text())
CS, I, T, R, AI = S["class_evals"], S["indist"], S["transfer"], S["a7"], S["a7_indist"]
RP = R["rolling"]["pooled"]
tex = (Path(__file__).parent / "main.tex").read_text()
f3 = lambda x: f"{x:.3f}"  # noqa: E731
ci = lambda v: f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"  # noqa: E731
sci = lambda v: "[{}, {}]".format(*(f"${x:+.3f}$" for x in v["ci95"]))  # noqa: E731
dm = lambda e, d: float(np.mean(list(R["evals"][e]["cells"][d].values())))  # noqa: E731
au = lambda e, d: I[e]["best"][d]["test"]["auc"]["mean"]  # noqa: E731
EV = ["loaco_D1fam", "loaco_D2", "loaco_D3", "loaco_D5pb", "nn_D1", "nn_D3", "nn_D5"]

# seed std with HoNE's pooling (LOACO D1-D3 vs natural D1, D3, D5)
v = defaultdict(list)
for p in glob.glob(str(ROOT / "results/final_v2/*/*/seed*.json")):
    r = json.loads(Path(p).read_text())
    if r["kind"] in ("loaco", "natural_novelty"):
        for d, bc in r["auc"]["best"].items():
            for c, a in bc.items():
                v[(r["eval_id"], d, c)].append(a)
lo, nn = ["loaco_D1fam", "loaco_D2", "loaco_D3"], ["nn_D1", "nn_D3", "nn_D5"]
ps = lambda evs, d: float(np.mean([np.std(x, ddof=1) for (e, dd, c), x in v.items() if e in evs and dd == d]))  # noqa: E731
row = lambda d: f"{ps(lo, d):.3f} & {ps(nn, d):.3f} & {ps(nn, d) / ps(lo, d):.1f}$\\times$"  # noqa: E731
viol = [(e, b) for e in R["evals"] for b in ("", "GRU_") if dm(e, b + "MSP") > dm(e, b + "P_attack")]
g3 = [f"indist_{d}_grouped_random" for d in ("D1", "D2", "D3")]
w = [json.loads(Path(p).read_text())["wall_seconds"] for p in glob.glob(str(ROOT / "results/final_v2/*/*/seed*.json"))]
ops = lambda st, f: st["ops"][f]["tpr_at_1pct_fpr"]["mean"]  # noqa: E731
ill = json.loads((ROOT / "paper/noms2027/figures/illusion.json").read_text())
win = json.loads((ROOT / "paper/noms2027/figures/winners.json").read_text())
FAM7 = ["confidence", "representation", "benign-only", "CLAD", "XGBoost", "GRU-confidence", "GRU-representation", "XGBoost-s"]

checks = [
    ("45 units, 19 classes", R["feasibility"]["n_units"] == 45 and R["feasibility"]["n_unique"] == 19, "45 units over 19 unique classes"),
    ("rolling per dataset 15", all(x["n_units"] == 15 for x in R["rolling"]["by_dataset"].values()), "15 per dataset"),
    ("GRU same leader 7/7", all(R["leaders"][e]["gru"].replace("GRU-", "") == R["leaders"][e]["ahsd"] for e in EV), "in all seven evaluations"),
    ("nnD3 GRU repr / AHSD repr / benign", tuple(f3(R["evals"]["nn_D3"]["families"][f]["mean"]) for f in ("GRU-representation", "representation", "benign-only")) == ("0.869", "0.522", "0.898"), "GRU representation 0.869 against AHSD's 0.522"),
    ("winners", [max(x, key=x.get) for x in win.values()] == ["representation", "XGBoost", "benign-only"], "representation scores win under LOACO, XGBoost under natural novelty"),
    ("benign rolling CI", True, ci(RP["families"]["benign-only"])),
    ("repr rolling CI", True, ci(RP["families"]["representation"])),
    ("benign-repr contrast", True, sci(RP["contrasts"]["benign-only − representation"])),
    ("XGB - leader", True, sci(RP["XGBoost_minus_leader_ahsd"])),
    ("XGB - leader mean", f"{RP['XGBoost_minus_leader_ahsd']['mean']:.3f}" == "-0.076", "$-0.076$"),
    ("XGB D3 f045", f3(R["evals"]["roll_D3_f045"]["families"]["XGBoost"]["mean"]) == "0.263", "0.263"),
    ("leaders by ds/cut", {k: x["leader_ahsd"] for k, x in R["rolling"]["by_dataset"].items()} == {"D1": "representation", "D3": "benign-only", "D5": "representation"}
     and [x["leader_ahsd"] for x in R["rolling"]["by_cut"].values()] == ["representation", "benign-only", "CLAD", "benign-only"], "representation, benign-only, CLAD, benign-only"),
    ("GRU seed rows", True, "GRU Mahal. & Repr. & " + row("GRU_Mahalanobis")),
    ("GRU kNN row", True, "GRU $k$NN & Repr. & " + row("GRU_kNN")),
    ("GRU MSP row", True, "GRU MSP & Conf. & " + row("GRU_MSP")),
    ("GRU P row", True, "GRU $P(\\text{attack})$ & Conf. & " + row("GRU_P_attack")),
    ("XGB-s row", True, "XGBoost-s & Ref. & " + row("XGBoost_s")),
    ("AHSD kNN row", True, "$k$NN & Repr. & " + row("kNN")),
    ("GRU ratios text", f"{ps(nn, 'GRU_kNN') / ps(lo, 'GRU_kNN'):.1f}" == "2.6" and f"{ps(nn, 'GRU_Mahalanobis') / ps(lo, 'GRU_Mahalanobis'):.1f}" == "2.6"
     and sorted(f"{ps(nn, d) / ps(lo, d):.1f}" for d in ("GRU_MSP", "GRU_Energy", "GRU_P_attack")) == ["1.0", "1.2", "1.3"], "confidence scores 1.0--1.3$\\times$"),
    ("XGB-s 0.014", f3(ps(nn, "XGBoost_s")) == "0.014", "only to 0.014"),
    ("GRU MSP D2 / nn D3", (f3(dm("loaco_D2", "GRU_MSP")), f3(dm("loaco_D2", "GRU_P_attack")), f3(dm("nn_D3", "GRU_MSP")), f3(dm("nn_D3", "GRU_P_attack"))) == ("0.162", "0.999", "0.245", "0.918"), "MSP 0.162 vs $P(\\text{attack})$ 0.999"),
    ("GRU in-dist MSP", f"{min(AI[e]['GRU_MSP']['auc'] for e in g3):.3f}--{max(AI[e]['GRU_MSP']['auc'] for e in g3):.3f}" == "0.065--0.340", "0.065--0.340"),
    ("5 of 34", len(viol) == 5 and 2 * len(R["evals"]) == 34 and all(e.startswith("roll_") for e, _ in viol)
     and f"{min(dm(e, b + 'P_attack') for e, b in viol):.3f}--{max(dm(e, b + 'P_attack') for e, b in viol):.3f}" == "0.345--0.548", "(0.345--0.548)"),
    ("GRU temporal D3", (f3(au("indist_D3_grouped_random", "GRU_P_attack")), f3(au("indist_D3_temporal_gap", "GRU_P_attack"))) == ("0.981", "0.857"), "0.981$\\rightarrow$0.857"),
    ("TPR LOACO D2 87-99", f"{round(100 * ops(R['evals']['loaco_D2'], 'representation'))}--{round(100 * ops(R['evals']['loaco_D2'], 'GRU-representation'))}" == "87--99", "87--99\\%"),
    ("TPR nn D3 12/38", (round(100 * ops(R["evals"]["nn_D3"], "benign-only")), round(100 * ops(R["evals"]["nn_D3"], "XGBoost"))) == (12, 38), "catch 12\\%, and XGBoost 38\\%"),
    ("rolling max 18%", round(100 * max(ops(RP, f) for f in FAM7)) == 18, "no family exceeds 18\\%"),
    ("rolling benign 0.09 @0.75", (f"{ops(RP, 'benign-only'):.2f}", f"{RP['families']['benign-only']['mean']:.2f}") == ("0.09", "0.75"), "benign-only 0.09 at AUC 0.75"),
    ("pooled AUC range", f"{min(RP['families'][f]['mean'] for f in FAM7):.2f}--{max(RP['families'][f]['mean'] for f in FAM7):.2f}" == "0.53--0.75", "AUCs of 0.53--0.75"),
    ("184 of 333, median 0.15", (ill["n_trap"], ill["n_auc_ge_075"], round(ill["median_tpr_auc_ge_075"], 2)) == (184, 333, 0.15), "of the 333 detector--class points"),
    ("CLAD purged D5", f3(R["evals"]["loaco_D5pb"]["families"]["CLAD"]["mean"]) == "0.972", "0.972 on purged D5"),
    ("210 runs 10.8 h 185 s", (len(w), round(sum(w) / 3600, 1), round(float(np.mean(w)))) == (210, 10.8, 185), "took 10.8\\,h (185\\,s per run"),
    ("single commit", S["code_commit"].startswith("94fbf32"), "All 210 final runs"),
    ("A6 reproduced", all(abs(CS[e]["families"][f]["mean"] - json.loads((ROOT / "results/final/summary.json").read_text())["class_evals"][e]["families"][f]["mean"]) < 1e-12
                          for e in CS for f in CS[e]["families"]), "whose results they reproduce exactly"),
]
ok = True
for name, cond, q in checks:
    found = q in tex
    print(("OK  " if cond and found else "FAIL"), name, "" if found else f"   [not in text: {q}]")
    ok &= bool(cond and found)
sys.exit(0 if ok else 1)
