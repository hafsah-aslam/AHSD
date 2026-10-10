#!/usr/bin/env python
"""Check that numbers quoted in main.tex prose match the results JSON (fails loudly)."""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "results/final/summary.json").read_text()); CS, I, T = S["class_evals"], S["indist"], S["transfer"]
tex = (Path(__file__).parent / "main.tex").read_text()
lo, nn = ["loaco_D1fam", "loaco_D2", "loaco_D3"], ["nn_D1", "nn_D3", "nn_D5"]
pooled = lambda ev, d: np.mean([CS[e]["cells"][d][c]["std"] for e in ev for c in CS[e]["classes"]])
sup = ["MSP", "Energy", "P_attack", "AHSD_stress", "Mahalanobis", "kNN"]
ratios = [pooled(nn, d) / pooled(lo, d) for d in sup]
lead = {e: max(["confidence", "representation", "benign-only", "CLAD"], key=lambda f: CS[e]["families"][f]["mean"]) for e in CS}
xgb_ge = sum(CS[e]["families"]["XGBoost"]["mean"] >= CS[e]["families"][lead[e]]["mean"] for e in CS)
base_ge = sum(max(CS[e]["families"]["benign-only"]["mean"], CS[e]["families"]["XGBoost"]["mean"]) >= max(CS[e]["families"][f]["mean"] for f in ("confidence", "representation", "CLAD")) for e in CS)
au = lambda ds, sch, d: I[f"indist_{ds}_{sch}"]["best"][d]["test"]["auc"]
drop = {ds: (au(ds, "grouped_random", "P_attack")["mean"] - au(ds, "temporal_gap", "P_attack")["mean"],
             au(ds, "grouped_random", "XGBoost")["mean"] - au(ds, "temporal_gap", "XGBoost")["mean"]) for ds in ("D1", "D2", "D3", "D5")}
off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
ck = {(e, f): CS[e]["checkpoint"][f]["final"] - CS[e]["checkpoint"][f]["best"] for e in CS for f in ("confidence", "representation", "CLAD")}
w = [json.loads(p.read_text())["wall_seconds"] for p in (ROOT / "results/final").glob("*/*/seed*.json")]
checks = {
    "ratio min 1.7": round(min(ratios), 1) == 1.7, "ratio max 17": round(max(ratios)) == 17,
    "benign seeded max <0.018": max(pooled(nn, d) for d in ("IF", "AE")) < 0.018,
    "supervised NN min 0.074": round(min(pooled(nn, d) for d in sup), 3) == 0.074,
    "repr leads 5, benign 2": sorted(lead.values()).count("representation") == 5 and list(lead.values()).count("benign-only") == 2,
    "XGB >= lead in 4 of 7": xgb_ge == 4, "benign or XGB >= in 5 of 7": base_ge == 5,
    "D1 drops 0.29/0.27": (round(drop["D1"][0], 2), round(drop["D1"][1], 2)) == (0.29, 0.27),
    "D2 change <= 0.014": max(abs(x) for x in drop["D2"]) <= 0.0145, "D5 both rise": all(x < 0 for x in drop["D5"]),
    "D3 0.664/0.202/0.978": (round(au("D3", "temporal_gap", "P_attack")["mean"], 3), round(au("D3", "temporal_gap", "P_attack")["std"], 3),
                              round(au("D3", "temporal_gap", "XGBoost")["mean"], 3)) == (0.664, 0.202, 0.978),
    "transfer 108, 67/12/21, median 0.500": (len(off), round(100 * np.mean((off >= .3) & (off <= .6))), round(100 * np.mean(off < .3)),
                                              round(100 * np.mean(off > .6)), round(float(np.median(off)), 3)) == (108, 67, 12, 21, 0.5),
    "top transfer CLAD 0.876 XGB 0.832": (round(T["Wilkie_CLAD"]["D1"]["D3"]["mean"], 3), round(T["XGBoost"]["D1"]["D3"]["mean"], 3)) == (0.876, 0.832),
    "ckpt repr/CLAD max 0.073": round(max(abs(v) for (e, f), v in ck.items() if f != "confidence"), 3) == 0.073,
    "ckpt conf max 0.205": round(max(abs(v) for (e, f), v in ck.items() if f == "confidence"), 3) == 0.205,
    "D2 LOACO ben-repr gap 0.17": round(CS["loaco_D2"]["families"]["representation"]["mean"] - CS["loaco_D2"]["families"]["benign-only"]["mean"], 2) == 0.17,
    "160 runs 4.27 h 96 s": (len(w), round(sum(w) / 3600, 2), round(sum(w) / len(w))) == (160, 4.27, 96),
    "contribution checks hold": all(S["contribution_checks"].values()),
}
for k, v in checks.items():
    print(("OK  " if v else "FAIL"), k)
for q in ["1.7", "17$\\times$", "0.018", "0.074", "four of seven", "five of seven", "0.27--0.29", "0.014", "0.664", "0.202", "0.978", "0.876",
          "0.832", "0.073", "0.205", "4.27", "96\\,s", "+0.376", "0.522 [0.402, 0.647]", "0.623 [0.562, 0.692]", "0.372", "0.997--1.000"]:
    if q not in tex:
        print("NOT IN TEXT:", q)
sys.exit(0 if all(checks.values()) else 1)
