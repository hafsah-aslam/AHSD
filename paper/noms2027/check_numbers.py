#!/usr/bin/env python
"""Check that numbers quoted in main.tex prose match the results JSON (fails loudly).

Each check recomputes a value from results/final_v2/summary.json, formats it as the text does,
and requires that exact string to appear in main.tex.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "results/final_v2/summary.json").read_text())
CS, I, T, R, AI = S["class_evals"], S["indist"], S["transfer"], S["a7"], S["a7_indist"]
tex = (Path(__file__).parent / "main.tex").read_text()
EV = ["loaco_D1fam", "loaco_D2", "loaco_D3", "loaco_D5pb", "nn_D1", "nn_D3", "nn_D5"]
RP = R["rolling"]["pooled"]


def f3(x):
    return f"{x:.3f}"


def ci(v, sign=False):
    f = "+.3f" if sign else ".3f"
    s = f"{v['mean']:{f}} [{v['ci95'][0]:{f}}, {v['ci95'][1]:{f}}]"
    return s


def sci(v):  # signed interval as typeset: $+0.125$ [$+0.048$, $+0.204$]
    g = lambda x: f"${x:+.3f}$".replace("$-", "$-")  # noqa: E731
    return f"{g(v['mean'])} [{g(v['ci95'][0])}, {g(v['ci95'][1])}]"


def dm(e, d):
    src = RP if e == "rolling" else R["evals"][e]
    return float(np.mean(list(src["cells"][d].values())))


au = lambda e, d: I[e]["best"][d]["test"]["auc"]["mean"]  # noqa: E731
ss = R["seed_std_ratio"]
viol = [(e, b) for e in R["evals"] for b in ("", "GRU_") if dm(e, b + "MSP") > dm(e, b + "P_attack")]
viol_p = [dm(e, b + "P_attack") for e, b in viol]
g3 = [f"indist_{d}_grouped_random" for d in ("D1", "D2", "D3")]
off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
w = [json.loads(p.read_text())["wall_seconds"] for p in (ROOT / "results/final_v2").glob("*/*/seed*.json")]
drop = {ds: (au(f"indist_{ds}_grouped_random", "P_attack") - au(f"indist_{ds}_temporal_gap", "P_attack"),
             au(f"indist_{ds}_grouped_random", "XGBoost") - au(f"indist_{ds}_temporal_gap", "XGBoost")) for ds in ("D1", "D2", "D5")}
repr_ratio = [ss[d]["nn_ratio"] for d in ("AHSD_stress", "Mahalanobis", "kNN")]
gru_repr_ratio = [ss[d]["nn_ratio"] for d in ("GRU_Mahalanobis", "GRU_kNN")]
conf_ratio = [ss[d]["nn_ratio"] for d in ("MSP", "Energy", "P_attack")]
gconf_ratio = [ss[d]["nn_ratio"] for d in ("GRU_MSP", "GRU_Energy", "GRU_P_attack")]
ops = lambda st, f: st["ops"][f]["tpr_at_1pct_fpr"]["mean"]  # noqa: E731
FAM7 = ["confidence", "representation", "benign-only", "CLAD", "XGBoost", "GRU-confidence", "GRU-representation", "XGBoost-s"]
leaders = [R["evals"][e]["leader_ahsd"] for e in EV]

quotes = {
    # abstract / intro
    "45 units, 19 classes": (R["feasibility"]["n_units"] == 45 and R["feasibility"]["n_unique"] == 19, "45 zero-day units over 19 attack classes"),
    "MSP 0.163 / P 0.999 (LOACO D2)": ((f3(dm("loaco_D2", "MSP")), f3(dm("loaco_D2", "P_attack"))) == ("0.163", "0.999"), "MSP 0.163 vs attack probability 0.999"),
    "GRU MSP LOACO D2 0.162": (f3(dm("loaco_D2", "GRU_MSP")) == "0.162" and f3(dm("loaco_D2", "GRU_P_attack")) == "0.999", "0.162 GRU"),
    "XGB >= leader in 4 of 7": (len(R["xgb_ge_leader"]["XGBoost"]) == 4, "four of seven"),
    "repr ratio 2.7--17.7 (both backbones)": (f"{min(gru_repr_ratio):.1f}--{max(repr_ratio):.1f}" == "2.7--17.7", "2.7--17.7-fold"),
    "repr ratio AHSD 6.1--17.7": (f"{min(repr_ratio):.1f}--{max(repr_ratio):.1f}" == "6.1--17.7", "6.1--17.7$\\times$"),
    "repr ratio GRU 2.7--2.8": (f"{min(gru_repr_ratio):.1f}--{max(gru_repr_ratio):.1f}" == "2.7--2.8", "2.7--2.8$\\times$"),
    "conf ratio 1.4--2.1 / 1.1--1.2": (f"{min(conf_ratio):.1f}--{max(conf_ratio):.1f}" == "1.4--2.1" and f"{min(gconf_ratio):.1f}--{max(gconf_ratio):.1f}" == "1.1--1.2", "(1.4--2.1$\\times$ and 1.1--1.2$\\times$)"),
    "pooled AUC range 0.53--0.75": (f"{min(RP['families'][f]['mean'] for f in FAM7):.2f}--{max(RP['families'][f]['mean'] for f in FAM7):.2f}" == "0.53--0.75", "0.53--0.75"),
    "max TPR@1% rolling 18%": (round(100 * max(ops(RP, f) for f in FAM7)) == 18 and max(FAM7, key=lambda f: ops(RP, f)) == "XGBoost", "18\\%"),
    "benign leads rolling 0.748": (RP["leader_ahsd"] == "benign-only" and ci(RP["families"]["benign-only"]) == "0.748 [0.699, 0.796]", "0.748 [0.699, 0.796]"),
    "repr rolling 0.623": (ci(RP["families"]["representation"]) == "0.623 [0.566, 0.683]", "0.623 [0.566, 0.683]"),
    "benign-repr rolling": (True, sci(RP["contrasts"]["benign-only − representation"])),
    "benign-GRUrepr rolling": (True, sci(RP["contrasts"]["benign-only − GRU-representation"])),
    "XGB - leader rolling": (True, sci(RP["XGBoost_minus_leader_ahsd"])),
    "CLAD - benign rolling": (True, sci(RP["contrasts"]["CLAD − benign-only"])),
    "XGB rolling short -0.076": (f"{RP['XGBoost_minus_leader_ahsd']['mean']:.3f}" == "-0.076", "$-0.076$"),
    # C1 details
    "nn_D3 MSP 0.262/0.245 vs 0.853/0.918": (tuple(f3(dm("nn_D3", d)) for d in ("MSP", "GRU_MSP", "P_attack", "GRU_P_attack")) == ("0.262", "0.245", "0.853", "0.918"), "0.262 and 0.245 against 0.853 and 0.918"),
    "LOACO D2/D3 MSP < 0.5 every class, both backbones": (all(R["evals"][e]["frac_units_below_0.5"][d] == 1.0 for e in ("loaco_D2", "loaco_D3") for d in ("MSP", "GRU_MSP")), "every class"),
    "in-dist MSP AHSD 0.075--0.394": (f"{min(AI[e]['MSP']['auc'] for e in g3):.3f}--{max(AI[e]['MSP']['auc'] for e in g3):.3f}" == "0.075--0.394", "0.075 to 0.394"),
    "in-dist MSP GRU 0.065--0.340": (f"{min(AI[e]['GRU_MSP']['auc'] for e in g3):.3f}--{max(AI[e]['GRU_MSP']['auc'] for e in g3):.3f}" == "0.065--0.340", "0.065 to 0.340"),
    "in-dist P 0.812--0.999 / 0.839--1.000": (f"{min(AI[e]['P_attack']['auc'] for e in g3):.3f}--{max(AI[e]['P_attack']['auc'] for e in g3):.3f}" == "0.812--0.999"
                                              and f"{min(AI[e]['GRU_P_attack']['auc'] for e in g3):.3f}--{max(AI[e]['GRU_P_attack']['auc'] for e in g3):.3f}" == "0.839--1.000", "0.812--0.999 and 0.839--1.000"),
    "MSP > P in 5 of 34, P 0.345--0.548": (len(viol) == 5 and 2 * len(R["evals"]) == 34 and f"{min(viol_p):.3f}--{max(viol_p):.3f}" == "0.345--0.548"
                                           and all(e.startswith("roll_") for e, _ in viol), "5 of 34"),
    "GRU purged D5 MSP 0.889": (f3(dm("loaco_D5pb", "GRU_MSP")) == "0.889", "(0.889)"),
    # C2
    "leaders repr 5 benign 2; GRU same 7/7": (leaders.count("representation") == 5 and leaders.count("benign-only") == 2
                                              and all(R["leaders"][e]["gru"].replace("GRU-", "") == R["leaders"][e]["ahsd"] for e in EV), "same leader in all seven"),
    "rolling ds leaders": ({k: v["leader_ahsd"] for k, v in R["rolling"]["by_dataset"].items()} == {"D1": "representation", "D3": "benign-only", "D5": "representation"}, "representation on D1 and D5 but benign-only on D3"),
    "cut leaders repr/benign/CLAD": (set(v["leader_ahsd"] for v in R["rolling"]["by_cut"].values()) == {"representation", "benign-only", "CLAD"}, "representation, benign-only and CLAD"),
    "XGB LOACO D2 0.998, NN D3 0.923": ((f3(R["evals"]["loaco_D2"]["families"]["XGBoost"]["mean"]), f3(R["evals"]["nn_D3"]["families"]["XGBoost"]["mean"])) == ("0.998", "0.923"), "LOACO D2 0.998, natural novelty D3 0.923"),
    "XGB D3 f045 0.263": (f3(R["evals"]["roll_D3_f045"]["families"]["XGBoost"]["mean"]) == "0.263", "0.263"),
    "CLAD never leads fixed": ("CLAD" not in leaders, "CLAD never leads"),
    "CLAD purged D5 0.972": (f3(R["evals"]["loaco_D5pb"]["families"]["CLAD"]["mean"]) == "0.972", "0.972"),
    # C3
    "kNN rolling D3/D1/D5 0.185/0.036/0.032, GRU 0.154": (tuple(f3(ss["kNN"]["rolling_by_dataset"][d]) for d in ("D3", "D1", "D5")) == ("0.185", "0.036", "0.032")
                                                          and f3(ss["GRU_kNN"]["rolling_by_dataset"]["D3"]) == "0.154", "0.185 on D3 against 0.036 on D1 and 0.032"),
    "IF 0.008, AE 0.017, XGB-s 0.014 (NN)": (tuple(f3(ss[d]["nn"]) for d in ("IF", "AE", "XGBoost_s")) == ("0.008", "0.017", "0.014"), "Isolation Forest 0.008, autoencoder 0.017"),
    # C4
    "LOACO D2 best TPR 87--99%": (f"{round(100 * ops(R['evals']['loaco_D2'], 'representation'))}--{round(100 * ops(R['evals']['loaco_D2'], 'GRU-representation'))}" == "87--99", "87--99\\%"),
    "nn D3 benign AUC 0.90 TPR 12%, XGB 38%": ((f"{R['evals']['nn_D3']['families']['benign-only']['mean']:.2f}", round(100 * ops(R["evals"]["nn_D3"], "benign-only")),
                                               round(100 * ops(R["evals"]["nn_D3"], "XGBoost"))) == ("0.90", 12, 38), "only 12\\% at 1\\% FPR, and XGBoost 38\\%"),
    "rolling benign 0.09 at AUC 0.75, PR-AUC <= 0.41": ((f"{ops(RP, 'benign-only'):.2f}", f"{RP['families']['benign-only']['mean']:.2f}",
                                                        f"{max(RP['ops'][f]['pr_auc']['mean'] for f in FAM7):.2f}") == ("0.09", "0.75", "0.41"), "benign-only 0.09 at AUC 0.75"),
    # ablations
    "nn D3 repr 0.522 / GRU 0.869 / benign 0.898": (tuple(f3(R["evals"]["nn_D3"]["families"][f]["mean"]) for f in ("representation", "GRU-representation", "benign-only")) == ("0.522", "0.869", "0.898"), "(0.522) while the GRU's reach 0.869"),
    "ckpt 0.073 / 0.205": ((round(max(abs(R["evals"][e]["checkpoint"][f]["mean"]) for e in EV for f in ("representation", "CLAD")), 3),
                            round(max(abs(R["evals"][e]["checkpoint"]["confidence"]["mean"]) for e in EV), 3)) == (0.073, 0.205), "at most 0.073"),
    "temporal D3 0.975->0.664, GRU 0.981->0.857": ((f3(au("indist_D3_grouped_random", "P_attack")), f3(au("indist_D3_temporal_gap", "P_attack")),
                                                    f3(au("indist_D3_grouped_random", "GRU_P_attack")), f3(au("indist_D3_temporal_gap", "GRU_P_attack")))
                                                   == ("0.975", "0.664", "0.981", "0.857"), "(0.975$\\rightarrow$0.664) much more than the GRU (0.981$\\rightarrow$0.857)"),
    # prior comparison prose
    "CLAD rolling D5 0.702; XGB D3 0.862 -> 0.699": ((f3(R["rolling"]["by_dataset"]["D5"]["families"]["CLAD"]["mean"]), f3(R["evals"]["loaco_D3"]["families"]["XGBoost"]["mean"]),
                                                      f3(R["rolling"]["by_dataset"]["D3"]["families"]["XGBoost"]["mean"])) == ("0.702", "0.862", "0.699"), "0.862 (LOACO) to 0.699 (rolling)"),
    "LOACO D3 MSP/energy 0.26--0.27": (f"{dm('loaco_D3', 'Energy'):.2f}--{dm('loaco_D3', 'MSP'):.2f}" == "0.26--0.27", "(0.26--0.27)"),
    # confirmatory
    "D1 drops 0.27--0.29; D2 <= 0.014; D5 rises": (sorted(round(x, 2) for x in drop["D1"]) == [0.27, 0.29] and max(abs(x) for x in drop["D2"]) <= 0.0145
                                                   and all(x < 0 for x in drop["D5"]), "0.27--0.29 AUC"),
    "transfer 108, median 0.500, 67%": ((len(off), round(float(np.median(off)), 3), round(100 * np.mean((off >= .3) & (off <= .6)))) == (108, 0.5, 67), "is 0.500, with 67\\%"),
    "transfer top CLAD 0.876 XGB 0.832": ((f3(T["Wilkie_CLAD"]["D1"]["D3"]["mean"]), f3(T["XGBoost"]["D1"]["D3"]["mean"])) == ("0.876", "0.832"), "CLAD 0.876, XGBoost 0.832"),
    "D2 LOACO repr 0.977; benign gap 0.17": ((f3(R["evals"]["loaco_D2"]["families"]["representation"]["mean"]),
                                             round(R["evals"]["loaco_D2"]["families"]["representation"]["mean"] - R["evals"]["loaco_D2"]["families"]["benign-only"]["mean"], 2)) == ("0.977", 0.17), "trail by 0.17"),
    # setup
    "210 runs, 10.78 h, 185 s": ((len(w), round(sum(w) / 3600, 2), round(float(np.mean(w)))) == (210, 10.78, 185), "took 10.78\\,h (185\\,s per run)"),
    "single commit": (S["code_commit"].startswith("94fbf32"), "210 final runs"),
}
ok = True
for k, (cond, q) in quotes.items():
    found = q in tex
    print(("OK  " if cond and found else "FAIL"), k, "" if found else f"  [not in text: {q}]")
    ok &= bool(cond and found)
sys.exit(0 if ok else 1)
