#!/usr/bin/env python
"""P1d report and gate G3 (AMENDMENT_04 A4.6 as amended by AMENDMENT_05), from results JSON only.

AMENDMENT_05: D5 LOACO uses purged_block (data/processed/D5/purged_block/meta.json); if the A5.2
decision (results/p1d/a52_decision.json) applies the fallback, G3 (b) is evaluated on D3 LOACO only
and D5 contributes (a) only. A5.3: D5_nn_final (final-epoch checkpoint) is a reported sensitivity
(primary B minus final-epoch S); it never enters the gate.

Inputs: results/p1d/** (code commit A), results/p1c/D3 (P1c natural_novelty, no re-anchoring;
code commit 702fc4d) and results/p1d/D3_nn_wilkie. Writes P1D_REPORT.md, GATE_G3_DECISION.md,
results/p1d/summary.json. Fails closed on mixed code commits within an experiment or a dirty tree.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import metrics  # noqa: E402
from nids.provenance import ROOT, provenance, rq_sha256, write_json  # noqa: E402

SUP = ["AHSD_stress", "MSP", "Energy", "Mahalanobis", "kNN"]
BEN = ["IF", "OCSVM", "PCA", "AE"]
W = "Wilkie_CLAD"
SEEDS = [17, 23, 42, 101, 202]
N_BOOT, BOOT_SEED = 2000, 0
A_MIN, B_MAX = 0.15, 0.05


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _load(pattern):
    rs = [json.loads(p.read_text()) for p in sorted(ROOT.glob(pattern))]
    return [r for r in rs if "seed" in r and not r.get("smoke")]


def _one_commit(rs, name):
    cs = {r["provenance"]["code_commit"] for r in rs}
    if len(cs) != 1 or any(r["provenance"]["code_dirty"] for r in rs):
        raise SystemExit(f"{name}: {sorted(cs)} or dirty tree")
    return cs.pop()


def tensor(rows: list[tuple[str, str, int, float]]):
    """rows (class, detector, seed, auc) -> {class: {detector: {seed: auc}}}"""
    A = defaultdict(lambda: defaultdict(dict))
    for c, d, s, a in rows:
        A[c][d][s] = a
    return A


def check_complete(A, dets, name):
    for c, by in A.items():
        for d in dets:
            got = sorted(by.get(d, {}))
            if got != SEEDS:
                raise SystemExit(f"{name}: {c}/{d} seeds {got} != {SEEDS}")


def diff_stats(A, A_sup=None):
    """B from A; S from A_sup (default A). A_sup = final-epoch tensor gives the A5.3 sensitivity."""
    A_sup = A if A_sup is None else A_sup
    classes = sorted(A)
    if sorted(A_sup) != classes:
        raise SystemExit(f"class sets differ: {classes} vs {sorted(A_sup)}")
    Mb = np.array([[[A[c][d][s] for s in SEEDS] for d in BEN] for c in classes])        # (C, 4, S)
    Ms = np.array([[[A_sup[c][d][s] for s in SEEDS] for d in SUP] for c in classes])    # (C, 5, S)
    per_cs = Mb.mean(1) - Ms.mean(1)                                                    # (C, S)
    point = float(per_cs.mean())
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        ci = rng.integers(0, len(classes), len(classes))
        si = rng.integers(0, len(SEEDS), len(SEEDS))
        boots.append(per_cs[np.ix_(ci, si)].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    Bc = {c: float(np.mean([np.mean([A[c][d][s] for s in SEEDS]) for d in BEN])) for c in classes}
    Sc = {c: float(np.mean([np.mean([A_sup[c][d][s] for s in SEEDS]) for d in SUP])) for c in classes}
    out = {"classes": classes, "B": float(np.mean(list(Bc.values()))), "S": float(np.mean(list(Sc.values()))),
           "B_minus_S": point, "ci95": [float(lo), float(hi)], "B_by_class": Bc, "S_by_class": Sc}
    if all(W in A_sup[c] for c in classes):
        Wc = {c: float(np.mean([A_sup[c][W][s] for s in SEEDS])) for c in classes}
        out["W"] = float(np.mean(list(Wc.values())))
        out["W_by_class"] = Wc
        out["W_minus_B"], out["W_minus_S"] = out["W"] - out["B"], out["W"] - out["S"]
    return out


def per_detector_table(A, dets):
    return {c: {d: {**metrics.mean_ci([A[c][d][s] for s in SEEDS]),
                    "frac_below_0.5": float(np.mean([A[c][d][s] < 0.5 for s in SEEDS]))}
                for d in dets if d in A[c]} for c in sorted(A)}


def main():
    d5nn = _load("results/p1d/D5_nn/*/seed*.json")
    d5nf = _load("results/p1d/D5_nn_final/*/seed*.json")
    d5lo = _load("results/p1d/D5_loaco/*/seed*.json")
    d3lo = _load("results/p1d/D3_loaco/*/seed*.json")
    d3w = _load("results/p1d/D3_nn_wilkie/*/seed*.json")
    runs = d5nn + d5nf + d5lo + d3lo + d3w
    p1d_commit = _one_commit(runs, "P1d")
    if any(r.get("checkpoint", "best") != "best" for r in d5nn + d5lo + d3lo + d3w) or \
            any(r.get("checkpoint") != "final" for r in d5nf):
        raise SystemExit("checkpoint labels inconsistent with A5.3")
    dec = json.loads((ROOT / "results/p1d/a52_decision.json").read_text())
    pmeta = json.loads((ROOT / "data/processed/D5/purged_block/meta.json").read_text())
    if sorted(dec["feasible_folds"]) != sorted(pmeta["feasible_folds"]) or \
            dec["fallback_applied"] != (len(pmeta["feasible_folds"]) < dec["threshold"]):
        raise SystemExit("A5.2 decision inconsistent with purged_block meta")
    fallback = dec["fallback_applied"]
    if fallback and d5lo:
        raise SystemExit("A5.2 fallback applied but D5_loaco runs exist")
    if fallback and not (ROOT / "results/p1d/D5_loaco/skipped.json").exists():
        raise SystemExit("A5.2 fallback applied but no D5_loaco skip record")
    if not fallback and sorted({r["unit"] for r in d5lo}) != sorted(dec["feasible_folds"]):
        raise SystemExit("D5_loaco folds run differ from the A5.2 feasible folds")
    p1c = [r for r in _load("results/p1c/D3/seed*.json")]
    p1c_commit = _one_commit(p1c, "P1c D3")

    def rows_nn(rs):
        return [(c, d, r["seed"], a) for r in rs for d, by in r["auc"].items() for c, a in by.items()]

    def rows_lo(rs):
        return [(r["unit"], d, r["seed"], by[r["unit"]]) for r in rs for d, by in r["auc"].items()]

    T = {"D5_nn": tensor(rows_nn(d5nn)), "D3_loaco": tensor(rows_lo(d3lo))}
    if not fallback:
        T["D5_loaco"] = tensor(rows_lo(d5lo))
    d3nn = [(c, k.split("/", 1)[1], r["seed"], a) for r in p1c for k, by in r["auc"].items() if k.startswith("none/")
            for c, a in by.items()]
    d3nn += rows_nn(d3w)
    T["D3_nn"] = tensor(d3nn)
    for k, A in T.items():
        check_complete(A, SUP + BEN + [W], k)
    T_final = tensor(rows_nn(d5nf))
    check_complete(T_final, SUP + [W], "D5_nn_final")
    G = {k: diff_stats(A) for k, A in T.items()}
    sens = diff_stats(T["D5_nn"], T_final)
    b_eval = "D3_loaco" if fallback else "D5_loaco"
    crit = {
        "a": G["D5_nn"]["B_minus_S"] >= A_MIN and G["D5_nn"]["ci95"][0] > 0,
        "b": G[b_eval]["B_minus_S"] <= B_MAX,
        "b_evaluated_on": b_eval,
        "c_nn": G["D3_nn"]["B_minus_S"] >= A_MIN and G["D3_nn"]["ci95"][0] > 0,
        "c_loaco": G["D3_loaco"]["B_minus_S"] <= B_MAX,
    }
    crit["c"] = crit["c_nn"] and crit["c_loaco"]
    pb = {"days": pmeta["days"], "blocks": pmeta["blocks"], "split_windows": pmeta["split_windows"],
          "fold_info": {c: {k: r.get(k) for k in ("eligible", "feasible", "train_windows_majority",
                                                  "test_windows_majority", "reason")} for c, r in pmeta["loaco"].items()},
          "feasible_folds": pmeta["feasible_folds"], "checks": [sum(x["ok"] for x in pmeta["verification"]), len(pmeta["verification"])],
          "code_commit": pmeta["provenance"]["code_commit"]}
    outcome = "GO" if crit["a"] and crit["b"] and crit["c"] else "STOP"
    tables = {k: per_detector_table(A, SUP + BEN + [W]) for k, A in T.items()}
    feas = json.loads((ROOT / "results/p1d/d5_feasibility.json").read_text())
    tables["D5_nn_final (A5.3 sensitivity)"] = per_detector_table(T_final, SUP + [W])
    S = {"gate": {"stats": G, "criteria": crit, "outcome": outcome}, "tables": tables,
         "a52": {**{k: dec[k] for k in ("feasible_folds", "n_feasible", "threshold", "fallback_applied", "rule")},
                 "code_commit": dec["provenance"]["code_commit"]},
         "a53_sensitivity_D5_nn": sens, "purged_block": pb,
         "code_commits": {"P1d": p1d_commit, "P1c (D3 natural_novelty, 9 detectors)": p1c_commit},
         "rq_sha256_in_runs": {"P1d": sorted({r["provenance"]["rq_sha256"] for r in runs}),
                               "P1c": sorted({r["provenance"]["rq_sha256"] for r in p1c})},
         "rq_sha256": rq_sha256(), "d5_feasibility": {k: feas[k] for k in ("evaluable_classes", "test_windows", "train_tail")},
         "n_runs": {"D5_nn": len(d5nn), "D5_nn_final": len(d5nf), "D5_loaco": len(d5lo), "D3_loaco": len(d3lo), "D3_nn_wilkie": len(d3w)},
         "provenance": provenance()}
    write_json(ROOT / "results/p1d/summary.json", S)
    (ROOT / "P1D_REPORT.md").write_text(render(S))
    (ROOT / "GATE_G3_DECISION.md").write_text(render_gate(S))
    print(f"G3: {outcome} {crit}")


def render_gate(S):
    G, c = S["gate"]["stats"], S["gate"]["criteria"]
    L = ["# GATE_G3_DECISION", "", f"**Outcome: {S['gate']['outcome']}**", "",
         "Rule (AMENDMENT_04 A4.6, frozen before any P1d processing; LOACO protocol for D5 amended by AMENDMENT_05 "
         "before any feasibility count): B = mean over classes of the mean AUC of the 4 "
         "benign-only detectors; S = same for the 5 supervised-backbone scores. GO only if (a) D5 natural_novelty "
         "B − S ≥ 0.15 with paired-bootstrap 95% CI > 0; (b) D5 LOACO (purged_block) B − S ≤ 0.05 — under the A5.2 "
         "fallback (< 3 feasible D5 folds) on D3 LOACO only; (c) the same (a)/(b) pattern on D3 "
         "(natural_novelty from P1c + new LOACO). Bootstrap: classes and seeds resampled independently, 2000 resamples. "
         "D5 natural_novelty primary checkpoint = best validation macro-F1 (A5.3); the final-epoch sensitivity does not "
         "enter the gate.", "",
         f"A5.2: {S['a52']['n_feasible']} feasible D5 purged_block LOACO folds ({', '.join(S['a52']['feasible_folds']) or 'none'}); "
         + ("**fallback applied** — (b) evaluated on D3 LOACO only; D5 contributes (a) only (protocol limitation)."
            if S["a52"]["fallback_applied"] else "fallback not applied — (b) evaluated on D5 LOACO."), "",
         "| Evaluation | classes | B | S | B − S | 95% CI | test | pass |", "|---|---|---|---|---|---|---|---|"]
    rows = [("D5_nn", "(a) ≥ 0.15 and CI > 0", c["a"])]
    if "D5_loaco" in G:
        rows.append(("D5_loaco", "(b) ≤ 0.05", c["b"]))
    rows += [("D3_nn", "(c) ≥ 0.15 and CI > 0", c["c_nn"]),
             ("D3_loaco", "(c) ≤ 0.05" + (" and (b) under A5.2" if S["a52"]["fallback_applied"] else ""), c["c_loaco"])]
    for k, test, ok in rows:
        g = G[k]
        L.append(f"| {k} | {len(g['classes'])} | {_f(g['B'])} | {_f(g['S'])} | {_f(g['B_minus_S'])} | "
                 f"[{_f(g['ci95'][0])}, {_f(g['ci95'][1])}] | {test} | {'yes' if ok else 'no'} |")
    sn = S["a53_sensitivity_D5_nn"]
    L += ["", f"(a) {'pass' if c['a'] else 'fail'}; (b) {'pass' if c['b'] else 'fail'} (on {c['b_evaluated_on']}); "
          f"(c) {'pass' if c['c'] else 'fail'}.", "",
          f"A5.3 sensitivity (not gating): D5 natural_novelty B (primary) − S (final-epoch checkpoint) = "
          f"{_f(sn['B_minus_S'])} [{_f(sn['ci95'][0])}, {_f(sn['ci95'][1])}].", "",
          "Wilkie et al. CLAD (not gating): " + "; ".join(
              f"{k}: W {_f(G[k].get('W'))}, W − B {_f(G[k].get('W_minus_B'))}, W − S {_f(G[k].get('W_minus_S'))}" for k in G),
          "", f"P1d code commit `{S['code_commits']['P1d']}`; P1c D3 code commit "
          f"`{S['code_commits']['P1c (D3 natural_novelty, 9 detectors)']}`. Generated by `scripts/report_p1d.py`.", ""]
    return "\n".join(L)


def render(S):
    L = ["# P1d report", "",
         "Generated by `scripts/report_p1d.py` from results JSON. Scope: AMENDMENT_04 (confirmatory H-REV). "
         "Seeds 17/23/42/101/202; mean AUC ± Student-t 95% CI (n = 5), fraction of seeds < 0.5 in brackets. "
         "Scores higher = more anomalous, never flipped. Supervised-backbone scores: " + ", ".join(SUP)
         + "; benign-only: " + ", ".join(BEN) + "; Wilkie et al. CLAD reported separately.", "",
         f"**Gate G3: {S['gate']['outcome']}** (see `GATE_G3_DECISION.md`).", "",
         "## D5 natural_novelty feasibility (before any model run)", "",
         f"Evaluable classes: {', '.join(S['d5_feasibility']['evaluable_classes'])}; test windows "
         f"{S['d5_feasibility']['test_windows']}; train-tail validation {S['d5_feasibility']['train_tail']}.", ""]
    pb, a52 = S["purged_block"], S["a52"]
    L += ["## D5 LOACO protocol purged_block (AMENDMENT_05 A5.1) and A5.2 decision", "",
          f"10-minute blocks; {pb['blocks']['non_empty']} non-empty blocks in {len(pb['days'])} capture days; day-stratified "
          f"60/20/20 seeded shuffle; purged {pb['blocks']['purged_rule1']} train/val blocks adjacent to test and "
          f"{pb['blocks']['purged_rule2']} train blocks adjacent to val; after purge "
          + ", ".join(f"{k} {v}" for k, v in pb["blocks"]["after_purge"].items()) + " blocks; windows "
          + ", ".join(f"{k} {v}" for k, v in pb["split_windows"].items()) + f". Data checks passed {pb['checks'][0]}/{pb['checks'][1]} "
          f"(code commit `{pb['code_commit']}`).", "",
          "| Day | blocks | train | val | test | purged (adj. test) | purged (train adj. val) | after purge train/val/test |",
          "|---|---|---|---|---|---|---|---|"]
    for d in pb["days"]:
        ap = d["after_purge"]
        L.append(f"| {d['day']} | {d['blocks']} | {d['train']} | {d['val']} | {d['test']} | {d['purged_rule1_adjacent_test']} | "
                 f"{d['purged_rule2_train_adjacent_val']} | {ap['train']}/{ap['val']}/{ap['test']} |")
    L += ["", "| Class | eligible | train windows (majority) | test windows (majority) | feasible | reason |", "|---|---|---|---|---|---|"]
    for cname, r in pb["fold_info"].items():
        L.append(f"| {cname} | {r['eligible']} | {r['train_windows_majority']} | {r['test_windows_majority']} | {r['feasible']} | {r['reason'] or ''} |")
    L += ["", f"Feasible folds (≥ 200 training-pool and ≥ 50 test windows): {a52['n_feasible']} "
          f"({', '.join(a52['feasible_folds']) or 'none'}). "
          + ("**A5.2 fallback applied** (< 3 folds): D5 LOACO was not run; G3 (b) is evaluated on D3 LOACO only and D5 "
             "contributes (a) only. Protocol limitation: the D5 natural_novelty vs LOACO contrast cannot be measured on D5 itself."
             if a52["fallback_applied"] else "A5.2 fallback not applied: D5 LOACO evaluated on the feasible folds."), ""]
    G = S["gate"]["stats"]
    L += ["## B, S and Wilkie per evaluation", "",
          "| Evaluation | B (benign-only) | S (supervised) | B − S [95% CI] | Wilkie CLAD | W − B | W − S |", "|---|---|---|---|---|---|---|"]
    for k, g in G.items():
        L.append(f"| {k} | {_f(g['B'])} | {_f(g['S'])} | {_f(g['B_minus_S'])} [{_f(g['ci95'][0])}, {_f(g['ci95'][1])}] | "
                 f"{_f(g.get('W'))} | {_f(g.get('W_minus_B'))} | {_f(g.get('W_minus_S'))} |")
    sn = S["a53_sensitivity_D5_nn"]
    L += ["", "## A5.3 checkpoint sensitivity (D5 natural_novelty; reported, not gating)", "",
          "Primary: AHSD backbone at best validation macro-F1 (CLAD at best validation AUROC). Sensitivity: AHSD backbone "
          "and CLAD re-trained with the final-epoch checkpoint; benign-only scores reused from the primary runs.", "",
          "| | B | S | B − S [95% CI] | Wilkie CLAD |", "|---|---|---|---|---|",
          f"| primary | {_f(G['D5_nn']['B'])} | {_f(G['D5_nn']['S'])} | {_f(G['D5_nn']['B_minus_S'])} "
          f"[{_f(G['D5_nn']['ci95'][0])}, {_f(G['D5_nn']['ci95'][1])}] | {_f(G['D5_nn'].get('W'))} |",
          f"| final-epoch S | {_f(sn['B'])} | {_f(sn['S'])} | {_f(sn['B_minus_S'])} [{_f(sn['ci95'][0])}, {_f(sn['ci95'][1])}] | "
          f"{_f(sn.get('W'))} |", ""]
    for k, tab in S["tables"].items():
        dets = [d for d in SUP + BEN + [W] if all(d in by for by in tab.values())]
        L += [f"## {k}: mean AUC per class", "", "| Class | " + " | ".join(dets) + " | B | S |", "|---|" + "---|" * (len(dets) + 2)]
        for c, by in tab.items():
            L.append(f"| {c} | " + " | ".join(f"{_f(by[d]['mean'])} ± {_f(by[d]['ci95'], 2)} ({by[d]['frac_below_0.5']:.1f})"
                                             for d in dets) + (f" | {_f(G[k]['B_by_class'][c])} | {_f(G[k]['S_by_class'][c])} |" if k in G
                                                else f" | — | {_f(sn['S_by_class'][c])} |"))
        L.append("")
    L += ["## Provenance", "",
          f"- P1d runs: code_commit `{S['code_commits']['P1d']}` (single, clean; preflight seeding test passed); runs "
          + ", ".join(f"{k} {v}" for k, v in S["n_runs"].items()) + ".",
          f"- D3 natural_novelty for the 9 non-Wilkie detectors: P1c results, code_commit "
          f"`{S['code_commits']['P1c (D3 natural_novelty, 9 detectors)']}` (no re-anchoring condition).",
          "- RESEARCH_QUESTIONS.md SHA-256 in runs: " + "; ".join(f"{k}: " + ", ".join(f"`{h}`" for h in v)
                                                          for k, v in S["rq_sha256_in_runs"].items()) + ".",
          f"- Report generated at `{S['provenance']['code_commit']}`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
