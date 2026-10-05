#!/usr/bin/env python
"""P1d report and gate G3 (AMENDMENT_04 A4.6), from results JSON only.

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


def diff_stats(A):
    classes = sorted(A)
    M = np.array([[[A[c][d][s] for s in SEEDS] for d in BEN + SUP] for c in classes])  # (C, 9, S)
    per_cs = M[:, :len(BEN), :].mean(1) - M[:, len(BEN):, :].mean(1)                   # (C, S)
    point = float(per_cs.mean())
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(N_BOOT):
        ci = rng.integers(0, len(classes), len(classes))
        si = rng.integers(0, len(SEEDS), len(SEEDS))
        boots.append(per_cs[np.ix_(ci, si)].mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    Bc = {c: float(np.mean([np.mean([A[c][d][s] for s in SEEDS]) for d in BEN])) for c in classes}
    Sc = {c: float(np.mean([np.mean([A[c][d][s] for s in SEEDS]) for d in SUP])) for c in classes}
    out = {"classes": classes, "B": float(np.mean(list(Bc.values()))), "S": float(np.mean(list(Sc.values()))),
           "B_minus_S": point, "ci95": [float(lo), float(hi)], "B_by_class": Bc, "S_by_class": Sc}
    if all(W in A[c] for c in classes):
        Wc = {c: float(np.mean([A[c][W][s] for s in SEEDS])) for c in classes}
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
    d5lo = _load("results/p1d/D5_loaco/*/seed*.json")
    d3lo = _load("results/p1d/D3_loaco/*/seed*.json")
    d3w = _load("results/p1d/D3_nn_wilkie/*/seed*.json")
    p1d_commit = _one_commit(d5nn + d5lo + d3lo + d3w, "P1d")
    p1c = [r for r in _load("results/p1c/D3/seed*.json")]
    p1c_commit = _one_commit(p1c, "P1c D3")

    def rows_nn(rs):
        return [(c, d, r["seed"], a) for r in rs for d, by in r["auc"].items() for c, a in by.items()]

    def rows_lo(rs):
        return [(r["unit"], d, r["seed"], by[r["unit"]]) for r in rs for d, by in r["auc"].items()]

    T = {"D5_nn": tensor(rows_nn(d5nn)), "D5_loaco": tensor(rows_lo(d5lo)), "D3_loaco": tensor(rows_lo(d3lo))}
    d3nn = [(c, k.split("/", 1)[1], r["seed"], a) for r in p1c for k, by in r["auc"].items() if k.startswith("none/")
            for c, a in by.items()]
    d3nn += rows_nn(d3w)
    T["D3_nn"] = tensor(d3nn)
    for k, A in T.items():
        check_complete(A, SUP + BEN + [W], k)
    G = {k: diff_stats(A) for k, A in T.items()}
    crit = {
        "a": G["D5_nn"]["B_minus_S"] >= A_MIN and G["D5_nn"]["ci95"][0] > 0,
        "b": G["D5_loaco"]["B_minus_S"] <= B_MAX,
        "c_nn": G["D3_nn"]["B_minus_S"] >= A_MIN and G["D3_nn"]["ci95"][0] > 0,
        "c_loaco": G["D3_loaco"]["B_minus_S"] <= B_MAX,
    }
    crit["c"] = crit["c_nn"] and crit["c_loaco"]
    outcome = "GO" if crit["a"] and crit["b"] and crit["c"] else "STOP"
    tables = {k: per_detector_table(A, SUP + BEN + [W]) for k, A in T.items()}
    feas = json.loads((ROOT / "results/p1d/d5_feasibility.json").read_text())
    S = {"gate": {"stats": G, "criteria": crit, "outcome": outcome}, "tables": tables,
         "code_commits": {"P1d": p1d_commit, "P1c (D3 natural_novelty, 9 detectors)": p1c_commit},
         "rq_sha256_in_runs": {"P1d": sorted({r["provenance"]["rq_sha256"] for r in d5nn + d5lo + d3lo + d3w}),
                               "P1c": sorted({r["provenance"]["rq_sha256"] for r in p1c})},
         "rq_sha256": rq_sha256(), "d5_feasibility": {k: feas[k] for k in ("evaluable_classes", "test_windows", "train_tail")},
         "n_runs": {"D5_nn": len(d5nn), "D5_loaco": len(d5lo), "D3_loaco": len(d3lo), "D3_nn_wilkie": len(d3w)},
         "provenance": provenance()}
    write_json(ROOT / "results/p1d/summary.json", S)
    (ROOT / "P1D_REPORT.md").write_text(render(S))
    (ROOT / "GATE_G3_DECISION.md").write_text(render_gate(S))
    print(f"G3: {outcome} {crit}")


def render_gate(S):
    G, c = S["gate"]["stats"], S["gate"]["criteria"]
    L = ["# GATE_G3_DECISION", "", f"**Outcome: {S['gate']['outcome']}**", "",
         "Rule (AMENDMENT_04 A4.6, frozen before any P1d processing): B = mean over classes of the mean AUC of the 4 "
         "benign-only detectors; S = same for the 5 supervised-backbone scores. GO only if (a) D5 natural_novelty "
         "B − S ≥ 0.15 with paired-bootstrap 95% CI > 0; (b) D5 LOACO B − S ≤ 0.05; (c) the same (a)/(b) pattern on D3 "
         "(natural_novelty from P1c + new LOACO). Bootstrap: classes and seeds resampled independently, 2000 resamples.", "",
         "| Evaluation | classes | B | S | B − S | 95% CI | test | pass |", "|---|---|---|---|---|---|---|---|"]
    for k, test, ok in (("D5_nn", "≥ 0.15 and CI > 0", c["a"]), ("D5_loaco", "≤ 0.05", c["b"]),
                        ("D3_nn", "≥ 0.15 and CI > 0", c["c_nn"]), ("D3_loaco", "≤ 0.05", c["c_loaco"])):
        g = G[k]
        L.append(f"| {k} | {len(g['classes'])} | {_f(g['B'])} | {_f(g['S'])} | {_f(g['B_minus_S'])} | "
                 f"[{_f(g['ci95'][0])}, {_f(g['ci95'][1])}] | {test} | {'yes' if ok else 'no'} |")
    L += ["", f"(a) {'pass' if c['a'] else 'fail'}; (b) {'pass' if c['b'] else 'fail'}; (c) {'pass' if c['c'] else 'fail'}.", "",
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
    G = S["gate"]["stats"]
    L += ["## B, S and Wilkie per evaluation", "",
          "| Evaluation | B (benign-only) | S (supervised) | B − S [95% CI] | Wilkie CLAD | W − B | W − S |", "|---|---|---|---|---|---|---|"]
    for k, g in G.items():
        L.append(f"| {k} | {_f(g['B'])} | {_f(g['S'])} | {_f(g['B_minus_S'])} [{_f(g['ci95'][0])}, {_f(g['ci95'][1])}] | "
                 f"{_f(g.get('W'))} | {_f(g.get('W_minus_B'))} | {_f(g.get('W_minus_S'))} |")
    L.append("")
    for k, tab in S["tables"].items():
        dets = SUP + BEN + [W]
        L += [f"## {k}: mean AUC per class", "", "| Class | " + " | ".join(dets) + " | B | S |", "|---|" + "---|" * (len(dets) + 2)]
        for c, by in tab.items():
            L.append(f"| {c} | " + " | ".join(f"{_f(by[d]['mean'])} ± {_f(by[d]['ci95'], 2)} ({by[d]['frac_below_0.5']:.1f})"
                                             for d in dets) + f" | {_f(G[k]['B_by_class'][c])} | {_f(G[k]['S_by_class'][c])} |")
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
