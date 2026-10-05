#!/usr/bin/env python
"""P1c report and gate G2 (AMENDMENT_03), from results/p1c JSON only.

Writes P1C_REPORT.md, GATE_G2_DECISION.md, results/p1c/summary.json, report/p1c/*.
Fails closed on mixed code commits, a dirty tree, or a seed set different from the amendment's.
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

SEEDS = [17, 23, 42, 101, 202]
CONDS = ["none", "N100", "N500"]
NEED_DET = 5          # ">= 5 of 9 detectors"
A_MIN_CLASSES = 3     # (a) mean AUC < 0.5 on >= 3 classes
B_MIN_DPRIME = 0.5    # (b) median-over-classes d'(test benign vs train benign) >= 0.5
C_GAIN, C_FLOOR = 0.15, 0.5


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def load():
    runs = [json.loads(p.read_text()) for p in sorted(ROOT.glob("results/p1c/*/seed*.json"))]
    runs = [r for r in runs if not r.get("smoke")]
    if not runs:
        raise SystemExit("no P1c results")
    cs = {r["provenance"]["code_commit"] for r in runs}
    if len(cs) != 1 or any(r["provenance"]["code_dirty"] for r in runs):
        raise SystemExit(f"P1c runs from {sorted(cs)} or a dirty tree")
    for ds in {r["dataset"] for r in runs}:
        got = sorted(r["seed"] for r in runs if r["dataset"] == ds)
        if got != SEEDS:
            raise SystemExit(f"{ds}: seeds {got} != {SEEDS}")
    return runs, cs.pop()


def aggregate(runs):
    dets = runs[0]["detectors"]
    no_ref = set(runs[0]["no_reference_detectors"])
    by_ds = defaultdict(list)
    for r in runs:
        by_ds[r["dataset"]].append(r)
    T = {"auc": {}, "drift_benign": {}, "drift_attack": {}, "classes": {}}
    for ds, rs in sorted(by_ds.items()):
        classes = rs[0]["classes"]
        T["classes"][ds] = classes
        for cond in CONDS:
            for d in dets:
                k = f"{cond}/{d}"
                T["drift_benign"][(ds, cond, d)] = metrics.mean_ci([r["d_prime_testbenign_vs_valbenign"][k] for r in rs])
                for c in classes:
                    aucs = [r["auc"][k][c] for r in rs]
                    T["auc"][(ds, c, cond, d)] = {**metrics.mean_ci(aucs), "frac_below_0.5": float(np.mean(np.array(aucs) < 0.5)),
                                                 "per_seed": aucs}
                    T["drift_attack"][(ds, c, cond, d)] = metrics.mean_ci([r["d_prime_attack_vs_valbenign"][k][c] for r in rs])
    return dets, no_ref, T


def gate(dets, no_ref, T):
    pooled = [(ds, c) for ds, cl in T["classes"].items() for c in cl]
    n_cls = len(pooled)
    rows = {}
    for d in dets:
        inv = [(ds, c) for ds, c in pooled if T["auc"][(ds, c, "none", d)]["mean"] < 0.5]
        drift_vals = [T["drift_benign"][(ds, "none", d)]["mean"] for ds, c in pooled]
        med = float(np.median(drift_vals))
        gains = []
        for ds, c in pooled:
            a0, a5 = T["auc"][(ds, c, "none", d)]["mean"], T["auc"][(ds, c, "N500", d)]["mean"]
            gains.append({"dataset": ds, "class": c, "auc_none": a0, "auc_N500": a5, "gain": a5 - a0,
                          "ok": (d not in no_ref) and (a5 - a0 >= C_GAIN) and (a5 > C_FLOOR)})
        n_ok = sum(g["ok"] for g in gains)
        rows[d] = {"a_inverted_classes": [f"{ds}/{c}" for ds, c in inv], "a_pass": len(inv) >= A_MIN_CLASSES,
                   "b_median_dprime": med, "b_pass": med >= B_MIN_DPRIME,
                   "c_classes_ok": n_ok, "c_needed": int(np.ceil(n_cls / 2)), "c_pass": n_ok >= int(np.ceil(n_cls / 2)),
                   "c_applicable": d not in no_ref, "c_detail": gains,
                   "per_dataset": {ds: {"a_inverted": sum(1 for x, c in inv if x == ds),
                                        "b_dprime": T["drift_benign"][(ds, "none", d)]["mean"],
                                        "c_ok": sum(1 for g in gains if g["dataset"] == ds and g["ok"]),
                                        "n_classes": len(T["classes"][ds])} for ds in T["classes"]}}
    crit = {k: sum(rows[d][f"{k}_pass"] for d in dets) for k in ("a", "b", "c")}
    passed = {k: v >= NEED_DET for k, v in crit.items()}
    return {"per_detector": rows, "detectors_passing": crit, "criteria_pass": passed, "n_classes_pooled": n_cls,
            "outcome": "GO" if all(passed.values()) else "STOP"}


def figure(dets, T, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pooled = [(ds, c) for ds, cl in T["classes"].items() for c in cl]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cols = {"none": "#2a78d6", "N100": "#eb6834", "N500": "#1baf7a"}
    mk = {"none": "o", "N100": "s", "N500": "^"}
    x = np.arange(len(dets))
    for j, cond in enumerate(CONDS):
        means = [np.mean([T["auc"][(ds, c, cond, d)]["mean"] for ds, c in pooled]) for d in dets]
        ax.plot(x + (j - 1) * 0.18, means, mk[cond], color=cols[cond], ms=7, mec="white", mew=1.2,
                label={"none": "no re-anchoring", "N100": "re-anchored, N=100", "N500": "re-anchored, N=500"}[cond])
    ax.axhline(0.5, color="#b0afa8", lw=1, ls="--", zorder=0)
    ax.set_xticks(x, dets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean AUC over 7 natural-novelty classes", fontsize=8.5)
    ax.set_ylim(0, 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def render(S, dets, T) -> str:
    G = S["gate"]
    L = ["# P1c report", "",
         "Generated by `scripts/report_p1c.py` from `results/p1c` JSON. Scope: AMENDMENT_03 (RESEARCH_QUESTIONS.md). "
         "D1 + D3 natural_novelty, train-tail validation, seeds 17/23/42/101/202, 9 detectors on identical windows. "
         "AUC: held-out class vs shared test-period benign (anchor groups removed for all conditions). "
         "Scores higher = more anomalous, never flipped. CI: Student-t, n = 5.", "",
         f"**Gate G2: {G['outcome']}** (see `GATE_G2_DECISION.md`).", ""]
    L += ["## (i) Mean AUC per class, no re-anchoring", ""]
    for ds, classes in T["classes"].items():
        L += [f"### {ds}", "", "| Detector | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for d in dets:
            L.append(f"| {d} | " + " | ".join(
                f"{_f(T['auc'][(ds, c, 'none', d)]['mean'])} ± {_f(T['auc'][(ds, c, 'none', d)]['ci95'])} "
                f"({T['auc'][(ds, c, 'none', d)]['frac_below_0.5']:.1f})" for c in classes) + " |")
        L += ["", "Cell: mean AUC ± t-CI (fraction of seeds < 0.5).", ""]
    L += ["## (ii) Drift diagnostic (no re-anchoring)", "",
          "d′(test-period benign vs train-period benign validation windows), mean over seeds:", "",
          "| Detector | " + " | ".join(T["classes"]) + " |", "|---|" + "---|" * len(T["classes"])]
    for d in dets:
        L.append(f"| {d} | " + " | ".join(f"{_f(T['drift_benign'][(ds, 'none', d)]['mean'])} ± "
                                          f"{_f(T['drift_benign'][(ds, 'none', d)]['ci95'])}" for ds in T["classes"]) + " |")
    L += ["", "d′(held-out attack vs train-period benign validation windows), mean over seeds:", ""]
    for ds, classes in T["classes"].items():
        L += [f"**{ds}**", "", "| Detector | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for d in dets:
            L.append(f"| {d} | " + " | ".join(_f(T["drift_attack"][(ds, c, "none", d)]["mean"]) for c in classes) + " |")
        L.append("")
    L += ["## (iii) Re-anchoring (benign reference refit only; no labels, no backbone retraining)", "",
          "Mean AUC per class: none → N=100 → N=500. MSP and Energy have no benign reference (not applicable).", ""]
    for ds, classes in T["classes"].items():
        L += [f"### {ds}", "", "| Detector | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
        for d in dets:
            L.append(f"| {d} | " + " | ".join(" → ".join(_f(T["auc"][(ds, c, cond, d)]["mean"], 2) for cond in CONDS)
                                              for c in classes) + " |")
        L += ["", "d′(test benign vs train benign) after re-anchoring (none → N100 → N500): " + "; ".join(
            f"{d} " + " → ".join(_f(T["drift_benign"][(ds, cond, d)]["mean"], 2) for cond in CONDS) for d in dets), ""]
    L += ["Figure: `report/p1c/auc_by_detector.pdf`.", "",
          "## Provenance", "",
          f"- code_commit `{S['code_commit']}` (single, clean), preflight seeding regression test passed at start.",
          f"- RESEARCH_QUESTIONS.md SHA-256 in runs: " + ", ".join(f"`{h}`" for h in S["rq_sha256_in_runs"])
          + " (AMENDMENT_03).",
          "- Data packages: `data/processed/<ds>/temporal_gap/p1c/` (`scripts/p1c_build.py`); archive SHA-256:"]
    for ds, h in S["archive_sha256"].items():
        L.append(f"  - {ds}: flows.npy `{h['flows.npy']}`, sets.npz `{h['sets.npz']}`, cleaner.json `{h['cleaner.json']}`")
    L += [f"- Report generated at `{S['provenance']['code_commit']}`.", ""]
    return "\n".join(L)


def render_gate(S, dets) -> str:
    G = S["gate"]
    L = ["# GATE_G2_DECISION", "", f"**Outcome: {G['outcome']}**", "",
         "Rule (AMENDMENT_03 A3.4, frozen before any P1c run): GO only if all three hold — "
         "(a) ≥ 5 of 9 detectors have mean AUC < 0.5 on ≥ 3 natural-novelty classes (D1 + D3 pooled, 7 classes); "
         "(b) ≥ 5 of 9 detectors have median-over-classes d′(test benign vs train benign) ≥ 0.5; "
         "(c) for ≥ 5 of 9 detectors, re-anchoring with N = 500 raises mean AUC by ≥ 0.15 over no re-anchoring "
         "AND above 0.5 on ≥ half of the classes (≥ 4 of 7).", "",
         "| Criterion | detectors passing | needed | pass |", "|---|---|---|---|"]
    for k in ("a", "b", "c"):
        L.append(f"| ({k}) | {G['detectors_passing'][k]} / 9 | {NEED_DET} | {'yes' if G['criteria_pass'][k] else 'no'} |")
    L += ["", "Per detector (pooled; per dataset in brackets D1 / D3):", "",
          "| Detector | (a) classes with AUC < 0.5 | (a) | (b) median d′ | (b) | (c) classes gaining ≥ 0.15 to > 0.5 | (c) |",
          "|---|---|---|---|---|---|---|"]
    for d in dets:
        r = G["per_detector"][d]
        pd = r["per_dataset"]
        L.append(f"| {d} | {len(r['a_inverted_classes'])} [{pd['D1']['a_inverted']} / {pd['D3']['a_inverted']}] | "
                 f"{'pass' if r['a_pass'] else 'fail'} | {_f(r['b_median_dprime'])} [{_f(pd['D1']['b_dprime'])} / "
                 f"{_f(pd['D3']['b_dprime'])}] | {'pass' if r['b_pass'] else 'fail'} | "
                 + (f"{r['c_classes_ok']} / 7 [{pd['D1']['c_ok']} / {pd['D3']['c_ok']}]" if r["c_applicable"] else "n/a (no benign reference)")
                 + f" | {'pass' if r['c_pass'] else 'fail'} |")
    L += ["", f"P1c code commit `{S['code_commit']}`; RQ SHA-256 " + ", ".join(f"`{h}`" for h in S["rq_sha256_in_runs"]) + ".",
          "Generated by `scripts/report_p1c.py` from `results/p1c` JSON.", ""]
    return "\n".join(L)


def main():
    runs, commit = load()
    dets, no_ref, T = aggregate(runs)
    G = gate(dets, no_ref, T)
    archive = {}
    for ds in T["classes"]:
        archive[ds] = next(r for r in runs if r["dataset"] == ds)["provenance"]["data_sha256"]
    S = {"code_commit": commit, "rq_sha256": rq_sha256(),
         "rq_sha256_in_runs": sorted({r["provenance"]["rq_sha256"] for r in runs}),
         "archive_sha256": archive, "gate": G,
         "tables": {"auc": {"|".join(k): v for k, v in T["auc"].items()},
                    "drift_benign": {"|".join(k): v for k, v in T["drift_benign"].items()},
                    "drift_attack": {"|".join(k): v for k, v in T["drift_attack"].items()}},
         "provenance": provenance()}
    write_json(ROOT / "results/p1c/summary.json", S)
    figure(dets, T, ROOT / "report/p1c/auc_by_detector.pdf")
    (ROOT / "P1C_REPORT.md").write_text(render(S, dets, T))
    (ROOT / "GATE_G2_DECISION.md").write_text(render_gate(S, dets))
    print(f"G2: {G['outcome']} {G['detectors_passing']}")


if __name__ == "__main__":
    main()
