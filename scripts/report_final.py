#!/usr/bin/env python
"""Final benchmark report (AMENDMENT_06), from results JSON only. Descriptive; no gate.

Inputs: results/final/** (one code commit), results/p1d/d5_purged_block_meta.json,
data/processed/*/index_meta.json (feasibility), the P1/P1b/P1c/P1d summaries (NEGATIVE_RESULTS.md).
Outputs:
  report/final/tables/*.tex, report/final/figures/F1..F6 *.pdf
  results/final/summary.json, RESULTS_SUMMARY.md, NEGATIVE_RESULTS.md
Fails closed on mixed code commits, a dirty run, smoke runs, a missing seed or class, or an RQ hash
other than the AMENDMENT_06 hash.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_run import BEN, CK_DEP, CLAD_, CONF, EVALS, REPR, TRANSFER_TARGETS, XGB  # noqa: E402
from nids import metrics  # noqa: E402
from nids.provenance import ROOT, provenance, write_json  # noqa: E402

SEEDS = [17, 23, 42, 101, 202]
DETS = CONF + REPR + BEN + [CLAD_, XGB]
FAMILIES = {"confidence": CONF, "representation": REPR, "benign-only": BEN, "CLAD": [CLAD_], "XGBoost": [XGB]}
FAM_CK = ["confidence", "representation", "CLAD"]
CONTRASTS = [("benign-only", "confidence"), ("benign-only", "representation"), ("representation", "confidence"),
             ("CLAD", "benign-only"), ("XGBoost", "benign-only")]
LABEL = {"P_attack": "P(attack)\u2020", "AHSD_stress": "AHSD stress", "Wilkie_CLAD": "CLAD", "XGBoost": "XGBoost"}
N_BOOT, BOOT_SEED = 2000, 0
CLASS_EVALS = [e["id"] for e in EVALS if e["kind"] in ("loaco", "natural_novelty")]
LOACO_EVALS = [e["id"] for e in EVALS if e["kind"] == "loaco"]
NN_EVALS = [e["id"] for e in EVALS if e["kind"] == "natural_novelty"]
IND_EVALS = [e["id"] for e in EVALS if e["kind"] == "indist"]
EVAL_NAME = {"loaco_D1fam": "LOACO D1 (families)", "loaco_D2": "LOACO D2", "loaco_D3": "LOACO D3",
             "loaco_D5pb": "LOACO D5 purged\\_block", "nn_D1": "Natural novelty D1", "nn_D3": "Natural novelty D3",
             "nn_D5": "Natural novelty D5"}
RES = ROOT / "results/final"
SHORT = {"loaco_D1fam": "LOACO\nD1 fam", "loaco_D2": "LOACO\nD2", "loaco_D3": "LOACO\nD3", "loaco_D5pb": "D5\npb",
         "nn_D1": "NN\nD1", "nn_D3": "NN\nD3", "nn_D5": "NN\nD5"}
OUT = ROOT / "report/final"
MD_SUMMARY, MD_NEG = ROOT / "RESULTS_SUMMARY.md", ROOT / "NEGATIVE_RESULTS.md"
CHECK_PROVENANCE = True
TAB, FIG = OUT / "tables", OUT / "figures"


def lab(d):
    return LABEL.get(d, d)


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def tex_escape(s: str) -> str:
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("\u2020", "$^\\dagger$") \
        .replace("−", "$-$").replace("—", "--")


# ------------------------------------------------------------------ loading
def load():
    runs = []
    for p in sorted(RES.glob("*/*/seed*.json")):
        r = json.loads(p.read_text())
        runs.append(r)
    if not runs:
        raise SystemExit("no final-run results")
    commits = {r["provenance"]["code_commit"] for r in runs}
    a6 = [a for a in json.loads((ROOT / "docs/AMENDMENTS.json").read_text())["amendments"] if a["id"] == "AMENDMENT_06"][0]
    if not CHECK_PROVENANCE:
        return runs, sorted(commits)[0], a6
    if len(commits) != 1 or any(r["provenance"]["code_dirty"] for r in runs) or any(r.get("smoke") for r in runs):
        raise SystemExit(f"final run: commits {sorted(commits)}, dirty or smoke runs present")
    rq = {r["provenance"]["rq_sha256"] for r in runs}
    if rq != {a6["new_sha256"]}:
        raise SystemExit(f"RQ hash in runs {rq} != AMENDMENT_06 {a6['new_sha256']}")
    return runs, commits.pop(), a6


def class_tensor(runs):
    """A[eval][ck][det][class] = {seed: auc}"""
    A = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    for r in runs:
        if r["kind"] in ("loaco", "natural_novelty"):
            for ck, by in r["auc"].items():
                for d, bc in by.items():
                    for c, a in bc.items():
                        A[r["eval_id"]][ck][d][c][r["seed"]] = a
    return A


def check_complete(A, runs):
    folds = {e: json.loads((RES / e / "folds.json").read_text()) for e in LOACO_EVALS}
    nn_classes = {r["eval_id"]: r["classes"] for r in runs if r["kind"] == "natural_novelty"}
    expect = {**{e: folds[e]["feasible"] for e in LOACO_EVALS}, **nn_classes}
    for e in CLASS_EVALS:
        if e not in A:
            raise SystemExit(f"{e}: no runs")
        for ck, dets in (("best", DETS), ("final", CK_DEP)):
            for d in dets:
                for c in expect[e]:
                    got = sorted(A[e][ck][d][c])
                    if got != SEEDS:
                        raise SystemExit(f"{e}/{ck}/{d}/{c}: seeds {got}")
    return folds, expect


# ------------------------------------------------------------------ statistics
def cell(by_seed: dict) -> dict:
    x = np.array([by_seed[s] for s in SEEDS], dtype=float)
    return {**metrics.mean_ci(x), "std": float(x.std(ddof=1)), "frac_below_0.5": float(np.mean(x < 0.5)),
            "values": x.tolist()}


def fam_matrix(A_ck, fam, classes):
    """(C, S) family mean per class and seed."""
    return np.array([[np.mean([A_ck[d][c][s] for d in FAMILIES[fam]]) for s in SEEDS] for c in classes])


def boot_mean_ci(M, rng_seed=BOOT_SEED):
    """Bootstrap of the grand mean of a (C, S) matrix, classes and seeds resampled independently (G3 procedure)."""
    rng = np.random.default_rng(rng_seed)
    C, S = M.shape
    b = [M[np.ix_(rng.integers(0, C, C), rng.integers(0, S, S))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(M.mean()), "ci95": [float(lo), float(hi)]}


def class_eval_stats(A, expect):
    S = {}
    for e in CLASS_EVALS:
        classes = expect[e]
        st = {"classes": classes, "n_classes": len(classes), "cells": {}, "cells_final": {}, "families": {},
              "contrasts": {}, "checkpoint": {}, "seed_std": {}}
        for d in DETS:
            st["cells"][d] = {c: cell(A[e]["best"][d][c]) for c in classes}
            st["seed_std"][d] = float(np.mean([st["cells"][d][c]["std"] for c in classes]))
        for d in CK_DEP:
            st["cells_final"][d] = {c: cell(A[e]["final"][d][c]) for c in classes}
        for fam in FAMILIES:
            st["families"][fam] = boot_mean_ci(fam_matrix(A[e]["best"], fam, classes))
        for a, b in CONTRASTS:
            M = fam_matrix(A[e]["best"], a, classes) - fam_matrix(A[e]["best"], b, classes)
            st["contrasts"][f"{a} − {b}"] = boot_mean_ci(M)
        for fam in FAM_CK:
            Mb, Mf = fam_matrix(A[e]["best"], fam, classes), fam_matrix(A[e]["final"], fam, classes)
            st["checkpoint"][fam] = {"best": float(Mb.mean()), "final": float(Mf.mean()),
                                     "final_minus_best": boot_mean_ci(Mf - Mb)}
        Bm = fam_matrix(A[e]["best"], "benign-only", classes)
        for fam in ("confidence", "representation"):
            st["checkpoint"][f"benign-only − {fam}"] = {
                "best": boot_mean_ci(Bm - fam_matrix(A[e]["best"], fam, classes)),
                "final": boot_mean_ci(Bm - fam_matrix(A[e]["final"], fam, classes))}
        Cm = fam_matrix(A[e]["best"], "CLAD", classes)
        st["clad_vs"] = {f: boot_mean_ci(Cm - fam_matrix(A[e]["best"], f, classes))
                         for f in ("confidence", "representation", "benign-only", "XGBoost")}
        st["checkpoint_per_detector"] = {
            d: {c: float(np.mean([A[e]["final"][d][c][s] - A[e]["best"][d][c][s] for s in SEEDS])) for c in classes}
            for d in CK_DEP}
        S[e] = st
    return S


def indist_stats(runs):
    """I[eval][ck][det][set][metric] = cell"""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))
    info = {}
    for r in runs:
        if r["kind"] != "indist":
            continue
        info[r["eval_id"]] = {"n": r["n"], "transfer_sha256": r.get("transfer_sha256", {})}
        for ck, by in r["metrics"].items():
            for d, bs in by.items():
                for k, mm in bs.items():
                    for m_, v in mm.items():
                        raw[r["eval_id"]][ck][d][k][m_][r["seed"]] = v
    I = {}
    for e in IND_EVALS:
        if e not in raw:
            raise SystemExit(f"{e}: no runs")
        I[e] = {}
        for ck in ("best", "final"):
            I[e][ck] = {}
            for d, bs in raw[e][ck].items():
                I[e][ck][d] = {}
                for k, mm in bs.items():
                    I[e][ck][d][k] = {}
                    for m_, by_seed in mm.items():
                        if sorted(by_seed) != SEEDS:
                            raise SystemExit(f"{e}/{ck}/{d}/{k}/{m_}: seeds {sorted(by_seed)}")
                        x = np.array([by_seed[s] for s in SEEDS], float)
                        I[e][ck][d][k][m_] = {**metrics.mean_ci(x), "std": float(np.nanstd(x, ddof=1))}
        for d in DETS:
            if d not in I[e]["best"]:
                raise SystemExit(f"{e}: detector {d} missing")
    return I, info


def transfer_matrix(I):
    """T[det][src][tgt] = cell (AUC); diagonal = in-domain test of the source temporal_gap model."""
    T = {}
    for d in DETS:
        T[d] = {}
        for src, tgts in TRANSFER_TARGETS.items():
            e = f"indist_{src}_temporal_gap"
            T[d][src] = {src: I[e]["best"][d]["test"]["auc"]}
            for t in tgts:
                T[d][src][t] = I[e]["best"][d][f"transfer/{t}"]["auc"]
    return T


# ------------------------------------------------------------------ feasibility
def feasibility(folds, runs):
    F = {"loaco": {}, "natural_novelty": {}, "transfer_targets": TRANSFER_TARGETS}
    for e in LOACO_EVALS:
        fi = folds[e]["fold_info"]
        F["loaco"][e] = {"classes_total": len(fi), "feasible": folds[e]["feasible"],
                         "infeasible": {c: r.get("reason", "") for c, r in fi.items() if c not in folds[e]["feasible"]}}
    for r in runs:
        if r["kind"] == "natural_novelty" and r["seed"] == SEEDS[0]:
            F["natural_novelty"][r["eval_id"]] = {"classes": r["classes"], "n": r["n"]}
    pm = json.loads((ROOT / "results/p1d/d5_purged_block_meta.json").read_text())
    F["d5_purged_block"] = {"blocks": pm["blocks"], "days": len(pm["days"]), "feasible": pm["feasible_folds"],
                            "split_windows": pm["split_windows"]}
    a52 = json.loads((ROOT / "results/p1d/a52_decision.json").read_text())
    F["a52_fallback_applied"] = a52["fallback_applied"]
    dsets = {}
    for ds in ("D1", "D2", "D3", "D5"):
        m = json.loads((ROOT / f"data/processed/{ds}/index_meta.json").read_text())
        row = {"flows": m["flows"], "classes": len(m["classes"]) - 1}
        for sch in ("temporal_gap", "grouped_random"):
            fi = m["schemes"][sch]["loaco"]
            row[f"loaco_{sch}"] = sum(1 for r in fi.values() if r.get("eligible") and r.get("feasible"))
        if "family" in m:
            fi = m["schemes"]["grouped_random"]["loaco_family"]
            row["loaco_family_grouped_random"] = sum(1 for r in fi.values() if r.get("eligible") and r.get("feasible"))
            row["families"] = len(m["family"]["classes"]) - 1
        dsets[ds] = row
    F["datasets"] = dsets
    return F


# ------------------------------------------------------------------ LaTeX
def write_tex(name, caption, label, header, rows, colspec=None, note=None):
    colspec = colspec or ("l" * len(header))
    L = ["% generated by scripts/report_final.py from results JSON; do not edit",
         "\\begin{table}[t]", "\\centering", "\\scriptsize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\hline", " & ".join(tex_escape(h) for h in header) + " \\\\", "\\hline"]
    for r in rows:
        if r == "---":
            L.append("\\hline")
        else:
            L.append(" & ".join(tex_escape(x) for x in r) + " \\\\")
    L += ["\\hline", "\\end{tabular}"]
    if note:
        L.append(f"\\\\[2pt]\\parbox{{\\linewidth}}{{\\tiny {tex_escape(note)}}}")
    L += ["\\end{table}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(L))


def tables(CS, I, T, F):
    note = "\u2020 P(attack) added post-G3 (AMENDMENT\\_06 A6.2). Mean over 5 seeds; seed std in brackets."
    # per-class AUC tables, one per class evaluation
    for e in CLASS_EVALS:
        st = CS[e]
        rows = []
        for fam, dets in FAMILIES.items():
            for d in dets:
                rows.append([lab(d), fam] + [f"{st['cells'][d][c]['mean']:.3f} ({st['cells'][d][c]['std']:.3f})"
                                             for c in st["classes"]])
            rows.append("---")
        write_tex(f"T_auc_{e}", f"{EVAL_NAME[e]}: per-class AUC (primary checkpoint).", f"tab:auc_{e}",
                  ["Detector", "Family"] + st["classes"], rows[:-1], "ll" + "r" * len(st["classes"]), note)
    # family means and contrasts
    rows = []
    for e in CLASS_EVALS:
        st = CS[e]
        rows.append([EVAL_NAME[e].replace("\\_", "_"), str(st["n_classes"])]
                    + [f"{st['families'][f]['mean']:.3f}" for f in FAMILIES]
                    + [f"{st['contrasts'][k]['mean']:+.3f} [{st['contrasts'][k]['ci95'][0]:+.3f}, {st['contrasts'][k]['ci95'][1]:+.3f}]"
                       for k in ("benign-only − confidence", "benign-only − representation")])
    write_tex("T_families", "Family mean AUC per protocol and descriptive contrasts (bootstrap over classes and seeds).",
              "tab:families", ["Evaluation", "classes"] + list(FAMILIES) + ["BEN − CONF", "BEN − REPR"], rows,
              "lr" + "r" * (len(FAMILIES) + 2),
              "Descriptive only (AMENDMENT_06 A6.3). D5 natural novelty has 2 classes; D5 purged_block has 1 fold.")
    # seed std
    rows = [[lab(d)] + [f"{CS[e]['seed_std'][d]:.3f}" for e in CLASS_EVALS] for d in DETS]
    write_tex("T_seed_std", "Mean seed standard deviation of AUC per detector and evaluation.", "tab:seed_std",
              ["Detector"] + [e for e in CLASS_EVALS], rows, "l" + "r" * len(CLASS_EVALS))
    # checkpoint
    rows = []
    for e in CLASS_EVALS:
        ck = CS[e]["checkpoint"]
        for fam in FAM_CK:
            fb = ck[fam]["final_minus_best"]
            rows.append([e, fam, f"{ck[fam]['best']:.3f}", f"{ck[fam]['final']:.3f}",
                         f"{fb['mean']:+.3f} [{fb['ci95'][0]:+.3f}, {fb['ci95'][1]:+.3f}]"])
        rows.append("---")
    write_tex("T_checkpoint", "Checkpoint sensitivity: family mean AUC at the best-validation and final-epoch state.",
              "tab:checkpoint", ["Evaluation", "Family", "best", "final", "final − best [95% CI]"], rows[:-1], "llrrr")
    # saturation
    rows = []
    for e in IND_EVALS:
        for d in ("P_attack", XGB):
            for k in ("test", "test_natural"):
                m = I[e]["best"][d][k]
                rows.append([e.replace("indist_", ""), lab(d), k] + [f"{m[x]['mean']:.3f} ± {m[x]['ci95']:.3f}"
                                                                    for x in ("auc", "pr_auc", "macro_f1", "mcc", "tpr_at_1pct_fpr")])
        rows.append("---")
    write_tex("T_saturation", "Temporal-split degradation: AHSD backbone P(attack) vs XGBoost, in-distribution binary detection "
              "(mean ± t-CI, 5 seeds). Single neural backbone (limitation).",
              "tab:saturation", ["Dataset/split", "Model", "Test set", "AUC", "PR-AUC", "macro-F1", "MCC", "TPR@1%FPR"],
              rows[:-1], "lllrrrrr", "Limitation: AHSD is the only neural backbone evaluated (authors' decision, DECISIONS_PENDING item 9); "
              "XGBoost is deterministic under the fixed parameters (seed std 0).")
    # transfer
    rows = []
    for d in DETS:
        for src in TRANSFER_TARGETS:
            rows.append([lab(d), src] + [f"{T[d][src][t]['mean']:.3f}" if t in T[d][src] else "—" for t in ("D1", "D2", "D3", "D4")])
    write_tex("T_transfer", "Zero-shot transfer AUC (rows: source; columns: target; diagonal: in-domain test).",
              "tab:transfer", ["Detector", "Source", "D1", "D2", "D3", "D4"], rows, "llrrrr")
    # feasibility
    rows = []
    for ds, r in F["datasets"].items():
        rows.append([ds, f"{r['flows']:,}", str(r["classes"]), str(r["loaco_grouped_random"]), str(r["loaco_temporal_gap"]),
                     str(r.get("loaco_family_grouped_random", "—")),
                     ", ".join(F["natural_novelty"].get(f"nn_{ds}", {}).get("classes", [])) or "—"])
    write_tex("T_feasibility", "Protocol feasibility per dataset (feasible LOACO folds; natural-novelty classes).",
              "tab:feasibility", ["Dataset", "flows", "attack classes", "LOACO gr", "LOACO tg", "LOACO families", "natural novelty"],
              rows, "lrrrrrl", f"D5 purged_block: {len(F['d5_purged_block']['feasible'])} feasible fold(s) "
                               f"({', '.join(F['d5_purged_block']['feasible'])}); A5.2 fallback applied: {F['a52_fallback_applied']}.")


# ------------------------------------------------------------------ figures
FAM_COLOR = {"confidence": "#d95f02", "representation": "#1b9e77", "benign-only": "#7570b3", "CLAD": "#e7298a",
             "XGBoost": "#666666"}


def fig_protocols(F):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 4.2))
    col = {"train": "#4c78a8", "val": "#f58518", "test": "#54a24b", "gap": "#dddddd", "purged": "#bbbbbb"}
    # LOACO grouped_random
    ax = axes[0]
    rng = np.random.default_rng(0)
    lab_ = rng.permutation(["train"] * 12 + ["val"] * 4 + ["test"] * 4)
    for i, s in enumerate(lab_):
        ax.add_patch(Rectangle((i, 0), 0.9, 1, color=col[s]))
    n_l = sum(len(F["loaco"][e]["feasible"]) for e in LOACO_EVALS if e != "loaco_D5pb")
    ax.set_title(f"LOACO (grouped_random): 1-h groups shuffled 60/20/20; held-out class removed from train/val "
                 f"({n_l} folds on D1 families, D2, D3)", fontsize=7, loc="left")
    # natural novelty
    ax = axes[1]
    segs = [("train", 0, 9.6), ("val", 9.6, 2.4), ("gap", 12, 1), ("val", 13, 3), ("test", 16, 4)]
    names = ["train head (80%)", "train tail = val", "gap", "(unused val)", "test: novel classes"]
    for (s, x, w), nm in zip(segs, names):
        ax.add_patch(Rectangle((x, 0), w - 0.05, 1, color=col[s]))
        ax.text(x + w / 2, 0.5, nm, ha="center", va="center", fontsize=5.5)
    n_nn = {e: len(F["natural_novelty"][e]["classes"]) for e in NN_EVALS}
    ax.set_title("Natural novelty (temporal_gap, train-tail validation): classes first seen in the test period "
                 f"(classes: {', '.join(f'{k[3:]} {v}' for k, v in n_nn.items())})", fontsize=7, loc="left")
    # purged block
    ax = axes[2]
    lab2 = np.array(rng.permutation(["train"] * 12 + ["val"] * 4 + ["test"] * 4), dtype=object)
    final = lab2.copy()
    for i in range(20):
        nb = [lab2[j] for j in (i - 1, i + 1) if 0 <= j < 20]
        if lab2[i] in ("train", "val") and "test" in nb:
            final[i] = "purged"
    for i in range(20):
        nb = [final[j] for j in (i - 1, i + 1) if 0 <= j < 20]
        if final[i] == "train" and "val" in nb:
            final[i] = "purged"
    for i, s in enumerate(final):
        ax.add_patch(Rectangle((i, 0), 0.9, 1, color=col[s]))
    pb = F["d5_purged_block"]
    ax.set_title(f"D5 purged_block: 10-min blocks per capture day, 60/20/20, neighbours purged "
                 f"(D5: {pb['blocks']['non_empty']} blocks, {pb['blocks']['purged_rule1'] + pb['blocks']['purged_rule2']} purged; "
                 f"{len(pb['feasible'])} feasible fold)", fontsize=7, loc="left")
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)
        ax.axis("off")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    fig.legend(handles, list(col), loc="lower center", ncol=5, fontsize=7, frameon=False)
    fig.text(0.5, 0.06, "Schematic (block layouts illustrative); counts from results JSON.", ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "F1_protocols.pdf")
    plt.close(fig)


def fig_feasibility(F):
    ds = list(F["datasets"])
    keys = [("loaco_grouped_random", "LOACO grouped_random"), ("loaco_temporal_gap", "LOACO temporal_gap"),
            ("loaco_family_grouped_random", "LOACO families (D1)")]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    w = 0.2
    for i, (k, nm) in enumerate(keys):
        ax.bar(np.arange(len(ds)) + (i - 1.5) * w, [F["datasets"][d].get(k, 0) for d in ds], w, label=nm)
    ax.bar(np.arange(len(ds)) + 1.5 * w, [len(F["natural_novelty"].get(f"nn_{d}", {}).get("classes", [])) for d in ds],
           w, label="natural-novelty classes")
    ax.bar(len(ds) - 1 + 2.5 * w, len(F["d5_purged_block"]["feasible"]), w, color="k", label="D5 purged_block folds")
    ax.set_xticks(np.arange(len(ds)))
    ax.set_xticklabels([f"{d}\n({F['datasets'][d]['classes']} attack classes)" for d in ds], fontsize=7)
    ax.set_ylabel("feasible folds / classes", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F2_feasibility.pdf")
    plt.close(fig)


def fig_heatmap(CS):
    cols, groups = [], []
    for e in CLASS_EVALS:
        for c in CS[e]["classes"]:
            cols.append((e, c))
        groups.append((e, len(CS[e]["classes"])))
    M = np.array([[CS[e]["cells"][d][c]["mean"] for (e, c) in cols] for d in DETS])
    fig, ax = plt.subplots(figsize=(max(8, 0.33 * len(cols) + 2), 4.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            t = f"{M[i, j]:.2f}"
            ax.text(j, i, t[1:] if t.startswith("0") else t, ha="center", va="center", fontsize=4.5,
                    color="w" if abs(M[i, j] - 0.5) > 0.35 else "k")
    ax.set_yticks(range(len(DETS)))
    ax.set_yticklabels([lab(d) for d in DETS], fontsize=6)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for _, c in cols], rotation=90, fontsize=5.5)
    x = -0.5
    for e, n in groups:
        ax.axvline(x, color="k", lw=0.8)
        ax.text(x + n / 2, -0.7, SHORT[e], ha="center", va="bottom", fontsize=5.5, rotation=0)
        x += n
    y = -0.5
    for fam, dets in FAMILIES.items():
        y += len(dets)
        ax.axhline(y, color="k", lw=0.5)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("mean AUC (5 seeds)", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "F3_auc_heatmap.pdf")
    plt.close(fig)


def fig_seed_std(CS):
    def pooled(evals, d):
        return [CS[e]["cells"][d][c]["std"] for e in evals for c in CS[e]["classes"]]
    lo_evals = [e for e in LOACO_EVALS if e != "loaco_D5pb"]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    x = np.arange(len(DETS))
    for k, (evals, nm, off) in enumerate(((lo_evals, "LOACO (D1 families, D2, D3)", -0.2), (NN_EVALS, "natural novelty (D1, D3, D5)", 0.2))):
        vals = [pooled(evals, d) for d in DETS]
        ax.bar(x + off, [np.mean(v) for v in vals], 0.38, label=nm, color=["#4c78a8", "#e45756"][k])
        for i, v in enumerate(vals):
            ax.scatter(np.full(len(v), x[i] + off), v, s=3, color="k", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([lab(d) for d in DETS], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("seed std of AUC", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F4_seed_std.pdf")
    plt.close(fig)


def fig_transfer(T):
    tg = ["D1", "D2", "D3", "D4"]
    srcs = list(TRANSFER_TARGETS)
    fig, axes = plt.subplots(3, 4, figsize=(8, 5.6))
    for ax, d in zip(axes.flat, DETS):
        M = np.array([[T[d][s][t]["mean"] if t in T[d][s] else np.nan for t in tg] for s in srcs])
        ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1)
        for i in range(len(srcs)):
            for j in range(len(tg)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="w" if abs(M[i, j] - 0.5) > 0.35 else "k", fontweight="bold" if srcs[i] == tg[j] else None)
        ax.set_xticks(range(len(tg)))
        ax.set_xticklabels(tg, fontsize=6)
        ax.set_yticks(range(len(srcs)))
        ax.set_yticklabels(srcs, fontsize=6)
        ax.set_title(lab(d), fontsize=7, color=FAM_COLOR[[f for f, ds in FAMILIES.items() if d in ds][0]])
    fig.text(0.5, 0.005, "rows: source (trained in-domain, temporal_gap); columns: target (zero-shot); bold diagonal: in-domain test",
             ha="center", fontsize=6)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "F5_transfer.pdf")
    plt.close(fig)


def fig_checkpoint(CS):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ax = axes[0]
    y = 0
    ticks = []
    for e in CLASS_EVALS:
        for fam in FAM_CK:
            ck = CS[e]["checkpoint"][fam]
            ax.plot([ck["best"], ck["final"]], [y, y], color=FAM_COLOR[fam], lw=1)
            ax.scatter([ck["best"]], [y], color=FAM_COLOR[fam], s=10)
            ax.scatter([ck["final"]], [y], facecolor="w", edgecolor=FAM_COLOR[fam], s=10, zorder=3)
            y += 1
        ticks.append((y - 2, e))
        y += 0.8
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([e for _, e in ticks], fontsize=6)
    ax.set_xlabel("family mean AUC (filled: best val; open: final epoch)", fontsize=7)
    ax.invert_yaxis()
    ax = axes[1]
    for e in CLASS_EVALS:
        for d in CK_DEP:
            fam = [f for f, ds in FAMILIES.items() if d in ds][0]
            for c in CS[e]["classes"]:
                ax.scatter(CS[e]["cells"][d][c]["mean"], CS[e]["cells_final"][d][c]["mean"], s=5, color=FAM_COLOR[fam])
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("AUC, best-validation checkpoint", fontsize=7)
    ax.set_ylabel("AUC, final-epoch checkpoint", fontsize=7)
    for fam in FAM_CK:
        ax.scatter([], [], color=FAM_COLOR[fam], label=fam, s=8)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "F6_checkpoint.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ markdown
def caveats(F, CS):
    return [
        f"D5 natural novelty rests on {CS['nn_D5']['n_classes']} classes ({', '.join(CS['nn_D5']['classes'])}); "
        "a class bootstrap over 2 classes understates between-class uncertainty.",
        f"D5 LOACO (purged_block) has {len(F['d5_purged_block']['feasible'])} feasible fold "
        f"({', '.join(F['d5_purged_block']['feasible'])}); under A5.2 it is a protocol limitation, not a LOACO estimate.",
        "Checkpoint fragility: the D5 train-tail validation set has 26 attack windows; checkpoint choice changes "
        "backbone scores (see the checkpoint section).",
        "P(attack) was added after G3 (A6.2); MSP and Energy measure classifier confidence and read confident attack "
        "predictions as normal in a binary task.",
        "Temporal-split degradation (R5) compares a single neural backbone (AHSD) with XGBoost; no other backbone was built "
        "or will be (authors' decision, DECISIONS_PENDING item 9). Conclusions about 'neural backbones' rest on AHSD alone.",
        "All analyses here are descriptive (A6.3); intervals are not tests and no threshold is applied.",
        "XGBoost uses subsample = colsample = 1 (fixed parameters, AMENDMENT_06), so its fit is deterministic: its seed std "
        "is 0 by construction and its interval reflects no training randomness.",
    ]


NOVELTY_FAMS = ["confidence", "representation", "benign-only", "CLAD"]


def _ci(v, sign=False):
    f = "+.3f" if sign else ".3f"
    return f"{v['mean']:{f}} [{v['ci95'][0]:{f}}, {v['ci95'][1]:{f}}]"


def _det_mean(st, d):
    return float(np.mean([st["cells"][d][c]["mean"] for c in st["classes"]]))


def verdict(items):
    """items: [(sub-claim, ok, detail)] -> evidence-check lines; 'holds' only if every sub-claim holds."""
    ok = all(x[1] for x in items)
    head = "*Evidence check:* **holds**" if ok else "*Evidence check:* **does NOT hold**"
    lines = [head + f" ({sum(x[1] for x in items)}/{len(items)} sub-claims verified against the JSON):", ""]
    lines += [f"- {'✓' if o else '✗'} {s}" + (f" — {d}" if d else "") for s, o, d in items]
    return lines + [""], ok


def contributions(S, CS, I, T, F):
    """C1–C6 (authors' text; DECISIONS_PENDING items 8 and 10) with evidence and automatic evidence checks.
    Returns (lines, {C: holds})."""
    L, V = [], {}
    nm = {e: EVAL_NAME[e].replace("\\_", "_") for e in CLASS_EVALS}
    # ---------------- C1
    D = F["datasets"]
    L += ["## C1. Protocol feasibility", "",
          "*Claim (authors):* On day-scheduled captures (D1, D3, D5), class hold-out (LOACO) is structurally limited: D5 "
          "purged_block yields 1 feasible fold; temporal_gap LOACO yields 0 folds on D1 and D3. Natural temporal novelty "
          "is available where LOACO is not.", "",
          "*Evidence* (Feasibility in the appendix; `T_feasibility.tex`; F1, F2; `P0_REPORT.md`, `P1D_REPORT.md`):", ""]
    for ds, r in D.items():
        nn = F["natural_novelty"].get(f"nn_{ds}", {}).get("classes", [])
        L.append(f"- {ds}: feasible LOACO folds temporal_gap {r['loaco_temporal_gap']}, grouped_random {r['loaco_grouped_random']}"
                 + (f", families (grouped_random) {r['loaco_family_grouped_random']}" if "loaco_family_grouped_random" in r else "")
                 + f"; natural-novelty classes {len(nn)}" + (f" ({', '.join(nn)})" if nn else "") + ".")
    pb = F["d5_purged_block"]
    L.append(f"- D5 purged_block: {len(pb['feasible'])} feasible fold ({', '.join(pb['feasible'])}) of "
             f"{pb['blocks']['non_empty']} ten-minute blocks ({pb['blocks']['purged_rule1'] + pb['blocks']['purged_rule2']} purged); "
             f"A5.2 fallback applied: {F['a52_fallback_applied']}.")
    L.append("")
    nncls = {ds: len(F["natural_novelty"].get(f"nn_{ds}", {}).get("classes", [])) for ds in D}
    lines, V["C1"] = verdict([
        ("D5 purged_block yields 1 feasible fold", len(pb["feasible"]) == 1, ", ".join(pb["feasible"])),
        ("temporal_gap LOACO yields 0 folds on D1 and D3", D["D1"]["loaco_temporal_gap"] == 0 and D["D3"]["loaco_temporal_gap"] == 0,
         f"D1 {D['D1']['loaco_temporal_gap']}, D3 {D['D3']['loaco_temporal_gap']} (D5 {D['D5']['loaco_temporal_gap']})"),
        ("natural novelty is available on D1, D3, D5, where temporal LOACO is not",
         all(nncls[ds] > 0 and D[ds]["loaco_temporal_gap"] == 0 for ds in ("D1", "D3", "D5")),
         ", ".join(f"{ds} {nncls[ds]} classes" for ds in ("D1", "D3", "D5")))])
    L += lines + [f"Note: LOACO stays feasible under grouped_random on D1 ({D['D1']['loaco_grouped_random']}) and D3 "
                  f"({D['D3']['loaco_grouped_random']}); D2 has {D['D2']['loaco_temporal_gap']} temporal_gap folds.", ""]
    # ---------------- C2
    L += ["## C2. No detector family dominates", "",
          "*Claim (authors, revised):* No novelty-detector family dominates across datasets and protocols: "
          "representation-distance scores lead under LOACO D2 and natural novelty D5; benign-only detectors lead under "
          "natural novelty D3 (+0.376 [+0.25, +0.50]). Under natural novelty D1 all families are weak (best: "
          "representation 0.623 [0.562, 0.692]). The supervised reference (XGBoost) exceeds the leading novelty family in "
          "LOACO D2, natural novelty D3 and natural novelty D5. Single-protocol evaluations can therefore crown different "
          "winners, and dedicated novelty scores do not reliably beat a plain supervised classifier on held-out or new "
          "attacks.", "",
          "*Evidence* (R1–R3; `T_families.tex`, `T_auc_*.tex`; F3). Family mean AUC [95% bootstrap CI]; novelty families "
          "= confidence, representation, benign-only, CLAD; XGBoost is the supervised reference:", "",
          "| Evaluation | classes | confidence | representation | benign-only | CLAD | XGBoost | leading novelty family |",
          "|---|---|---|---|---|---|---|---|"]
    lead = {}
    for e in CLASS_EVALS:
        st = CS[e]
        lead[e] = max(NOVELTY_FAMS, key=lambda f: st["families"][f]["mean"])
        L.append(f"| {nm[e]} | {st['n_classes']} | " + " | ".join(_ci(st["families"][f]) for f in FAMILIES) + f" | {lead[e]} |")
    L += ["", "Wilkie et al. CLAD vs each family (CLAD − family, paired bootstrap):", "",
          "| Evaluation | CLAD | − confidence | − representation | − benign-only | − XGBoost |", "|---|---|---|---|---|---|"]
    for e in CLASS_EVALS:
        st = CS[e]
        L.append(f"| {nm[e]} | {_ci(st['families']['CLAD'])} | " + " | ".join(_ci(st["clad_vs"][f], True)
                                                                          for f in ("confidence", "representation", "benign-only", "XGBoost")) + " |")
    L.append("")
    fam = lambda e, f: CS[e]["families"][f]  # noqa: E731
    c3 = CS["nn_D3"]["contrasts"]["benign-only − representation"]
    d1 = CS["nn_D1"]["families"]
    d1_best = max(d1, key=lambda f: d1[f]["mean"])
    xgb_wins = [e for e in CLASS_EVALS if fam(e, "XGBoost")["mean"] > fam(e, lead[e])["mean"]]
    leads = sorted({lead[e] for e in CLASS_EVALS})
    lines, V["C2"] = verdict([
        ("no single novelty family leads everywhere", len(leads) > 1, f"leading families across evaluations: {', '.join(leads)}"),
        ("representation leads under LOACO D2", lead["loaco_D2"] == "representation", f"{fam('loaco_D2', 'representation')['mean']:.3f}"),
        ("representation leads under natural novelty D5", lead["nn_D5"] == "representation", f"{fam('nn_D5', 'representation')['mean']:.3f}"),
        ("benign-only leads under natural novelty D3 by +0.376 [+0.25, +0.50] over representation",
         lead["nn_D3"] == "benign-only" and round(c3["mean"], 3) == 0.376 and round(c3["ci95"][0], 2) == 0.25
         and round(c3["ci95"][1], 2) == 0.50, _ci(c3, True)),
        ("natural novelty D1: best family is representation 0.623 [0.562, 0.692]",
         d1_best == "representation" and _ci(d1["representation"]) == "0.623 [0.562, 0.692]", _ci(d1[d1_best])),
        ("XGBoost exceeds the leading novelty family in LOACO D2, natural novelty D3 and natural novelty D5",
         all(e in xgb_wins for e in ("loaco_D2", "nn_D3", "nn_D5")),
         "; ".join(f"{nm[e]} {fam(e, 'XGBoost')['mean']:.3f} vs {fam(e, lead[e])['mean']:.3f}" for e in ("loaco_D2", "nn_D3", "nn_D5"))),
        ("novelty scores do not reliably beat XGBoost", len(xgb_wins) >= len(CLASS_EVALS) / 2,
         f"XGBoost ≥ leading novelty family in {len(xgb_wins)} of {len(CLASS_EVALS)} evaluations "
         f"({', '.join(nm[e] for e in xgb_wins)})")])
    L += lines
    # ---------------- C3
    lo = [e for e in LOACO_EVALS if e != "loaco_D5pb"]

    def pooled(evals, d):
        return float(np.mean([CS[e]["cells"][d][c]["std"] for e in evals for c in CS[e]["classes"]]))
    L += ["## C3. Natural novelty exposes instability that LOACO hides", "",
          "*Claim (authors):* Seed std of every supervised score is higher under natural novelty than under LOACO; benign-only "
          "detectors stay stable throughout.", "",
          "*Evidence* (R7; `T_seed_std.tex`; F4). Mean seed std of AUC over classes, LOACO (D1 families, D2, D3) vs natural "
          "novelty (D1, D3, D5):", "", "| Detector | family | LOACO | natural novelty | ratio |", "|---|---|---|---|---|"]
    for fm, dets in FAMILIES.items():
        for d in dets:
            a, b = pooled(lo, d), pooled(NN_EVALS, d)
            L.append(f"| {lab(d)} | {fm} | {a:.4f} | {b:.4f} | {(f'{b / a:.1f}' if a > 1e-6 else 'n/a (deterministic)')} |")
    L.append("")
    sup = CONF + REPR
    higher = [d for d in sup if pooled(NN_EVALS, d) > pooled(lo, d)]
    bmax = max(max(pooled(lo, d), pooled(NN_EVALS, d)) for d in BEN)
    smin = min(pooled(NN_EVALS, d) for d in sup)
    det0 = [d for d in BEN if pooled(lo, d) < 1e-6 and pooled(NN_EVALS, d) < 1e-6]
    lines, V["C3"] = verdict([
        ("every supervised score has higher seed std under natural novelty", len(higher) == len(sup), f"{len(higher)}/{len(sup)}"),
        ("benign-only detectors stay stable (seed std below every supervised score's natural-novelty std)", bmax < smin,
         f"benign-only max {bmax:.4f} vs supervised min {smin:.4f}")])
    L += lines + [f"Note: CLAD also rises ({pooled(lo, CLAD_):.4f} → {pooled(NN_EVALS, CLAD_):.4f}). "
                  + (f"{', '.join(det0)} are deterministic (seed std 0 by construction); the seeded benign-only detectors "
                     "(IF, AE) carry the stability claim." if det0 else ""), ""]
    # ---------------- C4
    ev4 = ["loaco_D2", "loaco_D3", "nn_D3", "nn_D5"]
    L += ["## C4. Confidence-magnitude scores are invalid novelty scores", "",
          "*Claim (authors, revised):* Confidence-magnitude scores (MSP, Energy) are invalid novelty scores for binary NIDS: "
          "below 0.5 in LOACO D2/D3 and natural novelty D3/D5, because confidently detected attacks read as \"normal\". The "
          "classifier's own attack probability P(attack) is not affected in the mean (0.999, 0.752, 0.853, 0.575 in the "
          "same evaluations) and falls below 0.5 only on individual classes (D3 LOACO ddos 0.35, scanning 0.46; D5 "
          "natural-novelty portscan 0.16). Novelty scoring for binary NIDS should use P(attack) or representation "
          "distance, not max-softmax or energy.", "",
          "*Evidence* (R1, R2; `T_families.tex`, `T_auc_*.tex`; A6.2). Mean AUC over classes:", "",
          "| Evaluation | MSP | Energy | P(attack)† | representation family | P(attack)† classes < 0.5 |", "|---|---|---|---|---|---|"]
    pc = {}
    for e in ev4:
        st = CS[e]
        pc[e] = {c: round(st["cells"]["P_attack"][c]["mean"], 2) for c in st["classes"] if st["cells"]["P_attack"][c]["mean"] < 0.5}
        L.append(f"| {nm[e]} | " + " | ".join(f"{_det_mean(st, d):.3f}" for d in CONF) + f" | {_ci(fam(e, 'representation'))} | "
                 + (", ".join(f"{c} {v:.2f}" for c, v in pc[e].items()) or "none") + " |")
    L.append("")
    pmeans = [round(_det_mean(CS[e], "P_attack"), 3) for e in ev4]
    expect_pc = {"loaco_D2": {}, "loaco_D3": {"ddos": 0.35, "scanning": 0.46}, "nn_D3": {}, "nn_D5": {"portscan": 0.16}}
    lines, V["C4"] = verdict([
        ("MSP below 0.5 in all four evaluations", all(_det_mean(CS[e], "MSP") < 0.5 for e in ev4),
         ", ".join(f"{_det_mean(CS[e], 'MSP'):.3f}" for e in ev4)),
        ("Energy below 0.5 in all four evaluations", all(_det_mean(CS[e], "Energy") < 0.5 for e in ev4),
         ", ".join(f"{_det_mean(CS[e], 'Energy'):.3f}" for e in ev4)),
        ("P(attack) mean AUC 0.999, 0.752, 0.853, 0.575 (none below 0.5)",
         pmeans == [0.999, 0.752, 0.853, 0.575], ", ".join(f"{v:.3f}" for v in pmeans)),
        ("P(attack) below 0.5 only on D3 LOACO ddos 0.35, scanning 0.46 and D5 natural-novelty portscan 0.16",
         pc == expect_pc, "; ".join(f"{nm[e]}: " + (", ".join(f"{c} {v:.2f}" for c, v in pc[e].items()) or "none") for e in ev4)),
        ("the recommended alternatives (P(attack), representation distance) have mean AUC above 0.5 in all four",
         all(_det_mean(CS[e], "P_attack") > 0.5 and fam(e, "representation")["mean"] > 0.5 for e in ev4),
         "representation " + ", ".join(f"{fam(e, 'representation')['mean']:.3f}" for e in ev4))])
    L += lines + [f"Note: on natural novelty D3 the representation family is {_ci(fam('nn_D3', 'representation'))}; its CI "
                  "includes 0.5, so there it is only marginally above chance.", ""]
    # ---------------- C5
    def au(e, d, k="test"):
        return I[e]["best"][d][k]["auc"]
    L += ["## C5. Temporal and cross-dataset generalization", "",
          "*Claim (authors, revised):* Temporal-split effects are dataset-specific. Relative to grouped_random: D1 — both AHSD "
          "P(attack) and XGBoost drop similarly (0.29, 0.27); D2 — neither drops; D5 — both rise; D3 — only the neural "
          "backbone degrades (P(attack) 0.66, seed std 0.20) while XGBoost holds 0.98. This is one backbone (stated as a "
          "limitation). Zero-shot cross-dataset transfer: 67% of 108 off-diagonal cells fall in 0.3–0.6 AUC.", "",
          "*Evidence* (R5 Temporal-split degradation, R4; `T_saturation.tex`, `T_transfer.tex`; F5). Limitation: a single "
          "neural backbone (AHSD).", "",
          "| Dataset | P(attack)† grouped_random | P(attack)† temporal_gap | XGBoost grouped_random | XGBoost temporal_gap | "
          "drop P(attack) | drop XGBoost |", "|---|---|---|---|---|---|---|"]
    drops = {}
    for ds in ("D1", "D2", "D3", "D5"):
        g, t = f"indist_{ds}_grouped_random", f"indist_{ds}_temporal_gap"
        dp = au(g, "P_attack")["mean"] - au(t, "P_attack")["mean"]
        dx = au(g, XGB)["mean"] - au(t, XGB)["mean"]
        drops[ds] = (dp, dx)
        L.append(f"| {ds} | " + " | ".join(f"{au(e, d)['mean']:.3f} (sd {au(e, d)['std']:.3f})"
                                           for e, d in ((g, "P_attack"), (t, "P_attack"), (g, XGB), (t, XGB)))
                 + f" | {dp:+.3f} | {dx:+.3f} |")
    off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
    frac = float(np.mean((off >= 0.3) & (off <= 0.6)))
    L += ["", f"Zero-shot transfer, all {len(off)} off-diagonal (detector × source × target) mean AUCs: "
          f"{frac:.0%} in [0.3, 0.6], {np.mean(off < 0.3):.0%} below 0.3, {np.mean(off > 0.6):.0%} above 0.6; median "
          f"{np.median(off):.3f}.", ""]
    TOL = 0.05  # "drops"/"holds"/"neither drops" = change larger / not larger than 0.05 AUC
    t3 = au("indist_D3_temporal_gap", "P_attack")
    lines, V["C5"] = verdict([
        ("D1: both drop similarly (0.29, 0.27)", round(drops["D1"][0], 2) == 0.29 and round(drops["D1"][1], 2) == 0.27,
         f"{drops['D1'][0]:+.3f} / {drops['D1'][1]:+.3f}"),
        (f"D2: neither drops (change ≤ {TOL})", abs(drops["D2"][0]) <= TOL and abs(drops["D2"][1]) <= TOL,
         f"{drops['D2'][0]:+.3f} / {drops['D2'][1]:+.3f}"),
        ("D5: both rise under temporal_gap", drops["D5"][0] < 0 and drops["D5"][1] < 0, f"{drops['D5'][0]:+.3f} / {drops['D5'][1]:+.3f}"),
        ("D3: only the neural backbone degrades — P(attack) 0.66 (seed std 0.20), XGBoost 0.98",
         drops["D3"][0] > TOL and abs(drops["D3"][1]) <= TOL and round(t3["mean"], 2) == 0.66 and round(t3["std"], 2) == 0.20
         and round(au("indist_D3_temporal_gap", XGB)["mean"], 2) == 0.98,
         f"P(attack) {t3['mean']:.3f} (sd {t3['std']:.3f}), XGBoost {au('indist_D3_temporal_gap', XGB)['mean']:.3f}"),
        ("transfer: 67% of 108 off-diagonal cells in 0.3–0.6 AUC", len(off) == 108 and round(frac * 100) == 67,
         f"{frac:.1%} of {len(off)}")])
    L += lines
    # ---------------- C6
    g = json.loads((ROOT / "results/p1/gate.json").read_text())
    b = json.loads((ROOT / "results/p1b/summary.json").read_text())
    c = json.loads((ROOT / "results/p1c/summary.json").read_text())
    d = json.loads((ROOT / "results/p1d/summary.json").read_text())
    am = json.loads((ROOT / "docs/AMENDMENTS.json").read_text())["amendments"]
    G = d["gate"]["stats"]
    L += ["## C6. Pre-registered negative results and release", "",
          "*Claim (authors):* Pre-registered negative results: spectral detectability (P1), adaptive-equilibrium collapse not "
          "reproduced (G1), drift re-anchoring (G2), benign-only superiority (G3). Released pipeline, splits, amendments and "
          "provenance.", "", "*Evidence* (`NEGATIVE_RESULTS.md`; `release/`):", "",
          f"- P1 ({g['outcome']}): pooled ρ_S2 = {g['primary_oriented']['rho_S2']:.3f} (needs ≥ 0.6); ρ_S2 − ρ_W 95% CI "
          f"[{g['primary_oriented']['diff_ci95'][0]:.3f}, {g['primary_oriented']['diff_ci95'][1]:.3f}].",
          f"- G1 ({b['g1']['outcome']}): {b['g1']['datasets_passing']} of 3 datasets pass; ρ(γ, d′) = "
          + " / ".join(f"{v['rho_gamma_dprime']:.3f}" for v in b["g1"]["datasets"].values()) + " (D1 / D2 / D3; needs ≤ −0.8).",
          f"- G2 ({c['gate']['outcome']}): detectors passing (a) {c['gate']['detectors_passing']['a']}/9, (b) "
          f"{c['gate']['detectors_passing']['b']}/9, (c) re-anchoring {c['gate']['detectors_passing']['c']}/9 (5 needed each).",
          f"- G3 ({d['gate']['outcome']}): D5 natural novelty B − S = {G['D5_nn']['B_minus_S']:.3f}; D3 LOACO B − S = "
          f"{G['D3_loaco']['B_minus_S']:.3f} (needs ≤ 0.05); criterion (b) on {d['gate']['criteria']['b_evaluated_on']} (A5.2).",
          f"- Release: {len(am)} dated, hashed amendments (`docs/AMENDMENTS.json`); splits, file hashes and checklist in "
          "`release/`; every final-run number from one code commit.", ""]
    rel = (ROOT / "release/splits/MANIFEST.json").exists() and (ROOT / "release/CHECKLIST.md").exists()
    lines, V["C6"] = verdict([
        ("P1, G1, G2, G3 are all negative (STOP) as recorded",
         [g["outcome"], b["g1"]["outcome"], c["gate"]["outcome"], d["gate"]["outcome"]] == ["STOP"] * 4, "STOP ×4"),
        ("G1: adaptive-equilibrium collapse not reproduced (criterion (a) fails on every dataset)",
         not any(v["criterion_a"] for v in b["g1"]["datasets"].values()), "criterion (a) 0/3"),
        ("release package present (splits manifest, checklist) and amendments dated and hashed",
         rel and all(a.get("new_sha256") and a.get("date") for a in am), f"{len(am)} amendments")])
    L += lines + ["Note: the G3 verdict stands as recorded although A6.2 records a flaw in its S definition.", ""]
    return L, V


def results_summary(S, CS, I, T, F, commit):
    L = ["# RESULTS_SUMMARY (final benchmark run)", "",
         "Generated by `scripts/report_final.py` from `results/final` JSON only. AMENDMENT_06: descriptive benchmark; "
         "no gate. Seeds 17/23/42/101/202; mean ± Student-t 95% CI (n = 5, t = 2.776); seed std reported explicitly; "
         "family contrasts use the G3 paired bootstrap (classes × seeds, 2,000 resamples, seed 0). Scores: higher = "
         "more anomalous, never flipped. † = added post-G3.", "",
         "Organised by the authors' contributions C1–C6 (`DECISIONS_PENDING.md` item 8, decided 2026-10-09). Each "
         "contribution states the authors' claim, the evidence with its R-section and table/figure, and an automatic "
         "**evidence check** wherever the JSON qualifies or contradicts the wording. Every number is computed from "
         "`results/final` (or, for C6, the recorded gate JSON); the full per-evaluation results follow in the appendix "
         "(R1–R7).", "",
         f"Single code commit for every number: `{commit}`. RQ SHA-256 in every run: `{S['rq_sha256']}`.", "",
         "## Caveats (apply throughout)", ""] + [f"- {c}" for c in caveats(F, CS)] + [""]
    cl, verdicts = contributions(S, CS, I, T, F)
    S["contribution_checks"] = verdicts
    L += ["## Evidence checks", "", "| Contribution | evidence check |", "|---|---|"] + [
        f"| {k} | {'holds' if v else 'does NOT hold'} |" for k, v in verdicts.items()] + [""]
    L += cl
    L += ["# Appendix: results by evaluation (R1–R7)", ""]

    def fam_line(e):
        st = CS[e]
        fams = "; ".join(f"{f} {st['families'][f]['mean']:.3f} [{st['families'][f]['ci95'][0]:.3f}, {st['families'][f]['ci95'][1]:.3f}]"
                         for f in FAMILIES)
        cons = "; ".join(f"{k} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for k, v in st["contrasts"].items())
        return [f"**{EVAL_NAME[e].replace(chr(92) + '_', '_')}** ({st['n_classes']} classes: {', '.join(st['classes'])})", "",
                f"- Family mean AUC [95% bootstrap CI]: {fams}.", f"- Contrasts: {cons}.", ""]

    def det_table(e):
        st = CS[e]
        H = ["| Detector | family | " + " | ".join(st["classes"]) + " | seed std (mean) |",
             "|---|---|" + "---|" * (len(st["classes"]) + 1)]
        for fam, dets in FAMILIES.items():
            for d in dets:
                H.append(f"| {lab(d)} | {fam} | " + " | ".join(
                    f"{_f(st['cells'][d][c]['mean'])} ± {_f(st['cells'][d][c]['ci95'], 2)} (sd {_f(st['cells'][d][c]['std'], 2)}; "
                    f"<0.5: {st['cells'][d][c]['frac_below_0.5']:.1f})" for c in st["classes"]) + f" | {_f(st['seed_std'][d])} |")
        return H + [""]

    L += ["## R1 LOACO (grouped_random): D1 families, D2, D3", ""]
    for e in ("loaco_D1fam", "loaco_D2", "loaco_D3"):
        L += fam_line(e) + det_table(e)
    L += ["## R2 Natural novelty (train-tail validation): D1, D3, D5", ""]
    for e in NN_EVALS:
        L += fam_line(e) + det_table(e)
    L += ["## R3 D5 purged_block (single fold; limitation)", ""] + fam_line("loaco_D5pb") + det_table("loaco_D5pb")
    L += ["## R4 Zero-shot transfer (AUC; rows source, columns target; diagonal in-domain test)", "",
          "| Detector | source | D1 | D2 | D3 | D4 |", "|---|---|---|---|---|---|"]
    for d in DETS:
        for src in TRANSFER_TARGETS:
            L.append(f"| {lab(d)} | {src} | " + " | ".join(
                f"{_f(T[d][src][t]['mean'])} ± {_f(T[d][src][t]['ci95'], 2)} (sd {_f(T[d][src][t]['std'], 2)})"
                if t in T[d][src] else "—" for t in ("D1", "D2", "D3", "D4")) + " |")
    L += ["", "## R5 Temporal-split degradation: AHSD vs XGBoost", "",
          "Limitation: a single neural backbone (AHSD, P(attack)†) is compared with XGBoost; no other backbone was built "
          "(authors' decision). XGBoost is deterministic under the fixed parameters (seed std 0).", "",
          "| Dataset / split | model | test set | AUC | PR-AUC | macro-F1 | MCC | TPR@1%FPR |", "|---|---|---|---|---|---|---|---|"]
    for e in IND_EVALS:
        for d in ("P_attack", XGB):
            for k in ("test", "test_natural"):
                m = I[e]["best"][d][k]
                L.append(f"| {e.replace('indist_', '')} | {lab(d)} | {k} | " + " | ".join(
                    f"{_f(m[x]['mean'])} ± {_f(m[x]['ci95'], 3)} (sd {_f(m[x]['std'], 3)})"
                    for x in ("auc", "pr_auc", "macro_f1", "mcc", "tpr_at_1pct_fpr")) + " |")
    L += ["", "In-distribution AUC on `test` for every detector (mean ± CI):", "",
          "| Detector | " + " | ".join(e.replace("indist_", "") for e in IND_EVALS) + " |", "|---|" + "---|" * len(IND_EVALS)]
    for d in DETS:
        L.append(f"| {lab(d)} | " + " | ".join(f"{_f(I[e]['best'][d]['test']['auc']['mean'])} ± {_f(I[e]['best'][d]['test']['auc']['ci95'], 2)}"
                                              for e in IND_EVALS) + " |")
    L += ["", "## R6 Checkpoint sensitivity (best validation vs final epoch; same training run)", "",
          "| Evaluation | family | best | final | final − best [95% CI] |", "|---|---|---|---|---|"]
    for e in CLASS_EVALS:
        for fam in FAM_CK:
            ck = CS[e]["checkpoint"][fam]
            fb = ck["final_minus_best"]
            L.append(f"| {e} | {fam} | {_f(ck['best'])} | {_f(ck['final'])} | {fb['mean']:+.3f} [{fb['ci95'][0]:+.3f}, {fb['ci95'][1]:+.3f}] |")
    L += ["", "Benign-only minus backbone family, at each checkpoint of the backbone:", "",
          "| Evaluation | contrast | best | final |", "|---|---|---|---|"]
    for e in CLASS_EVALS:
        for fam in ("confidence", "representation"):
            c = CS[e]["checkpoint"][f"benign-only − {fam}"]
            L.append(f"| {e} | benign-only − {fam} | {c['best']['mean']:+.3f} [{c['best']['ci95'][0]:+.3f}, {c['best']['ci95'][1]:+.3f}] | "
                     f"{c['final']['mean']:+.3f} [{c['final']['ci95'][0]:+.3f}, {c['final']['ci95'][1]:+.3f}] |")
    L += ["", "In-distribution P(attack)† AUC on `test`, best vs final:", "", "| Dataset / split | best | final |", "|---|---|---|"]
    for e in IND_EVALS:
        L.append(f"| {e.replace('indist_', '')} | {_f(I[e]['best']['P_attack']['test']['auc']['mean'])} | "
                 f"{_f(I[e]['final']['P_attack']['test']['auc']['mean'])} |")
    L += ["", "## R7 Seed variability (mean seed std of AUC over classes)", "",
          "| Detector | " + " | ".join(CLASS_EVALS) + " |", "|---|" + "---|" * len(CLASS_EVALS)]
    for d in DETS:
        L.append(f"| {lab(d)} | " + " | ".join(_f(CS[e]["seed_std"][d]) for e in CLASS_EVALS) + " |")
    L += ["", "## Feasibility", "", "| Dataset | flows | attack classes | LOACO gr | LOACO tg | LOACO families | natural novelty |",
          "|---|---|---|---|---|---|---|"]
    for ds, r in F["datasets"].items():
        L.append(f"| {ds} | {r['flows']:,} | {r['classes']} | {r['loaco_grouped_random']} | {r['loaco_temporal_gap']} | "
                 f"{r.get('loaco_family_grouped_random', '—')} | {', '.join(F['natural_novelty'].get(f'nn_{ds}', {}).get('classes', [])) or '—'} |")
    L += ["", "Infeasible LOACO classes (reason):", ""]
    for e, r in F["loaco"].items():
        L.append(f"- {e}: " + ("; ".join(f"{c}: {why}" for c, why in r["infeasible"].items()) or "none"))
    L += ["", "## Outputs", "", "- Tables: `report/final/tables/*.tex`.",
          "- Figures: `report/final/figures/` F1_protocols, F2_feasibility, F3_auc_heatmap, F4_seed_std, F5_transfer, F6_checkpoint (PDF).",
          "- Machine-readable: `results/final/summary.json`.", ""]
    return "\n".join(L)


def negative_results():
    g = json.loads((ROOT / "results/p1/gate.json").read_text())
    s1 = json.loads((ROOT / "results/p1/summary.json").read_text())
    b = json.loads((ROOT / "results/p1b/summary.json").read_text())
    c = json.loads((ROOT / "results/p1c/summary.json").read_text())
    d = json.loads((ROOT / "results/p1d/summary.json").read_text())
    L = ["# NEGATIVE_RESULTS", "",
         "Generated by `scripts/report_final.py` from the recorded gate JSON of each phase. These are historical "
         "records with their own code commits; none of their numbers enters the final-run tables. The gate verdicts "
         "remain as recorded (AMENDMENT_06 A6.2).", "",
         "| Phase | Gate | Outcome | Source |", "|---|---|---|---|",
         f"| P1 | S2 predictor gate | **{g['outcome']}** | `results/p1/gate.json`, `GATE_DECISION.md` |",
         f"| P1b | G1 | **{b['g1']['outcome']}** | `results/p1b/summary.json`, `GATE_G1_DECISION.md` |",
         f"| P1c | G2 | **{c['gate']['outcome']}** | `results/p1c/summary.json`, `GATE_G2_DECISION.md` |",
         f"| P1d | G3 | **{d['gate']['outcome']}** | `results/p1d/summary.json`, `GATE_G3_DECISION.md` |", ""]
    # P1
    L += ["## P1: spectral detectability predictor (RQ1)", "",
          "Rule: GO iff ρ_S2 ≥ 0.6 and the lower 95% bound of ρ_S2 − ρ_W > 0 (pooled D1 families + D2 + D3 LOACO, "
          "paired bootstrap 2,000).", "", "| Variant | ρ_S2 | ρ_W | ρ_S2 − ρ_W | 95% CI | n | outcome |", "|---|---|---|---|---|---|---|"]
    for k in ("primary_oriented", "raw_sign_variant"):
        v = g[k]
        L.append(f"| {k} | {v['rho_S2']:.3f} | {v['rho_W']:.3f} | {v['diff']:.3f} | [{v['diff_ci95'][0]:.3f}, {v['diff_ci95'][1]:.3f}] | "
                 f"{v['n_points']} | {v['outcome']} |")
    L += ["", "Spearman ρ per predictor (oriented by the pre-registered sign) [bootstrap 95% CI]:", "",
          "| Set | predictor | ρ | 95% CI | n |", "|---|---|---|---|---|"]
    for setk, preds in s1["correlations"].items():
        for p, v in preds.items():
            L.append(f"| {setk} | {p} | {v['oriented_rho']:.3f} | [{v['oriented_ci95'][0]:.3f}, {v['oriented_ci95'][1]:.3f}] | {v['n']} |")
    L += ["", "Pooled points (D_pred, Wasserstein, measured stress AUC):", "", "| Evaluation | class | D_pred | Wasserstein | AUC |",
          "|---|---|---|---|---|"]
    for p in g["points"]:
        L.append(f"| {p['eval_id']} | {p['class']} | {p['D_pred']:.3f} | {p['wasserstein']:.3f} | {p['auc_stress']:.3f} |")
    L += ["", f"RQ-file decision rule on the pooled set: \"{s1['decision']}\". P1 code commit "
          f"`{s1['provenance_runs']['code_commit']}`.", ""]
    # G1
    L += ["## P1b: G1 (γ sweep and S1; RQ3/RQ4)", "",
          "Rule: GO iff on ≥ 2 of 3 datasets (a) Spearman(γ, mean test d′) ≤ −0.8 and (b) S1 d′ ≥ 0.8 × fixed and "
          "macro-F1 drop < 1 point.", "", "| Dataset | ρ(γ, d′) | (a) | S1 selected | S1/fixed d′ | macro-F1 drop | (b) |",
          "|---|---|---|---|---|---|---|"]
    for ds, v in b["g1"]["datasets"].items():
        L.append(f"| {ds} | {v['rho_gamma_dprime']:.3f} | {'pass' if v['criterion_a'] else 'fail'} | {v['s1_selected']} | "
                 f"{v['s1_dprime_ratio']:.3f} | {v['s1_macro_f1_drop']:.2f} | {'pass' if v['criterion_b'] else 'fail'} |")
    L += ["", "Test d′ (stress) per frozen γ, mean ± t-CI over seeds:", "",
          "| Dataset | " + " | ".join(f"γ={x}" for x in b["g1"]["gammas"]) + " |", "|---|" + "---|" * len(b["g1"]["gammas"])]
    for ds, v in b["g1"]["datasets"].items():
        L.append(f"| {ds} | " + " | ".join(f"{s['d_prime']['mean']:.3f} ± {s['d_prime']['ci95']:.3f}" for s in v["sweep"]) + " |")
    L += ["", f"G1 code commit `{b['code_commits']['P1b-G1']}`.", ""]
    # G2
    gg = c["gate"]
    L += ["## P1c: G2 (natural novelty, benign drift, re-anchoring)", "",
          f"Detectors passing: (a) {gg['detectors_passing']['a']}/9, (b) {gg['detectors_passing']['b']}/9, "
          f"(c) {gg['detectors_passing']['c']}/9 (5 needed each).", "",
          "| Detector | (a) classes with AUC < 0.5 | (b) median d′ | (c) classes ok |", "|---|---|---|---|"]
    for det_, v in gg["per_detector"].items():
        L.append(f"| {det_} | {len(v['a_inverted_classes'])} ({', '.join(v['a_inverted_classes'])}) | {v['b_median_dprime']:.3f} | "
                 f"{v['c_classes_ok']}/{v['c_needed']}{'' if v['c_applicable'] else ' (n/a)'} |")
    L += ["", "Re-anchoring (N = 500) AUC change per class:", "", "| Detector | dataset/class | none | N500 | gain |", "|---|---|---|---|---|"]
    for det_, v in gg["per_detector"].items():
        if v["c_applicable"]:
            for x in v["c_detail"]:
                L.append(f"| {det_} | {x['dataset']}/{x['class']} | {x['auc_none']:.3f} | {x['auc_N500']:.3f} | {x['gain']:+.3f} |")
    L += ["", f"P1c code commit `{c['code_commit']}`.", ""]
    # G3
    G = d["gate"]["stats"]
    L += ["## P1d: G3 (H-REV)", "", "| Evaluation | classes | B | S | B − S | 95% CI | Wilkie CLAD |", "|---|---|---|---|---|---|---|"]
    for k, v in G.items():
        L.append(f"| {k} | {len(v['classes'])} | {v['B']:.3f} | {v['S']:.3f} | {v['B_minus_S']:.3f} | "
                 f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] | {v.get('W', float('nan')):.3f} |")
    cr = d["gate"]["criteria"]
    sn = d["a53_sensitivity_D5_nn"]
    L += ["", f"Criteria: (a) {cr['a']}; (b) {cr['b']} (on {cr['b_evaluated_on']}, A5.2 fallback "
          f"{d['a52']['fallback_applied']}); (c) {cr['c']}. A5.3 final-epoch sensitivity B − S = {sn['B_minus_S']:.3f} "
          f"[{sn['ci95'][0]:.3f}, {sn['ci95'][1]:.3f}].",
          "", "Recorded gate flaw (A6.2): S mixes classifier-confidence scores (MSP, Energy), which read confident attack "
          "predictions as normal, with representation-distance scores. The verdict stands as recorded.",
          "", f"P1d code commit `{d['code_commits']['P1d']}`.", ""]
    return "\n".join(L)


def main():
    runs, commit, a6 = load()
    A = class_tensor(runs)
    folds, expect = check_complete(A, runs)
    CS = class_eval_stats(A, expect)
    I, iinfo = indist_stats(runs)
    T = transfer_matrix(I)
    F = feasibility(folds, runs)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tables(CS, I, T, F)
    fig_protocols(F)
    fig_feasibility(F)
    fig_heatmap(CS)
    fig_seed_std(CS)
    fig_transfer(T)
    fig_checkpoint(CS)
    S = {"code_commit": commit, "rq_sha256": a6["new_sha256"], "n_runs": len(runs),
         "n_runs_by_eval": {e["id"]: sum(1 for r in runs if r["eval_id"] == e["id"]) for e in EVALS},
         "class_evals": CS, "indist": I, "indist_info": iinfo, "transfer": T, "feasibility": F,
         "caveats": None, "provenance": provenance()}
    S["caveats"] = caveats(F, CS)
    MD_SUMMARY.write_text(results_summary(S, CS, I, T, F, commit))  # also records S["contribution_checks"]
    write_json(RES / "summary.json", S)
    MD_NEG.write_text(negative_results())
    print(f"final report: {len(runs)} runs, commit {commit}; tables {len(list(TAB.glob('*.tex')))}, "
          f"figures {len(list(FIG.glob('*.pdf')))}")


if __name__ == "__main__":
    main()
