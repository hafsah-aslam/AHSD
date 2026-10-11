#!/usr/bin/env python
"""Generate the paper's figures from results JSON only (NOMS 2027).

fig_pipeline.pdf    methodology diagram: inputs -> processing -> detectors -> evaluation/outputs
fig_families.pdf    C2: family mean AUC with 95% bootstrap intervals, per evaluation
fig_seedstd.pdf     C3: seed std of AUC, LOACO vs natural novelty, per detector
fig_confidence.pdf  C4: MSP/Energy vs P(attack) and representation distance
fig_shift.pdf       C5: grouped vs temporal in-distribution AUC + zero-shot transfer distribution

Palette: validated categorical order (dataviz reference palette, slots 1-5); every series also
carries a distinct marker and a direct label or legend, so identity never rests on colour alone.

  python paper/noms2027/make_figures.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "scripts"))
from final_run import TRANSFER_TARGETS  # noqa: E402

S = json.loads((ROOT / "results/final/summary.json").read_text())
CS, I, T, F = S["class_evals"], S["indist"], S["transfer"], S["feasibility"]

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
C = {"confidence": "#2a78d6", "representation": "#eb6834", "benign-only": "#1baf7a", "CLAD": "#eda100",
     "XGBoost": "#e87ba4"}
MK = {"confidence": "o", "representation": "s", "benign-only": "D", "CLAD": "^", "XGBoost": "v"}
FAMS = list(C)
EV = ["loaco_D1fam", "loaco_D2", "loaco_D3", "loaco_D5pb", "nn_D1", "nn_D3", "nn_D5"]
EVN = {"loaco_D1fam": "LOACO D1 (fam.)", "loaco_D2": "LOACO D2", "loaco_D3": "LOACO D3", "loaco_D5pb": "Purged LOACO D5",
       "nn_D1": "Natural nov. D1", "nn_D3": "Natural nov. D3", "nn_D5": "Natural nov. D5"}
DETN = {"MSP": "MSP", "Energy": "Energy", "P_attack": "P(attack)$^\\dagger$", "AHSD_stress": "Stress",
        "Mahalanobis": "Mahalanobis", "kNN": "kNN", "IF": "Isol. Forest", "AE": "Autoencoder", "Wilkie_CLAD": "CLAD"}
FAM_OF = {**{d: "confidence" for d in ("MSP", "Energy", "P_attack")},
          **{d: "representation" for d in ("AHSD_stress", "Mahalanobis", "kNN")},
          **{d: "benign-only" for d in ("IF", "OCSVM", "PCA", "AE")}, "Wilkie_CLAD": "CLAD", "XGBoost": "XGBoost"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
COL_W = 3.45


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def chance(ax, orient="v"):
    (ax.axvline if orient == "v" else ax.axhline)(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)


# ------------------------------------------------------------------ pipeline diagram
def fig_pipeline():
    meta = {ds: json.loads((ROOT / f"data/processed/{ds}/index_meta.json").read_text()) for ds in ("D1", "D2", "D3", "D4", "D5")}
    aud = {ds: json.loads((ROOT / f"audit/{ds}.json").read_text())["timestamp_range_ms"] for ds in meta}
    day = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%b %Y")  # noqa: E731
    name = {"D1": "CSE-CIC-IDS2018", "D2": "UNSW-NB15", "D3": "ToN-IoT", "D4": "BoT-IoT", "D5": "LYCOS-IDS2017"}
    fig = plt.figure(figsize=(7.16, 2.95))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 41)
    ax.axis("off")
    panels = [(0.6, "1  Inputs", "#e8f0fb", "#2a78d6"), (25.6, "2  Leakage-controlled processing", "#fdeee7", "#eb6834"),
              (50.6, "3  Backbone & 12 detectors", "#e6f6ef", "#1baf7a"), (75.6, "4  Protocols & outputs", "#fdf3dd", "#c98500")]
    W = 23.8
    for x, title, fill, edge in panels:
        ax.add_patch(FancyBboxPatch((x, 0.8), W, 39.2, boxstyle="round,pad=0,rounding_size=1.2", fc=fill, ec=edge, lw=0.9))
        ax.text(x + 1.2, 37.6, title, fontsize=8, weight="bold", color=INK, va="center")
    for x in (24.4, 49.4, 74.4):
        ax.add_patch(FancyArrowPatch((x, 20.4), (x + 1.2, 20.4), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=1))

    def box(x, y, w, h, head, body, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=ec, lw=0.6))
        ax.text(x + 0.8, y + h - 0.7, head, fontsize=6.3, weight="bold", color=INK, va="top")
        if body:
            ax.text(x + 0.8, y + h - 2.55, body, fontsize=5.4, color=INK2, va="top", linespacing=1.2)

    # 1 inputs
    y = 31.2
    for ds in ("D1", "D2", "D3", "D4", "D5"):
        m = meta[ds]
        box(1.6, y, 21.8, 5.0, f"{ds}  {name[ds]}",
            f"{m['flows'] / 1e6:.1f}M flows · {len(m['classes']) - 1} attack classes · {day(aud[ds][0])}", "#2a78d6")
        y -= 6.0
    ax.text(1.8, 3.6, "Official sources, SHA-256 manifest.\nD1–D4: NetFlow v3, 53 features;\nD5: own 73-feature space.",
            fontsize=5.6, color=INK2, va="center", linespacing=1.25)
    # 2 processing
    steps = [("Audit & drop", "identifiers, TTL leakage; ±inf → NaN"),
             ("Clean (train statistics only)", "median impute · log(1+x) · standardise"),
             ("Sort & group in time", "1-h blocks (D5 purged: 10-min blocks)"),
             ("Window & label", "T = 32 flows, stride 16, never cross blocks"),
             ("Deduplicate globally", "SHA-256 per window (D3 −9.7%, D4 −15.8%)"),
             ("Split & balance", "grouped / temporal / purged · 1:1 train, 5:1 test"),
             ("Fail-closed checks", "row & block disjointness, balance, features")]
    y = 31.6
    for i, (h, b) in enumerate(steps):
        box(26.6, y, 21.8, 4.3, f"{i + 1}. {h}", b, "#eb6834")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((37.5, y), (37.5, y - 0.7), arrowstyle="-|>", mutation_scale=4, color=MUTED, lw=0.6))
        y -= 5.0
    # 3 models
    box(51.6, 27.0, 21.8, 9.2, "Recurrent backbone (AHSD, fixed E)",
        "z = LayerNorm(Wx), gated state h_t around\nbenign mean E; stress = mean|h − E|;\nMLP head → 2 logits (best val. macro-F1)", "#1baf7a")
    fam_rows = [("confidence", "Confidence", "MSP · Energy · P(attack)†"),
                ("representation", "Representation", "Stress · Mahalanobis · kNN"),
                ("benign-only", "Benign-only", "IF · OCSVM · PCA · AE"),
                ("CLAD", "Contrastive", "CLAD (benign + known attacks)"),
                ("XGBoost", "Supervised ref.", "XGBoost on flattened windows")]
    y = 21.6
    for f, h, b in fam_rows:
        ax.add_patch(FancyBboxPatch((51.6, y), 21.8, 4.0, boxstyle="round,pad=0,rounding_size=0.6", fc="white", ec=C[f], lw=0.9))
        ax.plot([53.0], [y + 2.0], marker=MK[f], color=C[f], ms=4.2, mec="white", mew=0.5)
        ax.text(54.4, y + 2.75, h, fontsize=6.2, weight="bold", color=INK, va="center")
        ax.text(54.4, y + 1.15, b, fontsize=5.6, color=INK2, va="center")
        y -= 4.5
    ax.text(51.8, 1.9, "5 seeds · best & final-epoch checkpoints", fontsize=5.6, color=INK2, va="center")
    # 4 outputs
    nfold = sum(len(F["loaco"][e]["feasible"]) for e in ("loaco_D1fam", "loaco_D2", "loaco_D3"))
    nn = sum(len(F["natural_novelty"][e]["classes"]) for e in ("nn_D1", "nn_D3", "nn_D5"))
    outs = [("LOACO (grouped)", f"{nfold} held-out classes: D1 fam., D2, D3"),
            ("Natural temporal novelty", f"{nn} test-only classes: D1, D3, D5"),
            ("Purged block LOACO", f"D5: {len(F['d5_purged_block']['feasible'])} feasible fold"),
            ("In-distribution & transfer", "4 datasets × 2 splits; 9 zero-shot pairs"),
            ("Metrics", "AUC · seed std · t-CI · class×seed bootstrap"),
            ("Findings (C1–C5)", "seed instability · score inversion ·\nranking flips · supervised ref. · prereg.")]
    y = 31.6
    for i, (h, b) in enumerate(outs):
        hh = 6.0 if i == len(outs) - 1 else 4.3
        if i == len(outs) - 1:
            y -= 1.2
        box(76.6, y, 21.8, hh, h, b, "#c98500")
        y -= 5.0
    fig.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C2 families
def fig_families():
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    ys = np.arange(len(EV))[::-1]
    off = np.linspace(-0.3, 0.3, len(FAMS))
    for k, f in enumerate(FAMS):
        m = [CS[e]["families"][f]["mean"] for e in EV]
        lo = [CS[e]["families"][f]["ci95"][0] for e in EV]
        hi = [CS[e]["families"][f]["ci95"][1] for e in EV]
        y = ys + off[k]
        ax.hlines(y, lo, hi, color=C[f], lw=1.0, zorder=2)
        ax.scatter(m, y, marker=MK[f], s=16, color=C[f], edgecolor="white", lw=0.5, zorder=3,
                   label={"benign-only": "Benign-only", "confidence": "Confidence", "representation": "Representation",
                          "CLAD": "CLAD", "XGBoost": "XGBoost (sup. ref.)"}[f])
    ax.axhline(2.5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_ylim(-0.6, 6.55)
    chance(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels([EVN[e] for e in EV])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Family mean AUC (5 seeds; bars: 95% bootstrap interval)")
    style(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_families.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C3 seed std
def fig_seedstd():
    lo_ev, nn_ev = ["loaco_D1fam", "loaco_D2", "loaco_D3"], ["nn_D1", "nn_D3", "nn_D5"]

    def pooled(evals, d):
        return float(np.mean([CS[e]["cells"][d][c]["std"] for e in evals for c in CS[e]["classes"]]))
    dets = ["MSP", "Energy", "P_attack", "AHSD_stress", "Mahalanobis", "kNN", "Wilkie_CLAD", "IF", "AE"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    ys = np.arange(len(dets))[::-1]
    for y, d in zip(ys, dets):
        a, b = pooled(lo_ev, d), pooled(nn_ev, d)
        col = C[FAM_OF[d]]
        ax.plot([a, b], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.scatter([a], [y], s=18, facecolor="white", edgecolor=col, lw=1.0, zorder=3, marker=MK[FAM_OF[d]])
        ax.scatter([b], [y], s=18, color=col, edgecolor="white", lw=0.5, zorder=3, marker=MK[FAM_OF[d]])
        ax.text(max(a, b) + 0.006, y, f"{b / a:.1f}×", fontsize=6, color=INK2, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([DETN[d] for d in dets])
    ax.set_xlim(0, 0.25)
    ax.set_xlabel("Mean seed std of AUC over classes (open: LOACO; filled: natural novelty)")
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_seedstd.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C4 confidence
def fig_confidence():
    ev = ["loaco_D2", "loaco_D3", "nn_D3", "nn_D5"]
    bars = [("MSP", "MSP", "#2a78d6", None), ("Energy", "Energy", "#2a78d6", "////"),
            ("P_attack", "P(attack)$^\\dagger$", "#6250d6", None), ("repr", "Representation", C["representation"], None)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    x = np.arange(len(ev))
    w = 0.19
    for k, (d, lab, col, hatch) in enumerate(bars):
        if d == "repr":
            v = [CS[e]["families"]["representation"]["mean"] for e in ev]
        else:
            v = [float(np.mean([CS[e]["cells"][d][c]["mean"] for c in CS[e]["classes"]])) for e in ev]
        ax.bar(x + (k - 1.5) * w, v, w * 0.9, color=col if hatch is None else "white", edgecolor=col,
               hatch=hatch, lw=0.8, label=lab, zorder=2)
    chance(ax, "h")
    ax.text(-0.48, 0.515, "chance", fontsize=6, color=MUTED, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([EVN[e].replace("Natural nov.", "Nat. nov.") for e in ev])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean AUC over classes")
    style(ax, "y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_confidence.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C5 temporal + transfer
def fig_shift():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw={"width_ratios": [1.15, 1]})
    dsc = {"D1": "#2a78d6", "D2": "#eb6834", "D3": "#1baf7a", "D5": "#6250d6"}
    for ds, col in dsc.items():
        for d, ls, mk in (("P_attack", "-", "o"), ("XGBoost", (0, (2, 1.5)), "v")):
            g = I[f"indist_{ds}_grouped_random"]["best"][d]["test"]["auc"]["mean"]
            t = I[f"indist_{ds}_temporal_gap"]["best"][d]["test"]["auc"]["mean"]
            a1.plot([0, 1], [g, t], color=col, ls=ls, lw=1.1, marker=mk, ms=3.2, mec="white", mew=0.4)
        a1.text(1.07, I[f"indist_{ds}_temporal_gap"]["best"]["P_attack"]["test"]["auc"]["mean"], ds, fontsize=6,
                color=INK2, va="center")
    a1.plot([], [], color=INK2, lw=1.1, marker="o", ms=3, label="AHSD P(attack)$^\\dagger$")
    a1.plot([], [], color=INK2, lw=1.1, ls=(0, (2, 1.5)), marker="v", ms=3, label="XGBoost")
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["grouped", "temporal"])
    a1.set_xlim(-0.15, 1.3)
    a1.set_ylim(0.45, 1.02)
    a1.set_ylabel("In-distribution AUC")
    a1.legend(loc="lower left", frameon=False, fontsize=5.6, handlelength=1.8)
    style(a1, "y")
    off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
    a2.hist(off, bins=np.arange(0, 1.0001, 0.05), color=C["confidence"], edgecolor="white", lw=0.6, zorder=2)
    a2.axvspan(0.3, 0.6, color=GRID, alpha=0.6, zorder=1)
    chance(a2)
    a2.text(0.98, a2.get_ylim()[1] * 0.97, f"{np.mean((off >= .3) & (off <= .6)):.0%} in shaded\nband [0.3, 0.6]", fontsize=6,
            color=INK, ha="right", va="top")
    a2.set_xlabel(f"Zero-shot transfer AUC (n = {len(off)})")
    a2.set_ylabel("Count")
    a2.set_xlim(0, 1)
    style(a2, "y")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "fig_shift.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_pipeline, fig_families, fig_seedstd, fig_confidence, fig_shift):
        f()
    print("figures:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
