#!/usr/bin/env python
"""Generate the paper's figures from results JSON only (NOMS 2027).

fig_pipeline.pdf    methodology diagram: inputs -> processing -> detectors -> evaluation/outputs
fig_families.pdf    C2: family mean AUC with 95% bootstrap intervals, per evaluation
fig_seedstd.pdf     C3: seed std of AUC, LOACO vs natural novelty, per detector
fig_confidence.pdf  C4: MSP/Energy vs P(attack) and representation distance
fig_shift.pdf       C5: grouped vs temporal in-distribution AUC + zero-shot transfer distribution
fig_rolling.pdf     AMENDMENT_07: family mean AUC per rolling cut-point, per dataset

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

S = json.loads((ROOT / "results/final_v2/summary.json").read_text())
CS, I, T, F = S["class_evals"], S["indist"], S["transfer"], S["feasibility"]
A7 = S["a7"]

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
C = {"confidence": "#2a78d6", "representation": "#eb6834", "benign-only": "#1baf7a", "CLAD": "#eda100",
     "XGBoost": "#e87ba4"}
MK = {"confidence": "o", "representation": "s", "benign-only": "D", "CLAD": "^", "XGBoost": "v"}
FAMS = list(C)
EV = ["loaco_D1fam", "loaco_D2", "loaco_D3", "loaco_D5pb", "nn_D1", "nn_D3", "nn_D5"]
EVN = {"loaco_D1fam": "LOACO D1 (fam.)", "loaco_D2": "LOACO D2", "loaco_D3": "LOACO D3", "loaco_D5pb": "Purged LOACO D5",
       "nn_D1": "Natural nov. D1", "nn_D3": "Natural nov. D3", "nn_D5": "Natural nov. D5"}
DETN = {"MSP": "MSP", "Energy": "Energy", "P_attack": "P(attack)$^\\dagger$", "AHSD_stress": "Stress",
        "Mahalanobis": "Mahalanobis", "kNN": "kNN", "IF": "Isol. Forest", "AE": "Autoencoder", "Wilkie_CLAD": "CLAD",
        "GRU_MSP": "GRU MSP", "GRU_Energy": "GRU Energy", "GRU_P_attack": "GRU P(attack)$^\\dagger$",
        "GRU_Mahalanobis": "GRU Mahalanobis", "GRU_kNN": "GRU kNN", "XGBoost_s": "XGBoost-s"}
FAM_OF = {**{d: "confidence" for d in ("MSP", "Energy", "P_attack")},
          **{d: "representation" for d in ("AHSD_stress", "Mahalanobis", "kNN")},
          **{d: "benign-only" for d in ("IF", "OCSVM", "PCA", "AE")}, "Wilkie_CLAD": "CLAD", "XGBoost": "XGBoost",
          **{d: "confidence" for d in ("GRU_MSP", "GRU_Energy", "GRU_P_attack")},
          **{d: "representation" for d in ("GRU_Mahalanobis", "GRU_kNN")}, "XGBoost_s": "XGBoost"}

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
    """Four-panel methodology overview: data -> protocols -> models -> evaluation & outputs (numbers from JSON)."""
    meta = {ds: json.loads((ROOT / f"data/processed/{ds}/index_meta.json").read_text()) for ds in ("D1", "D2", "D3", "D4", "D5")}
    aud = {ds: json.loads((ROOT / f"audit/{ds}.json").read_text())["timestamp_range_ms"] for ds in meta}
    yr = lambda ms: dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y")  # noqa: E731
    name = {"D1": "CSE-CIC-IDS2018", "D2": "UNSW-NB15", "D3": "ToN-IoT", "D4": "BoT-IoT", "D5": "LYCOS-IDS2017"}
    nfold = sum(len(F["loaco"][e]["feasible"]) for e in ("loaco_D1fam", "loaco_D2", "loaco_D3"))
    nn = sum(len(F["natural_novelty"][e]["classes"]) for e in ("nn_D1", "nn_D3", "nn_D5"))
    nflows = sum(m["flows"] for m in meta.values())

    fig = plt.figure(figsize=(7.16, 3.25))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)
    ax.axis("off")
    PAN = [("1", "Traffic & data", f"{len(meta)} public captures, {nflows / 1e6:.0f}M flows, time-ordered", "#2a78d6", "#eef4fc"),
           ("2", "Novelty protocols", "Where the unseen attack comes from", "#1baf7a", "#ecf8f2"),
           ("3", "Backbones & detectors", "2 recurrent backbones, 18 detectors, 5 families", "#0f8fa8", "#e8f5f8"),
           ("4", "Evaluation & outputs", "5 seeds per detector, one code commit", "#6250d6", "#f1effc")]
    W, X0 = 23.6, [0.5, 25.5, 50.5, 75.5]

    def rbox(x, y, w, h, fc, ec, lw=0.7, r=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))

    def card(x, y, w, h, head, body, ec, fs=6.0, bfs=5.3):
        rbox(x, y, w, h, "white", ec, 0.6, 0.6)
        ax.text(x + 0.8, y + h - 0.65, head, fontsize=fs, weight="bold", color=INK, va="top")
        if body:
            ax.text(x + 0.8, y + h - 2.35, body, fontsize=bfs, color=INK2, va="top", linespacing=1.2)

    for (num, title, sub, col, fill), x in zip(PAN, X0):
        rbox(x, 0.5, W, 44.0, fill, col, 1.0, 1.2)
        ax.add_patch(plt.Circle((x + 2.4, 41.6), 1.55, color=col, zorder=3))
        ax.text(x + 2.4, 41.55, num, fontsize=8, weight="bold", color="white", ha="center", va="center", zorder=4)
        ax.text(x + 4.6, 41.6, title, fontsize=8.2, weight="bold", color=col, va="center")
        ax.text(x + 1.0, 38.6, sub, fontsize=5.6, color=INK2, va="center", style="italic")
    for x in (24.15, 49.15, 74.15):
        ax.add_patch(FancyArrowPatch((x, 22.5), (x + 1.3, 22.5), arrowstyle="-|>", mutation_scale=10, color=INK2, lw=1.1))

    # ---- 1 traffic & data
    c1, x = PAN[0][3], X0[0] + 1.0
    y = 33.6
    for ds in ("D1", "D2", "D3", "D4", "D5"):
        m = meta[ds]
        rbox(x, y, W - 2.0, 3.5, "white", c1, 0.5, 0.5)
        ax.text(x + 0.7, y + 2.4, f"{ds}", fontsize=6.0, weight="bold", color=c1, va="center")
        ax.text(x + 3.4, y + 2.4, name[ds], fontsize=6.0, weight="bold", color=INK, va="center")
        ax.text(x + 3.4, y + 0.9, f"{m['flows'] / 1e6:.1f}M flows · {len(m['classes']) - 1} attack cl. · {yr(aud[ds][0])}"
                + (" · transfer" if ds == "D4" else ""), fontsize=5.0, color=INK2, va="center")
        y -= 3.85
    ax.text(x, y + 2.7, "D1–D4 NetFlow v3, 41 features; D5: 73", fontsize=5.0, color=INK2, va="center")
    ax.text(x, y + 0.6, "Leakage-controlled processing", fontsize=6.0, weight="bold", color=INK, va="center")
    steps = ["Drop IDs & TTL", "Train-only stats", "1-h time blocks", "Window T = 32", "SHA-256 dedup", "Fail-closed checks"]
    cw1 = (W - 2.6) / 2
    for i, s_ in enumerate(steps):
        xx, yy = x + (i % 2) * (cw1 + 0.6), y - 1.6 - (i // 2) * 3.0
        rbox(xx, yy - 2.4, cw1, 2.4, "#f7fafe", c1, 0.4, 0.4)
        ax.text(xx + 0.6, yy - 1.2, f"{i + 1}", fontsize=5.4, weight="bold", color=c1, va="center")
        ax.text(xx + 1.9, yy - 1.2, s_, fontsize=5.1, color=INK, va="center")

    # ---- 2 protocols
    c2, x = PAN[1][3], X0[1] + 1.0
    cw = (W - 2.6) / 2
    for k, (head, sub, items, ec, fc) in enumerate([
            ("LOACO", "artificial novelty", ["Remove one\nattack class", "Train on\nthe rest", "Test on the\nheld-out class"], "#c98500", "#fff7e6"),
            ("Natural novelty", "temporal", ["Train on\nearlier traffic", "Skip one\ngap block", "Test classes\nfirst seen later"], c2, "#e6f6ef")]):
        xx = x + k * (cw + 0.6)
        rbox(xx, 18.6, cw, 18.4, fc, ec, 0.7, 0.6)
        ax.text(xx + cw / 2, 35.5, head, fontsize=6.4, weight="bold", color=INK, ha="center", va="center")
        ax.text(xx + cw / 2, 33.6, sub, fontsize=5.2, color=INK2, ha="center", va="center", style="italic")
        yy = 31.6
        for j, it in enumerate(items):
            rbox(xx + 0.5, yy - 3.0, cw - 1.0, 3.0, "white", ec, 0.5, 0.4)
            ax.text(xx + cw / 2, yy - 1.5, it, fontsize=5.0, color=INK, ha="center", va="center", linespacing=1.05)
            ax.text(xx + 1.0, yy - 0.6, f"{j + 1}", fontsize=4.8, weight="bold", color=ec, va="center")
            if j < 2:
                ax.add_patch(FancyArrowPatch((xx + cw / 2, yy - 3.0), (xx + cw / 2, yy - 3.8), arrowstyle="-|>",
                                             mutation_scale=5, color=MUTED, lw=0.6))
            yy -= 3.8
        ax.text(xx + cw / 2, 19.8, f"{nfold} + 1 folds" if k == 0 else f"{nn} classes", fontsize=5.4,
                weight="bold", color=ec, ha="center", va="center")
    nroll = A7["feasibility"]["n_units"]
    card(x, 11.6, W - 2.0, 6.2, "Rolling cut-points (4 cuts)",
         f"f = 0.30 · 0.45 · 0.60 · 0.75 of each dataset\n{nroll} (dataset, cut, class) units, {A7['feasibility']['n_unique']} classes", c2)
    card(x, 6.3, W - 2.0, 4.6, "Purged block LOACO (D5)", "10-min blocks, neighbours purged", "#c98500")
    card(x, 1.3, W - 2.0, 4.4, "In-distribution & transfer", "2 split schemes · 9 zero-shot pairs", MUTED)

    # ---- 3 backbones & detectors
    c3, x = PAN[2][3], X0[2] + 1.0
    rbox(x, 29.4, W - 2.0, 7.8, "white", c3, 0.7, 0.6)
    ax.text(x + 0.8, 36.4, "Backbones (trained per seed)", fontsize=6.0, weight="bold", color=INK, va="top")
    for k, (bn, bd) in enumerate([("AHSD", "gated state around\nfixed benign mean"), ("GRU", "1 layer,\nhidden 128")]):
        xx = x + 0.6 + k * 10.6
        rbox(xx, 30.0, 10.0, 4.4, "#e8f5f8", c3, 0.5, 0.4)
        ax.text(xx + 5.0, 33.4, bn, fontsize=6.0, weight="bold", color=c3, ha="center", va="center")
        ax.text(xx + 5.0, 31.4, bd, fontsize=4.9, color=INK2, ha="center", va="center", linespacing=1.1)
    fams = [("confidence", "Classifier confidence  ×2", "MSP · Energy · P(attack)†"),
            ("representation", "Representation distance  ×2", "Mahalanobis · kNN · stress (AHSD)"),
            ("benign-only", "Benign-only", "Isolation Forest · OCSVM · PCA · AE"),
            ("CLAD", "Contrastive", "CLAD (benign + known attacks)"),
            ("XGBoost", "Supervised reference", "XGBoost fixed · XGBoost-s seeded")]
    y = 24.6
    for f, h, b in fams:
        rbox(x, y, W - 2.0, 4.1, "white", C[f], 0.9, 0.5)
        ax.plot([x + 1.3], [y + 2.05], marker=MK[f], color=C[f], ms=4.4, mec="white", mew=0.5)
        ax.text(x + 2.7, y + 2.85, h, fontsize=6.0, weight="bold", color=INK, va="center")
        ax.text(x + 2.7, y + 1.2, b, fontsize=5.2, color=INK2, va="center")
        y -= 4.7
    ax.annotate("", xy=(x + 11, 28.75), xytext=(x + 11, 29.4), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.6))
    ax.text(x + 0.2, 2.6, "Scores oriented in advance (higher = attack);\nbest and final-epoch checkpoints",
            fontsize=5.0, color=INK2, va="center", linespacing=1.2)

    # ---- 4 evaluation & outputs
    c4, x = PAN[3][3], X0[3] + 1.0
    rbox(x, 24.0, W - 2.0, 13.2, "white", c4, 0.7, 0.6)
    ax.text(x + 0.8, 36.4, "Metrics per detector × class × seed", fontsize=6.0, weight="bold", color=INK, va="top")
    chips = ["ROC AUC", "TPR @ 1% FPR", "PR-AUC", "seed std", "t-interval", "class × seed\nbootstrap"]
    for i, ch in enumerate(chips):
        xx, yy = x + 0.6 + (i % 3) * 7.05, 30.0 - (i // 3) * 5.2
        rbox(xx, yy, 6.6, 4.3, "#f1effc", c4, 0.5, 0.4)
        ax.text(xx + 3.3, yy + 2.15, ch, fontsize=5.2, color=INK, ha="center", va="center", linespacing=1.05)
    rbox(x, 1.3, W - 2.0, 21.9, "white", c4, 0.7, 0.6)
    ax.text(x + 0.8, 22.4, "Findings", fontsize=6.0, weight="bold", color=INK, va="top")
    finds = [("C1", "Confidence scores misaligned", "MSP/energy invert in binary NIDS"),
             ("C2", "Rankings shift", "with protocol, dataset and cut-point"),
             ("C3", "Hidden instability", "representation scores vary by seed"),
             ("C4", "AUC overstates coverage", "low TPR at 1% false-positive rate"),
             ("C5", "Pre-registration", "4 failed hypotheses reported")]
    y = 18.6
    for k, h, b in finds:
        ax.add_patch(plt.Circle((x + 1.7, y + 0.45), 1.15, color=c4, zorder=3))
        ax.text(x + 1.7, y + 0.42, k, fontsize=4.9, weight="bold", color="white", ha="center", va="center", zorder=4)
        ax.text(x + 3.4, y + 1.05, h, fontsize=5.8, weight="bold", color=INK, va="center")
        ax.text(x + 3.4, y - 0.45, b, fontsize=5.1, color=INK2, va="center")
        y -= 3.55
    fig.savefig(OUT / "fig_pipeline.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C2 families
def fig_families():
    fig, ax = plt.subplots(figsize=(COL_W, 3.0))
    rows = EV + ["rolling"]
    src = {**{e: A7["evals"][e]["families"] for e in EV}, "rolling": A7["rolling"]["pooled"]["families"]}
    ys = np.arange(len(rows))[::-1]
    off = np.linspace(-0.3, 0.3, len(FAMS))
    for k, f in enumerate(FAMS):
        m = [src[e][f]["mean"] for e in rows]
        lo = [src[e][f]["ci95"][0] for e in rows]
        hi = [src[e][f]["ci95"][1] for e in rows]
        y = ys + off[k]
        ax.hlines(y, lo, hi, color=C[f], lw=1.0, zorder=2)
        ax.scatter(m, y, marker=MK[f], s=16, color=C[f], edgecolor="white", lw=0.5, zorder=3,
                   label={"benign-only": "Benign-only", "confidence": "Confidence", "representation": "Representation",
                          "CLAD": "CLAD", "XGBoost": "XGBoost (sup. ref.)"}[f])
    ax.axhline(3.5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
    ax.set_ylim(-0.6, 7.55)
    chance(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels([EVN.get(e, f"Rolling cuts ({A7['feasibility']['n_units']} units)") for e in rows])
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

    R = A7["seed_std_ratio"]
    dets = ["MSP", "Energy", "P_attack", "AHSD_stress", "Mahalanobis", "kNN", "GRU_MSP", "GRU_Energy", "GRU_P_attack",
            "GRU_Mahalanobis", "GRU_kNN", "Wilkie_CLAD", "XGBoost_s", "IF", "AE"]
    fig, ax = plt.subplots(figsize=(COL_W, 3.15))
    ys = np.arange(len(dets))[::-1]
    for y, d in zip(ys, dets):
        a, b = R[d]["loaco"], R[d]["nn"]
        assert abs(R[d]["nn_ratio"] - b / a) < 1e-9
        col = C[FAM_OF[d]]
        ax.plot([a, b], [y, y], color=col, lw=1.4, solid_capstyle="round", zorder=2)
        ax.scatter([a], [y], s=18, facecolor="white", edgecolor=col, lw=1.0, zorder=3, marker=MK[FAM_OF[d]])
        ax.scatter([b], [y], s=18, color=col, edgecolor="white", lw=0.5, zorder=3, marker=MK[FAM_OF[d]])
        ax.text(max(a, b) + 0.006, y, f"{b / a:.1f}×", fontsize=6, color=INK2, va="center")
    ax.set_yticks(ys)
    ax.set_yticklabels([DETN[d] for d in dets])
    ax.set_xlim(0, 0.25)
    ax.axhline(8.5, color=MUTED, lw=0.5, ls=(0, (1, 1.5)))
    ax.axhline(3.5, color=MUTED, lw=0.5, ls=(0, (1, 1.5)))
    ax.set_xlabel("Mean seed std of AUC (open: LOACO; filled: natural novelty)")
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_seedstd.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C4 confidence
def fig_confidence():
    ev = ["loaco_D2", "loaco_D3", "nn_D3", "nn_D5", "rolling"]
    bars = [("MSP", "MSP (AHSD)", "#2a78d6", None), ("GRU_MSP", "MSP (GRU)", "#2a78d6", "////"),
            ("P_attack", "P(attack)$^\\dagger$ (AHSD)", "#6250d6", None), ("GRU_P_attack", "P(attack) (GRU)", "#6250d6", "////")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    x = np.arange(len(ev))
    w = 0.19
    cells = lambda e: A7["rolling"]["pooled"]["cells"] if e == "rolling" else A7["evals"][e]["cells"]  # noqa: E731
    for k, (d, lab, col, hatch) in enumerate(bars):
        v = [float(np.mean(list(cells(e)[d].values()))) for e in ev]
        ax.bar(x + (k - 1.5) * w, v, w * 0.9, color=col if hatch is None else "white", edgecolor=col,
               hatch=hatch, lw=0.8, label=lab, zorder=2)
    chance(ax, "h")
    ax.text(-0.48, 0.515, "chance", fontsize=6, color=MUTED, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([EVN.get(e, "Rolling (45)").replace("Natural nov.", "Nat. nov.").replace(" ", "\n", 1) for e in ev])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean AUC over classes")
    style(ax, "y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_confidence.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ C5 temporal + transfer
def fig_shift():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
    models = [("P_attack", "AHSD", C["confidence"], "o", "full"), ("GRU_P_attack", "GRU", C["confidence"], "s", "none"),
              ("XGBoost", "XGBoost", C["XGBoost"], "v", "full")]
    dss = ["D1", "D2", "D3", "D5"]
    y = 0
    yt, yl = [], []
    for ds in dss:
        for d, lab_, col, mk, fs in models:
            g = I[f"indist_{ds}_grouped_random"]["best"][d]["test"]["auc"]["mean"]
            t = I[f"indist_{ds}_temporal_gap"]["best"][d]["test"]["auc"]["mean"]
            a1.annotate("", xy=(t, -y), xytext=(g, -y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                                                        mutation_scale=6, shrinkA=0, shrinkB=0))
            a1.scatter([g], [-y], s=10, marker=mk, facecolor="white", edgecolor=col, lw=0.8, zorder=3)
            y += 1
        yt.append(-(y - 2))
        yl.append(ds)
        y += 0.8
    a1.set_yticks(yt)
    a1.set_yticklabels(yl)
    a1.set_xlim(0.45, 1.02)
    a1.set_xlabel("AUC: grouped → temporal split")
    for d, lab_, col, mk, fs in models:
        a1.plot([], [], color=col, marker=mk, ms=3.2, lw=1, mfc=col if fs == "full" else "white", label=lab_)
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=5.5, handlelength=1.3,
              columnspacing=0.8, borderaxespad=0.1)
    style(a1)
    off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
    a2.hist(off, bins=np.arange(0, 1.0001, 0.05), color=C["confidence"], edgecolor="white", lw=0.6, zorder=2)
    a2.axvspan(0.3, 0.6, color=GRID, alpha=0.6, zorder=1)
    chance(a2)
    a2.set_ylim(0, a2.get_ylim()[1] * 1.3)
    a2.text(0.02, 0.97, f"{np.mean((off >= .3) & (off <= .6)):.0%} in\n[0.3, 0.6]", fontsize=6,
            color=INK, ha="left", va="top", transform=a2.transAxes)
    a2.set_xlabel(f"Zero-shot transfer AUC (n = {len(off)})")
    a2.set_ylabel("Count")
    a2.set_xlim(0, 1)
    style(a2, "y")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(OUT / "fig_shift.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ AMENDMENT_07 rolling cut-points
def fig_rolling():
    dss = ["D1", "D3", "D5"]
    fig, axs = plt.subplots(1, 3, figsize=(COL_W, 1.85), sharey=True)
    for ax, ds in zip(axs, dss):
        evs = sorted((e for e in A7["evals"] if e.startswith(f"roll_{ds}_")), key=lambda e: e[-3:])
        xs = [int(e[-3:]) / 100 for e in evs]
        for f in FAMS:
            ax.plot(xs, [A7["evals"][e]["families"][f]["mean"] for e in evs], color=C[f], marker=MK[f], ms=3.2, lw=1.0,
                    mec="white", mew=0.4, label=f)
        chance(ax, "h")
        n = A7["rolling"]["by_dataset"][ds]["n_units"]
        ax.set_title(f"{ds} ({n} units)", pad=2)
        ax.set_xticks([0.3, 0.45, 0.6, 0.75])
        ax.set_xticklabels([".30", ".45", ".60", ".75"])
        ax.set_xlim(0.25, 0.8)
        ax.set_ylim(0.2, 1.02)
        style(ax, "y")
    axs[0].set_ylabel("Family mean AUC")
    axs[1].set_xlabel("Cut fraction f (share of windows before the cut)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, ["Confidence", "Representation", "Benign-only", "CLAD", "XGBoost"], loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=0.6)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9), w_pad=0.3)
    fig.savefig(OUT / "fig_rolling.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ creative result figures (AMENDMENT_07)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from scipy.stats import norm  # noqa: E402

# diverging around chance: red arm = below 0.5 (inverted), neutral gray at 0.5, blue arm = detects
DIV = LinearSegmentedColormap.from_list("auc_div", ["#7d1d1d", "#e34948", "#f3b3ad", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
DNORM = TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=1.0)
ROLL = sorted((e for e in A7["evals"] if e.startswith("roll_")), key=lambda e: (e[5:7], e[-3:]))
FAM_ROWS = [("confidence", "Confidence"), ("representation", "Representation"), ("GRU-confidence", "GRU confidence"),
            ("GRU-representation", "GRU representation"), ("benign-only", "Benign-only"), ("CLAD", "CLAD (contrastive)"),
            ("XGBoost", "XGBoost"), ("XGBoost-s", "XGBoost-s")]
FAM_ROW_COLOR = {"GRU-confidence": C["confidence"], "GRU-representation": C["representation"], "XGBoost-s": C["XGBoost"]}


def _runs():
    out = []
    for p in sorted((ROOT / "results/final_v2").glob("*/*/seed*.json")):
        r = json.loads(p.read_text())
        if r["kind"] in ("loaco", "natural_novelty"):
            out.append(r)
    return out


def fig_heatmap():
    cols = EV + ROLL + ["pooled"]
    src = {**{e: A7["evals"][e] for e in EV + ROLL}, "pooled": A7["rolling"]["pooled"]}
    M = np.array([[src[e]["families"][f]["mean"] for e in cols] for f, _ in FAM_ROWS])
    fig, ax = plt.subplots(figsize=(7.16, 2.35))
    im = ax.imshow(M, cmap=DIV, norm=DNORM, aspect="auto")
    for j in range(M.shape[1]):  # white column gaps between cells
        ax.axvline(j + 0.5, color=SURF, lw=1.2)
    for i in range(M.shape[0]):
        ax.axhline(i + 0.5, color=SURF, lw=1.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.3,
                    color="white" if abs(v - 0.5) > 0.27 else INK)
    nov = [k for k, (f, _) in enumerate(FAM_ROWS) if f in ("confidence", "representation", "benign-only", "CLAD")]
    for j in range(M.shape[1]):  # leading AHSD novelty family per column
        i = max(nov, key=lambda k: M[k, j])
        ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=INK, lw=1.1))
    grp = [(0, 3, "LOACO"), (4, 6, "Natural novelty"), (7, 10, "Rolling D1"), (11, 13, "Rolling D3"), (14, 16, "Rolling D5"),
           (17, 17, "All")]
    for a, b, lab in grp:
        ax.annotate("", xy=(a - 0.42, -0.95), xytext=(b + 0.42, -0.95), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.text((a + b) / 2, -1.25, lab, ha="center", va="bottom", fontsize=6.2, color=INK, weight="bold")
    for x in (3.5, 6.5, 16.5):
        ax.axvline(x, color=INK2, lw=1.0)
    ax.axhline(3.5, color=INK2, lw=0.8)
    ax.axhline(5.5, color=INK2, lw=0.8)
    xl = ["D1 fam.", "D2", "D3", "D5 pb", "D1", "D3", "D5"] + [f".{e[-2:]}" if e[-3] == "0" else e[-3:] for e in ROLL] + ["45 units"]
    xl = [x if not x.startswith(".") else f"f=.{x[1:]}" for x in xl]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(xl, fontsize=5.6, rotation=0)
    ax.set_yticks(range(len(FAM_ROWS)))
    ax.set_yticklabels([n for _, n in FAM_ROWS], fontsize=6.2)
    for t, (f, _) in zip(ax.get_yticklabels(), FAM_ROWS):
        t.set_color(INK)
    ax.tick_params(length=0, pad=6)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012, ticks=[0.2, 0.35, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_visible(False)
    cb.set_label("family mean AUC  (red: below chance · blue: detects)", fontsize=5.8, color=INK2)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_heatmap.pdf")
    plt.close(fig)


def fig_confidence():
    """Each point: one (evaluation, class, backbone): x = P(attack) AUC, y = MSP AUC on the same models."""
    kinds = {"loaco": ("LOACO", C["CLAD"]), "nn": ("Natural novelty", C["benign-only"]), "roll": ("Rolling cut", C["confidence"])}
    fig, (ins, ax) = plt.subplots(1, 2, figsize=(COL_W, 2.45), gridspec_kw={"width_ratios": [1, 2.35]})
    ax.add_patch(Polygon([[0.5, 0], [1.02, 0], [1.02, 0.5], [0.5, 0.5]], closed=True, fc="#f9dedb", ec="none", zorder=0))
    ax.text(0.52, 0.02, "inverted: detected, but\nMSP reads it as normal", ha="left", va="bottom", fontsize=5.4, color="#7d1d1d")
    ax.plot([0, 1.02], [0, 1.02], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.04, 0.09, "MSP = P(attack)", rotation=45, fontsize=5.4, color=MUTED, rotation_mode="anchor")
    chance(ax)
    chance(ax, "h")
    for e, st in A7["evals"].items():
        k = "loaco" if e.startswith("loaco") else "nn" if e.startswith("nn") else "roll"
        for bb, mk in (("", "o"), ("GRU_", "s")):
            xs = [st["cells"][bb + "P_attack"][u] for u in st["cells"][bb + "P_attack"]]
            ys = [st["cells"][bb + "MSP"][u] for u in st["cells"][bb + "P_attack"]]
            ax.scatter(xs, ys, s=11 if mk == "o" else 9, marker=mk, color=kinds[k][1], edgecolor="white", lw=0.35,
                       alpha=0.9, zorder=3 if k != "roll" else 2)
    for k, (lab, col) in kinds.items():
        ax.scatter([], [], s=12, color=col, label=lab)
    ax.scatter([], [], s=12, marker="o", facecolor="white", edgecolor=INK2, label="AHSD")
    ax.scatter([], [], s=10, marker="s", facecolor="white", edgecolor=INK2, label="GRU")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("AUC of P(attack)")
    ax.set_ylabel("AUC of MSP (same model)", labelpad=1)
    ax.set_title("b) every class × evaluation × backbone", fontsize=6.3, loc="left", pad=3)
    style(ax, "both")
    ax.legend(loc="upper left", frameon=False, ncol=1, handletextpad=0.1, fontsize=5.4, borderaxespad=0.1, labelspacing=0.3)
    d = np.linspace(-6, 6, 200)
    ins.axvspan(2, 6, color="#f9dedb", zorder=0)
    ins.plot(d, 1 / (1 + np.exp(-d)), color="#6250d6", lw=1.3)
    ins.plot(d, 1 / (1 + np.exp(np.abs(d))), color=C["confidence"], lw=1.3)
    ins.text(-5.6, 0.86, "P(attack)\n" + r"$=\sigma(\Delta)$", fontsize=5.4, color="#6250d6", ha="left", va="top")
    ins.text(-5.6, 0.32, "MSP score\n" + r"$=\sigma(-|\Delta|)$", fontsize=5.4, color=C["confidence"], ha="left", va="top")
    ins.text(4, 0.62, "confident\nattack", fontsize=5.2, color="#7d1d1d", ha="center")
    ins.set_xlabel(r"logit margin $\Delta = f_1 - f_0$", labelpad=1)
    ins.set_ylabel("score", labelpad=1)
    ins.set_title("a) mechanism", fontsize=6.3, loc="left", pad=3)
    ins.set_xticks([-5, 0, 5])
    ins.set_yticks([0, 0.5, 1])
    ins.set_ylim(0, 1.02)
    style(ins, "y")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_confidence.pdf")
    plt.close(fig)


def fig_illusion():
    """AUC vs TPR at 1% FPR for every detector x natural-novelty unit (mean over seeds), with the binormal expectation."""
    acc = {}
    for r in _runs():
        if r["kind"] != "natural_novelty":
            continue
        for d, bc in r["ops"]["best"].items():
            for c, mm in bc.items():
                acc.setdefault((r["eval_id"], d, c), []).append((r["auc"]["best"][d][c], mm["tpr_at_1pct_fpr"]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    a = np.linspace(0.5, 0.9999, 300)
    ax.plot(a, norm.cdf(np.sqrt(2) * norm.ppf(a) - norm.ppf(0.99)), color=INK2, lw=0.9, ls=(0, (3, 1.5)), zorder=4)
    ax.plot([], [], color=INK2, lw=0.9, ls=(0, (3, 1.5)), label="binormal reference")
    ax.axhspan(0, 0.2, xmin=0.75, xmax=1.0, color="#f9dedb", zorder=0)
    pts = {f: [] for f in FAMS}
    for (e, d, c), v in acc.items():
        if len(v) != 5:
            raise SystemExit(f"{e}/{d}/{c}: {len(v)} seeds")
        pts[FAM_OF[d]].append((np.mean([x for x, _ in v]), np.mean([y for _, y in v])))
    lab = {"confidence": "Confidence", "representation": "Representation", "benign-only": "Benign-only", "CLAD": "CLAD",
           "XGBoost": "XGBoost"}
    for f in ["confidence", "representation", "benign-only", "CLAD", "XGBoost"]:
        P = np.array(pts[f])
        ax.scatter(P[:, 0], P[:, 1], s=7, marker=MK[f], color=C[f], alpha=0.75, edgecolor="none", label=lab[f], zorder=2)
    all_ = np.array([p for f in pts for p in pts[f]])
    hi = all_[all_[:, 0] >= 0.75]
    trap = int(np.sum((all_[:, 0] >= 0.75) & (all_[:, 1] < 0.2)))
    ax.text(0.02, 0.97, f"{trap} of {len(hi)} points with AUC ≥ 0.75\ncatch < 20% of attacks (shaded)",
            transform=ax.transAxes, fontsize=6.4, color="#7d1d1d", va="top", weight="bold", linespacing=1.25)
    chance(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("ROC AUC (natural novelty: primary splits and rolling cuts)")
    ax.set_ylabel("TPR at 1% FPR")
    style(ax, "both")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False, fontsize=5.6, handletextpad=0.1, markerscale=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_illusion.pdf")
    plt.close(fig)
    return {"n_points": int(len(all_)), "median_tpr_auc_ge_075": float(np.median(hi[:, 1])), "n_auc_ge_075": int(len(hi)),
            "n_trap": trap}


def fig_seedstd():
    """Every seed drawn: AUC of each seed per D3 class, LOACO vs natural novelty, for kNN (both backbones) and IF."""
    vals = {}
    for r in _runs():
        if r["eval_id"] in ("loaco_D3", "nn_D3"):
            for d in ("kNN", "GRU_kNN", "IF"):
                for c, v in r["auc"]["best"][d].items():
                    vals.setdefault((r["eval_id"], c, d), []).append(v)
    rows = [("loaco_D3", c) for c in A7["evals"]["loaco_D3"]["classes"]] + [("nn_D3", c) for c in A7["evals"]["nn_D3"]["classes"]]
    dets = [("kNN", "kNN (AHSD)", C["representation"], "s", "full"), ("GRU_kNN", "kNN (GRU)", C["representation"], "s", "none"),
            ("IF", "Isolation Forest", C["benign-only"], "D", "full")]
    fig, ax = plt.subplots(figsize=(COL_W, 2.75))
    ys = np.arange(len(rows))[::-1]
    for y, (e, c) in zip(ys, rows):
        for k, (d, _, col, mk, fs) in enumerate(dets):
            v = np.array(vals[(e, c, d)])
            if len(v) != 5:
                raise SystemExit(f"{e}/{c}/{d}: {len(v)} seeds")
            yy = y + (k - 1) * 0.25
            ax.hlines(yy, v.min(), v.max(), color=col, lw=2.6 if fs == "full" else 1.2, alpha=0.35, zorder=1)
            ax.scatter(v, np.full(5, yy), s=9, marker=mk, facecolor=col if fs == "full" else "white", edgecolor=col,
                       lw=0.6, zorder=2)
    ax.axhline(len(rows) - len(A7["evals"]["loaco_D3"]["classes"]) - 0.5, color=INK2, lw=0.7)
    ax.text(0.01, ys[0] + 0.55, "LOACO D3", fontsize=6, weight="bold", color=INK, transform=ax.get_yaxis_transform())
    ax.text(0.01, ys[len(A7['evals']['loaco_D3']['classes'])] + 0.55, "Natural novelty D3", fontsize=6, weight="bold",
            color=INK, transform=ax.get_yaxis_transform())
    chance(ax)
    ax.set_yticks(ys)
    ax.set_yticklabels([c for _, c in rows], fontsize=6)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.7, len(rows) + 0.3)
    ax.set_xlabel("AUC of each of the five seeds (bar: seed range)")
    for d, lab_, col, mk, fs in dets:
        ax.scatter([], [], s=12, marker=mk, facecolor=col if fs == "full" else "white", edgecolor=col, label=lab_)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=5.8, handletextpad=0.2)
    style(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_seedstd.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ simplified, one-message figures
plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7})
FAM5 = [("confidence", "Confidence", CONF_D := ["MSP", "Energy", "P_attack"]),
        ("representation", "Representation", ["AHSD_stress", "Mahalanobis", "kNN"]),
        ("benign-only", "Benign-only", ["IF", "OCSVM", "PCA", "AE"]),
        ("CLAD", "CLAD", ["Wilkie_CLAD"]), ("XGBoost", "XGBoost", ["XGBoost"])]
PROTO = [("LOACO\n(hold-out)", ["loaco_D1fam", "loaco_D2", "loaco_D3", "loaco_D5pb"]),
         ("Natural\nnovelty", ["nn_D1", "nn_D3", "nn_D5"]),
         ("Rolling\ncut-points", None)]


def pooled_family(evs):
    """Unit-weighted family mean AUC over every (evaluation, class) unit of the given evaluations."""
    out = {}
    for f, _, ds in FAM5:
        v = [np.mean([A7["evals"][e]["cells"][d][u] for d in ds]) for e in evs for u in A7["evals"][e]["cells"][ds[0]]]
        out[f] = (float(np.mean(v)), len(v))
    return out


def protocol_winners():
    roll = [e for e in A7["evals"] if e.startswith("roll_")]
    return [(lab, pooled_family(evs if evs else roll)) for lab, evs in PROTO]


def title(ax, t):
    ax.set_title(t, loc="left", fontsize=8, weight="bold", color=INK, pad=4)


def fig_heatmap():  # kept name for the build; now "who wins under each protocol"
    W = protocol_winners()
    fig, axs = plt.subplots(1, 3, figsize=(COL_W, 2.15), sharey=True)
    for ax, (lab, P) in zip(axs, W):
        win = max(P, key=lambda f: P[f][0])
        ys = np.arange(len(FAM5))[::-1]
        for y, (f, name, _) in zip(ys, FAM5):
            v = P[f][0]
            ax.barh(y, v, height=0.62, color=C[f], alpha=1.0 if f == win else 0.28, zorder=2)
            ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=6.8, color=INK, weight="bold" if f == win else "normal")
        n = next(iter(P.values()))[1]
        ax.set_title(f"{lab}\n{n} attack classes" if n < 45 else f"{lab}\n{n} class-cut units", fontsize=7.2, color=INK, pad=3)
        ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", ".5", "1"])
        style(ax)
    axs[0].set_yticks(np.arange(len(FAM5))[::-1])
    axs[0].set_yticklabels([n for _, n, _ in FAM5])
    axs[1].set_xlabel("Mean AUC (bold bar = winner)")
    fig.suptitle("Each protocol crowns a different winner", x=0.02, ha="left", fontsize=8.2, weight="bold", color=INK)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.94), w_pad=0.4)
    fig.savefig(OUT / "fig_heatmap.pdf")
    plt.close(fig)
    return {lab.replace("\n", " "): {f: round(v[0], 3) for f, v in P.items()} for lab, P in W}


def fig_confidence():
    rows = [("LOACO D2", "loaco_D2"), ("LOACO D3", "loaco_D3"), ("Natural nov. D3", "nn_D3"),
            ("Natural nov. D5", "nn_D5"), ("Rolling (45 units)", "rolling")]
    cells = lambda e: A7["rolling"]["pooled"]["cells"] if e == "rolling" else A7["evals"][e]["cells"]  # noqa: E731
    cM, cP = "#eb6834", "#2a78d6"
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.axvspan(0, 0.5, color="#fbeae7", zorder=0)
    ax.text(0.25, -0.45, "worse than chance", ha="center", fontsize=6.6, color="#9b2c2c")
    ys = np.arange(len(rows))[::-1]
    for y, (lab, e) in zip(ys, rows):
        for dy, pre, fs in ((0.16, "", "full"), (-0.16, "GRU_", "none")):
            m = float(np.mean(list(cells(e)[pre + "MSP"].values())))
            p = float(np.mean(list(cells(e)[pre + "P_attack"].values())))
            ax.plot([m, p], [y + dy, y + dy], color=GRID if abs(p - m) < 0.02 else "#c9c8c3", lw=1.6, zorder=1)
            ax.scatter([m], [y + dy], s=26, marker="o", facecolor=cM if fs == "full" else "white", edgecolor=cM, lw=1, zorder=3)
            ax.scatter([p], [y + dy], s=26, marker="o", facecolor=cP if fs == "full" else "white", edgecolor=cP, lw=1, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r for r, _ in rows])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, 4.85)
    ax.set_xlabel("Mean AUC on the same trained models")
    for lab_, col in (("MSP", cM), ("P(attack)", cP)):
        ax.scatter([], [], s=26, color=col, label=lab_)
    ax.scatter([], [], s=26, facecolor=INK2, edgecolor=INK2, label="AHSD")
    ax.scatter([], [], s=26, facecolor="white", edgecolor=INK2, label="GRU")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False, handletextpad=0.1, columnspacing=0.9)
    style(ax)
    fig.suptitle("MSP drops below chance where P(attack) detects", x=0.02, ha="left", fontsize=8, weight="bold", color=INK)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_confidence.pdf")
    plt.close(fig)


def _illusion_stats():
    acc = {}
    for r in _runs():
        if r["kind"] != "natural_novelty":
            continue
        for d, bc in r["ops"]["best"].items():
            for c, mm in bc.items():
                acc.setdefault((r["eval_id"], d, c), []).append((r["auc"]["best"][d][c], mm["tpr_at_1pct_fpr"]))
    P = []
    for k, v in acc.items():
        if len(v) != 5:
            raise SystemExit(f"{k}: {len(v)} seeds")
        P.append((np.mean([x for x, _ in v]), np.mean([y for _, y in v])))
    P = np.array(P)
    hi = P[P[:, 0] >= 0.75]
    return {"n_points": int(len(P)), "median_tpr_auc_ge_075": float(np.median(hi[:, 1])), "n_auc_ge_075": int(len(hi)),
            "n_trap": int(np.sum(hi[:, 1] < 0.2))}


def fig_illusion():
    RPf = A7["rolling"]["pooled"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ys = np.arange(len(FAM5))[::-1]
    for y, (f, name, _) in zip(ys, FAM5):
        a = RPf["families"][f]["mean"]
        t = RPf["ops"][f]["tpr_at_1pct_fpr"]["mean"]
        ax.barh(y + 0.19, a, height=0.34, color=C[f], alpha=0.3, zorder=2)
        ax.barh(y - 0.19, t, height=0.34, color=C[f], zorder=2)
        ax.text(a + 0.015, y + 0.19, f"AUC {a:.2f}", va="center", fontsize=6.4, color=INK2)
        ax.text(t + 0.015, y - 0.19, f"{t:.0%} caught", va="center", fontsize=6.6, color=INK, weight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n, _ in FAM5])
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel("light: AUC   ·   dark: share of attacks caught at 1% false alarms")
    style(ax)
    title(ax, "A good AUC still misses most new attacks")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_illusion.pdf")
    plt.close(fig)
    return _illusion_stats()


def fig_seedstd():
    vals = {}
    for r in _runs():
        if r["eval_id"] in ("loaco_D3", "nn_D3"):
            for d in ("kNN", "IF"):
                for c, v in r["auc"]["best"][d].items():
                    vals.setdefault((r["eval_id"], c, d), []).append(v)
    blocks = [("loaco_D3", "LOACO (hold-out)"), ("nn_D3", "Natural novelty")]
    dets = [("kNN", "kNN distance (learned)", C["representation"], "s"), ("IF", "Isolation Forest (benign-only)", C["benign-only"], "D")]
    fig, axs = plt.subplots(1, 2, figsize=(COL_W, 2.2), sharey=False)
    for ax, (e, lab) in zip(axs, blocks):
        cls = A7["evals"][e]["classes"]
        ys = np.arange(len(cls))[::-1]
        for y, c in zip(ys, cls):
            for k, (d, _, col, mk) in enumerate(dets):
                v = np.array(vals[(e, c, d)])
                if len(v) != 5:
                    raise SystemExit(f"{e}/{c}/{d}: {len(v)} seeds")
                yy = y + (0.14 if k == 0 else -0.14)
                ax.hlines(yy, v.min(), v.max(), color=col, lw=3, alpha=0.3, zorder=1)
                ax.scatter(v, np.full(5, yy), s=12, marker=mk, color=col, edgecolor="white", lw=0.4, zorder=2)
        ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_yticks(ys)
        ax.set_yticklabels(cls)
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(lab, fontsize=7.4, color=INK, pad=3)
        style(ax)
    fig.supxlabel("AUC of each of 5 training seeds, ToN-IoT", fontsize=7, color=INK2, y=0.02)
    for d, lab_, col, mk in dets:
        axs[0].scatter([], [], s=14, marker=mk, color=col, label=lab_)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
               handletextpad=0.1, fontsize=6.6)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 0.9), w_pad=0.6)
    fig.savefig(OUT / "fig_seedstd.pdf")
    plt.close(fig)


def fig_shift():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 1.95), gridspec_kw={"width_ratios": [1.2, 1]})
    models = [("P_attack", "Neural (AHSD)", C["confidence"], "o"), ("XGBoost", "XGBoost", C["XGBoost"], "v")]
    y, yt = 0, []
    for ds in ("D1", "D2", "D3", "D5"):
        for d, _, col, mk in models:
            g = I[f"indist_{ds}_grouped_random"]["best"][d]["test"]["auc"]["mean"]
            t = I[f"indist_{ds}_temporal_gap"]["best"][d]["test"]["auc"]["mean"]
            a1.annotate("", xy=(t, -y), xytext=(g, -y), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=7,
                                                                        shrinkA=0, shrinkB=0))
            a1.scatter([g], [-y], s=14, marker=mk, facecolor="white", edgecolor=col, lw=1, zorder=3)
            y += 1
        yt.append(-(y - 1.5))
        y += 0.7
    a1.set_yticks(yt)
    a1.set_yticklabels(["D1", "D2", "D3", "D5"])
    a1.set_xlim(0.45, 1.02)
    a1.set_xlabel("AUC, random → time split")
    for _, lab_, col, mk in models:
        a1.plot([], [], color=col, marker=mk, ms=3.5, lw=1.2, mfc="white", label=lab_)
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1.3, columnspacing=0.8)
    style(a1)
    off = np.array([T[d][s][t]["mean"] for d in T for s in T[d] for t in T[d][s] if s != t])
    a2.hist(off, bins=np.arange(0, 1.0001, 0.1), color=C["confidence"], edgecolor="white", lw=1.0, zorder=2)
    a2.axvline(0.5, color=INK2, lw=0.9, ls=(0, (3, 2)))
    a2.text(0.47, a2.get_ylim()[1] * 0.95, f"median {np.median(off):.2f}", fontsize=6.6, color=INK, va="top", ha="right")
    a2.set_xlabel(f"Cross-dataset AUC ({len(off)} results)")
    a2.set_ylabel("Count")
    a2.set_xlim(0, 1)
    a2.set_title("Transfer ≈ chance", fontsize=7.4, color=INK, pad=3)
    style(a2, "y")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "fig_shift.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig_pipeline, fig_seedstd, fig_confidence, fig_shift):
        f()
    (OUT / "illusion.json").write_text(json.dumps(fig_illusion(), indent=1))
    (OUT / "winners.json").write_text(json.dumps(fig_heatmap(), indent=1))
    print("figures:", sorted(p.name for p in OUT.glob("fig_*.pdf")))
