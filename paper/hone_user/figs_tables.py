"""Figures that replace the HoNE result tables. Every value is read from results/final_v2 (no hand-typed numbers).

fig_agree.pdf     LOACO vs natural-novelty AUC per detector (agreement.json), fixed split and rolling cuts
fig_family.pdf    family mean AUC per evaluation (summary.json class_evals)
fig_seedsd.pdf    mean seed s.d. of AUC, LOACO vs natural novelty (per-run JSON)
fig_conf.pdf      confidence scores vs P(attack) and representation (summary.json a7)
fig_temporal.pdf  in-distribution AUC, grouped vs temporal split (summary.json indist)
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from report_a7 import SEEDS, tensors  # noqa: E402

RES = ROOT / "results/final_v2"
OUT = Path(__file__).parent
S = json.loads((RES / "summary.json").read_text())
AG = json.loads((RES / "agreement.json").read_text())
CS, R = S["class_evals"], S["a7"]

# categorical palette (validated: scripts/validate_palette.js, light mode); protocol pair = slots 1 and 2
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"
LO_C, NN_C = BLUE, ORANGE
FAM_C = {"confidence": BLUE, "representation": ORANGE, "benign-only": AQUA, "CLAD": YELLOW, "XGBoost": MAGENTA}
FAM_OF = {"MSP": "confidence", "Energy": "confidence", "P_attack": "confidence",
          "AHSD_stress": "representation", "Mahalanobis": "representation", "kNN": "representation",
          "IF": "benign-only", "OCSVM": "benign-only", "PCA": "benign-only", "AE": "benign-only",
          "Wilkie_CLAD": "CLAD", "XGBoost": "XGBoost",
          "GRU_MSP": "confidence", "GRU_Energy": "confidence", "GRU_P_attack": "confidence",
          "GRU_Mahalanobis": "representation", "GRU_kNN": "representation"}

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.4)
        ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- agreement (replaces the agreement table)
def fig_agree():
    C = AG["configurations"]
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.15), sharex=True, sharey=True)
    for r, view in enumerate(("fixed", "rolling")):
        for c, ds in enumerate(("D1", "D3", "D5")):
            ax = axes[r, c]
            ax.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--", zorder=1)
            ax.axhline(0.5, color=GRID, lw=0.5, zorder=0)
            ax.axvline(0.5, color=GRID, lw=0.5, zorder=0)
            rho = []
            for bb, filled in (("AHSD", True), ("GRU", False)):
                cf = C[f"{ds}|{bb}|{view}"]
                rho.append(cf["spearman"])
                for d, x in cf["auc_loaco"].items():
                    if bb == "GRU" and not d.startswith("GRU_"):
                        continue  # backbone-free detectors are drawn once (AHSD row)
                    col = FAM_C[FAM_OF[d]]
                    ax.scatter(x, cf["auc_natural"][d], s=16, marker="o" if filled else "D",
                               facecolor=col if filled else "white", edgecolor=col if not filled else "white",
                               linewidth=0.9 if not filled else 0.5, zorder=3)
                if bb == "AHSD":
                    wl, wn = cf["winner_loaco"], cf["winner_natural"]
            ax.set_title(f"{ds}, {'fixed split' if view == 'fixed' else 'rolling cuts'}:  "
                         f"$\\rho$ = {rho[0]:.2f} / {rho[1]:.2f}", fontsize=7)
            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
            clean(ax, None)
            if c == 0:
                ax.set_ylabel("AUC, natural novelty")
            if r == 1:
                ax.set_xlabel("AUC, LOACO")
            ax.text(0.97, 0.04, "below the line:\nlower under natural novelty", ha="right", va="bottom",
                    fontsize=5.5, color=MUTED) if (r, c) == (0, 0) else None
    h = [Line2D([], [], marker="o", ls="", ms=4, color=FAM_C[f], label=f) for f in FAM_C]
    h += [Line2D([], [], marker="o", ls="", ms=4, color=MUTED, label="AHSD backbone"),
          Line2D([], [], marker="D", ls="", ms=3.5, mfc="white", mec=MUTED, label="GRU backbone")]
    fig.legend(handles=h, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5,
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8)
    save(fig, "fig_agree")


# ---------------------------------------------------------------- family heatmap (replaces the family table)
EV = [("loaco_D1fam", "LOACO D1"), ("loaco_D2", "LOACO D2"), ("loaco_D3", "LOACO D3"), ("loaco_D5pb", "Purged D5"),
      ("nn_D1", "Natural D1"), ("nn_D3", "Natural D3"), ("nn_D5", "Natural D5")]
FAMS = [("confidence", "Conf."), ("representation", "Repr."), ("benign-only", "Benign"), ("CLAD", "CLAD"),
        ("XGBoost", "XGB")]
NOVELTY = ("confidence", "representation", "benign-only", "CLAD")


def fig_family():
    M = np.array([[CS[e]["families"][f]["mean"] for f, _ in FAMS] for e, _ in EV])
    fig, ax = plt.subplots(figsize=(3.4, 2.05))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
    ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i, (e, n) in enumerate(EV):
        lead = max(NOVELTY, key=lambda f: CS[e]["families"][f]["mean"])
        for j, (f, _) in enumerate(FAMS):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.62 else INK,
                    fontweight="bold" if f == lead else "normal")
            if f == lead:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=ORANGE, lw=1.4))
    ax.set_xticks(range(len(FAMS)), [lab for _, lab in FAMS])
    ax.set_yticks(range(len(EV)), [f"{n} ({CS[e]['n_classes']})" for e, n in EV])
    ax.axhline(3.5, color="white", lw=2.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_family")


# ---------------------------------------------------------------- seed s.d. dumbbell (replaces the seed table)
SD_ROWS = [("MSP", "MSP"), ("Energy", "Energy"), ("P_attack", "$P$(attack)$^\\dagger$"), ("AHSD_stress", "Stress"),
           ("Mahalanobis", "Mahalanobis"), ("kNN", "$k$NN"), ("GRU_Mahalanobis", "GRU Mahal."),
           ("GRU_kNN", "GRU $k$NN"), ("GRU_MSP", "GRU MSP"), ("GRU_P_attack", "GRU $P$(attack)"),
           ("Wilkie_CLAD", "CLAD"), ("XGBoost_s", "XGBoost-s"), ("IF", "Isolation Forest"), ("AE", "Autoencoder")]
LO_EVALS, NN_EVALS = ["loaco_D1fam", "loaco_D2", "loaco_D3"], ["nn_D1", "nn_D3", "nn_D5"]


def seed_sd():
    runs = [json.loads(p.read_text()) for p in sorted(RES.glob("*/*/seed*.json"))]
    A, _, classes = tensors(runs)

    def pooled(d, evals):
        return float(np.mean([np.std([A[e]["best"][d][c][s] for s in SEEDS], ddof=1)
                              for e in evals for c in classes[e]]))
    return {d: (pooled(d, LO_EVALS), pooled(d, NN_EVALS)) for d, _ in SD_ROWS}


def fig_seedsd():
    V = seed_sd()
    (OUT / "seedsd.json").write_text(json.dumps(V, indent=1))
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    n = len(SD_ROWS)
    for i, (d, lab) in enumerate(SD_ROWS):
        y = n - 1 - i
        lo, nn = V[d]
        ax.plot([lo, nn], [y, y], color=GRID, lw=1.6, zorder=1, solid_capstyle="round")
        ax.scatter([lo], [y], s=22, color=LO_C, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter([nn], [y], s=22, color=NN_C, zorder=3, edgecolor="white", linewidth=0.6)
        ax.text(0.62, y, f"{nn / lo:.1f}$\\times$", va="center", ha="right", fontsize=6.5,
                fontweight="bold" if nn / lo >= 5 else "normal")
    ax.set_yticks(range(n), [lab for _, lab in SD_ROWS][::-1])
    ax.set_xscale("log")
    ax.set_xlim(0.0015, 0.65)
    ax.set_xticks([0.002, 0.01, 0.05, 0.2], ["0.002", "0.01", "0.05", "0.2"])
    ax.set_xlabel("Mean seed s.d. of AUC (log scale)")
    ax.text(0.62, n - 0.35, "ratio", ha="right", fontsize=6, color=MUTED)
    for yy in (3.5, 9.5):
        ax.axhline(yy, color=GRID, lw=0.5, ls=":")
    clean(ax, "x")
    ax.tick_params(axis="y", length=0)
    h = [Line2D([], [], marker="o", ls="", ms=4, color=LO_C, label="LOACO (D1--D3)"),
         Line2D([], [], marker="o", ls="", ms=4, color=NN_C, label="Natural novelty (D1, D3, D5)")]
    ax.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=6.5)
    save(fig, "fig_seedsd")


# ---------------------------------------------------------------- confidence scores and temporal shift
def dmean(e, d):
    return float(np.mean(list(R["evals"][e]["cells"][d].values())))


def fig_conf():
    ev = [("loaco_D2", "LOACO D2"), ("loaco_D3", "LOACO D3"), ("nn_D3", "Natural D3"), ("nn_D5", "Natural D5")]
    series = [("MSP", BLUE), ("Energy", AQUA), ("$P$(attack)$^\\dagger$", ORANGE), ("Repr. family", MAGENTA)]
    vals = [[dmean(e, "MSP"), dmean(e, "Energy"), dmean(e, "P_attack"), CS[e]["families"]["representation"]["mean"]]
            for e, _ in ev]
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    w = 0.19
    for k, (name, col) in enumerate(series):
        xs = np.arange(len(ev)) + (k - 1.5) * (w + 0.015)
        ax.bar(xs, [v[k] for v in vals], width=w, color=col, label=name, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color=INK, lw=0.6, ls="--")
    ax.text(-0.5, 0.515, "chance", fontsize=5.5, color=MUTED, ha="left", va="bottom")
    ax.set_xticks(range(len(ev)), [n for _, n in ev])
    ax.set_xlim(-0.55, len(ev) - 0.45)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Mean AUC")
    clean(ax)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5,
              handlelength=1.0, columnspacing=0.9)
    save(fig, "fig_conf")


def fig_temporal():
    I = S["indist"]
    ds = ["D1", "D2", "D3", "D5"]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.45), sharey=True)
    for ax, (d, name) in zip(axes, (("P_attack", "$P$(attack)$^\\dagger$ (AHSD)"), ("XGBoost", "XGBoost"))):
        for i, x in enumerate(ds):
            g = I[f"indist_{x}_grouped_random"]["best"][d]["test"]["auc"]["mean"]
            t = I[f"indist_{x}_temporal_gap"]["best"][d]["test"]["auc"]["mean"]
            ax.annotate("", xy=(i, t), xytext=(i, g),
                        arrowprops=dict(arrowstyle="-|>", color=GRID if abs(t - g) < 0.02 else MUTED, lw=0.9,
                                        mutation_scale=7, shrinkA=3, shrinkB=3))
            ax.scatter([i], [g], s=18, color=LO_C, zorder=3, edgecolor="white", linewidth=0.5)
            ax.scatter([i], [t], s=18, color=NN_C, zorder=3, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(ds)), ds)
        ax.set_xlim(-0.5, len(ds) - 0.5)
        ax.set_title(name, fontsize=7)
        ax.set_ylim(0.45, 1.03)
        clean(ax)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("In-distribution AUC")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=LO_C, label="grouped (random) split"),
         Line2D([], [], marker="o", ls="", ms=4, color=NN_C, label="temporal split")]
    fig.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_temporal")


if __name__ == "__main__":
    fig_agree(); fig_family(); fig_seedsd(); fig_conf(); fig_temporal()
    print("figures written to", OUT)
