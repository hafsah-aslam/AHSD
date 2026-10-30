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


# ---------------------------------------------------------------- summary figure: does LOACO predict natural novelty?
DS_C = {"D1": BLUE, "D3": ORANGE, "D5": AQUA}


def fig_agree():
    C, G = AG["configurations"], AG["aggregate"]
    keys = [f"{ds}|{bb}|{v}" for ds in DS_C for bb in ("AHSD", "GRU") for v in ("fixed", "rolling")]
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.25, 1.05]})
    rng = np.random.default_rng(0)
    # (a) AUC vs AUC
    a = ax[0]
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--")
    for k in keys:
        cf = C[k]
        a.scatter(list(cf["auc_loaco"].values()), [cf["auc_natural"][d] for d in cf["auc_loaco"]], s=6,
                  color=DS_C[cf["dataset"]], alpha=0.75, linewidth=0)
    a.set_xlim(0, 1.02); a.set_ylim(0, 1.02); a.set_xlabel("AUC, LOACO"); a.set_ylabel("AUC, natural novelty")
    a.text(0.98, 0.03, f"{G['cells_frac_lower_natural'] * 100:.0f}% of {G['n_cells']}\nbelow diagonal",
           ha="right", va="bottom", fontsize=5.8, color=INK)
    a.set_title("(a) AUC per detector", fontsize=7)
    # (b) rank vs rank
    b = ax[1]
    nmax = max(C[k]["n_detectors"] for k in keys)
    b.plot([1, nmax], [1, nmax], color=MUTED, lw=0.6, ls="--")
    for k in keys:
        cf = C[k]
        rl, rn = cf["rank_loaco"], cf["rank_natural"]
        jit = rng.uniform(-0.22, 0.22, (2, len(rl)))
        b.scatter(np.array([rl[d] for d in rl]) + jit[0], np.array([rn[d] for d in rl]) + jit[1], s=5,
                  color=DS_C[cf["dataset"]], alpha=0.7, linewidth=0)
    b.set_xlim(nmax + 0.7, 0.3); b.set_ylim(nmax + 0.7, 0.3)
    b.set_xticks([1, 4, 8, 12]); b.set_yticks([1, 4, 8, 12])
    b.set_xlabel("Rank, LOACO"); b.set_ylabel("Rank, natural novelty")
    b.set_title(f"(b) Ranks: median $\\rho$ = {G['spearman_median']:.2f}", fontsize=7)
    # (c) bootstrap probability that both protocols pick the same best detector
    c = ax[2]
    xs, lab = [], []
    for i, k in enumerate(keys):
        cf = C[k]
        x = i + 0.6 * (i // 4)
        xs.append(x)
        c.bar(x, cf["p_same_winner"], width=0.8, color=DS_C[cf["dataset"]],
              hatch=None if cf["view"] == "fixed" else "////", edgecolor="white", linewidth=0.4)
        if not cf["winner_flip"]:
            c.text(x, cf["p_same_winner"] + 0.02, "same", ha="center", fontsize=5, color=INK)
    c.set_xticks([np.mean(xs[j:j + 4]) for j in (0, 4, 8)], list(DS_C))
    c.set_ylim(0, 1); c.set_ylabel("P(same best detector)")
    c.axhline(G["p_same_winner_mean"], color=INK, lw=0.6, ls=":")
    c.text(xs[-1] + 0.5, G["p_same_winner_mean"] + 0.02, f"mean {G['p_same_winner_mean']:.2f}", ha="right",
           fontsize=5.8)
    c.set_title(f"(c) Best detector kept in {G['loaco_winner_retained']} of {G['n_configurations']}", fontsize=7)
    clean(c); c.tick_params(axis="x", length=0)
    # (d) delta AUC distribution per dataset
    d = ax[3]
    for j, ds in enumerate(DS_C):
        D = [C[k]["delta_auc"][x] for k in keys if C[k]["dataset"] == ds for x in C[k]["delta_auc"]]
        d.scatter(rng.uniform(-0.18, 0.18, len(D)) + j, D, s=5, color=DS_C[ds], alpha=0.7, linewidth=0)
        d.plot([j - 0.28, j + 0.28], [np.median(D)] * 2, color=INK, lw=1.2)
    d.axhline(0, color=MUTED, lw=0.6, ls="--")
    d.set_xticks(range(3), list(DS_C)); d.set_ylabel("$\\Delta$AUC (natural $-$ LOACO)")
    d.set_title(f"(d) $\\Delta$AUC: median {G['cells_median_delta_auc']:+.3f}", fontsize=7)
    clean(d); d.tick_params(axis="x", length=0)
    for a_ in (ax[0], ax[1]):
        clean(a_, None)
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="o", ls="", ms=4, color=DS_C[x], label=x) for x in DS_C]
    h += [Patch(facecolor="white", edgecolor=MUTED, label="fixed split"),
          Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="rolling cuts (in c)")]
    fig.legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.2, w_pad=1.0)
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
