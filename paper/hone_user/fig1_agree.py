"""Fig. 1 of the HoNE paper: does LOACO predict natural novelty?

Reads only results/final_v2/agreement.json (written by scripts/report_agreement.py) and writes
fig_agree.pdf / fig_agree.png next to this file. No hand-typed numbers: every value and label comes from the JSON.

Usage:  python fig1_agree.py [path/to/agreement.json]
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

try:  # run as a script: paths relative to this file
    HERE = Path(__file__).resolve().parent
    _DEFAULT = HERE.parents[1] / "results/final_v2/agreement.json"
except NameError:  # Jupyter notebook: no __file__, use the notebook's folder
    HERE = Path.cwd()
    _DEFAULT = HERE / "agreement.json"
_ARGS = [a for a in sys.argv[1:] if a.endswith("agreement.json")]  # ignore Jupyter's own arguments
AGREEMENT = Path(_ARGS[0]) if _ARGS else _DEFAULT

# colours: validated categorical palette (one colour per dataset) and neutral inks
DS_C = {"D1": "#2a78d6", "D3": "#eb6834", "D5": "#1baf7a"}
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2})


def clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.4)
        ax.set_axisbelow(True)


def main():
    AG = json.loads(AGREEMENT.read_text())
    C, G = AG["configurations"], AG["aggregate"]
    # 12 paired comparisons: dataset x backbone x natural-novelty view
    keys = [f"{ds}|{bb}|{v}" for ds in DS_C for bb in ("AHSD", "GRU") for v in ("fixed", "rolling")]
    fig, ax = plt.subplots(2, 2, figsize=(3.45, 3.35))
    ax = ax.ravel()
    rng = np.random.default_rng(0)  # fixed seed: jitter is identical on every run

    # (a) each detector's AUC under LOACO (x) and natural novelty (y)
    a = ax[0]
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--")
    for k in keys:
        cf = C[k]
        a.scatter(list(cf["auc_loaco"].values()), [cf["auc_natural"][d] for d in cf["auc_loaco"]], s=6,
                  color=DS_C[cf["dataset"]], alpha=0.75, linewidth=0)
    a.set_xlim(0, 1.02); a.set_ylim(0, 1.02)
    a.set_xlabel("AUC, LOACO"); a.set_ylabel("AUC, natural novelty")
    a.text(0.98, 0.03, f"{G['cells_frac_lower_natural'] * 100:.0f}% of {G['n_cells']}\nbelow diagonal",
           ha="right", va="bottom", fontsize=5.8, color=INK)
    a.set_title("(a) AUC per detector", fontsize=7)

    # (b) each detector's rank under both protocols (1 = best)
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

    # (c) bootstrap probability that both protocols select the same best detector
    c = ax[2]
    xs = []
    for i, k in enumerate(keys):
        cf = C[k]
        x = i + 0.6 * (i // 4)  # gap between datasets
        xs.append(x)
        c.bar(x, cf["p_same_winner"], width=0.8, color=DS_C[cf["dataset"]],
              hatch=None if cf["view"] == "fixed" else "////", edgecolor="white", linewidth=0.4)
        if not cf["winner_flip"]:
            c.text(x, cf["p_same_winner"] + 0.02, "same", ha="center", fontsize=5, color=INK)
    c.set_xticks([np.mean(xs[j:j + 4]) for j in (0, 4, 8)], list(DS_C))
    c.set_ylim(0, 1); c.set_ylabel("P(same best detector)")
    c.axhline(G["p_same_winner_mean"], color=INK, lw=0.6, ls=":")
    c.text(xs[-1] + 0.5, 0.30, f"mean {G['p_same_winner_mean']:.2f}", ha="right", va="bottom", fontsize=5.8)
    c.annotate("", xy=(xs[-1] + 0.3, G["p_same_winner_mean"]), xytext=(xs[-1] + 0.3, 0.30),
               arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))
    c.set_title(f"(c) Best detector kept in {G['loaco_winner_retained']} of {G['n_configurations']}", fontsize=7)
    clean(c); c.tick_params(axis="x", length=0)

    # (d) delta AUC (natural - LOACO) per detector, by dataset; bar = median
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
    h = [Line2D([], [], marker="o", ls="", ms=4, color=DS_C[x], label=x) for x in DS_C]
    h += [Patch(facecolor="white", edgecolor=MUTED, label="fixed split"),
          Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="rolling cuts (in c)")]
    fig.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.985), frameon=False, fontsize=6.3,
               columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.9)
    fig.savefig(HERE / "fig_agree.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(HERE / "fig_agree.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", HERE / "fig_agree.pdf")


if __name__ == "__main__":
    main()
