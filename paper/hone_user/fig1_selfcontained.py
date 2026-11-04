# HoNE Fig. 1 - self-contained (data embedded from results/final_v2/agreement.json). Paste into one notebook cell.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# each row: "dataset|backbone|view", AUC under LOACO, AUC under natural novelty (one value per detector),
#           rank under LOACO, rank under natural novelty, P(same best detector) from bootstrap, best detector changed (1/0)
ROWS = [
    ["D1|AHSD|fixed", [0.5173, 0.4805, 0.6616, 0.7171, 0.7367, 0.7327, 0.6299, 0.5752, 0.7475, 0.6864, 0.6534, 0.6818], [0.5221, 0.5195, 0.5466, 0.6195, 0.6276, 0.6208, 0.4975, 0.4374, 0.5576, 0.527, 0.5743, 0.5235], [11, 12, 7, 4, 2, 3, 9, 10, 1, 5, 8, 6], [9, 10, 6, 3, 1, 2, 11, 12, 5, 7, 4, 8], 0.215, 1],
    ["D1|AHSD|rolling", [0.5173, 0.4805, 0.6616, 0.7171, 0.7367, 0.7327, 0.6299, 0.5752, 0.7475, 0.6864, 0.6534, 0.6818], [0.5501, 0.55, 0.5571, 0.6359, 0.6693, 0.6585, 0.5753, 0.5619, 0.6934, 0.6628, 0.6221, 0.622], [11, 12, 7, 4, 2, 3, 9, 10, 1, 5, 8, 6], [11, 12, 10, 5, 2, 4, 8, 9, 1, 3, 6, 7], 0.437, 0],
    ["D1|GRU|fixed", [0.5067, 0.5067, 0.6274, 0.7332, 0.7022, 0.6299, 0.5752, 0.7475, 0.6864, 0.6534, 0.6818], [0.5619, 0.5609, 0.5832, 0.6428, 0.6279, 0.4975, 0.4374, 0.5576, 0.527, 0.5743, 0.5235], [10, 11, 8, 2, 3, 7, 9, 1, 4, 6, 5], [5, 6, 3, 1, 2, 10, 11, 7, 8, 4, 9], 0.3475, 1],
    ["D1|GRU|rolling", [0.5067, 0.5067, 0.6274, 0.7332, 0.7022, 0.6299, 0.5752, 0.7475, 0.6864, 0.6534, 0.6818], [0.6692, 0.6685, 0.6793, 0.7071, 0.7052, 0.5753, 0.5619, 0.6934, 0.6628, 0.6221, 0.622], [10, 11, 8, 2, 3, 7, 9, 1, 4, 6, 5], [5, 6, 4, 1, 2, 10, 11, 3, 7, 8, 9], 0.378, 1],
    ["D3|AHSD|fixed", [0.2688, 0.2584, 0.7522, 0.8817, 0.9047, 0.8552, 0.9003, 0.8982, 0.9062, 0.9066, 0.7802, 0.8624], [0.2623, 0.2871, 0.8529, 0.539, 0.528, 0.5, 0.916, 0.8956, 0.8858, 0.8944, 0.8225, 0.9225], [11, 12, 10, 6, 3, 8, 4, 5, 2, 1, 9, 7], [12, 11, 6, 8, 9, 10, 2, 3, 5, 4, 7, 1], 0.121, 1],
    ["D3|AHSD|rolling", [0.2688, 0.2584, 0.7522, 0.8817, 0.9047, 0.8552, 0.9003, 0.8982, 0.9062, 0.9066, 0.7802, 0.8624], [0.4549, 0.4698, 0.5665, 0.454, 0.4539, 0.5229, 0.9095, 0.9101, 0.8984, 0.8923, 0.6324, 0.6987], [11, 12, 10, 6, 3, 8, 4, 5, 2, 1, 9, 7], [10, 9, 7, 11, 12, 8, 2, 1, 3, 4, 6, 5], 0.0935, 1],
    ["D3|GRU|fixed", [0.2821, 0.2867, 0.804, 0.9023, 0.8573, 0.9003, 0.8982, 0.9062, 0.9066, 0.7802, 0.8624], [0.2448, 0.2534, 0.9182, 0.8588, 0.8792, 0.916, 0.8956, 0.8858, 0.8944, 0.8225, 0.9225], [11, 10, 8, 3, 7, 4, 5, 2, 1, 9, 6], [11, 10, 2, 8, 7, 3, 4, 6, 5, 9, 1], 0.0665, 1],
    ["D3|GRU|rolling", [0.2821, 0.2867, 0.804, 0.9023, 0.8573, 0.9003, 0.8982, 0.9062, 0.9066, 0.7802, 0.8624], [0.3668, 0.3833, 0.725, 0.6157, 0.6467, 0.9095, 0.9101, 0.8984, 0.8923, 0.6324, 0.6987], [11, 10, 8, 3, 7, 4, 5, 2, 1, 9, 6], [11, 10, 5, 9, 7, 2, 1, 3, 4, 8, 6], 0.0435, 1],
    ["D5|AHSD|fixed", [0.4012, 0.34, 0.9994, 0.998, 0.9983, 0.9984, 0.9614, 0.5064, 0.8913, 0.9744, 0.9723, 1.0], [0.1032, 0.0904, 0.5752, 0.9822, 0.986, 0.8506, 0.8821, 0.7343, 0.8373, 0.7205, 0.8636, 0.9678], [11, 12, 2, 5, 4, 3, 8, 10, 9, 6, 7, 1], [11, 12, 10, 2, 1, 6, 4, 8, 7, 9, 5, 3], 0.2535, 1],
    ["D5|AHSD|rolling", [0.4012, 0.34, 0.9994, 0.998, 0.9983, 0.9984, 0.9614, 0.5064, 0.8913, 0.9744, 0.9723, 1.0], [0.5081, 0.494, 0.637, 0.6913, 0.7593, 0.762, 0.7141, 0.6876, 0.7387, 0.736, 0.702, 0.6961], [11, 12, 2, 5, 4, 3, 8, 10, 9, 6, 7, 1], [11, 12, 10, 8, 2, 1, 5, 9, 3, 4, 6, 7], 0.0415, 1],
    ["D5|GRU|fixed", [0.8892, 0.8353, 0.9991, 0.9995, 0.9748, 0.9614, 0.5064, 0.8913, 0.9744, 0.9723, 1.0], [0.5212, 0.5282, 0.9659, 0.9804, 0.9337, 0.8821, 0.7343, 0.8373, 0.7205, 0.8636, 0.9678], [9, 10, 3, 2, 4, 7, 11, 8, 5, 6, 1], [11, 10, 3, 1, 4, 5, 8, 7, 9, 6, 2], 0.0, 1],
    ["D5|GRU|rolling", [0.8892, 0.8353, 0.9991, 0.9995, 0.9748, 0.9614, 0.5064, 0.8913, 0.9744, 0.9723, 1.0], [0.629, 0.6293, 0.7677, 0.7697, 0.7685, 0.7141, 0.6876, 0.7387, 0.736, 0.702, 0.6961], [9, 10, 3, 2, 4, 7, 11, 8, 5, 6, 1], [11, 10, 3, 1, 2, 6, 9, 4, 5, 7, 8], 0.0105, 1],
]
AGG = {"cells_frac_lower_natural": 0.7536, "n_cells": 138, "spearman_median": 0.5455, "loaco_winner_retained": 1, "n_configurations": 12, "p_same_winner_mean": 0.1673, "cells_median_delta_auc": -0.0572}

OUT = Path.home() / "Downloads"   # where the figure is saved
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


G = AGG
fig, ax = plt.subplots(2, 2, figsize=(3.45, 3.35))
ax = ax.ravel()
rng = np.random.default_rng(0)

# (a) AUC under both protocols
a = ax[0]
a.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls="--")
for key, lo, nn, rl, rn, p, flip in ROWS:
    a.scatter(lo, nn, s=6, color=DS_C[key[:2]], alpha=0.75, linewidth=0)
a.set_xlim(0, 1.02); a.set_ylim(0, 1.02)
a.set_xlabel("AUC, LOACO"); a.set_ylabel("AUC, natural novelty")
a.text(0.98, 0.03, f"{G['cells_frac_lower_natural'] * 100:.0f}% of {G['n_cells']}\nbelow diagonal",
       ha="right", va="bottom", fontsize=5.8, color=INK)
a.set_title("(a) AUC per detector", fontsize=7)

# (b) rank under both protocols (1 = best)
b = ax[1]
nmax = max(len(r[1]) for r in ROWS)
b.plot([1, nmax], [1, nmax], color=MUTED, lw=0.6, ls="--")
for key, lo, nn, rl, rn, p, flip in ROWS:
    jit = rng.uniform(-0.22, 0.22, (2, len(rl)))
    b.scatter(np.array(rl) + jit[0], np.array(rn) + jit[1], s=5, color=DS_C[key[:2]], alpha=0.7, linewidth=0)
b.set_xlim(nmax + 0.7, 0.3); b.set_ylim(nmax + 0.7, 0.3)
b.set_xticks([1, 4, 8, 12]); b.set_yticks([1, 4, 8, 12])
b.set_xlabel("Rank, LOACO"); b.set_ylabel("Rank, natural novelty")
b.set_title(f"(b) Ranks: median $\\rho$ = {G['spearman_median']:.2f}", fontsize=7)

# (c) bootstrap probability that both protocols pick the same best detector
c = ax[2]
xs = []
for i, (key, lo, nn, rl, rn, p, flip) in enumerate(ROWS):
    x = i + 0.6 * (i // 4)
    xs.append(x)
    c.bar(x, p, width=0.8, color=DS_C[key[:2]], hatch=None if key.endswith("fixed") else "////",
          edgecolor="white", linewidth=0.4)
    if not flip:
        c.text(x, p + 0.02, "same", ha="center", fontsize=5, color=INK)
c.set_xticks([np.mean(xs[j:j + 4]) for j in (0, 4, 8)], list(DS_C))
c.set_ylim(0, 1); c.set_ylabel("P(same best detector)")
c.axhline(G["p_same_winner_mean"], color=INK, lw=0.6, ls=":")
c.text(xs[-1] + 0.5, G["p_same_winner_mean"] + 0.02, f"mean {G['p_same_winner_mean']:.2f}", ha="right", fontsize=5.8)
c.set_title(f"(c) Best detector kept in {G['loaco_winner_retained']} of {G['n_configurations']}", fontsize=7)
clean(c); c.tick_params(axis="x", length=0)

# (d) delta AUC per detector, by dataset (bar = median)
d = ax[3]
for j, ds in enumerate(DS_C):
    D = [n - l for key, lo, nn, rl, rn, p, flip in ROWS if key[:2] == ds for l, n in zip(lo, nn)]
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
fig.savefig(OUT / "fig_agree.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_agree.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
plt.show()
print("saved", OUT / "fig_agree.pdf")
