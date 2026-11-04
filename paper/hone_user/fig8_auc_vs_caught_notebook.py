%matplotlib inline
# HoNE Fig. 8 - AUC vs attacks caught at 1% false-positive rate, per detector family (paired panels, data embedded).
# Rolling cut-points: 45 class-cut units under natural novelty. Bars = family mean; whiskers = 95% bootstrap CI
# over class-cut units and seeds. Source: results/final_v2/summary.json (a7.rolling.pooled).
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# family -> (AUC mean, CI low, CI high), (TPR@1%FPR mean, CI low, CI high)
DATA = [
    ("Benign-only",    (0.7483, 0.6994, 0.7958), (0.0917, 0.0511, 0.1404)),
    ("XGBoost",        (0.6723, 0.5993, 0.7409), (0.1831, 0.1064, 0.2701)),
    ("CLAD",           (0.6522, 0.5947, 0.7078), (0.1009, 0.0475, 0.1684)),
    ("Representation", (0.6230, 0.5658, 0.6831), (0.1251, 0.0544, 0.2127)),
    ("Confidence",     (0.5319, 0.4834, 0.5780), (0.0443, 0.0216, 0.0760)),
]
COLOR = {"Confidence": "#2a78d6", "Representation": "#eb6834", "Benign-only": "#1baf7a", "CLAD": "#eda100",
         "XGBoost": "#e87ba4"}
OUT = Path.home() / "Downloads"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

fig, (axl, axr) = plt.subplots(1, 2, figsize=(3.45, 1.6), sharey=True)
ys = np.arange(len(DATA))[::-1]
for y, (name, (a, alo, ahi), (t, tlo, thi)) in zip(ys, DATA):
    c = COLOR[name]
    axl.barh(y, a, height=0.6, color=c, alpha=0.35, edgecolor=c, linewidth=0.6)
    axl.errorbar(a, y, xerr=[[a - alo], [ahi - a]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    axl.text(ahi + 0.02, y, f"{a:.2f}", ha="left", va="center", fontsize=6.0)
    axr.barh(y, t * 100, height=0.6, color=c, edgecolor=c, linewidth=0.6)
    axr.errorbar(t * 100, y, xerr=[[(t - tlo) * 100], [(thi - t) * 100]], fmt="none", ecolor=INK,
                 elinewidth=0.6, capsize=1.5)
    axr.text(thi * 100 + 1.2, y, f"{t * 100:.0f}%", ha="left", va="center", fontsize=6.0, weight="bold")
axl.set_xlim(0, 1.0)
axl.axvline(0.5, color=MUTED, lw=0.6, ls="--")
axl.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
axl.set_title("AUC", fontsize=6.8, pad=3)
axl.set_yticks(ys, [d[0] for d in DATA])
axl.tick_params(axis="y", length=0, labelsize=6.4)
axr.set_xlim(0, 40)
axr.set_xticks([0, 10, 20, 30, 40], ["0", "10", "20", "30", "40%"])
axr.set_title("Attacks caught at 1% FPR", fontsize=6.8, pad=3)
axr.tick_params(axis="y", length=0)
for ax in (axl, axr):
    ax.tick_params(axis="x", labelsize=6)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "left", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.2)
fig.subplots_adjust(wspace=0.12)
fig.savefig(OUT / "fig_operating.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_operating.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_operating.pdf")
plt.show()
