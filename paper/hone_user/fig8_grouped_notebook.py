%matplotlib inline
# HoNE Fig. 8 - AUC vs attacks caught at 1% false-positive rate, per detector family (grouped bars, data embedded).
# Rolling cut-points: 45 class-cut units under natural novelty. Bars = family mean; whiskers = 95% bootstrap CI
# over class-cut units and seeds. Source: results/final_v2/summary.json (a7.rolling.pooled).
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# family -> (AUC mean, CI low, CI high), (share caught at 1% FPR: mean, CI low, CI high); ordered by AUC
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
W = 0.36                                   # bar width

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "ytick.major.width": 0.5, "ytick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.45, 1.9))
for i, (name, (a, alo, ahi), (t, tlo, thi)) in enumerate(DATA):
    c = COLOR[name]
    ax.bar(i - W / 2, a, W, color=c, alpha=0.35, edgecolor=c, linewidth=0.6)      # AUC (light)
    ax.bar(i + W / 2, t, W, color=c, edgecolor=c, linewidth=0.6)                   # caught (dark)
    ax.errorbar(i - W / 2, a, yerr=[[a - alo], [ahi - a]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    ax.errorbar(i + W / 2, t, yerr=[[t - tlo], [thi - t]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    ax.text(i - W / 2, ahi + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.text(i + W / 2, thi + 0.02, f"{t * 100:.0f}%", ha="center", va="bottom", fontsize=6.2, weight="bold")
ax.axhline(0.5, color=MUTED, lw=0.6, ls="--", zorder=0)
ax.set_xticks(range(len(DATA)), [d[0].replace("Representation", "Represent.") for d in DATA], fontsize=6.4)
ax.set_xlim(-0.55, len(DATA) - 0.45)
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".50", ".75", "1"])
ax.set_ylabel("AUC  /  share caught")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[Patch(facecolor="#9a9890", alpha=0.4, edgecolor="#9a9890", label="AUC"),
                   Patch(facecolor="#5f5e5a", label="share caught at 1% FPR"),
                   plt.Line2D([], [], color=MUTED, lw=0.6, ls="--", label="AUC chance")],
          ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.3, handlelength=1.2)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_operating.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_operating.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_operating.pdf")
plt.show()
