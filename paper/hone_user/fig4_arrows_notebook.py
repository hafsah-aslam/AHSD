%matplotlib inline
# HoNE Fig. 4 - seed variability moves from LOACO to natural novelty (arrow chart, data embedded).
# Arrow: from a detector's mean seed s.d. of AUC under LOACO (circle) to its value under natural novelty (head).
# LOACO pools D1-D3; natural novelty pools D1, D3, D5. Colour depth = size of the increase.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

GROUPS = [
    ("AHSD backbone", [("MSP", 0.061328, 0.103010), ("Energy", 0.054902, 0.100012),
                       ("$P$(attack)$^\\dagger$", 0.041906, 0.082294), ("Stress", 0.018927, 0.110020),
                       ("Mahalanobis", 0.009356, 0.073890), ("$k$NN", 0.012609, 0.214969)]),
    ("GRU backbone", [("MSP", 0.049327, 0.061690), ("$P$(attack)", 0.026544, 0.026883),
                      ("Mahalanobis", 0.008164, 0.021591), ("$k$NN", 0.019107, 0.050538)]),
    ("Other", [("CLAD", 0.019751, 0.057821), ("XGBoost-s", 0.005962, 0.014463),
               ("Isolation Forest", 0.007320, 0.007883), ("Autoencoder", 0.001863, 0.017229)]),
]
OUT = Path.home() / "Downloads"
LO_C, INK, MUTED, GRID, STRIPE = "#2a78d6", "#2b2b2a", "#6b6b67", "#d9d8d3", "#f6f5f2"
CMAP = LinearSegmentedColormap.from_list("rise", ["#f6c3a8", "#eb6834", "#9c3412"])   # light = small rise
NORM = LogNorm(1, 17)

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.45, 3.0))
y, ticks, labels, row = 0.0, [], [], 0
for g, (group, rows) in enumerate(GROUPS):
    ax.text(0.00125, y, group, fontsize=6.6, weight="bold", va="center", color=MUTED)
    y += 1
    for lab, lo, nn in rows:
        if row % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=STRIPE, zorder=0, lw=0)
        r = nn / lo
        col = CMAP(NORM(max(r, 1)))
        ax.annotate("", xy=(nn, y), xytext=(lo, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=col, lw=1.6,
                                    shrinkA=2.5, shrinkB=0), zorder=2)
        ax.scatter(lo, y, s=20, facecolor="white", edgecolor=LO_C, linewidth=1.0, zorder=3)
        ax.text(nn * 1.12, y, f"{r:.1f}×", va="center", fontsize=6.2, color=INK,
                weight="bold" if r >= 5 else "normal")
        ticks.append(y); labels.append(lab)
        y += 1; row += 1
    y += 0.4
ax.set_yticks(ticks, labels)
ax.set_ylim(y - 0.9, -0.7)
ax.set_xscale("log")
ax.set_xlim(0.0012, 0.5)
ax.set_xticks([0.002, 0.01, 0.05, 0.2], ["0.002", "0.01", "0.05", "0.2"])
ax.minorticks_off()
ax.set_xlabel("Mean seed s.d. of AUC (log)  → less stable")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.scatter([], [], s=20, facecolor="white", edgecolor=LO_C, linewidth=1.0, label="LOACO")
ax.plot([], [], color="#eb6834", lw=1.6, marker=">", ms=4, label="→ natural novelty")
ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.4, handlelength=1.6)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
