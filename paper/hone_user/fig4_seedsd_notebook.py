%matplotlib inline
# HoNE Fig. 4 - seed variability under LOACO vs natural novelty (dumbbell plot, data embedded).
# Value = mean over attack classes of the standard deviation of AUC across the 5 seeds.
# LOACO pools D1-D3; natural novelty pools D1, D3, D5. Source: per-run results in results/final_v2.
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (detector, seed s.d. under LOACO, seed s.d. under natural novelty)
ROWS = [
    ("MSP", 0.061328, 0.103010),
    ("Energy", 0.054902, 0.100012),
    ("$P$(attack)$^\\dagger$", 0.041906, 0.082294),
    ("Stress", 0.018927, 0.110020),
    ("Mahalanobis", 0.009356, 0.073890),
    ("$k$NN", 0.012609, 0.214969),
    ("GRU Mahal.", 0.008164, 0.021591),
    ("GRU $k$NN", 0.019107, 0.050538),
    ("GRU MSP", 0.049327, 0.061690),
    ("GRU $P$(attack)", 0.026544, 0.026883),
    ("CLAD", 0.019751, 0.057821),
    ("XGBoost-s", 0.005962, 0.014463),
    ("Isolation Forest", 0.007320, 0.007883),
    ("Autoencoder", 0.001863, 0.017229),
]
OUT = Path.home() / "Downloads"
LO_C, NN_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.4, 2.35))
n = len(ROWS)
for i, (lab, lo, nn) in enumerate(ROWS):
    y = n - 1 - i
    ax.plot([lo, nn], [y, y], color=GRID, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter([lo], [y], s=22, color=LO_C, zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter([nn], [y], s=22, color=NN_C, zorder=3, edgecolor="white", linewidth=0.6)
    ax.text(0.62, y, f"{nn / lo:.1f}$\\times$", va="center", ha="right", fontsize=6.5,
            fontweight="bold" if nn / lo >= 5 else "normal")      # bold: at least 5x more variable
ax.set_yticks(range(n), [lab for lab, _, _ in ROWS][::-1])
ax.set_xscale("log")
ax.set_xlim(0.0015, 0.65)
ax.set_xticks([0.002, 0.01, 0.05, 0.2], ["0.002", "0.01", "0.05", "0.2"])
ax.minorticks_off()
ax.set_xlabel("Mean seed s.d. of AUC (log scale)")
ax.text(0.62, n - 0.35, "ratio", ha="right", fontsize=6, color=MUTED)
for yy in (3.5, 7.5):                     # separates AHSD scores, GRU scores, and the rest
    ax.axhline(yy, color=GRID, lw=0.5, ls=":")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
h = [Line2D([], [], marker="o", ls="", ms=4, color=LO_C, label="LOACO (D1\u2013D3)"),
     Line2D([], [], marker="o", ls="", ms=4, color=NN_C, label="Natural novelty (D1, D3, D5)")]
ax.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
