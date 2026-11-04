%matplotlib inline
# HoNE Fig. 4 (alternative style) - how much more seed-variable each detector is under natural novelty.
# Ratio = (mean seed s.d. of AUC under natural novelty) / (mean seed s.d. under LOACO).
# LOACO pools D1-D3; natural novelty pools D1, D3, D5. Source: per-run results in results/final_v2.
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (detector, group, seed s.d. under LOACO, seed s.d. under natural novelty)
ROWS = [
    ("MSP", "AHSD", 0.061328, 0.103010),
    ("Energy", "AHSD", 0.054902, 0.100012),
    ("$P$(attack)$^\\dagger$", "AHSD", 0.041906, 0.082294),
    ("Stress", "AHSD", 0.018927, 0.110020),
    ("Mahalanobis", "AHSD", 0.009356, 0.073890),
    ("$k$NN", "AHSD", 0.012609, 0.214969),
    ("GRU Mahal.", "GRU", 0.008164, 0.021591),
    ("GRU $k$NN", "GRU", 0.019107, 0.050538),
    ("GRU MSP", "GRU", 0.049327, 0.061690),
    ("GRU $P$(attack)", "GRU", 0.026544, 0.026883),
    ("CLAD", "Other", 0.019751, 0.057821),
    ("XGBoost-s", "Other", 0.005962, 0.014463),
    ("Isolation Forest", "Other", 0.007320, 0.007883),
    ("Autoencoder", "Other", 0.001863, 0.017229),
]
GROUP_C = {"AHSD": "#eb6834", "GRU": "#2a78d6", "Other": "#9a9890"}
GROUP_LAB = {"AHSD": "AHSD backbone scores", "GRU": "GRU backbone scores", "Other": "Other detectors"}
OUT = Path.home() / "Downloads"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

rows = sorted(ROWS, key=lambda r: r[3] / r[2])          # smallest ratio at the bottom
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for y, (lab, g, lo, nn) in enumerate(rows):
    r = nn / lo
    ax.barh(y, r - 1, left=1, height=0.68, color=GROUP_C[g], edgecolor="white", linewidth=0.5, zorder=2)
    ax.text(r * 1.06, y, f"{r:.1f}$\\times$", va="center", ha="left", fontsize=6.4,
            fontweight="bold" if r >= 5 else "normal")
ax.axvline(1, color=INK, lw=0.8, zorder=3)
ax.text(1.04, len(rows) - 0.35, "no change", fontsize=5.8, color=MUTED, va="bottom")
ax.set_xscale("log")
ax.set_xlim(0.9, 30)
ax.set_xticks([1, 2, 5, 10, 20], ["1×", "2×", "5×", "10×", "20×"])
ax.minorticks_off()
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xlabel("Seed variability, natural novelty / LOACO (log scale)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=GROUP_C[g], label=GROUP_LAB[g]) for g in GROUP_C], ncol=3, loc="lower center",
          bbox_to_anchor=(0.42, 1.0), frameon=False, fontsize=6.1, handlelength=1.0, columnspacing=0.8)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
