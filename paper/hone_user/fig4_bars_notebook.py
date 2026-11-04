%matplotlib inline
# HoNE Fig. 4 - seed variability of AUC under LOACO and natural novelty (paired horizontal bars, data embedded).
# Value = mean over attack classes of the s.d. of AUC across 5 seeds. LOACO pools D1-D3; natural novelty pools D1, D3, D5.
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# (group, [(detector, s.d. LOACO, s.d. natural novelty), ...])
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
LO_C, NN_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"
BAR, GAP = 0.36, 0.7          # bar height; extra space before each group

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.45, 3.1))
y, ticks, labels = 0.0, [], []
for g, (group, rows) in enumerate(GROUPS):
    ax.text(-0.005, y - 0.15, group, fontsize=6.6, weight="bold", ha="right", va="center", color=INK)
    y += 0.75
    for lab, lo, nn in rows:
        ax.barh(y - BAR / 2, lo, height=BAR, color=LO_C, edgecolor="white", linewidth=0.4)
        ax.barh(y + BAR / 2, nn, height=BAR, color=NN_C, edgecolor="white", linewidth=0.4)
        r = nn / lo
        ax.text(nn + 0.003, y + BAR / 2, f"{r:.1f}×", va="center", fontsize=6.2,
                weight="bold" if r >= 5 else "normal", color=INK)
        ticks.append(y); labels.append(lab)
        y += 1.0
    if g < len(GROUPS) - 1:
        ax.axhline(y - 0.5 + GAP / 2, color=GRID, lw=0.6)
    y += GAP
ax.set_yticks(ticks, labels)
ax.set_ylim(y - GAP - 0.4, -0.6)
ax.set_xlim(0, 0.25)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20], ["0", "0.05", "0.10", "0.15", "0.20"])
ax.set_xlabel("Mean seed s.d. of AUC (higher = less stable)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(handles=[Patch(color=LO_C, label="LOACO"), Patch(color=NN_C, label="Natural novelty")],
          ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6.5, handlelength=1.2)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
