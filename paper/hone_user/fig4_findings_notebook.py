%matplotlib inline
# HoNE Fig. 4 - seed variability, LOACO vs natural novelty, with the finding of each detector group written on the plot.
# Value = mean over attack classes of the s.d. of AUC across 5 seeds. LOACO pools D1-D3; natural novelty pools D1, D3, D5.
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# group header (the finding), then (detector, s.d. LOACO, s.d. natural novelty)
GROUPS = [
    ("AHSD backbone: every score less stable, up to 17×", [
        ("$k$NN", 0.012609, 0.214969), ("Mahalanobis", 0.009356, 0.073890), ("Stress", 0.018927, 0.110020),
        ("$P$(attack)$^\\dagger$", 0.041906, 0.082294), ("Energy", 0.054902, 0.100012), ("MSP", 0.061328, 0.103010)]),
    ("GRU backbone: smaller rise, at most 2.6×", [
        ("GRU Mahal.", 0.008164, 0.021591), ("GRU $k$NN", 0.019107, 0.050538), ("GRU MSP", 0.049327, 0.061690),
        ("GRU $P$(attack)", 0.026544, 0.026883)]),
    ("Other detectors: benign-only stay below 0.02", [
        ("CLAD", 0.019751, 0.057821), ("XGBoost-s", 0.005962, 0.014463), ("Autoencoder", 0.001863, 0.017229),
        ("Isolation Forest", 0.007320, 0.007883)]),
]
OUT = Path.home() / "Downloads"
LO_C, NN_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BAND = "#2b2b2a", "#6b6b67", "#d9d8d3", "#f6f5f2"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.45, 3.25))
y, ticks, labels = 0, [], []
for k, (header, rows) in enumerate(GROUPS):
    top = y
    ax.text(0.0013, y, header, fontsize=6.4, weight="bold", color=INK, va="center")   # the group's finding
    y += 1
    for lab, lo, nn in rows:
        r = nn / lo
        ax.plot([lo, nn], [y, y], color=NN_C if r >= 2 else GRID, lw=1.8, alpha=0.45 if r >= 2 else 1,
                solid_capstyle="round", zorder=1)
        ax.scatter(lo, y, s=24, color=LO_C, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(nn, y, s=24, color=NN_C, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(0.62, y, f"{r:.1f}×", va="center", ha="right", fontsize=6.4, weight="bold" if r >= 5 else "normal")
        ticks.append(y); labels.append(lab)
        if lab == "$k$NN":     # callout for the strongest effect
            ax.annotate("17× more variable:\nretraining moves AUC a lot", xy=(nn, y), xytext=(0.0065, y + 1.55),
                        fontsize=5.6, color=NN_C, arrowprops=dict(arrowstyle="->", color=NN_C, lw=0.6))
        y += 1
    if k % 2 == 0:
        ax.axhspan(top - 0.5, y - 0.5, color=BAND, zorder=0, lw=0)
    y += 0.3
ax.set_yticks(ticks, labels)
ax.set_ylim(y - 0.6, -0.6)
ax.set_xscale("log")
ax.set_xlim(0.0012, 0.65)
ax.set_xticks([0.002, 0.01, 0.05, 0.2], ["0.002", "0.01", "0.05", "0.2"])
ax.minorticks_off()
ax.set_xlabel("Seed s.d. of AUC (log scale)  → less stable")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
h = [Line2D([], [], marker="o", ls="", ms=4, color=LO_C, label="LOACO"),
     Line2D([], [], marker="o", ls="", ms=4, color=NN_C, label="Natural novelty"),
     Line2D([], [], ls="", label="ratio →")]
ax.legend(handles=h[:2], ncol=2, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
