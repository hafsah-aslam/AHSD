%matplotlib inline
# HoNE Fig. 7 - in-distribution AUC on the grouped (random) and temporal splits (arrow plot, data embedded).
# Isolates temporal shift without novelty: same attack classes in training and test, only the split changes.
# Source: results/final_v2/summary.json (indist[...]["best"][detector]["test"]["auc"]["mean"]).
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# detector -> dataset -> (AUC on grouped split, AUC on temporal split)
DATA = {
    "$P$(attack)$^\\dagger$ (AHSD)": {"D1": (0.81160, 0.52084), "D2": (0.99886, 0.99690),
                                       "D3": (0.97466, 0.66391), "D5": (0.83063, 0.94838)},
    "XGBoost": {"D1": (0.81455, 0.54712), "D2": (0.99833, 0.98416),
                "D3": (0.97662, 0.97820), "D5": (0.70450, 0.98977)},
}
DATASETS = ["D1", "D2", "D3", "D5"]
G_C, T_C = "#2a78d6", "#eb6834"
OUT = Path.home() / "Downloads"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "ytick.major.width": 0.5, "ytick.major.size": 2})

fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.45), sharey=True)
for ax, (name, vals) in zip(axes, DATA.items()):
    for i, ds in enumerate(DATASETS):
        g, t = vals[ds]
        ax.annotate("", xy=(i, t), xytext=(i, g),            # arrow: grouped -> temporal
                    arrowprops=dict(arrowstyle="-|>", color=GRID if abs(t - g) < 0.02 else MUTED, lw=0.9,
                                    mutation_scale=7, shrinkA=3, shrinkB=3))
        ax.scatter([i], [g], s=18, color=G_C, zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter([i], [t], s=18, color=T_C, zorder=3, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.set_xlim(-0.5, len(DATASETS) - 0.5)
    ax.set_title(name, fontsize=7)
    ax.set_ylim(0.45, 1.03)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("In-distribution AUC")
h = [Line2D([], [], marker="o", ls="", ms=4, color=G_C, label="grouped (random) split"),
     Line2D([], [], marker="o", ls="", ms=4, color=T_C, label="temporal split")]
fig.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=6.5)
fig.tight_layout(pad=0.2, w_pad=0.6)
fig.savefig(OUT / "fig_temporal.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_temporal.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_temporal.pdf")
plt.show()
