%matplotlib inline
# HoNE Fig. 3 - family mean AUC per evaluation (heatmap, data embedded).
# Each value: mean AUC of the family over the evaluation's attack classes and 5 seeds.
# Source: results/final_v2/summary.json (class_evals[...]["families"]).
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# (evaluation, number of attack classes, [Confidence, Representation, Benign-only, CLAD, XGBoost])
DATA = [
    ("LOACO D1", 6, [0.55315, 0.72882, 0.65975, 0.65337, 0.6818]),
    ("LOACO D2", 8, [0.43909, 0.97684, 0.80692, 0.94922, 0.9976]),
    ("LOACO D3", 6, [0.42647, 0.88057, 0.90283, 0.78024, 0.86243]),
    ("Purged D5", 1, [0.58022, 0.99823, 0.83339, 0.97226, 1.0]),
    ("Natural D1", 2, [0.52943, 0.62263, 0.50489, 0.57433, 0.52348]),
    ("Natural D3", 5, [0.46743, 0.52233, 0.89795, 0.82251, 0.92255]),
    ("Natural D5", 2, [0.25626, 0.93962, 0.79356, 0.86357, 0.96778]),
]
FAMILIES = ["Conf.", "Repr.", "Benign", "CLAD", "XGB"]
NOVELTY = range(4)          # the leader is chosen among the four novelty families (XGBoost is a reference)
OUT = Path.home() / "Downloads"
INK, ORANGE = "#2b2b2a", "#eb6834"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK})

M = np.array([v for _, _, v in DATA])
fig, ax = plt.subplots(figsize=(3.4, 2.05))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9ec5f4", "#2a78d6", "#0d366b"])
ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
for i in range(len(DATA)):
    lead = max(NOVELTY, key=lambda j: M[i, j])
    for j in range(len(FAMILIES)):
        v = M[i, j]
        ax.text(j, i, f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v > 0.62 else INK, fontweight="bold" if j == lead else "normal")
        if j == lead:   # box the leading novelty family of each evaluation
            ax.add_patch(plt.Rectangle((j - 0.48, i - 0.46), 0.96, 0.92, fill=False, ec=ORANGE, lw=1.4))
ax.set_xticks(range(len(FAMILIES)), FAMILIES)
ax.set_yticks(range(len(DATA)), [f"{name} ({n})" for name, n, _ in DATA])
ax.axhline(3.5, color="white", lw=2.5)   # separates LOACO evaluations from natural-novelty ones
ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_family.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_family.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_family.pdf")
plt.show()
