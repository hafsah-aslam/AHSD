%matplotlib inline
# HoNE Fig. 2 - ranking of detector families under each protocol (bump chart).
# Mean AUC of each family, pooled over all classes of each protocol (rolling: all 45 class-cut units),
# averaged over the family's detectors and 5 seeds. Source: results/final_v2/summary.json (paper Sec. RQ2).
from pathlib import Path
import matplotlib.pyplot as plt

MEAN_AUC = {  # protocol -> family -> mean AUC
    "LOACO":   {"Confidence": 0.475, "Representation": 0.880, "Benign-only": 0.794, "CLAD": 0.818, "XGBoost": 0.869},
    "Natural": {"Confidence": 0.434, "Representation": 0.637, "Benign-only": 0.787, "CLAD": 0.776, "XGBoost": 0.844},
    "Rolling": {"Confidence": 0.532, "Representation": 0.623, "Benign-only": 0.748, "CLAD": 0.652, "XGBoost": 0.672},
}
PROTOCOLS = [("LOACO", "LOACO\nhold-out\n21 classes"), ("Natural", "Natural\nnovelty\n9 classes"),
             ("Rolling", "Rolling\ncuts\n45 units")]
COLOR = {"Confidence": "#2a78d6", "Representation": "#eb6834", "Benign-only": "#1baf7a",
         "CLAD": "#eda100", "XGBoost": "#e87ba4"}
INK, MUTED, BAND = "#2b2b2a", "#6b6b67", "#f3f2ee"
OUT = Path.home() / "Downloads"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "text.color": INK})

# rank 1 = highest mean AUC under that protocol
rank = {p: {f: r for r, f in enumerate(sorted(MEAN_AUC[p], key=MEAN_AUC[p].get, reverse=True), 1)}
        for p, _ in PROTOCOLS}

fig, ax = plt.subplots(figsize=(3.45, 2.35))
ax.axhspan(0.5, 1.5, color=BAND, zorder=0)                       # the "winner" row
for f, col in COLOR.items():
    xs = range(len(PROTOCOLS))
    ys = [rank[p][f] for p, _ in PROTOCOLS]
    ax.plot(xs, ys, color=col, lw=2.2, alpha=0.85, zorder=1, solid_capstyle="round")
    for x, (p, _) in zip(xs, PROTOCOLS):
        v = MEAN_AUC[p][f]
        ax.scatter(x, rank[p][f], s=300, color=col, zorder=2, edgecolor="white", linewidth=1.2)
        ax.text(x, rank[p][f], f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.2,
                color="white", weight="bold", zorder=3)
    # direct labels at both ends: no legend needed
    ax.text(-0.22, ys[0], f, ha="right", va="center", fontsize=6.6, color=col, weight="bold")
    ax.text(len(PROTOCOLS) - 1 + 0.22, ys[-1], f, ha="left", va="center", fontsize=6.6, color=col, weight="bold")

ax.set_xticks(range(len(PROTOCOLS)), [lab for _, lab in PROTOCOLS])
ax.xaxis.tick_top()
ax.set_yticks([])
ax.set_ylim(5.6, 0.4)                                            # rank 1 at the top
ax.set_xlim(-1.05, len(PROTOCOLS) - 1 + 1.05)
ax.tick_params(axis="x", length=0, labelsize=6.6, colors=INK)
for s in ax.spines.values():
    s.set_visible(False)
ax.text(len(PROTOCOLS) - 1 + 1.05, 5.55, "value = mean AUC; shaded row = winner", fontsize=5.4,
        color=MUTED, ha="right", va="bottom")
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_winners.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_winners.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("winners:", {p: min(rank[p], key=rank[p].get) for p, _ in PROTOCOLS})
plt.show()
