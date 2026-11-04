%matplotlib inline
# HoNE Fig. 6 - confidence scores in binary detectors (grouped bars, data embedded).
# Mean AUC over the evaluation's attack classes and 5 seeds. MSP and energy invert (fall below chance) where the
# classifier detects the unseen attacks; P(attack) on the same models does not.
# Source: results/final_v2/summary.json (a7 evals cells; class_evals representation family).
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# evaluation -> [MSP, Energy, P(attack), Representation family]
DATA = [
    ("LOACO D2",   [0.16304, 0.15501, 0.99922, 0.97684]),
    ("LOACO D3",   [0.26880, 0.25836, 0.75224, 0.88057]),
    ("Natural D3", [0.26234, 0.28709, 0.85287, 0.52233]),
    ("Natural D5", [0.10322, 0.09040, 0.57517, 0.93962]),
]
SERIES = [("MSP", "#2a78d6"), ("Energy", "#1baf7a"), ("$P$(attack)$^\\dagger$", "#eb6834"),
          ("Repr. family", "#e87ba4")]
OUT = Path.home() / "Downloads"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "ytick.major.width": 0.5, "ytick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.4, 1.55))
w = 0.19
for k, (name, col) in enumerate(SERIES):
    xs = np.arange(len(DATA)) + (k - 1.5) * (w + 0.015)
    ax.bar(xs, [v[k] for _, v in DATA], width=w, color=col, label=name, edgecolor="white", linewidth=0.5)
ax.axhline(0.5, color=INK, lw=0.6, ls="--")
ax.text(-0.5, 0.515, "chance", fontsize=5.5, color=MUTED, ha="left", va="bottom")
ax.set_xticks(range(len(DATA)), [n for n, _ in DATA])
ax.set_xlim(-0.55, len(DATA) - 0.45)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Mean AUC")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.5,
          handlelength=1.0, columnspacing=0.9)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_conf.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_conf.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_conf.pdf")
plt.show()
