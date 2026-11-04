%matplotlib inline
# HoNE Fig. 5 - D3 (ToN-IoT) classes evaluated under both protocols: mean AUC +/- seed s.d. (data embedded).
# DATA[protocol][detector] = [[mean, s.d.] for Backdoor, password, xss], over the 5 seeds.
# Source: results/final_v2/summary.json (class_evals loaco_D3 / nn_D3).
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = {
    "LOACO": {
        "AHSD stress": [[0.97944, 0.00592], [0.99335, 0.00223], [0.97449, 0.0094]],
        "Mahalanobis": [[0.98429, 0.00272], [0.99262, 0.00332], [0.98324, 0.00653]],
        "kNN": [[0.98994, 0.00097], [0.96922, 0.01213], [0.97288, 0.02879]],
        "IF": [[0.9966, 0.00053], [0.99584, 7e-05], [0.9903, 0.00041]],
        "AE": [[0.99932, 4e-05], [0.97651, 0.00097], [0.9784, 0.00075]],
        "CLAD": [[0.97081, 0.00312], [0.93989, 0.03266], [0.94671, 0.01545]],
    },
    "Natural novelty": {
        "AHSD stress": [[0.66997, 0.2667], [0.67393, 0.24832], [0.44286, 0.15145]],
        "Mahalanobis": [[0.66842, 0.17235], [0.5877, 0.09927], [0.45708, 0.12008]],
        "kNN": [[0.51007, 0.38382], [0.46243, 0.36586], [0.49342, 0.39769]],
        "IF": [[0.98269, 0.00242], [0.9883, 0.00049], [0.95766, 0.0042]],
        "AE": [[0.96781, 0.00105], [0.88559, 0.00131], [0.87704, 0.00022]],
        "CLAD": [[0.89027, 0.02761], [0.91364, 0.05425], [0.82504, 0.08161]],
    },
}
CLASSES = ["Backdoor", "password", "xss"]
MARKER = {"Backdoor": "o", "password": "s", "xss": "^"}
DETS = [("AHSD stress", "Stress"), ("Mahalanobis", "Mahal."), ("kNN", "$k$NN"), ("IF", "IF"), ("AE", "AE"),
        ("CLAD", "CLAD")]
COLOR = {"LOACO": "#2a78d6", "Natural novelty": "#eb6834"}
OFFSET = {"LOACO": -0.18, "Natural novelty": 0.18}
OUT = Path.home() / "Downloads"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "ytick.major.width": 0.5, "ytick.major.size": 2})

fig, ax = plt.subplots(figsize=(3.45, 1.75))
for i, (d, _) in enumerate(DETS):
    for proto, col in COLOR.items():
        for k, c in enumerate(CLASSES):
            m, s = DATA[proto][d][k]
            ax.errorbar(i + OFFSET[proto] + (k - 1) * 0.07, m, yerr=s, fmt=MARKER[c], ms=2.6, color=col,
                        elinewidth=0.6, capsize=0, mew=0)
ax.axhline(0.5, color=MUTED, lw=0.5, ls="--")
ax.axvline(2.5, color="#999999", lw=0.4)
ax.text(1, 0.03, "supervised backbone", ha="center", fontsize=6, color=MUTED)
ax.text(4.5, 0.03, "benign-only / contrastive", ha="center", fontsize=6, color=MUTED)
ax.set_xticks(range(len(DETS)), [lab for _, lab in DETS])
ax.tick_params(axis="x", length=0)
ax.set_ylabel("AUC (mean $\\pm$ seed s.d.)")
ax.set_ylim(0, 1.05)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.4)
ax.set_axisbelow(True)
h = [Line2D([], [], color=COLOR["LOACO"], marker="o", ls="", ms=3, label="LOACO"),
     Line2D([], [], color=COLOR["Natural novelty"], marker="o", ls="", ms=3, label="Natural novelty")]
h += [Line2D([], [], color=INK, marker=MARKER[c], ls="", ms=3, label=c) for c in CLASSES]
ax.legend(handles=h, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6,
          handletextpad=0.2, columnspacing=0.7)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_d3.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_d3.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_d3.pdf")
plt.show()
