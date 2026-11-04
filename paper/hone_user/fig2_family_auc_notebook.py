%matplotlib inline
# HoNE Fig. 2 - detector-family AUC under each protocol: every class shown, mean with 95% bootstrap CI.
# UNITS[protocol][family] = AUC of that family on each evaluated attack class (rolling: class-cut unit),
# averaged over the family's detectors and the 5 seeds. Source: results/final_v2/summary.json.
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

UNITS = {
    "LOACO": {
        "confidence": [0.65935, 0.56291, 0.50448, 0.43828, 0.50204, 0.65181, 0.37262, 0.3729, 0.39703, 0.42968, 0.38257, 0.55675, 0.51248, 0.48871, 0.47069, 0.31595, 0.42338, 0.38831, 0.44632, 0.51415, 0.58022],
        "representation": [0.82257, 0.86578, 0.57638, 0.94108, 0.56342, 0.60369, 0.9917, 0.97925, 0.98329, 0.97187, 0.98636, 0.97184, 0.96294, 0.96746, 0.98456, 0.79236, 0.98406, 0.98506, 0.56049, 0.97687, 0.99823],
        "benign-only": [0.49448, 0.61459, 0.83054, 0.8581, 0.49622, 0.6646, 0.82576, 0.75482, 0.77316, 0.78825, 0.83788, 0.86442, 0.77654, 0.83455, 0.99868, 0.87248, 0.98726, 0.98418, 0.59285, 0.98151, 0.83339],
        "CLAD": [0.66224, 0.58371, 0.67034, 0.94148, 0.49723, 0.56521, 0.96353, 0.94492, 0.93953, 0.941, 0.95734, 0.95446, 0.95238, 0.94061, 0.97081, 0.37159, 0.95054, 0.93989, 0.50188, 0.94671, 0.97226],
        "XGBoost": [0.73121, 0.80879, 0.58244, 0.83352, 0.54288, 0.59195, 0.99953, 0.99509, 0.99857, 0.99795, 0.99965, 0.99752, 0.9954, 0.99711, 0.96657, 0.69428, 0.99803, 0.99516, 0.52515, 0.9954, 1.0],
    },
    "Natural": {
        "confidence": [0.59119, 0.46767, 0.40663, 0.57221, 0.43702, 0.52145, 0.39986, 0.36302, 0.1495],
        "representation": [0.68115, 0.56412, 0.61616, 0.53461, 0.57468, 0.42174, 0.46445, 0.97677, 0.90247],
        "benign-only": [0.50676, 0.50302, 0.96487, 0.7932, 0.91298, 0.92302, 0.89566, 0.74888, 0.83824],
        "CLAD": [0.62952, 0.51913, 0.89027, 0.64139, 0.91364, 0.84223, 0.82504, 0.98606, 0.74109],
        "XGBoost": [0.51196, 0.535, 0.95783, 0.76604, 0.98659, 0.91595, 0.98634, 0.99631, 0.93925],
    },
    "Rolling": {
        "confidence": [0.3478, 0.52165, 0.65172, 0.39379, 0.79924, 0.44327, 0.59318, 0.46944, 0.60219, 0.75527, 0.45508, 0.70409, 0.58874, 0.46763, 0.49287, 0.58184, 0.40814, 0.5222, 0.53436, 0.45868, 0.38098, 0.48539, 0.42574, 0.46893, 0.37899, 0.53816, 0.52821, 0.55207, 0.59482, 0.59731, 0.4901, 0.94857, 0.625, 0.17098, 0.67015, 0.59312, 0.71625, 0.49702, 0.35803, 0.21434, 0.68833, 0.61441, 0.73935, 0.36227, 0.50764],
        "representation": [0.65508, 0.65483, 0.70674, 0.97317, 0.76245, 0.56182, 0.60095, 0.64085, 0.57978, 0.65868, 0.5506, 0.58869, 0.69102, 0.55273, 0.64112, 0.33598, 0.6052, 0.39427, 0.33841, 0.16559, 0.76648, 0.72233, 0.71611, 0.64186, 0.67114, 0.39045, 0.486, 0.35555, 0.28169, 0.28272, 0.51456, 0.98549, 0.56165, 0.97125, 0.60738, 0.57025, 0.64884, 0.51603, 0.97675, 0.92705, 0.61836, 0.56045, 0.64007, 0.99248, 0.97304],
        "benign-only": [0.53533, 0.61428, 0.84992, 0.47856, 0.89946, 0.51011, 0.78174, 0.50139, 0.6004, 0.83566, 0.4964, 0.76785, 0.50427, 0.48614, 0.48862, 0.99157, 0.71583, 0.91927, 0.94836, 0.87629, 0.98223, 0.82717, 0.94903, 0.95474, 0.93465, 0.96136, 0.77574, 0.90139, 0.91643, 0.88436, 0.52269, 0.75608, 0.80873, 0.87875, 0.69095, 0.58272, 0.68717, 0.5407, 0.77182, 0.85488, 0.6999, 0.61129, 0.69826, 0.77255, 0.91026],
        "CLAD": [0.60441, 0.5425, 0.57806, 0.9846, 0.76595, 0.51013, 0.50623, 0.61762, 0.51999, 0.75075, 0.51029, 0.64971, 0.63734, 0.52312, 0.63106, 0.7803, 0.37828, 0.79506, 0.56449, 0.32438, 0.93001, 0.7478, 0.92407, 0.88402, 0.87797, 0.45675, 0.55933, 0.37209, 0.47245, 0.41861, 0.46213, 0.88968, 0.71825, 0.31095, 0.69112, 0.59085, 0.70913, 0.5247, 0.9861, 0.79301, 0.69493, 0.62169, 0.7038, 0.98809, 0.84605],
        "XGBoost": [0.5031, 0.52575, 0.68799, 0.9981, 0.74496, 0.43632, 0.61139, 0.44091, 0.60851, 0.89864, 0.50723, 0.8411, 0.48142, 0.50682, 0.53847, 0.05947, 0.54312, 0.30427, 0.26253, 0.14513, 0.97087, 0.84981, 0.98657, 0.93686, 0.98755, 0.91087, 0.69702, 0.98246, 0.85916, 0.98535, 0.51509, 0.51606, 0.52364, 0.44666, 0.68556, 0.48511, 0.70436, 0.55025, 0.99595, 0.9727, 0.72896, 0.59024, 0.74749, 0.99732, 0.98179],
    },
}

OUT = Path.home() / "Downloads"
FAMILIES = [("confidence", "Confidence"), ("representation", "Represent."), ("benign-only", "Benign-only"),
            ("CLAD", "CLAD"), ("XGBoost", "XGBoost")]
PROTOCOLS = [("LOACO", "LOACO (21 classes)", "#2a78d6"), ("Natural", "Natural novelty (9 classes)", "#eb6834"),
             ("Rolling", "Rolling cuts (45 units)", "#1baf7a")]
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"
N_BOOT = 2000

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": INK, "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "ytick.major.width": 0.5, "ytick.major.size": 2})

rng = np.random.default_rng(0)


def mean_ci(v):
    v = np.asarray(v)
    b = v[rng.integers(0, len(v), (N_BOOT, len(v)))].mean(axis=1)
    return v.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


stats = {p: {f: mean_ci(UNITS[p][f]) for f, _ in FAMILIES} for p, _, _ in PROTOCOLS}
winner = {p: max(stats[p], key=lambda f: stats[p][f][0]) for p, _, _ in PROTOCOLS}

fig, ax = plt.subplots(figsize=(3.45, 2.6))
ax.axhline(0.5, color=MUTED, lw=0.6, ls="--", zorder=0.5)
offsets = (-0.27, 0.0, 0.27)
for j, (f, _) in enumerate(FAMILIES):
    if j % 2:                                              # light background band for every other family
        ax.axvspan(j - 0.5, j + 0.5, color="#f6f5f2", zorder=0, lw=0)
    for (p, _, col), dx in zip(PROTOCOLS, offsets):
        v = np.asarray(UNITS[p][f])
        x = j + dx
        ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, color=col, alpha=0.45, linewidth=0, zorder=1)
        m, lo, hi = stats[p][f]
        ax.vlines(x, lo, hi, color=INK, lw=0.9, zorder=3)
        ax.scatter(x, m, s=22, color=col, edgecolor=INK, linewidth=0.7, zorder=4)
        if winner[p] == f:                                 # winner of this protocol
            ax.scatter(x, 1.05, marker="*", s=55, color=col, edgecolor=INK, linewidth=0.4, zorder=4)
ax.set_xticks(range(len(FAMILIES)), [n for _, n in FAMILIES], fontsize=6.6)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.5, len(FAMILIES) - 0.5)
ax.set_ylim(0, 1.1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("AUC per attack class")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.4)
ax.set_axisbelow(True)
h = [Line2D([], [], ls="", marker="o", ms=4.5, mfc=col, mec=INK, mew=0.6, label=lab) for _, lab, col in PROTOCOLS]
h.append(Line2D([], [], ls="", marker="*", ms=7, mfc=INK, mec=INK, label="protocol winner"))
h.append(Line2D([], [], ls="--", lw=0.6, color=MUTED, label="chance (AUC 0.5)"))
fig.legend(handles=h, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.965), frameon=False, fontsize=6.2,
           columnspacing=1.0, handletextpad=0.3)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_winners.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_winners.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
for p, _, _ in PROTOCOLS:
    print(p, {f: f"{stats[p][f][0]:.3f} [{stats[p][f][1]:.3f}, {stats[p][f][2]:.3f}]" for f, _ in FAMILIES})
plt.show()
