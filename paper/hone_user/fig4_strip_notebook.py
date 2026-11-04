%matplotlib inline
# HoNE Fig. 4 - seed variability per attack class, LOACO vs natural novelty (strip plot, data embedded).
# Each dot = one attack class: standard deviation of the detector's AUC across the 5 seeds.
# LOACO: 20 classes (D1-D3); natural novelty: 9 classes (D1, D3, D5). Large marker = mean over classes.
# Source: per-run results in results/final_v2 (best checkpoint).
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (detector, per-class seed s.d. under LOACO, per-class seed s.d. under natural novelty)
ROWS = [
    ("MSP", [0.02568, 0.12186, 0.23514, 0.02299, 0.11434, 0.03824, 0.02175, 0.01291, 0.02039, 0.01386, 0.01258, 0.01148, 0.01794, 0.05892, 0.18917, 0.02235, 0.01259, 0.08638, 0.08258, 0.10543],
     [0.10148, 0.00652, 0.08745, 0.09064, 0.1427, 0.196, 0.06547, 0.00891, 0.22792]),
    ("Energy", [0.01168, 0.05173, 0.17665, 0.01935, 0.14168, 0.03213, 0.03237, 0.00787, 0.04347, 0.02221, 0.00958, 0.01738, 0.00925, 0.05586, 0.1878, 0.04446, 0.01615, 0.07033, 0.07356, 0.0745],
     [0.09907, 0.00823, 0.10485, 0.06217, 0.12554, 0.18471, 0.10602, 0.0063, 0.20323]),
    ("$P$(attack)$^\\dagger$", [0.02473, 0.1156, 0.32592, 0.02294, 0.08867, 0.03907, 0.0003, 9e-05, 0.00017, 8e-05, 0.00029, 0.00135, 0.00014, 0.00018, 0.01254, 0.08134, 0.01013, 0.00569, 0.01944, 0.08948],
     [0.10442, 0.00641, 0.0625, 0.03777, 0.06866, 0.14683, 0.08441, 0.00185, 0.22779]),
    ("Stress", [0.08049, 0.0039, 0.04679, 0.02322, 0.02162, 0.0169, 0.01359, 0.01338, 0.01719, 0.02336, 0.00877, 0.01859, 0.01316, 0.01811, 0.0094, 0.00912, 0.02692, 0.00592, 0.00223, 0.00591],
     [0.02502, 0.00479, 0.2667, 0.10736, 0.24832, 0.18058, 0.15145, 0.00156, 0.00439]),
    ("Mahalanobis", [0.03543, 0.00205, 0.02759, 0.00946, 0.02926, 0.0318, 0.0021, 0.0001, 0.00049, 0.00108, 0.00038, 0.00447, 0.00011, 0.00114, 0.00653, 0.01169, 0.01264, 0.00272, 0.00332, 0.00475],
     [0.02563, 0.00504, 0.17235, 0.13836, 0.09927, 0.09802, 0.12008, 0.00111, 0.00513]),
    ("$k$NN", [0.02725, 0.00511, 0.07966, 0.0163, 0.01059, 0.02825, 0.00082, 0.00019, 0.00025, 0.00016, 0.00034, 0.00133, 0.00019, 0.00037, 0.02879, 0.01426, 0.00897, 0.00097, 0.01213, 0.01624],
     [0.03047, 0.00929, 0.38382, 0.13955, 0.36586, 0.32948, 0.39769, 0.09143, 0.18711]),
    ("GRU Mahal.", [0.04338, 0.00485, 0.02418, 0.00929, 0.02716, 0.01635, 8e-05, 4e-05, 4e-05, 5e-05, 9e-05, 0.00164, 1e-05, 0.00017, 0.00177, 0.01657, 0.00846, 0.00252, 0.00143, 0.00518],
     [0.02586, 0.00713, 0.00787, 0.04065, 0.01396, 0.06416, 0.01737, 0.0013, 0.01602]),
    ("GRU $k$NN", [0.05185, 0.01346, 0.03653, 0.01424, 0.07222, 0.03747, 0.00051, 0.00011, 0.00015, 0.00313, 0.00078, 0.00458, 0.00072, 0.00022, 0.02021, 0.08327, 0.00955, 0.0071, 0.01138, 0.01461],
     [0.04569, 0.00787, 0.01574, 0.13341, 0.03959, 0.05194, 0.01348, 0.12086, 0.02626]),
    ("GRU MSP", [0.06192, 0.07896, 0.09271, 0.02126, 0.08288, 0.05975, 0.07631, 0.01992, 0.01554, 0.04064, 0.01959, 0.02298, 0.03842, 0.03674, 0.05892, 0.04124, 0.02551, 0.05454, 0.03698, 0.10175],
     [0.05627, 0.011, 0.04633, 0.12239, 0.03481, 0.14814, 0.01749, 0.09985, 0.01893]),
    ("GRU $P$(attack)", [0.06304, 0.00707, 0.09271, 0.0205, 0.10887, 0.06004, 7e-05, 8e-05, 4e-05, 5e-05, 4e-05, 0.00047, 0.0, 0.00017, 0.00154, 0.08426, 0.02627, 0.01416, 0.02161, 0.02989],
     [0.05427, 0.00741, 0.00641, 0.06043, 0.01476, 0.0228, 0.01546, 0.00235, 0.05806]),
    ("CLAD", [0.02269, 0.00368, 0.12367, 0.01711, 0.05987, 0.02027, 0.00256, 0.00287, 0.00574, 0.008, 0.0055, 0.00228, 0.00188, 0.00234, 0.01545, 0.03614, 0.01332, 0.00312, 0.03266, 0.01585],
     [0.04555, 0.01291, 0.02761, 0.06676, 0.05425, 0.044, 0.08161, 0.00182, 0.18588]),
    ("XGBoost-s", [0.00861, 0.01686, 0.03973, 0.00808, 0.01864, 0.01052, 0.00029, 3e-05, 5e-05, 0.00035, 0.00019, 0.00021, 2e-05, 0.0001, 0.00031, 0.00971, 0.00297, 0.002, 0.00033, 0.00022],
     [0.01073, 0.00692, 0.0073, 0.00703, 0.00632, 0.02202, 0.00371, 0.00065, 0.0655]),
    ("Isolation Forest", [0.00893, 0.00809, 0.02009, 0.00356, 0.02303, 0.01316, 0.01085, 0.00454, 0.00889, 0.01361, 0.00391, 0.00329, 0.00469, 0.00798, 0.00041, 0.00651, 0.00409, 0.00053, 7e-05, 0.00016],
     [0.01614, 0.00395, 0.00242, 0.00677, 0.00049, 0.00196, 0.0042, 0.0304, 0.00461]),
    ("Autoencoder", [0.0018, 0.00167, 0.00732, 0.00097, 0.0024, 0.00074, 0.00152, 0.00217, 0.00105, 0.00453, 0.00088, 0.0016, 0.00121, 0.00165, 0.00075, 0.00284, 0.00095, 4e-05, 0.00097, 0.00219],
     [0.00164, 0.00129, 0.00105, 0.00755, 0.00131, 0.00021, 0.00022, 0.01845, 0.12332]),
]
GROUPS = [(0, 6, "AHSD"), (6, 10, "GRU"), (10, 14, "Other")]   # row ranges of each detector group
OUT = Path.home() / "Downloads"
LO_C, NN_C = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#2b2b2a", "#6b6b67", "#d9d8d3"
FLOOR = 3e-4            # s.d. below this (near-deterministic classes) is drawn at the floor on the log axis

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.major.width": 0.5, "xtick.major.size": 2})

rng = np.random.default_rng(0)
n = len(ROWS)
fig, ax = plt.subplots(figsize=(3.45, 3.0))
for k, (a, b, name) in enumerate(GROUPS):              # alternate background bands per detector group
    if k % 2 == 0:
        ax.axhspan(n - b - 0.5, n - a - 0.5, color="#f6f5f2", zorder=0, lw=0)
for i, (lab, lo, nn) in enumerate(ROWS):
    y = n - 1 - i
    for vals, dy, col in ((lo, 0.17, LO_C), (nn, -0.17, NN_C)):
        v = np.maximum(vals, FLOOR)
        ax.scatter(v, y + dy + rng.uniform(-0.07, 0.07, len(v)), s=5, color=col, alpha=0.5, linewidth=0, zorder=2)
    mlo, mnn = np.mean(lo), np.mean(nn)
    ax.plot([mlo, mnn], [y + 0.17, y - 0.17], color=INK, lw=0.6, zorder=3)
    ax.scatter([mlo], [y + 0.17], s=26, marker="D", color=LO_C, edgecolor=INK, linewidth=0.6, zorder=4)
    ax.scatter([mnn], [y - 0.17], s=26, marker="D", color=NN_C, edgecolor=INK, linewidth=0.6, zorder=4)
    r = mnn / mlo
    ax.text(1.35, y, f"{r:.1f}$\\times$", va="center", ha="right", fontsize=6.4,
            fontweight="bold" if r >= 5 else "normal")
ax.text(1.35, n - 0.3, "mean\nratio", ha="right", va="bottom", fontsize=5.6, color=MUTED, linespacing=0.9)
ax.set_xscale("log")
ax.set_xlim(FLOOR * 0.7, 1.4)
ax.set_xticks([FLOOR, 1e-3, 1e-2, 1e-1], ["\u22640.0003", "0.001", "0.01", "0.1"])
ax.minorticks_off()
ax.set_yticks(range(n), [lab for lab, _, _ in ROWS][::-1])
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlabel("Seed s.d. of AUC per attack class (log scale)")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
h = [Line2D([], [], ls="", marker="o", ms=3, color=LO_C, alpha=0.6, label="LOACO class"),
     Line2D([], [], ls="", marker="o", ms=3, color=NN_C, alpha=0.6, label="Natural-novelty class"),
     Line2D([], [], ls="", marker="D", ms=4, mfc="white", mec=INK, label="mean")]
ax.legend(handles=h, ncol=3, loc="lower center", bbox_to_anchor=(0.38, 1.0), frameon=False, fontsize=6.2,
          handletextpad=0.2, columnspacing=0.9)
fig.tight_layout(pad=0.2)
fig.savefig(OUT / "fig_seedsd.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_seedsd.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_seedsd.pdf")
plt.show()
