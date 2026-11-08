"""AHSD architecture, single-column version (3.45 in wide) for the appendix.
Drawn from src/nids/models/ahsd.py, variant 'fixed'.  Usage: python fig_ahsd_arch_col.py
Main data path runs down the right column; inputs/outputs that branch off sit in the left column.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED = "#2b2b2a", "#6b6b67"
C_IN, C_REC, C_REF, C_SC, C_OUT = "#e8f1fc", "#fde9df", "#e3f5ec", "#fdf3d6", "#f1f0ec"

W, H = 38, 13                       # box size (canvas units; canvas is 100 wide)
XL, XR = 2, 49                      # left / right column x
Y = [70, 47, 26, 3]                 # rows top -> bottom: input, gates, update, head

fig, ax = plt.subplots(figsize=(3.45, 3.0))
ax.set_xlim(0, 100); ax.set_ylim(-1, 90); ax.axis("off")


def box(x, y, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=INK, lw=0.6))
    ax.text(x + W / 2, y + H - 3.2, title, ha="center", va="center", fontsize=7, weight="bold")
    ax.text(x + W / 2, y + 4.3, sub, ha="center", va="center", fontsize=5.8, linespacing=1.2)


def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.65, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


box(XL, Y[0], "Flow window", "$T{=}32$ flows,\n41 features each", C_IN)
box(XR, Y[0], "Embedding", "Linear, LayerNorm\n$z_t\\in\\mathbb{R}^{128}$", C_IN)
box(XL, Y[1], "Reference $E$", "mean benign\nembedding (fixed)", C_REF)
box(XR, Y[1], "Gates", "$\\alpha_t,\\beta_t=$\n$\\sigma(W_g[z_t,h_{t-1}])$", C_REC)
box(XL, Y[2], "Stress $s$", "mean $|h_t-E|$\n(also a score)", C_SC)
box(XR, Y[2], "State update", "pull $h_t$ toward $z_t$,\nrelax back to $E$", C_REC)
box(XL, Y[3], "Hidden layer", "128-d features\nMahalanobis, $k$NN", C_OUT)
box(XR, Y[3], "Head", "MLP on $[h_T,\\bar h,s]$\n$P$(attack), MSP, Energy", C_OUT)

# dashed frame around the recurrence (rows 2-3)
ax.add_patch(FancyBboxPatch((XL - 1.5, Y[2] - 2.6), 99.3 - XL, Y[1] + H - Y[2] + 5.4,
                            boxstyle="round,pad=0,rounding_size=1.5", fc="none", ec=MUTED, lw=0.6, ls="--"))
ax.text(XL - 1, Y[1] + H + 6.2, "recurrence, for each flow $t=1,\\dots,T$; $h_0=E$", ha="left", va="center",
        fontsize=5.8, color=MUTED, style="italic")

arrow((XL + W, Y[0] + H / 2), (XR, Y[0] + H / 2))                         # window -> embedding
arrow((XR + W / 2, Y[0]), (XR + W / 2, Y[1] + H))                         # z_t -> gates
ax.text(XR + W / 2 + 1.5, Y[0] - 5, "$z_t$", fontsize=6, va="center")
arrow((XL + W, Y[1] + H / 2), (XR, Y[1] + H / 2))                         # E -> gates
arrow((XR + W / 2, Y[1]), (XR + W / 2, Y[2] + H))                         # gates -> update
ax.text(XR + W / 2 - 1.5, Y[1] - 4, "$\\alpha_t,\\beta_t$", fontsize=6, ha="right", va="center")
arrow((XR + W, Y[2] + H / 2), (XR + W, Y[1] + H / 2), rad=0.75)           # feedback h_{t-1}
ax.text(XR + W - 1, (Y[1] + Y[2] + H) / 2, "$h_{t-1}$", fontsize=6, ha="right", va="center")
arrow((XR, Y[2] + H / 2), (XL + W, Y[2] + H / 2))                         # h -> stress
arrow((XR + W / 2, Y[2]), (XR + W / 2, Y[3] + H))                         # h_T, mean h -> head
ax.text(XR + W / 2 + 1.5, Y[2] - 5.3, "$h_T,\\ \\bar h$", fontsize=6, va="center")
arrow((XL + W - 4, Y[2]), (XR + 3, Y[3] + H))                             # stress -> head
arrow((XR, Y[3] + H / 2), (XL + W, Y[3] + H / 2))                         # head -> hidden layer

out = Path(__file__).resolve().parent
fig.savefig(out / "fig_ahsd_arch_col.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_ahsd_arch_col.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", out / "fig_ahsd_arch_col.pdf")
