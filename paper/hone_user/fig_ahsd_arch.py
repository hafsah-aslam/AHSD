"""AHSD architecture diagram (appendix figure). Drawn from src/nids/models/ahsd.py, variant 'fixed'.

Usage:  python fig_ahsd_arch.py   -> fig_ahsd_arch.pdf / .png next to this file
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
W, H = 17, 11                      # box size in canvas units (canvas is 100 wide)
X = [0, 21, 42, 63, 84]            # five columns, 4-unit gaps
YT, YM, YB = 25, 14, 3             # top, middle, bottom rows

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ax.set_xlim(-0.5, 101.8); ax.set_ylim(-3, 46); ax.axis("off")


def box(x, y, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.2,rounding_size=0.9", fc=fc, ec=INK, lw=0.7))
    ax.text(x + W / 2, y + H - 2.4, title, ha="center", va="center", fontsize=7, weight="bold")
    ax.text(x + W / 2, y + 3.9, sub, ha="center", va="center", fontsize=5.7, linespacing=1.25)


def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.75, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


box(X[0], YM, "Flow window", "$T{=}32$ flows,\n41 features each", C_IN)
box(X[1], YM, "Embedding", "Linear, LayerNorm\n$z_t\\in\\mathbb{R}^{128}$", C_IN)
box(X[2], YT, "Gates", "$\\alpha_t,\\beta_t=$\n$\\sigma(W_g[z_t,h_{t-1}])$", C_REC)
box(X[3], YT, "State update", "pull $h_t$ toward $z_t$,\nrelax back to $E$", C_REC)
box(X[2], YB, "Reference $E$", "mean benign\nembedding (fixed)", C_REF)
box(X[3], YB, "Stress $s$", "mean $|h_t-E|$\nover $t$ and channels", C_SC)
box(X[4], YT, "Head", "MLP on $[h_T,\\bar h,s]$\n$P$(attack), MSP, Energy", C_OUT)
box(X[4], YB, "Hidden layer", "128-d features\nMahalanobis, $k$NN", C_OUT)

# dashed frame: the recurrence, repeated for every flow in the window
top = YT + H + 7
ax.add_patch(FancyBboxPatch((X[2] - 2, YB - 2), X[3] + W - X[2] + 4, top - YB + 2,
                            boxstyle="round,pad=0,rounding_size=1.2", fc="none", ec=MUTED, lw=0.7, ls="--"))
ax.text((X[2] + X[3] + W) / 2, top + 1.2, "recurrence, repeated for each flow $t=1,\\dots,T$ (starts at $h_0=E$)",
        ha="center", va="bottom", fontsize=6.3, color=MUTED, style="italic")

arrow((X[0] + W, YM + H / 2), (X[1], YM + H / 2))                       # window -> embedding
arrow((X[1] + W, YM + H / 2), (X[2], YT + H / 2))                   # z_t -> gates
arrow((X[2] + W, YT + H / 2 + 2), (X[3], YT + H / 2 + 2))               # gates -> update
arrow((X[2] + W / 2, YB + H), (X[2] + W / 2, YT))                       # E -> gates
arrow((X[3] + W / 2, YT), (X[3] + W / 2, YB + H))                       # h -> stress
arrow((X[3] + W / 2, YT + H), (X[2] + W / 2, YT + H), rad=0.32)         # feedback h_{t-1}
ax.text((X[2] + X[3] + W) / 2, YT + H + 4.6, "$h_{t-1}$", fontsize=6.2, ha="center", va="bottom")
arrow((X[3] + W, YT + H / 2), (X[4], YT + H / 2))                       # h_T, mean h -> head
arrow((X[3] + W, YB + H / 2), (X[4], YT + 1.5))                         # stress -> head
arrow((X[4] + W / 2, YT), (X[4] + W / 2, YB + H))                       # head -> hidden layer
ax.text(X[3] + W / 2, YB - 2.6, "also scored directly (Stress detector)", fontsize=5.8, ha="center", va="top",
        color=MUTED, style="italic")

out = Path(__file__).resolve().parent
fig.savefig(out / "fig_ahsd_arch.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_ahsd_arch.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", out / "fig_ahsd_arch.pdf")
