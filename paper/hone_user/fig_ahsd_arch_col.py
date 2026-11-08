"""AHSD architecture, single-column (3.45 in) grid diagram for the appendix.
Main path runs down the right column (embedding -> gates -> state update -> head); the left column holds what
feeds in or branches off (flow window, reference E, stress, hidden layer). Drawn from src/nids/models/ahsd.py,
variant 'fixed'.  Usage: python fig_ahsd_arch_col.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
INK, MUTED = "#1f1f1f", "#5f5f5b"
BLUE = ("#e8f1fc", "#2a78d6")
ORANGE = ("#fde9df", "#eb6834")
GREEN = ("#e3f5ec", "#1baf7a")
YELLOW = ("#fdf3d6", "#c99700")
GREY = ("#f1f0ec", "#6b6b67")

W, H = 40, 15                       # box size (canvas is 100 wide)
XL, XR = 2, 51                      # left / right column x
Y = [77, 50, 27, 2]                 # rows (bottom edge): input, gates, update, head
F_T, F_S = 7.6, 6.5                 # title / subtitle font sizes

fig, ax = plt.subplots(figsize=(3.45, 3.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 93); ax.axis("off")


def box(x, y, title, sub, col):
    fill, edge = col
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=2.2", fc=fill, ec=edge, lw=0.9))
    ax.text(x + W / 2, y + H - 4.0, title, ha="center", va="center", fontsize=F_T, weight="bold", color=INK)
    ax.text(x + W / 2, y + 5.0, sub, ha="center", va="center", fontsize=F_S, color=INK, linespacing=1.2)


def arrow(p, q, rad=0.0, label=None, lxy=None, ha="left"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, lw=0.75, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    if label:
        ax.text(*lxy, label, fontsize=F_S, ha=ha, va="center", color=INK)


box(XL, Y[0], "Flow window", "$T{=}32$ flows,\n41 features each", BLUE)
box(XR, Y[0], "Embedding", "Linear + LayerNorm\n$z_t\\in\\mathbb{R}^{128}$", BLUE)
box(XL, Y[1], "Reference $E$", "mean benign embedding,\nfixed; initial state $h_0$", GREEN)
box(XR, Y[1], "Gates", "$\\alpha_t,\\beta_t=$\n$\\sigma(W_g[z_t,h_{t-1}])$", ORANGE)
box(XL, Y[2], "Stress $s$", "mean $|h_t-E|$\n(also a score)", YELLOW)
box(XR, Y[2], "State update", "pull $h_t$ toward $z_t$,\nrelax it back to $E$", ORANGE)
box(XL, Y[3], "Hidden layer", "128-d features:\nMahalanobis, $k$NN", GREY)
box(XR, Y[3], "Head", "MLP on $[h_T,\\bar h,s]$:\n$P$(attack), MSP, energy", GREY)

# dashed frame around the per-flow recurrence, with its label in the gap above it
fy0, fy1 = Y[2] - 2.5, Y[1] + H + 3.0
ax.add_patch(FancyBboxPatch((XL - 1.3, fy0), 98.8 - XL, fy1 - fy0, boxstyle="round,pad=0,rounding_size=2",
                            fc="none", ec=ORANGE[1], lw=0.8, ls=(0, (4, 2.5))))
ax.text(XL, fy1 + 2.6, "recurrence over flows $t=1,\\dots,T$", ha="left", va="center", fontsize=F_S,
        style="italic", color=ORANGE[1])

cx = XR + W / 2
arrow((XL + W, Y[0] + H / 2), (XR, Y[0] + H / 2))                                         # window -> embedding
arrow((cx, Y[0]), (cx, Y[1] + H), label="$z_t$", lxy=(cx + 1.6, Y[0] - 4.5))             # embedding -> gates
arrow((XL + W, Y[1] + H / 2), (XR, Y[1] + H / 2))                                         # E -> gates
arrow((cx, Y[1]), (cx, Y[2] + H), label="$\\alpha_t,\\beta_t$", lxy=(cx - 1.6, (Y[1] + Y[2] + H) / 2), ha="right")
arrow((XR + W, Y[2] + H / 2), (XR + W, Y[1] + H / 2), rad=0.8)                            # feedback h_{t-1}
ax.text(XR + W - 1.2, (Y[1] + Y[2] + H) / 2, "$h_{t-1}$", fontsize=F_S, ha="right", va="center", color=INK)
arrow((XR, Y[2] + H / 2), (XL + W, Y[2] + H / 2))                                         # h -> stress
arrow((cx, Y[2]), (cx, Y[3] + H), label="$h_T,\\ \\bar h$", lxy=(cx + 1.6, (Y[2] + Y[3] + H) / 2))
# stress -> head: elbow below the frame (down, right, down) so it crosses nothing
ex, ey = XR + 7, (Y[2] - 2.5 + Y[3] + H) / 2
ax.plot([XL + W / 2, XL + W / 2, ex], [Y[2], ey, ey], color=INK, lw=0.75, solid_capstyle="butt")
arrow((ex, ey), (ex, Y[3] + H))
ax.text((XL + W / 2 + ex) / 2, ey + 0.6, "$s$", fontsize=F_S, ha="center", va="bottom", color=INK)
arrow((XR, Y[3] + H / 2), (XL + W, Y[3] + H / 2))                                         # head -> hidden layer

out = Path(__file__).resolve().parent
fig.savefig(out / "fig_ahsd_arch_col.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_ahsd_arch_col.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", out / "fig_ahsd_arch_col.pdf")
