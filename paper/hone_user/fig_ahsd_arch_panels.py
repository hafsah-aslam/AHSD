"""AHSD architecture, single-column (3.45 in), monochrome three-panel block diagram for the appendix.
Panels: inputs (rounded) | AHSD stages (square, italic sub-rows) | anomaly scores (tall, rotated).
Drawn from src/nids/models/ahsd.py, variant 'fixed'.  Usage: python fig_ahsd_arch_panels.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
K, LW = "black", 0.8
F_T, F_S, F_I = 7.4, 6.6, 6.4            # title, subtitle, italic sub-row font sizes

XI, WI = 0.3, 24.5                         # panel 1: inputs
XM, WM = 30, 48                          # panel 2: AHSD stages
XO, WO = 87.5, 9.5                          # panel 3: scores
HB, HS = 12, 6.5                         # main box / sub-row heights
YE, YR, YH = 70, 40, 5                   # rows: embedding, recurrence, head (bottom of main boxes)

fig, ax = plt.subplots(figsize=(3.45, 3.05))
ax.set_xlim(-0.6, 100); ax.set_ylim(-12, 95); ax.axis("off")

# three coloured background panels: (1) input, (2) AHSD backbone, (3) scores
PANELS = [((-0.4, 26.3), "#e8f1fc", "#2a78d6", "(1) Input"),
          ((28.2, 79.8), "#fdf0e6", "#eb6834", "(2) AHSD backbone"),
          ((84.6, 99.6), "#e6f4ec", "#1baf7a", "(3)\nScores")]
PY0, PY1 = -11, 94
for (x0, x1), fill, edge, title in PANELS:
    ax.add_patch(FancyBboxPatch((x0, PY0), x1 - x0, PY1 - PY0, boxstyle="round,pad=0,rounding_size=2.5",
                                fc=fill, ec=edge, lw=0.8, zorder=0))
    ax.text((x0 + x1) / 2, PY1 - 2.0, title, ha="center", va="top", fontsize=7, weight="bold", color=edge, linespacing=1.0)
PANEL_BG = "#fdf0e6"


def rbox(x, y, w, h, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.4", fc="white", ec=K, lw=LW))
    ax.text(x + w / 2, y + h / 2, lines, ha="center", va="center", fontsize=F_S - 0.3, linespacing=1.15)


def sbox(y, title, sub):
    ax.add_patch(Rectangle((XM, y), WM, HB, fc="white", ec=K, lw=LW))
    ax.text(XM + WM / 2, y + HB * 0.69, title, ha="center", va="center", fontsize=F_T, weight="bold")
    ax.text(XM + WM / 2, y + HB * 0.27, sub, ha="center", va="center", fontsize=F_S)


def subrow(y, text, inset=2.0):
    ax.add_patch(Rectangle((XM + inset, y), WM - 2 * inset, HS, fc="white", ec=K, lw=LW))
    ax.text(XM + WM / 2, y + HS / 2, text, ha="center", va="center", fontsize=F_I, style="italic")
    return y + HS / 2


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=LW, color=K, shrinkA=0, shrinkB=0))


# panel 1: inputs
# flow windows drawn as three stacked panels (successive sliding windows); front panel carries the label
for k in (2, 1):
    ax.add_patch(FancyBboxPatch((XI + 1.6 * k, YE + 1 + 1.6 * k), WI - 3.2, HB - 2, boxstyle="round,pad=0,rounding_size=2.4",
                                fc="white", ec=K, lw=LW))
ax.add_patch(FancyBboxPatch((XI, YE + 1), WI - 3.2, HB - 2, boxstyle="round,pad=0,rounding_size=2.4", fc="white", ec=K, lw=LW))
ax.text(XI + (WI - 3.2) / 2, YE + 1 + (HB - 2) / 2, "Flow windows\n$T{=}32$ flows", ha="center", va="center",
        fontsize=F_S - 0.6, linespacing=1.15)
rbox(XI, YR + 1, WI - 3.2, HB - 2, "Benign\nreference $E$")

# panel 2: AHSD stages
sbox(YE, "Embedding", "Linear + LayerNorm, $z_t\\in\\mathbb{R}^{128}$")
sbox(YR, "Gated recurrence", "$h_t$ starts at $h_0=E$")
subrow(YR - HS, "Gates $\\alpha_t,\\beta_t=\\sigma(W_g[z_t,h_{t-1}])$")
y_stress = subrow(YR - 2 * HS, "Stress $s=\\mathrm{mean}\\,|h_t-E|$")
sbox(YH, "Head", "two-layer MLP on $[h_T,\\bar h,s]$")
y_out = subrow(YH - HS, "Output: $P$(attack), MSP, energy")
y_hid = subrow(YH - 2 * HS, "Hidden: Mahalanobis, $k$NN")

# panel 3: scores (tall box, rotated text)
y0, y1 = YH - 2 * HS, YR + HB
ax.add_patch(Rectangle((XO, y0), WO, y1 - y0, fc="white", ec=K, lw=LW))
ax.text(XO + WO / 2, (y0 + y1) / 2, "Anomaly scores", rotation=90, ha="center", va="center", fontsize=F_T + 0.4)

# arrows
arrow((XI + WI, YE + HB / 2 + 1.6), (XM, YE + HB / 2 + 1.6))
arrow((XI + WI - 3.2, YR + HB / 2), (XM, YR + HB / 2))
arrow((XM + WM / 2, YE), (XM + WM / 2, YR + HB))
ax.text(XM + WM / 2, (YE + YR + HB) / 2, "$z_t$, for each flow $t=1,\\dots,T$", ha="center", va="center",
        fontsize=F_I, style="italic", bbox=dict(fc=PANEL_BG, ec="none", pad=0.6))
arrow((XM + WM / 2, YR - 2 * HS), (XM + WM / 2, YH + HB))
ax.text(XM + WM / 2 + 1.5, (YR - 2 * HS + YH + HB) / 2, "$h_T,\\ \\bar h,\\ s$", ha="left", va="center",
        fontsize=F_I)
for y in (y_stress, y_out, y_hid):
    arrow((XM + WM - 2.0, y), (XO, y))

out = Path(__file__).resolve().parent
fig.savefig(out / "fig_ahsd_arch_panels.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out / "fig_ahsd_arch_panels.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", out / "fig_ahsd_arch_panels.pdf")
