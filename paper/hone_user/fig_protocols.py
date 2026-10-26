"""How the two primary zero-day protocols build their splits (Fig. 2).

Writes fig_protocols.pdf (used by main.tex) and fig_protocols.png (preview).
The same 20-block toy capture is used in both panels; class C appears only late,
as in the day-scheduled captures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

COL = {"N": "#d0d5dd", "A": "#3b82d6", "B": "#2e9e57", "C": "#d92d20"}
DARK, SUB, GAPC = "#1d2939", "#475467", "#ffffff"
SEQ = "NANBANBNANBNNCBNCNCN"          # 20 one-hour blocks in time order
CUT = 12                              # natural novelty: first 60% train, block 12 = gap

fig = plt.figure(figsize=(3.5, 2.85))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
BW, BH, X0 = 4.0, 5.2, 10.0


def t(x, y, s, size=6, weight="normal", color=DARK, ha="left", va="center"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, linespacing=1.1)


def block(x, y, c, removed=False, faded=False, ec="white"):
    ax.add_patch(plt.Rectangle((x, y), BW, BH, fc=COL[c] if c in COL else c, ec=ec, lw=0.6,
                               alpha=0.3 if faded else 1))
    if c in "ABC" and not removed:
        t(x + BW / 2, y + BH / 2, c, size=4.6, weight="bold", color="white", ha="center")
    if removed:
        ax.plot([x + 0.6, x + BW - 0.6], [y + 0.6, y + BH - 0.6], color=DARK, lw=0.8)
        ax.plot([x + 0.6, x + BW - 0.6], [y + BH - 0.6, y + 0.6], color=DARK, lw=0.8)


def timeline(y):
    for i, c in enumerate(SEQ):
        block(X0 + i * BW, y, c)
    ax.annotate("", xy=(X0 + 20 * BW + 2.5, y - 1.6), xytext=(X0, y - 1.6),
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.6, mutation_scale=5))
    t(X0 + 20 * BW + 3, y - 1.6, "time", size=5, color=SUB)


def down(x, y0, y1, label):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.7, mutation_scale=6))
    t(x + 1.5, (y0 + y1) / 2, label, size=5.4, color=SUB)


def brace(x0, x1, y, label, color=SUB):
    ax.plot([x0, x0, x1, x1], [y - 1, y, y, y - 1], color=color, lw=0.6)
    t((x0 + x1) / 2, y + 2.2, label, size=5.4, color=color, ha="center")


# ---------------------------------------------------------------- (a) LOACO
t(2, 96.5, "(a) LOACO on the grouped_random split", size=7, weight="bold")
t(2, 88.6, "capture", size=5.4, color=SUB)
timeline(86)
down(53.5, 82.5, 74.5, "shuffle 1-h blocks, 60/20/20")

# a fixed shuffle of the 20 blocks, then 12 / 4 / 4
order = [7, 0, 16, 3, 9, 1, 5, 12, 15, 2, 10, 6, 4, 17, 19, 8, 13, 11, 18, 14]
groups = [("training", order[:12], 4.0), ("val.", order[12:16], 4.0 + 12 * BW + 3),
          ("test", order[16:], 4.0 + 16 * BW + 6)]
y = 66
for name, idx, x in groups:
    for j, i in enumerate(idx):
        c = SEQ[i]
        if name == "test":
            block(x + j * BW, y, c, faded=c not in "NC")
        else:
            block(x + j * BW, y, c, removed=c == "C")
    brace(x, x + len(idx) * BW, y + BH + 1.6, name)
t(4.0, 61.5, "✕ every window of held-out class C is removed\n   from training and validation", size=5.4, color=DARK, va="top")
tx = 4.0 + 16 * BW + 6
t(tx + 2 * BW, 61.5, "score C against\nbenign windows", size=5.4, color=COL["C"], ha="center", va="top")
t(2, 52.3, "Benign traffic of the test period stays in training; repeated for each eligible class.",
  size=5.0, color=SUB)
ax.plot([2, 98], [49.2, 49.2], color="#d0d5dd", lw=0.6)

# ---------------------------------------------------------------- (b) natural novelty
t(2, 45.5, "(b) Natural novelty on the temporal_gap split", size=7, weight="bold")
yb = 25.5
for i, c in enumerate(SEQ):
    if i == CUT:
        ax.add_patch(plt.Rectangle((X0 + i * BW, yb), BW, BH, fc="white", ec=SUB, lw=0.6, hatch="////"))
    else:
        block(X0 + i * BW, yb, c, faded=(i > CUT and c not in "NC"))
ax.annotate("", xy=(X0 + 20 * BW + 2.5, yb - 1.6), xytext=(X0, yb - 1.6),
            arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.6, mutation_scale=5))
t(X0 + 20 * BW + 3, yb - 1.6, "time", size=5, color=SUB)
brace(X0, X0 + 10 * BW, yb + BH + 1.6, "training")
brace(X0 + 10 * BW, X0 + CUT * BW, yb + BH + 1.6, "val.", color=SUB)
ax.plot([X0, X0 + CUT * BW], [yb + BH + 7.2] * 2, color=SUB, lw=0.4)
t(X0 + CUT * BW / 2, yb + BH + 9.0, "training period = first 60% (val. = its last 20%)", size=5.0, color=SUB, ha="center")
brace(X0 + (CUT + 1) * BW, X0 + 20 * BW, yb + BH + 1.6, "test (after the cut)")
t(X0 + CUT * BW + BW / 2, yb - 4.6, "gap", size=5.0, color=SUB, ha="center")
cx = X0 + CUT * BW
ax.plot([cx, cx], [yb - 2.5, yb + BH + 8], color="#ef8a22", lw=1.0, ls=(0, (2, 1.5)))
t(cx + 3.2, yb + BH + 9.0, "cut", size=5.2, color="#ef8a22", ha="center", weight="bold")
t(2, 15.6, "C occurs only after the cut → novel class, scored against benign windows", size=5.4, color=COL["C"])
t(2, 11.0, "B also occurs in training → not novel, not scored (faded)", size=5.4, color=SUB)

# rolling cut-points
yr = 3.6
ax.plot([X0, X0 + 20 * BW], [yr, yr], color=SUB, lw=0.6)
for f in (0.30, 0.45, 0.60, 0.75):
    xx = X0 + f * 20 * BW
    ax.plot([xx, xx], [yr - 1.4, yr + 1.4], color="#ef8a22", lw=1.1)
    t(xx, yr + 3.2, f"{int(f * 100)}%", size=4.8, color="#ef8a22", ha="center")
t(2, yr, "rolling", size=5.4, color=SUB)
t(X0 + 20 * BW + 1, yr, "move cut", size=4.8, color=SUB)

fig.savefig("fig_protocols.pdf")
fig.savefig("fig_protocols.png", dpi=300)
