"""HoNE methodology overview (Fig. 1), drawn without icons.

Writes fig_methodology.pdf (used by main.tex) and fig_methodology.png (preview).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

W, H = 300, 100
fig = plt.figure(figsize=(14, 14 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

C = {  # header, light fill, edge
    "blue": ("#1f6fc5", "#eef5fc", "#9cc2ea"),
    "orange": ("#ef8a22", "#fff5ea", "#f5c08a"),
    "purple": ("#7b52c7", "#f5f0fd", "#c3aee9"),
    "green": ("#2e9e57", "#edf8f1", "#9fd6b3"),
}
TRAIN, VAL, TEST, GAP = "#3b82d6", "#f0913a", "#9b5fd0", "#b9bec7"
DARK = "#1d2939"
SUB = "#475467"


def box(x, y, w, h, fc="white", ec="#c9d2dd", lw=0.8, ls="-", r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def t(x, y, s, size=7, weight="normal", color=DARK, ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            linespacing=1.15, **kw)


def arrow(x0, y0, x1, y1, color="#667085", lw=0.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=6))


def bar(x, y, segs, h=2.6, labels=None):
    """segs: list of (width, colour); labels: list of (x_offset, text)."""
    cx = x
    for w, c in segs:
        ax.add_patch(plt.Rectangle((cx, y), w, h, fc=c, ec="white", lw=0.5))
        cx += w
    for off, s in labels or []:
        t(x + off, y - 2.2, s, size=5.6, color=SUB, ha="center")


def panel(x, w, key, title, size=9):
    head, light, edge = C[key]
    box(x, 1, w, 98, fc=light, ec=head, lw=1.2, r=1.5)
    box(x, 91.5, w, 7.5, fc=head, ec=head, lw=1.2, r=1.5)
    ax.add_patch(plt.Rectangle((x + 0.1, 91.5), w - 0.2, 2, fc=head, ec=head, lw=0))
    t(x + w / 2, 95.2, title, size=size, weight="bold", color="white", ha="center")


def between(x):
    ax.add_patch(Polygon([[x - 1.1, 52.5], [x - 1.1, 47.5], [x + 1.3, 50]], color="#1f4e8c"))


# ---------------------------------------------------------------- panel 1
panel(2, 75, "blue", "1. DATASETS & LEAKAGE CONTROL")
hb, lb, eb = C["blue"]
box(4.5, 34, 33, 55, ec=eb)
t(6.5, 85.5, "Public datasets", size=7.6, weight="bold")
for i, (k, name) in enumerate([("D1", "CSE-CIC-IDS2018"), ("D2", "UNSW-NB15"), ("D3", "ToN-IoT"),
                               ("D4", "BoT-IoT (transfer)"), ("D5", "Lycos2017")]):
    y = 77 - i * 9.3
    box(6.5, y - 3.3, 29, 6.6, fc="#f7fafe", ec="#d5e3f3", r=0.6)
    t(8.3, y, k, size=7, weight="bold", color=hb)
    t(13.5, y, name, size=7)
box(4.5, 4, 33, 26, ec=eb)
t(6.5, 26, "Flow timestamps", size=7.6, weight="bold")
t(6.5, 17, "Used only for ordering\nand splitting", size=6.6, color=SUB)

box(40, 4, 35, 85, ec=eb)
t(42, 85.5, "Leakage control", size=7.6, weight="bold")
steps1 = [("Drop identifiers", "IPs, ports, DNS IDs, TTL"),
          ("Train-only scaling", "Median impute, log(1+x),\nz-score; 41 common features"),
          ("Time blocks", "1-hour (all);\n10-min (D5 purged split)"),
          ("Windows", "T = 32, stride 16"),
          ("SHA-256 deduplication", "Remove duplicate windows"),
          ("Integrity checks", "Disjoint splits, class ratios,\nfinite values")]
bh, gap, top = 10.4, 2.4, 81.5
for i, (a, b) in enumerate(steps1):
    y = top - i * (bh + gap) - bh
    box(42, y, 31, bh, ec="#d5e3f3")
    ax.add_patch(plt.Rectangle((42, y + 0.6), 0.9, bh - 1.2, fc=hb, lw=0))
    t(44.5, y + bh - 2.6, a, size=6.9, weight="bold")
    t(44.5, y + 3.2 if "\n" in b else y + 2.9, b, size=5.9, color=SUB)
    if i:
        arrow(57.5, y + bh + gap, 57.5, y + bh, color=hb)
arrow(37.6, 61, 39.9, 61, color=hb)
between(78.5)

# ---------------------------------------------------------------- panel 2
panel(80, 79, "orange", "2. PAIRED ZERO-DAY PROTOCOLS")
ho, lo, eo = C["orange"]
box(82, 80, 45, 9, ec=eo)
t(104.5, 84.5, "Same data, detectors, seeds", size=7.6, weight="bold", ha="center")
box(129.5, 80, 27.5, 9, fc="#fff0dc", ec=ho)
t(143.25, 84.5, "Report infeasible\nLOACO folds", size=6.6, weight="bold", color="#9a4b00", ha="center")

box(82, 41, 12, 25, ec=eo)
t(88, 53.5, "Leakage-\ncontrolled\nwindows", size=6.4, weight="bold", ha="center")
ax.plot([94, 95.5, 95.5], [53.5, 53.5, 68.5], color=ho, lw=0.8)
ax.plot([95.5, 95.5], [53.5, 39], color=ho, lw=0.8)
arrow(95.5, 68.5, 97.4, 68.5, color=ho)
arrow(95.5, 39, 97.4, 39, color=ho)

# LOACO
box(97.5, 59.5, 59.5, 18.5, ec=TRAIN, lw=1)
t(99.5, 75, "LOACO (hold-out)", size=7.6, weight="bold")
t(99.5, 69.5, "Grouped random (60/20/20)", size=6.4)
bar(99.5, 63.3, [(3.6, TRAIN)] * 6 + [(3.6, VAL)] * 2 + [(3.6, TEST)] * 2,
    labels=[(10.8, "Train"), (25.2, "Val"), (32.4, "Test")])
arrow(137, 64.6, 140, 64.6)
t(148.5, 64.6, "Remove one\nattack class\nfrom all training", size=6.1, ha="center", color=DARK)

# Natural novelty
box(97.5, 27.5, 59.5, 29.5, ec="#2e9e57", lw=1)
t(99.5, 54, "Natural novelty", size=7.6, weight="bold")
t(99.5, 48.5, "Temporal ordering", size=6.4)
bar(99.5, 42.3, [(3.6, TRAIN)] * 6 + [(2.4, GAP)] + [(3.6, VAL)] * 2 + [(3.6, TEST)] * 2,
    labels=[(10.8, "Train"), (22.8, "Gap"), (27.6, "Val"), (34.8, "Test")])
arrow(139, 43.6, 142, 43.6)
t(149.5, 43.6, "New classes\nfirst seen\nafter training", size=6.1, ha="center")
ax.plot([99.5, 154.5], [33.5, 33.5], color=SUB, lw=0.7)
for f in (0.30, 0.45, 0.60, 0.75):
    xx = 99.5 + f * 55
    ax.plot([xx, xx], [32.4, 34.6], color=ho, lw=1.2)
t(127, 30, "4 rolling cut-offs (30 / 45 / 60 / 75%)", size=6.1, ha="center", color=SUB)

# Purged-block fallback
box(82, 3.5, 75, 21, ec=ho, ls=(0, (3, 2)))
t(84, 21, "Purged-block fallback (short captures)", size=7.2, weight="bold")
psteps = [(84, "10-min blocks"), (103, None), (122, "Purge adjacent\nblocks"), (140.5, "Min. test\nwindows")]
for x, s in psteps:
    box(x, 6, 15, 10.5, ec=eo)
    if s:
        t(x + 7.5, 11.25, s, size=6.1, ha="center")
t(110.5, 14.2, "60 / 20 / 20", size=6.1, weight="bold", ha="center")
bar(104.2, 9.5, [(2.1, TRAIN)] * 3 + [(2.1, VAL), (2.1, TEST), (2.1, TRAIN)],
    h=2.2, labels=[])
t(110.5, 7.6, "train / val / test", size=5.4, color=SUB, ha="center")
for x in (99, 118, 137):
    arrow(x, 11.25, x + 4, 11.25)
between(160.5)

# ---------------------------------------------------------------- panel 3
panel(162, 61, "purple", "3. DETECTOR PANEL & MULTI-SEED EVALUATION", size=7.3)
hp, lp, ep = C["purple"]
box(164, 36, 57, 53, ec=ep)
t(166, 85.5, "12 detectors / 5 families", size=7.6, weight="bold")
fams = [("Confidence", "MSP, Energy, P(attack)"), ("Representation", "Stress, Mahalanobis, kNN"),
        ("Benign-only", "IF, OCSVM, PCA, AE"), ("Contrastive", "CLAD"),
        ("Reference", "XGBoost, XGBoost-s")]
for i, (a, b) in enumerate(fams):
    y = 75 - i * 8.6
    box(166, y - 3.6, 34, 7.6, ec="#ddd2f2")
    ax.add_patch(plt.Rectangle((166, y - 3.0), 0.9, 6.4, fc=hp, lw=0))
    t(168.5, y + 1.6, a, size=6.9, weight="bold")
    t(168.5, y - 1.6, b, size=5.9, color=SUB)
box(202, 38.5, 17, 44, fc="#faf8fe", ec=hp, ls=(0, (3, 2)))
t(210.5, 78.2, "Recurrent\nbackbones", size=6.4, weight="bold", ha="center")
t(210.5, 72.2, "(confidence and\nrepresentation)", size=5.4, color=SUB, ha="center")
for y, s in ((58, "AHSD"), (44, "GRU")):
    box(204, y, 13, 9, fc="white", ec=hp)
    t(210.5, y + 4.5, s, size=7.4, weight="bold", ha="center", color=hp)

box(164, 4, 57, 29, ec=ep)
t(166, 29.5, "Multi-seed evaluation", size=7.6, weight="bold")
ms = [("5", "seeds"), ("210", "runs"), ("1", "code commit"), ("Pre-", "registered")]
for i, (a, b) in enumerate(ms):
    x = 166 + i * 13.6
    box(x, 9, 11.2, 15, ec="#ddd2f2")
    t(x + 5.6, 19, a, size=8.6, weight="bold", ha="center", color=hp)
    t(x + 5.6, 13, b, size=5.5, ha="center")
    if i:
        arrow(x - 2.3, 16.5, x - 0.1, 16.5, color=hp)
t(207.4, 6.5, "hypotheses, metrics, gates", size=5.0, color=SUB, ha="center")
between(223.5)

# ---------------------------------------------------------------- panel 4
panel(225, 73, "green", "4. EVIDENCE & ACCEPTANCE")
hg, lg, eg = C["green"]
box(227, 69, 69, 20, ec=eg)
t(229, 85.5, "Metrics", size=7.6, weight="bold")
mets = ["AUC", "TPR at\n1% FPR", "PR-AUC", "Seed s.d.", "Family\nranking", "Checkpoint\nsensitivity"]
tw = 10.6
for i, s in enumerate(mets):
    x = 229 + i * (tw + 0.4)
    box(x, 71, tw, 10.5, ec="#cfe9d8")
    t(x + tw / 2, 76.25, s, size=5.9, weight="bold", ha="center")

box(227, 38, 69, 28.5, ec=eg)
t(229, 62.5, "Three main findings", size=7.6, weight="bold")
finds = [("Protocol changes\nthe winner", "LOACO winner differs\nfrom natural novelty"),
         ("High AUC, most\nattacks missed", "AUC <= 0.75: <= 18%\ncaught at 1% FPR"),
         ("One run is\nunreliable", "Seed variation\nup to 17x")]
for i, (a, b) in enumerate(finds):
    x = 229 + i * 22.4
    box(x, 40, 21.4, 19.5, ec="#cfe9d8")
    t(x + 10.7, 54, a, size=6.6, weight="bold", ha="center")
    t(x + 10.7, 45.3, b.replace("<=", "≤").replace("17x", "17×"), size=5.8,
      color=SUB, ha="center")

box(227, 4, 69, 31, ec=eg)
t(229, 31.2, "HoNE acceptance protocol: 8 checks", size=7.6, weight="bold")
checks = ["Leakage controls", "Natural novelty + LOACO", "≥ 5 seeds", "Valid novelty scores",
          "Benign-only + XGBoost", "Operating-point metrics", "Checkpoint sensitivity",
          "Pre-registration / log"]
for i, s in enumerate(checks):
    col, row = divmod(i, 4)
    x, y = 229 + col * 24, 25.5 - row * 5.6
    t(x, y, "✓", size=7, weight="bold", color=hg)
    t(x + 2.2, y, s, size=5.5)
box(277.5, 20.5, 17, 7, fc="white", ec=hg)
t(286, 24, "Operator review", size=6.1, weight="bold", ha="center")
box(276.5, 6.5, 9, 8.5, fc=hg, ec=hg)
t(281, 10.75, "TRUST", size=6.1, weight="bold", color="white", ha="center")
box(286.5, 6.5, 9, 8.5, fc="#f08a24", ec="#f08a24")
t(291, 10.75, "REQUEST\nEVIDENCE", size=4.9, weight="bold", color="white", ha="center")
arrow(283, 20.4, 281, 15.2, color=hg)
arrow(289, 20.4, 291, 15.2, color="#f08a24")

fig.savefig("fig_methodology.pdf")
fig.savefig("fig_methodology.png", dpi=300)
