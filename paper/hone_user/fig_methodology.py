"""HoNE methodology overview (Fig. 1), drawn without icons.

Writes fig_methodology.pdf (used by main.tex) and fig_methodology.png (preview).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

W, H = 300, 100
fig = plt.figure(figsize=(14, 5.6))
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
t(6.5, 20.5, "Used only for ordering\nand splitting", size=6.6, color=SUB)
box(6.5, 6, 29, 8, fc=hb, ec=hb, r=0.6)
t(21, 10, "Output: leakage-controlled\nwindows per dataset", size=6.2, weight="bold", color="white", ha="center")

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
box(82, 82, 75, 7.5, ec=eo)
t(84, 87.3, "Primary: (1) LOACO vs (2) natural novelty, same data, detectors, seeds", size=5.9, weight="bold")
t(84, 84.0, "Stress tests: (3) purged blocks · (4) rolling cuts · (5) time split · (6) transfer", size=5.8, color=SUB)

# toy capture: grey = benign, A-C = attack classes; C occurs only late
TOY = {"N": "#d0d5dd", "A": "#3b82d6", "B": "#2e9e57", "C": "#d92d20"}
SEQ = "NANBANBNANBNNCBNCNCN"
BW, BH, XS = 2.75, 4.2, 93.5


def tblock(x, y, c, removed=False, faded=False, hatch=False):
    if hatch:
        ax.add_patch(plt.Rectangle((x, y), BW, BH, fc="white", ec=SUB, lw=0.5, hatch="////"))
        return
    ax.add_patch(plt.Rectangle((x, y), BW, BH, fc=TOY[c], ec="white", lw=0.5,
                               alpha=0.3 if faded else 1))
    if c != "N" and not removed:
        t(x + BW / 2, y + BH / 2, c, size=4.6, weight="bold", color="white", ha="center")
    if removed:
        ax.plot([x + 0.4, x + BW - 0.4], [y + 0.4, y + BH - 0.4], color=DARK, lw=0.8)
        ax.plot([x + 0.4, x + BW - 0.4], [y + BH - 0.4, y + 0.4], color=DARK, lw=0.8)


def tbrace(x0, x1, y, label, color=SUB):
    ax.plot([x0, x0, x1, x1], [y - 0.8, y, y, y - 0.8], color=color, lw=0.6)
    t((x0 + x1) / 2, y + 1.7, label, size=5.6, color=color, ha="center")


def taxis(y):
    ax.annotate("", xy=(XS + 20 * BW + 2, y), xytext=(XS, y),
                arrowprops=dict(arrowstyle="-|>", color=SUB, lw=0.6, mutation_scale=5))
    t(XS + 20 * BW + 2.6, y, "time", size=5.4, color=SUB)


# LOACO
box(82, 49, 75, 31, ec=TRAIN, lw=1)
t(84, 77, "(1) LOACO, grouped_random split", size=7.4, weight="bold")
t(155.5, 77, "grey = benign; A–C = attack classes", size=5.2, color=SUB, ha="right")
t(84, 70.6, "capture", size=5.8, color=SUB)
for k, c in enumerate(SEQ):
    tblock(XS + k * BW, 68.5, c)
taxis(67.2)
arrow(118, 66, 118, 60.6)
t(119.4, 63.6, "shuffle 1-h blocks, 60/20/20", size=5.6, color=SUB)
order = [7, 0, 16, 3, 9, 1, 5, 12, 15, 2, 10, 6, 4, 17, 19, 8, 13, 11, 18, 14]
for name, idx, x0 in (("training", order[:12], 84), ("val.", order[12:16], 119.5),
                      ("test", order[16:], 132.5)):
    for k, i in enumerate(idx):
        c = SEQ[i]
        tblock(x0 + k * BW, 54.3, c, removed=(c == "C" and name != "test"),
               faded=(name == "test" and c not in "NC"))
    tbrace(x0, x0 + len(idx) * BW, 59.6, name)
t(150.2, 56.4, "score C\nvs benign", size=5.6, color=TOY["C"], ha="center", weight="bold")
t(84, 51.2, "✕ class C removed from training and validation; test-period benign traffic stays in training",
  size=5.2, color=DARK)

# Natural novelty
box(82, 21, 75, 25.5, ec="#2e9e57", lw=1)
t(84, 43.5, "(2) Natural novelty, temporal split", size=7.4, weight="bold")
CUT = 12
for k, c in enumerate(SEQ):
    tblock(XS + k * BW, 31.6, c, faded=(k > CUT and c not in "NC"), hatch=(k == CUT))
tbrace(XS, XS + 10 * BW, 37.0, "training")
tbrace(XS + 10 * BW, XS + CUT * BW, 37.0, "val.")
tbrace(XS + (CUT + 1) * BW, XS + 20 * BW, 37.0, "test (after the cut)")
cx = XS + CUT * BW
ax.plot([cx, cx], [29.4, 37.6], color=ho, lw=1.1, ls=(0, (2, 1.4)))
taxis(30.0)
for f in (0.30, 0.45, 0.60, 0.75):
    xx = XS + f * 20 * BW
    ax.plot([xx, xx], [29.0, 31.0], color=ho, lw=1.1)
t(84, 30.0, "(4) rolling\ncuts", size=5.0, color=ho, va="center")
t(84, 26.2, "C first seen after the cut (orange) → novel, scored vs benign", size=5.6, color=TOY["C"])
t(84, 23.2, "B also in training → not scored (faded); val. = last 20% of training period",
  size=5.2, color=SUB)

# Purged-block fallback
box(82, 3.5, 75, 15.5, ec=ho, ls=(0, (3, 2)))
t(84, 16.2, "(3) Purged-block LOACO (short captures, D5)", size=6.9, weight="bold")
psteps = [(84, "10-min blocks"), (103, None), (122, "Purge adjacent\nblocks"), (140.5, "Min. test\nwindows")]
for x, s in psteps:
    box(x, 5, 15, 8.5, ec=eo)
    if s:
        t(x + 7.5, 9.25, s, size=5.9, ha="center")
t(110.5, 11.6, "60 / 20 / 20", size=5.9, weight="bold", ha="center")
bar(104.2, 7.0, [(2.1, TRAIN)] * 3 + [(2.1, VAL), (2.1, TEST), (2.1, TRAIN)], h=2.0, labels=[])
for x in (99, 118, 137):
    arrow(x, 9.25, x + 4, 9.25)
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
t(192.5, 6.3, "7 hashed amendments; 4 failed gates reported", size=5.0, color=SUB, ha="center")
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
