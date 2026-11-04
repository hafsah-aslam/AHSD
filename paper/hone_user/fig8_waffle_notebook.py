%matplotlib inline
# HoNE Fig. 8 - out of 100 new attacks, how many does each detector family catch at 1% false-positive rate?
# Rolling cut-points (45 class-cut units), natural novelty. Family mean AUC vs family mean TPR at 1% FPR.
# Source: results/final_v2/summary.json (a7.rolling.pooled: families[...].mean, ops[...].tpr_at_1pct_fpr.mean).
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# family -> (mean AUC, share of attacks caught at 1% FPR); ordered by AUC
DATA = [
    ("Benign-only",    0.7483, 0.0917),
    ("XGBoost",        0.6723, 0.1831),
    ("CLAD",           0.6522, 0.1009),
    ("Representation", 0.6230, 0.1251),
    ("Confidence",     0.5319, 0.0443),
]
COLOR = {"Confidence": "#2a78d6", "Representation": "#eb6834", "Benign-only": "#1baf7a", "CLAD": "#eda100",
         "XGBoost": "#e87ba4"}
MISSED = "#e6e5e0"
OUT = Path.home() / "Downloads"
INK, MUTED = "#2b2b2a", "#6b6b67"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "text.color": INK})

fig, axes = plt.subplots(1, len(DATA), figsize=(3.45, 1.32))
for ax, (name, auc, tpr) in zip(axes, DATA):
    caught = round(tpr * 100)
    for k in range(100):                              # 10 x 10 grid, filled from the bottom-left
        r, c = divmod(k, 10)
        ax.add_patch(Rectangle((c, r), 0.86, 0.86, color=COLOR[name] if k < caught else MISSED, lw=0))
    ax.set_xlim(-0.2, 10); ax.set_ylim(-0.2, 10)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{name}\nAUC {auc:.2f}", fontsize=6.4, pad=2, linespacing=1.1)
    ax.text(5, -1.6, f"{caught} caught", ha="center", va="top", fontsize=6.6, weight="bold", color=COLOR[name])
fig.text(0.5, -0.05, "Each grid = 100 new attacks; coloured squares = caught at 1% false-positive rate",
         ha="center", fontsize=5.8, color=MUTED)
fig.tight_layout(pad=0.1, w_pad=0.5)
fig.savefig(OUT / "fig_operating.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig_operating.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT / "fig_operating.pdf")
plt.show()
