"""Reviewer check: realised temporal_gap shares (train / gap / val / test) as % of all windows.

Blocks are indivisible 1-hour groups (src/nids/splits.py::_assign_cumulative), so realised shares
deviate from the nominal 60/80 cumulative cut-offs. Counts are assign_split's own output, recorded in
results/p0_summary.json (D1-D3) and results/p1d/d5_feasibility.json (D5). Cross-check: the 60% rolling
cut (scripts/rolling_build.py::cut_split, same rule, one gap group) must give identical train and gap
counts, and its test must equal val + test.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
p0 = json.loads((R / "p0_summary.json").read_text())["datasets"]
rows = {d: p0[d]["schemes"]["temporal_gap"]["split_windows"] for d in ("D1", "D2", "D3")}
rows["D5"] = json.loads((R / "p1d/d5_feasibility.json").read_text())["split_windows"]
roll = json.loads((R / "final_v2/rolling_feasibility.json").read_text())["datasets"]
for d, w in rows.items():
    T = sum(w.values())
    r = roll[d]["f060"]["split"]
    ok = (r["windows"]["train"] == w["train"] and r["windows"]["gap"] == w["gap"]
          and r["windows"]["test"] == w["val"] + w["test"] and isinstance(r["gap_group"], int))
    print(d, f"total {T:,}", "  ".join(f"{k} {w[k]:,} ({w[k] / T * 100:.2f}%)" for k in ("train", "gap", "val", "test")),
          f"| train+gap+val {100 * (w['train'] + w['gap'] + w['val']) / T:.2f}% | one gap block, matches 60% cut: {ok}")
    assert ok
