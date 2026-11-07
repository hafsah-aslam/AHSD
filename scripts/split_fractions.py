"""Reviewer check: realised temporal_gap shares (train / gap / val / test) as % of all windows.

Blocks are indivisible 1-hour groups (src/nids/splits.py::_assign_cumulative), so realised shares
deviate from the nominal 60/20/20 cut-offs. Reads results/p0_summary.json (D1-D3) and
results/p1d/d5_feasibility.json (D5).
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
p0 = json.loads((R / "p0_summary.json").read_text())["datasets"]
rows = {d: p0[d]["schemes"]["temporal_gap"]["split_windows"] for d in ("D1", "D2", "D3")}
rows["D5"] = json.loads((R / "p1d/d5_feasibility.json").read_text())["split_windows"] \
    if "split_windows" in json.loads((R / "p1d/d5_feasibility.json").read_text()) else None
if rows["D5"] is None:  # nested record
    def find(o):
        if isinstance(o, dict):
            if isinstance(o.get("split_windows"), dict):
                return o["split_windows"]
            for v in o.values():
                r = find(v)
                if r:
                    return r
    rows["D5"] = find(json.loads((R / "p1d/d5_feasibility.json").read_text()))
for d, w in rows.items():
    T = sum(w.values())
    print(d, f"total {T:,}", "  ".join(f"{k} {w[k]:,} ({w[k] / T * 100:.2f}%)" for k in ("train", "gap", "val", "test")))
