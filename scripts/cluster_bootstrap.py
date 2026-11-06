"""Reviewer check: cluster bootstrap for the rolling-cut family means (A7).

Rolling class-cut units overlap: the same attack class reappears at several cut points.
This resamples whole (dataset, class) clusters instead of single units and compares the
95% CI with the unit-level bootstrap. Reads results/final_v2/summary.json; writes
results/final_v2/cluster_bootstrap.json. No hand-typed numbers.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/final_v2/summary.json").read_text())
A7 = S["a7"]
FAM = {"confidence": ["MSP", "Energy", "P_attack"], "representation": ["AHSD_stress", "Mahalanobis", "kNN"],
       "benign-only": ["IF", "OCSVM", "PCA", "AE"], "CLAD": ["Wilkie_CLAD"], "XGBoost": ["XGBoost"]}
roll = [e for e in A7["evals"] if e.startswith("roll_")]
units = [(e, u) for e in roll for u in A7["evals"][e]["cells"]["MSP"]]
clusters = sorted({(e.split("_")[1], u.split("|", 1)[1]) for e, u in units})
cid = np.array([clusters.index((e.split("_")[1], u.split("|", 1)[1])) for e, u in units])
V = {f: np.array([np.mean([A7["evals"][e]["cells"][d][u] for d in ds]) for e, u in units]) for f, ds in FAM.items()}
rng, B = np.random.default_rng(0), 2000


def unit_ci(x):
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)], [2.5, 97.5]).tolist()


def clus_ci(x):
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(clusters), len(clusters))
        out.append(x[np.concatenate([np.flatnonzero(cid == k) for k in pick])].mean())
    return np.percentile(out, [2.5, 97.5]).tolist()


res = {"n_units": len(units), "n_clusters": len(clusters), "B": B, "families": {}, "contrasts": {}}
for f, x in V.items():
    res["families"][f] = {"mean": float(x.mean()), "unit_ci95": unit_ci(x), "cluster_ci95": clus_ci(x)}
for name, x in (("benign-only - representation", V["benign-only"] - V["representation"]),
                ("XGBoost - benign-only", V["XGBoost"] - V["benign-only"])):
    res["contrasts"][name] = {"mean": float(x.mean()), "unit_ci95": unit_ci(x), "cluster_ci95": clus_ci(x)}
(ROOT / "results/final_v2/cluster_bootstrap.json").write_text(json.dumps(res, indent=1))
print(f"{res['n_units']} units, {res['n_clusters']} clusters")
for k, r in {**res["families"], **res["contrasts"]}.items():
    print(f"{k:30s} {r['mean']:+.3f}  unit [{r['unit_ci95'][0]:+.3f},{r['unit_ci95'][1]:+.3f}]"
          f"  cluster [{r['cluster_ci95'][0]:+.3f},{r['cluster_ci95'][1]:+.3f}]")
