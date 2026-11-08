"""Reviewer check: cluster bootstrap for the pooled rolling-cut family comparisons (A7).

Rolling class-cut units overlap: one attack class reappears at several cut points. This script
  (1) rebuilds the per-seed (unit x seed) AUC matrices from the run files, exactly as
      scripts/report_a7.py does, and reproduces the paper's unit x seed bootstrap (check);
  (2) runs a cluster x seed bootstrap that resamples whole (dataset, class) clusters, with all
      units of a drawn cluster, and seeds independently.
Pairing: the RNG is re-seeded for every quantity (as in report_a7.boot), so every family and every
contrast is evaluated on the same sequence of resampled clusters and seeds; contrasts are bootstrapped
on the per-unit, per-seed difference matrix. Reads results/final_v2/roll_*/*/seed*.json; writes
results/final_v2/cluster_bootstrap.json. No hand-typed numbers.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/final_v2"
SEEDS = [17, 23, 42, 101, 202]
N_BOOT, BOOT_SEED = 2000, 0          # same as scripts/report_a7.py
FAM = {"confidence": ["MSP", "Energy", "P_attack"], "representation": ["AHSD_stress", "Mahalanobis", "kNN"],
       "benign-only": ["IF", "OCSVM", "PCA", "AE"], "CLAD": ["Wilkie_CLAD"], "XGBoost": ["XGBoost"]}
CONTRASTS = [("benign-only", "representation"), ("XGBoost", "benign-only")]

# ---- per-seed AUCs, best checkpoint: A[eval][det][class][seed]
A = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
classes = defaultdict(list)
for p in sorted(RES.glob("roll_*/*/seed*.json")):
    r = json.loads(p.read_text())
    e = r["eval_id"]
    for c in r["classes"]:
        if c not in classes[e]:
            classes[e].append(c)
    for d, bc in r["auc"]["best"].items():
        for c, a in bc.items():
            A[e][d][c][r["seed"]] = a
units = [(e, c) for e in sorted(classes) for c in classes[e]]
for e, c in units:
    for f in FAM.values():
        for d in f:
            assert sorted(A[e][d][c]) == SEEDS, (e, d, c)
cl_key = [(e.split("_")[1], c) for e, c in units]          # (dataset, class)
clusters = sorted(set(cl_key))
members = [np.array([i for i, k in enumerate(cl_key) if k == g]) for g in clusters]


def fmat(fam):
    """(U, S): family mean AUC per unit and seed (report_a7.fmat)."""
    return np.array([[np.mean([A[e][d][c][s] for d in FAM[fam]]) for s in SEEDS] for e, c in units])


def unit_boot(M):                     # identical to report_a7.boot
    rng = np.random.default_rng(BOOT_SEED)
    C, S = M.shape
    b = [M[np.ix_(rng.integers(0, C, C), rng.integers(0, S, S))].mean() for _ in range(N_BOOT)]
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def cluster_boot(M):
    rng = np.random.default_rng(BOOT_SEED)
    G, S = len(clusters), M.shape[1]
    b = []
    for _ in range(N_BOOT):
        rows = np.concatenate([members[g] for g in rng.integers(0, G, G)])
        b.append(M[np.ix_(rows, rng.integers(0, S, S))].mean())
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


M = {f: fmat(f) for f in FAM}
res = {"n_units": len(units), "n_clusters": len(clusters), "n_seeds": len(SEEDS), "B": N_BOOT,
       "families": {}, "contrasts": {}}
for f, X in M.items():
    res["families"][f] = {"mean": float(X.mean()), "unit_seed_ci95": unit_boot(X), "cluster_seed_ci95": cluster_boot(X)}
for a, b in CONTRASTS:
    X = M[a] - M[b]
    res["contrasts"][f"{a} - {b}"] = {"mean": float(X.mean()), "unit_seed_ci95": unit_boot(X),
                                      "cluster_seed_ci95": cluster_boot(X)}

# ---- check (1): unit x seed intervals must equal those in summary.json (paper)
P = json.loads((RES / "summary.json").read_text())["a7"]["rolling"]["pooled"]
paper = {**{f: P["families"][f]["ci95"] for f in FAM},
         "benign-only - representation": P["contrasts"]["benign-only − representation"]["ci95"],
         "XGBoost - benign-only": P["XGBoost_minus_leader_ahsd"]["ci95"]}
assert P["leader_ahsd"] == "benign-only"
mine = {**{f: v["unit_seed_ci95"] for f, v in res["families"].items()},
        **{k: v["unit_seed_ci95"] for k, v in res["contrasts"].items()}}
res["reproduces_summary"] = all(np.allclose(mine[k], paper[k], atol=1e-12) for k in paper)


# ---- check (2): sensitivity of the cluster x seed contrast bounds to the bootstrap RNG seed (0..19)
def cluster_boot_seed(X, seed):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(N_BOOT):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(clusters), len(clusters))])
        b.append(X[np.ix_(rows, rng.integers(0, X.shape[1], X.shape[1]))].mean())
    return np.percentile(b, [2.5, 97.5])


for a, b in CONTRASTS:
    ci = np.array([cluster_boot_seed(M[a] - M[b], k) for k in range(20)])
    res["contrasts"][f"{a} - {b}"]["cluster_seed_rng_sensitivity"] = {
        "rng_seeds": 20, "lower_min": float(ci[:, 0].min()), "lower_max": float(ci[:, 0].max()),
        "upper_min": float(ci[:, 1].min()), "upper_max": float(ci[:, 1].max()),
        "frac_excluding_zero": float(np.mean((ci[:, 0] > 0) | (ci[:, 1] < 0)))}
(RES / "cluster_bootstrap.json").write_text(json.dumps(res, indent=1))

print(f"{res['n_units']} units, {res['n_clusters']} clusters, {res['n_seeds']} seeds, B={N_BOOT}; "
      f"unit x seed CIs reproduce summary.json: {res['reproduces_summary']}")
for k, r in {**res["families"], **res["contrasts"]}.items():
    u, c = r["unit_seed_ci95"], r["cluster_seed_ci95"]
    print(f"{k:30s} {r['mean']:+.3f}  unit x seed [{u[0]:+.3f},{u[1]:+.3f}]  cluster x seed [{c[0]:+.3f},{c[1]:+.3f}]")
    if "cluster_seed_rng_sensitivity" in r:
        z = r["cluster_seed_rng_sensitivity"]
        print(f"{'':30s} over 20 RNG seeds: lower {z['lower_min']:+.3f}..{z['lower_max']:+.3f}, "
              f"upper {z['upper_min']:+.3f}..{z['upper_max']:+.3f}, CI excludes 0 in {z['frac_excluding_zero']:.0%}")
