#!/usr/bin/env python
"""Rolling temporal cut-points (AMENDMENT_07 A7.2): feasibility count and data packages.

For each dataset D1, D2, D3, D5 and cut fraction f in {0.30, 0.45, 0.60, 0.75}, with the 1-hour
groups in time order:
  TRAIN = groups whose cumulative window count before the group is < f x total windows
  GAP   = the next group (discarded)
  TEST  = every later group
Natural-novelty classes (frozen A4.3 rule): majority class of >= 20 TEST windows and of no TRAIN window.
A cut is feasible if it has >= 1 such class and its train-tail validation holds attack windows.

  python scripts/rolling_build.py --count    # data only, before any model run -> results/final_v2/rolling_feasibility.json
  python scripts/rolling_build.py --build    # packages under data/processed/<ds>/rolling/f<pct>/ (cleaner refit per cut)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1c_build import CFG as PKG_CFG, build  # noqa: E402
from nids import splits  # noqa: E402
from nids.preprocess import Cleaner  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.windowing import WindowIndex  # noqa: E402

DATASETS = ["D1", "D2", "D3", "D5"]
CUTS = [0.30, 0.45, 0.60, 0.75]
MIN_TEST_WINDOWS = 20  # frozen A4.3 rule
OUT = ROOT / "results/final_v2/rolling_feasibility.json"


def cut_name(f: float) -> str:
    return f"f{int(round(f * 100)):03d}"


def cut_split(wi: WindowIndex, f: float) -> tuple[np.ndarray, dict]:
    g, n = np.unique(wi.group, return_counts=True)  # group ids are in time order
    before = np.cumsum(n) - n
    is_tr = before < f * n.sum()
    k = int(is_tr.sum())  # TRAIN groups are a prefix
    if not np.all(is_tr[:k]) or k == 0 or k + 1 >= len(g):
        raise SystemExit(f"cut {f}: degenerate split")
    lut = np.full(int(g.max()) + 1, -1, np.int8)
    lut[g[:k]] = splits.TRAIN
    lut[g[k]] = splits.GAP
    lut[g[k + 1:]] = splits.TEST
    split = lut[wi.group]
    return split, {"train_groups": [int(x) for x in g[:k]], "gap_group": int(g[k]),
                   "test_groups": [int(g[k + 1]), int(g[-1])],
                   "windows": {s: int((split == v).sum()) for s, v in (("train", 0), ("gap", 1), ("test", 3))}}


def feasibility(ds: str, wi: WindowIndex, classes: list[str], f: float) -> dict:
    split, sinfo = cut_split(wi, f)
    tr, te = split == splits.TRAIN, split == splits.TEST
    tr_cnt = np.bincount(wi.y_cls[tr], minlength=len(classes))
    te_cnt = np.bincount(wi.y_cls[te], minlength=len(classes))
    nn = [c for k, c in enumerate(classes) if k > 0 and te_cnt[k] >= MIN_TEST_WINDOWS and tr_cnt[k] == 0]
    near = {c: {"test": int(te_cnt[k]), "train": int(tr_cnt[k])} for k, c in enumerate(classes)
            if k > 0 and tr_cnt[k] == 0 and 0 < te_cnt[k] < MIN_TEST_WINDOWS}
    # train-tail validation (A3.2 mechanics, as p1c_build): last 20% of TRAIN windows by whole groups, no NN flows
    bits = np.uint64(0)
    for c in nn:
        bits |= np.uint64(1) << np.uint64(classes.index(c))
    has_nn = (wi.class_mask & bits) != 0
    g, n = np.unique(wi.group[tr], return_counts=True)
    cum = np.cumsum(n)
    head_groups = g[(cum - n) < PKG_CFG["head_frac"] * cum[-1]]
    head = tr & np.isin(wi.group, head_groups) & ~has_nn
    tail = tr & ~np.isin(wi.group, head_groups) & ~has_nn
    r = {"cut": f, "split": sinfo, "natural_novelty": nn,
         "test_windows": {c: int(te_cnt[classes.index(c)]) for c in nn}, "below_threshold": near,
         "head": {"benign": int((head & (wi.y_bin == 0)).sum()), "attack": int((head & (wi.y_bin == 1)).sum())},
         "tail": {"benign": int((tail & (wi.y_bin == 0)).sum()), "attack": int((tail & (wi.y_bin == 1)).sum())},
         "test_benign": int((te & (wi.y_bin == 0)).sum())}
    r["feasible"] = bool(nn) and r["tail"]["attack"] > 0 and r["tail"]["benign"] > 0 and r["head"]["attack"] > 0
    return r, split


def load(ds: str, dcfg: dict):
    base = ROOT / dcfg["processed_dir"] / ds
    wi = WindowIndex.from_npz(np.load(base / "windows.npz"))
    classes = json.loads((base / "index_meta.json").read_text())["classes"]
    return base, wi, classes


def count(dcfg: dict) -> dict:
    res = {"rule": {"cuts": CUTS, "min_test_windows": MIN_TEST_WINDOWS, "train_windows": 0,
                    "validation": "train-tail (A3.2), must hold benign and attack windows"}, "datasets": {}}
    for ds in DATASETS:
        _, wi, classes = load(ds, dcfg)
        res["datasets"][ds] = {cut_name(f): feasibility(ds, wi, classes, f)[0] for f in CUTS}
        for f in CUTS:
            r = res["datasets"][ds][cut_name(f)]
            print(f"[{ds} {cut_name(f)}] feasible={r['feasible']} NN={r['test_windows']} tail={r['tail']}", flush=True)
    feas = [(ds, c, k) for ds, v in res["datasets"].items() for c, r in v.items() if r["feasible"]
            for k in r["natural_novelty"]]
    res["units"] = [{"dataset": ds, "cut": c, "class": k} for ds, c, k in feas]
    res["n_units"] = len(feas)
    res["n_unique_classes"] = len({(ds, k) for ds, _, k in feas})
    res["provenance"] = provenance(config=res["rule"])
    write_json(OUT, res)
    print(f"{res['n_units']} (dataset, cut, class) units, {res['n_unique_classes']} unique (dataset, class)")
    return res


def refit_cleaner(ds: str, ref: Cleaner, df: pd.DataFrame, flow_group: np.ndarray, wi: WindowIndex,
                  split: np.ndarray, frozen: list[str]) -> Cleaner:
    """Same procedure as the original temporal_gap cleaner (same candidate columns), on this cut's TRAIN rows."""
    train_groups = np.unique(wi.group[split == splits.TRAIN])
    rows = np.isin(flow_group, train_groups)
    cl = Cleaner(keep_ttl=ref.keep_ttl).fit(df, ref.corr_sample_rows or 1_000_000, 0, rows=rows)
    try:
        cl.restrict(frozen)  # fail closed: ValueError if a frozen feature was dropped on this cut's TRAIN rows
    except ValueError as e:
        raise SystemExit(f"{ds}: {e}")
    return cl


def build_all(dcfg: dict):
    feas = json.loads(OUT.read_text())
    proc = ROOT / dcfg["processed_dir"]
    for ds in DATASETS:
        base, wi, classes = load(ds, dcfg)
        if dcfg["datasets"][ds].get("own_feature_space"):
            frozen = json.loads((base / "own_features.json").read_text())["features"]
        else:
            frozen = json.loads((proc / "common_features.json").read_text())["features"]
        ref = Cleaner.from_dict(json.loads((base / "temporal_gap" / "archive" / "cleaner.json").read_text()))
        df = pq.read_table(ROOT / dcfg["interim_dir"] / ds / "sorted.parquet",
                           columns=ref.candidate_features + ["group"]).to_pandas()
        flow_group = df.pop("group").to_numpy()
        for f in CUTS:
            r = feas["datasets"][ds][cut_name(f)]
            if not r["feasible"]:
                continue
            r2, split = feasibility(ds, wi, classes, f)
            if r2["natural_novelty"] != r["natural_novelty"]:
                raise SystemExit(f"{ds} {cut_name(f)}: natural-novelty classes differ from the recorded count")
            cl = refit_cleaner(ds, ref, df, flow_group, wi, split, frozen)
            out_dir = base / "rolling" / cut_name(f)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "split.npy", split)
            build(ds, r["natural_novelty"], dcfg, anchors=[], split=split, cleaner=cl, out_dir=out_dir,
                  extra_meta={"rolling": {"cut": f, "split": r["split"], "cleaner_fit_rows": cl.fit_rows,
                                          "cleaner_dropped": cl.dropped}})
        del df, flow_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    dcfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    require_clean_tree("rolling build")
    if a.count:
        count(dcfg)
    if a.build:
        build_all(dcfg)


if __name__ == "__main__":
    main()
