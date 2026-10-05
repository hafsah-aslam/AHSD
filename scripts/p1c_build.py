#!/usr/bin/env python
"""Build the P1c data packages (AMENDMENT_03 A3.2 / A3.3(iii)) for D1 and D3.

Sets (data seed 0; identical for every detector and seed):
  train      balanced 1:1 from the first 80% (by time, whole 1-h groups) of the
             temporal_gap TRAIN period, cap 14,000; no natural-novelty flows
  val        5:1 from the last 20% of the TRAIN period ("train-tail validation"), cap 3,000
  anchor_N   first N benign windows of the TEST period in time order (N = 100, 500; nested)
  test_benign  2,500 benign windows from the TEST period minus every group holding an anchor_500 window
  test/<c>   up to 500 windows whose majority class is c, same test pool
Windows are materialised with the dataset's frozen temporal_gap cleaner (41 features).
Every check is fail-closed.

  python scripts/p1c_build.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import pipeline, splits, verify  # noqa: E402
from nids.preprocess import Cleaner  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.windowing import WindowIndex  # noqa: E402

CFG = {"head_frac": 0.8, "train_cap": 14000, "val_cap": 3000, "val_ratio": 5, "anchors": [100, 500],
       "test_benign": 2500, "test_attack_cap": 500, "data_seed": 0}


def build(ds: str, nn_classes: list[str], dcfg: dict, anchors: list[int] | None = None,
          out_name: str = "p1c") -> dict:
    """anchors=None -> CFG anchors (P1c). anchors=[] -> no re-anchoring pool; test pool = whole test period."""
    anchors = CFG["anchors"] if anchors is None else anchors
    proc = ROOT / dcfg["processed_dir"]
    base = proc / ds
    wi = WindowIndex.from_npz(np.load(base / "windows.npz"))
    split = np.load(base / "temporal_gap" / "split.npy")
    meta = json.loads((base / "index_meta.json").read_text())
    classes = meta["classes"]
    rng = np.random.default_rng(CFG["data_seed"])
    log = []
    bits = np.uint64(0)
    for c in nn_classes:
        bits |= np.uint64(1) << np.uint64(classes.index(c))
    has_nn = (wi.class_mask & bits) != 0

    tr = split == splits.TRAIN
    g, n = np.unique(wi.group[tr], return_counts=True)  # group ids are in time order
    cum = np.cumsum(n)
    head_groups = g[(cum - n) < CFG["head_frac"] * cum[-1]]
    head = tr & np.isin(wi.group, head_groups) & ~has_nn
    tail = tr & ~np.isin(wi.group, head_groups) & ~has_nn
    sets = {}
    sets["train"], i_tr = splits.balance(np.flatnonzero(head & (wi.y_bin == 0)), np.flatnonzero(head & (wi.y_bin == 1)),
                                         1, CFG["train_cap"], rng)
    sets["val"], i_va = splits.balance(np.flatnonzero(tail & (wi.y_bin == 0)), np.flatnonzero(tail & (wi.y_bin == 1)),
                                       CFG["val_ratio"], CFG["val_cap"], rng)
    te = split == splits.TEST
    tb = np.flatnonzero(te & (wi.y_bin == 0))
    tb = tb[np.argsort(wi.start[tb], kind="stable")]
    for k in anchors:
        sets[f"anchor_{k}"] = np.sort(tb[:k])
    anchor_groups = np.unique(wi.group[sets[f"anchor_{max(anchors)}"]]) if anchors else np.zeros(0, np.int64)
    pool = te & ~np.isin(wi.group, anchor_groups)
    pb = np.flatnonzero(pool & (wi.y_bin == 0))
    sets["test_benign"] = np.sort(rng.choice(pb, min(CFG["test_benign"], len(pb)), replace=False))
    test_info = {}
    for c in nn_classes:
        k = classes.index(c)
        pc = np.flatnonzero(pool & (wi.y_cls == k))
        sets[f"test/{c}"] = np.sort(rng.choice(pc, min(CFG["test_attack_cap"], len(pc)), replace=False))
        test_info[c] = {"windows": int(len(sets[f"test/{c}"])), "available": int(len(pc))}

    # ---- fail-closed checks
    grp = lambda idx: set(wi.group[idx].tolist())  # noqa: E731
    test_all = np.concatenate([sets["test_benign"]] + [sets[f"test/{c}"] for c in nn_classes])
    verify._check(bool(np.all(split[sets["train"]] == splits.TRAIN)) and bool(np.all(split[sets["val"]] == splits.TRAIN)),
                  f"{ds}: train and val inside the temporal_gap TRAIN period", log)
    verify._check(not grp(sets["train"]) & grp(sets["val"]), f"{ds}: train head and train tail share no group", log)
    verify._check(max(wi.group[sets["train"]]) < min(wi.group[sets["val"]]), f"{ds}: train tail is later than train head", log)
    verify._check(bool(np.all((wi.class_mask[np.r_[sets['train'], sets['val']]] & bits) == 0)),
                  f"{ds}: no natural-novelty flow in train or val", log)
    verify._check(i_va["attack"] > 0 and i_va["benign"] > 0, f"{ds}: train-tail validation has benign and attack", log)
    verify._check(bool(np.all(split[test_all] == splits.TEST)), f"{ds}: test windows from the TEST period", log)
    verify._check(not grp(test_all) & set(anchor_groups.tolist()), f"{ds}: test excludes every anchor group", log)
    if anchors:
        a_lo, a_hi = f"anchor_{min(anchors)}", f"anchor_{max(anchors)}"
        verify._check(bool(np.all(wi.y_bin[sets[a_hi]] == 0)) and set(sets[a_lo]) <= set(sets[a_hi]),
                      f"{ds}: anchors benign, {a_lo} nested in {a_hi}", log)
    verify._check(bool(np.all(wi.y_bin[sets['test_benign']] == 0)), f"{ds}: test benign windows are benign", log)
    for c in nn_classes:
        k = classes.index(c)
        verify._check(test_info[c]["windows"] > 0 and bool(np.all(wi.y_cls[sets[f'test/{c}']] == k)),
                      f"{ds}: test/{c} holds only {c} windows, at least one", log)
        verify._check(k not in set(np.unique(wi.y_cls[np.r_[sets['train'], sets['val']]]).tolist()),
                      f"{ds}: {c} absent from train and val", log)

    cl = Cleaner.from_dict(json.loads((base / "temporal_gap" / "archive" / "cleaner.json").read_text()))
    if dcfg["datasets"][ds].get("own_feature_space"):
        common = json.loads((base / "own_features.json").read_text())["features"]
    else:
        common = json.loads((proc / "common_features.json").read_text())["features"]
    if cl.features != common:
        raise SystemExit(f"{ds}: cleaner features differ from its frozen feature list")
    n_rows = meta["flows"]
    arch = pipeline._write_archive(base / "temporal_gap" / out_name, ROOT / dcfg["interim_dir"] / ds / "sorted.parquet",
                                   n_rows, cl, wi, split, sets, {}, {})
    info = {"dataset": ds, "classes": classes, "natural_novelty": nn_classes, "cfg": {**CFG, "anchors": anchors},
            "head_groups": [int(x) for x in head_groups], "tail_groups": sorted(int(x) for x in grp(sets["val"])),
            "anchor_groups": anchor_groups.tolist(), "sets": {"train": i_tr, "val": i_va,
                                                              "anchor": {k: int(len(sets[f'anchor_{k}'])) for k in anchors},
                                                              "test_benign": int(len(sets["test_benign"])), "test": test_info},
            "verification": log, "archive": arch, "provenance": provenance(config=CFG)}
    write_json(base / "temporal_gap" / out_name / "p1c_meta.json", info)
    print(f"[{ds}] P1c package: {len(log)} checks passed; train {i_tr['benign']}+{i_tr['attack']}, "
          f"val {i_va['benign']}+{i_va['attack']}, test benign {len(sets['test_benign'])}, {test_info}")
    return info


def main():
    require_clean_tree("P1c build")
    dcfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    for ds in ("D1", "D3"):
        build(ds, dcfg["datasets"][ds]["natural_novelty"], dcfg)


if __name__ == "__main__":
    main()
