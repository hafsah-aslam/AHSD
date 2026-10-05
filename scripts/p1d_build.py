#!/usr/bin/env python
"""P1d D5 natural-novelty package + feasibility report (AMENDMENT_04 A4.3).

Natural-novelty classes (pre-registered rule): classes present as a window's majority class
in the temporal_gap TEST split and absent from its TRAIN split; evaluated if >= 20 test
windows. Feasibility: at least one evaluable class AND train-tail validation holds attack
windows. Writes results/p1d/d5_feasibility.json; exits with code 2 (STOP) if infeasible.
Package: train-tail validation, no re-anchoring pool (test pool = whole test period),
written to data/processed/D5/temporal_gap/p1d/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1c_build  # noqa: E402
from nids import splits  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.windowing import WindowIndex  # noqa: E402

MIN_TEST_WINDOWS = 20


def feasibility(ds: str, dcfg: dict) -> dict:
    base = ROOT / dcfg["processed_dir"] / ds
    meta = json.loads((base / "index_meta.json").read_text())
    comp = meta["schemes"]["temporal_gap"]["composition"]
    test_only = comp["test_classes_absent_from_train"]
    tab = comp["windows_by_majority_class"]
    evaluable = [c for c in test_only if tab["test"][c] >= MIN_TEST_WINDOWS]
    wi = WindowIndex.from_npz(np.load(base / "windows.npz"))
    split = np.load(base / "temporal_gap" / "split.npy")
    classes = meta["classes"]
    bits = np.uint64(0)
    for c in evaluable:
        bits |= np.uint64(1) << np.uint64(classes.index(c))
    has_nn = (wi.class_mask & bits) != 0
    tr = split == splits.TRAIN
    g, n = np.unique(wi.group[tr], return_counts=True)
    cum = np.cumsum(n)
    head_groups = g[(cum - n) < p1c_build.CFG["head_frac"] * cum[-1]]
    tail = tr & ~np.isin(wi.group, head_groups) & ~has_nn
    tail_att = int((tail & (wi.y_bin == 1)).sum())
    out = {"dataset": ds, "rule": f"test-only majority classes with >= {MIN_TEST_WINDOWS} test windows",
           "test_only_classes": test_only, "test_windows": {c: tab["test"][c] for c in test_only},
           "evaluable_classes": evaluable,
           "train_tail": {"groups": int(len(g) - len(head_groups)), "benign": int((tail & (wi.y_bin == 0)).sum()),
                          "attack": tail_att,
                          "attack_classes": {classes[c]: int((tail & (wi.y_cls == c)).sum())
                                             for c in np.unique(wi.y_cls[tail]) if c}},
           "train_head_attack_classes": {classes[c]: int((tr & np.isin(wi.group, head_groups) & (wi.y_cls == c)).sum())
                                         for c in np.unique(wi.y_cls[tr & np.isin(wi.group, head_groups)]) if c},
           "split_windows": meta["schemes"]["temporal_gap"]["split"]["windows"],
           "composition": tab}
    out["feasible"] = bool(evaluable) and tail_att > 0
    out["reasons"] = ([] if evaluable else ["no evaluable test-only class"]) + ([] if tail_att else ["train-tail validation has no attack windows"])
    out["provenance"] = provenance(config={"min_test_windows": MIN_TEST_WINDOWS})
    return out


def main():
    require_clean_tree("P1d build")
    dcfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    f = feasibility("D5", dcfg)
    write_json(ROOT / "results/p1d/d5_feasibility.json", f)
    print(json.dumps({k: f[k] for k in ("test_only_classes", "test_windows", "evaluable_classes", "train_tail",
                                        "feasible", "reasons")}, indent=1))
    if not f["feasible"]:
        print("D5 natural_novelty INFEASIBLE -> STOP (AMENDMENT_04 A4.3)")
        sys.exit(2)
    p1c_build.build("D5", f["evaluable_classes"], dcfg, anchors=[], out_name="p1d")


if __name__ == "__main__":
    main()
