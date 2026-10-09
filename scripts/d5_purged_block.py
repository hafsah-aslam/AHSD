#!/usr/bin/env python
"""D5 LOACO protocol "purged_block" (AMENDMENT_05 A5.1) + feasibility count + A5.2 decision.

10-minute blocks; capture days = runs of non-empty blocks separated by < 6 h; per day a
seeded shuffle assigns blocks 60/20/20 (round half up) to train/val/test; purge train/val
blocks adjacent (±1, same day) to test, then train blocks adjacent to val. Windows rebuilt
within blocks, deduped with the D5 index hash columns. Folds: grouped_random mechanics plus
held-out class >= 200 train windows (majority) and >= 50 test windows. Cleaner fitted on
purged_block train rows, restricted to the frozen D5 own feature list (fail closed).

Writes data/processed/D5/purged_block/{meta.json, archive/} and results/p1d/a52_decision.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import pipeline, splits, verify  # noqa: E402
from nids.preprocess import Cleaner  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.windowing import dedupe, make_windows  # noqa: E402

BLOCK_MS = 600_000
DAY_GAP_BLOCKS = 36          # 6 h
N_DAYS = 5
FRAC = (0.6, 0.2)
MIN_TEST = 50
FALLBACK_BELOW = 3


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def main():
    require_clean_tree("D5 purged_block")
    dcfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    base = ROOT / dcfg["processed_dir"] / "D5"
    imeta = json.loads((base / "index_meta.json").read_text())
    classes = imeta["classes"]
    hash_cols = imeta["hash_columns"]
    parquet = ROOT / dcfg["interim_dir"] / "D5" / "sorted.parquet"
    df = pd.read_parquet(parquet)
    t = df["FLOW_START_MILLISECONDS"].to_numpy(np.float64)
    if np.any(np.diff(t) < 0):
        raise SystemExit("interim D5 parquet is not time-sorted")
    log = []

    # ---- 10-minute blocks, windows, dedupe
    blk = np.floor((t - t.min()) / BLOCK_MS).astype(np.int64)
    uniq, group = np.unique(blk, return_inverse=True)
    group = group.astype(np.int32)
    arrs = [df[c].to_numpy() for c in hash_cols]
    wi, wstats = make_windows(group, df["cls"].to_numpy().astype(np.int16),
                              lambda a, b: np.stack([x[a:b] for x in arrs], 1).astype(np.float32),
                              dcfg["window"]["T"], dcfg["window"]["stride"])
    wi, dstats = dedupe(wi)

    # ---- capture days
    day_of_group = np.r_[0, np.cumsum(np.diff(uniq) >= DAY_GAP_BLOCKS)]
    n_days = int(day_of_group.max()) + 1
    verify._check(n_days == N_DAYS, f"D5: exactly {N_DAYS} capture days found ({n_days})", log)

    # ---- per-day seeded shuffle split, then purge
    rng = np.random.default_rng(dcfg["data_seed"])
    assign = np.full(len(uniq), -1, np.int8)
    day_info = []
    for d in range(n_days):
        g_d = np.flatnonzero(day_of_group == d)
        perm = rng.permutation(g_d)
        n = len(perm)
        ntr, nva = _round_half_up(FRAC[0] * n), _round_half_up(FRAC[1] * n)
        assign[perm[:ntr]] = splits.TRAIN
        assign[perm[ntr:ntr + nva]] = splits.VAL
        assign[perm[ntr + nva:]] = splits.TEST
        day_info.append({"day": d, "blocks": n, "train": ntr, "val": nva, "test": n - ntr - nva,
                         "first_block_utc_ms": float(t.min() + uniq[g_d[0]] * BLOCK_MS)})
    abs_idx = uniq  # absolute 10-min index per group
    pos = {int(a): i for i, a in enumerate(abs_idx)}

    def neighbours(i):
        return [pos[a] for a in (int(abs_idx[i]) - 1, int(abs_idx[i]) + 1)
                if a in pos and day_of_group[pos[a]] == day_of_group[i]]

    purged = np.zeros(len(uniq), bool)
    rule1 = [i for i in range(len(uniq)) if assign[i] in (splits.TRAIN, splits.VAL)
             and any(assign[j] == splits.TEST for j in neighbours(i))]
    purged[rule1] = True
    rule2 = [i for i in range(len(uniq)) if assign[i] == splits.TRAIN and not purged[i]
             and any(assign[j] == splits.VAL and not purged[j] for j in neighbours(i))]
    purged[rule2] = True
    final = assign.copy()
    final[purged] = splits.GAP  # purged blocks are used nowhere
    for di in day_info:
        g_d = day_of_group == di["day"]
        di["purged_rule1_adjacent_test"] = int(np.isin(np.flatnonzero(g_d), rule1).sum())
        di["purged_rule2_train_adjacent_val"] = int(np.isin(np.flatnonzero(g_d), rule2).sum())
        di["after_purge"] = {splits.SPLIT_NAMES[s]: int((final[g_d] == s).sum()) for s in (splits.TRAIN, splits.VAL, splits.TEST)}
    split = final[wi.group]

    # ---- adjacency checks (fail closed)
    viol = [i for i in range(len(uniq))
            if (final[i] in (splits.TRAIN, splits.VAL) and splits.TEST in [final[j] for j in neighbours(i)])
            or (final[i] == splits.TRAIN and splits.VAL in [final[j] for j in neighbours(i)])]
    verify._check(not viol, f"D5 purge: no train/val block adjacent to a test block; no train block adjacent "
                            f"to a val block (violations: {viol[:10]})", log)

    # ---- splits, sets, folds, checks
    bal = dict(dcfg["balance"])
    log += verify.verify_splits(wi, split, df["row_id"].to_numpy())
    sets, set_info = splits.make_sets(wi, split, bal, dcfg["data_seed"])
    log += verify.verify_sets(wi, split, sets, set_info, bal)
    folds, finfo = splits.loaco_folds(wi, split, classes, bal, dcfg["data_seed"])
    comp = splits.composition(wi, split, classes)
    for c, rec in finfo.items():
        n_test = comp["windows_by_majority_class"]["test"][c]
        rec["test_windows_majority"] = n_test
        if rec.get("eligible") and rec.get("feasible") and n_test < MIN_TEST:
            rec["feasible"] = False
            rec["reason"] = f"< {MIN_TEST} test windows ({n_test})"
    folds = {c: f for c, f in folds.items() if finfo[c].get("feasible")}
    log += verify.verify_loaco(wi, split, folds, classes)
    feasible = sorted(folds)

    # ---- cleaner on purged_block train rows, frozen D5 feature list
    own = json.loads((base / "own_features.json").read_text())["features"]
    train_groups = np.flatnonzero(final == splits.TRAIN)
    cl = Cleaner(keep_ttl=dcfg["keep_ttl"]).fit(df, dcfg["corr_sample_rows"], dcfg["data_seed"],
                                                rows=np.isin(group, train_groups))
    missing = [f for f in own if f not in cl.features]
    verify._check(not missing, f"D5 purged_block cleaner keeps every frozen D5 feature (missing: {missing})", log)
    cl.restrict(own)
    out = base / "purged_block"
    arch = pipeline._write_archive(out / "archive", parquet, len(df), cl, wi, split, sets, folds, {})
    meta = {"dataset": "D5", "scheme": "purged_block", "classes": classes, "block_ms": BLOCK_MS,
            "day_gap_blocks": DAY_GAP_BLOCKS, "rounding": "half up", "days": day_info,
            "blocks": {"non_empty": int(len(uniq)), "purged_rule1": len(rule1), "purged_rule2": len(rule2),
                       "after_purge": {splits.SPLIT_NAMES[s]: int((final == s).sum()) for s in (splits.TRAIN, splits.VAL, splits.TEST)}},
            "windowing": wstats, "dedupe": dstats, "split_windows": {splits.SPLIT_NAMES[s]: int((split == s).sum()) for s in splits.SPLIT_NAMES},
            "composition": comp, "sets": set_info, "loaco": finfo, "feasible_folds": feasible,
            "verification": log, "archive": arch, "cleaner_kept_before_restrict": len(cl.features),
            "provenance": provenance(config={"block_ms": BLOCK_MS, "min_test": MIN_TEST, "balance": bal})}
    write_json(out / "meta.json", meta)
    decision = {"feasible_folds": feasible, "n_feasible": len(feasible), "threshold": FALLBACK_BELOW,
                "fallback_applied": len(feasible) < FALLBACK_BELOW,
                "rule": "A5.2: if < 3 feasible D5 LOACO folds, G3 (b) on D3 LOACO only; D5 contributes (a) only",
                "provenance": provenance()}
    write_json(ROOT / "results/p1d/a52_decision.json", decision)
    print(json.dumps({"days": day_info, "blocks": meta["blocks"], "windows": wstats["windows"],
                      "dedupe_removed": dstats["duplicates_removed"], "feasible_folds": feasible,
                      "fold_info": {c: (r["train_windows_majority"], r.get("test_windows_majority"), r.get("feasible"),
                                        r.get("reason", "")) for c, r in finfo.items()},
                      "fallback_applied": decision["fallback_applied"], "checks": f"{sum(x['ok'] for x in log)}/{len(log)}"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
