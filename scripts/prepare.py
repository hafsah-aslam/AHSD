#!/usr/bin/env python
"""Prepare validated archives for every dataset (spec §4).

  python scripts/prepare.py                       # all datasets present in data/raw
  python scripts/prepare.py --datasets D1 D2 --resume
  python scripts/prepare.py --smoke               # synthetic NF-v3-schema data, small caps,
                                                  # written under smoke_out/ (never results)

The common feature list is the intersection over every dataset in the run. A
run over a subset of D1–D4 is refused unless --allow-partial-common is given,
because adding a dataset later can shrink the list and change every archive.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import pipeline  # noqa: E402
from nids.preprocess_log import render_preprocess_log  # noqa: E402
from nids.provenance import ROOT, write_json  # noqa: E402


def smoke_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    base = "smoke_out"
    c.update(raw_dir=f"{base}/raw", interim_dir=f"{base}/interim", processed_dir=f"{base}/processed",
             audit_dir=f"{base}/audit", corr_sample_rows=20000)
    c["balance"].update(train_cap=2000, val_cap=600, test_cap=600, natural_cap=2000, loaco_min_train=30)
    c["datasets"] = {k: v for k, v in c["datasets"].items() if k in ("D1", "D2")}
    for d in c["datasets"].values():  # real-data-only options (AMENDMENT_01)
        for k in ("class_map", "natural_novelty", "target_only"):
            d.pop(k, None)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-partial-common", action="store_true")
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--stage", choices=["all", "index", "finalize"], default="all",
                    help="index: per-dataset stage only (run each dataset in its own process to "
                         "measure its peak RAM); finalize: common list + archives + transfer")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))

    if a.smoke:
        from nids.synthetic import generate
        cfg = smoke_cfg(cfg)
        frames = {"D1": generate(40_000, hours=14, seed=1), "D2": generate(30_000, hours=12, seed=2, shift=0.4)}
        ds_list = ["D1", "D2"]
        for ds in ds_list:
            pipeline.stage_index(ds, cfg, df=frames[ds], resume=a.resume, source_file=f"synthetic:{ds}")
    else:
        present = [ds for ds, d in cfg["datasets"].items()
                   if any((ROOT / cfg["raw_dir"]).glob(d["raw_glob"]))]
        ds_list = a.datasets or present
        missing = [ds for ds in ds_list if ds not in present]
        if missing or not ds_list:
            raise SystemExit(f"raw files missing for {missing or 'all datasets'}; run scripts/download.py "
                             f"or place the files in {cfg['raw_dir']}")
        in_domain_all = [d for d, v in cfg["datasets"].items()
                         if not v.get("target_only") and not v.get("own_feature_space")]
        if a.stage != "index" and not set(in_domain_all) <= set(ds_list) and not a.allow_partial_common:
            raise SystemExit(f"only {ds_list} requested; the common feature list must span "
                             f"{list(cfg['datasets'])}. Pass --allow-partial-common to proceed anyway.")
        manifest = ROOT / "data" / "MANIFEST.json"
        shas = json.loads(manifest.read_text())["datasets"] if manifest.exists() else {}
        if a.stage in ("all", "index"):
            for ds in ds_list:
                pipeline.stage_index(ds, cfg, resume=a.resume, raw_sha256=shas.get(ds, {}).get("sha256"),
                                     n_rows_hint=shas.get(ds, {}).get("row_count"))
        if a.stage == "index":
            return
        if a.stage == "finalize":
            ds_list = present if not a.datasets else ds_list

    targets = [d for d in ds_list if cfg["datasets"][d].get("target_only")]
    own_space = [d for d in ds_list if cfg["datasets"][d].get("own_feature_space")]
    for ds in own_space:  # own feature space: own common list, no transfer (AMENDMENT_04)
        own_common = pipeline.common_features(cfg, [ds])
        write_json(ROOT / cfg["processed_dir"] / ds / "own_features.json", {"dataset": ds, "features": own_common})
        print(f"[{ds}] own feature space ({len(own_common)}): {own_common}")
        pipeline.stage_finalize(ds, cfg, own_common, resume=a.resume)
    in_domain = [d for d in ds_list if d not in targets and d not in own_space]
    if not in_domain:
        print("no shared-feature-space datasets in this run")
        return
    common = pipeline.common_features(cfg, in_domain)
    frozen_path = ROOT / cfg["processed_dir"] / "common_features.json"
    if frozen_path.exists() and not a.smoke:
        frozen = json.loads(frozen_path.read_text())
        if frozen["features"] != common:
            raise SystemExit(f"recomputed common features differ from the frozen list in {frozen_path}: "
                             f"{sorted(set(frozen['features']) ^ set(common))}")
        print(f"common features: {len(common)}, identical to the frozen list")
    else:
        write_json(frozen_path, {"datasets": in_domain,
                                 "partial": set(in_domain) != {d for d, v in cfg["datasets"].items()
                                                               if not v.get("target_only")
                                                               and not v.get("own_feature_space")} and not a.smoke,
                                 "features": common})
        print(f"common features ({len(common)}): {common}")
    for ds in in_domain:
        pipeline.stage_finalize(ds, cfg, common, resume=a.resume)
    if not a.skip_transfer:
        for src in in_domain:
            for tgt in in_domain:
                if src != tgt:
                    pipeline.stage_transfer(src, tgt, cfg, resume=a.resume)
            for tgt in targets:
                pipeline.stage_transfer_target_only(src, tgt, cfg, common, resume=a.resume)
    log_path = ROOT / ("smoke_out/PREPROCESS_LOG.md" if a.smoke else "PREPROCESS_LOG.md")
    log_path.write_text(render_preprocess_log(cfg, in_domain, smoke=a.smoke))
    print(f"wrote {log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
