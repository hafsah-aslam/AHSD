"""Data preparation stages (spec §4).

  index     audit -> sort -> 1-h groups -> windows -> global dedupe -> splits
            -> balanced sets + natural-prior test -> LOACO folds -> fit cleaner
            on train rows (one per split scheme)
  finalize  common feature list across datasets -> transform -> compact
            archive of the rows any selected window needs -> verify -> SHA-256
  transfer  rebuild each target with the source's cleaner (medians, mean/std,
            feature order); nothing is fitted on the target

Every stage writes JSON next to its arrays and is skipped under --resume when
its outputs exist with the same config hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import audit as audit_mod
from . import io, schema, splits, verify
from .preprocess import Cleaner, common_feature_list
from .provenance import ROOT, sha256_bytes, sha256_file, write_json, provenance
from .windowing import WindowIndex, class_codes, dedupe, hour_groups, make_windows, relabel, sort_by_time


def cfg_hash(cfg: dict) -> str:
    keys = ("keep_ttl", "window", "splits", "data_seed", "balance", "corr_sample_rows")
    return sha256_bytes(json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True).encode())[:16]


def _p(cfg, key) -> Path:
    p = Path(cfg[key])
    return p if p.is_absolute() else ROOT / p


def _done(meta_path: Path, h: str) -> bool:
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text()).get("cfg_hash") == h
    except Exception:
        return False


def _save_folds(path: Path, folds: dict):
    np.savez(path, **{f"{c}/{part}": idx for c, f in folds.items() for part, idx in f.items()})


def family_codes(codes: np.ndarray, classes: list[str], map_path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Map flow class codes to family codes via a {family: [sub-labels]} YAML. Fails closed."""
    import yaml
    fams = yaml.safe_load(open(map_path))["families"]
    owner = {}
    for f, subs in fams.items():
        for x in subs:
            if x in owner:
                raise RuntimeError(f"class map: {x!r} in two families")
            owner[x] = f
    missing = [c for c in classes if c not in owner]
    if missing:
        raise RuntimeError(f"class map {map_path.name}: unmapped file labels {missing}")
    if owner.get(classes[0]) != schema.BENIGN:
        raise RuntimeError("class map must map Benign to Benign")
    fam_classes = [schema.BENIGN] + sorted(f for f in fams if f != schema.BENIGN)
    lut = np.array([fam_classes.index(owner[c]) for c in classes], dtype=np.int16)
    counts = np.bincount(lut[codes], minlength=len(fam_classes))
    return lut[codes], fam_classes, {"map": str(map_path.relative_to(ROOT)) if map_path.is_relative_to(ROOT) else str(map_path),
                                     "map_sha256": sha256_file(map_path),
                                     "flow_counts": {c: int(n) for c, n in zip(fam_classes, counts)},
                                     "sub_to_family": {c: owner[c] for c in classes}}


def load_folds(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for k in z.files:
        c, part = k.rsplit("/", 1)
        out.setdefault(c, {})[part] = z[k]
    return out


def _write_parquet(df: pd.DataFrame, path: Path, rows_per_group: int = 2_000_000):
    """Row-group-wise write: Arrow conversion never holds the whole frame twice."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    try:
        for a in range(0, max(len(df), 1), rows_per_group):
            t = pa.Table.from_pandas(df.iloc[a:a + rows_per_group], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, t.schema)
            writer.write_table(t)
    finally:
        if writer is not None:
            writer.close()


def _read_rows(path: Path, columns: list[str], rows: np.ndarray) -> pd.DataFrame:
    """Read only `columns` and only `rows` (sorted positions) of the interim parquet."""
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns, memory_map=True)
    return t.take(rows).to_pandas()


def rss_gb() -> float:
    """Current resident set size (Linux /proc)."""
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def peak_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KiB -> GB


# --------------------------------------------------------------------- index
def stage_index(ds: str, cfg: dict, df: pd.DataFrame | None = None, resume: bool = False,
                raw_sha256: str | None = None, source_file: str | None = None,
                n_rows_hint: int | None = None) -> dict:
    dcfg = cfg["datasets"][ds]
    h = cfg_hash(cfg) + sha256_bytes(json.dumps(dcfg, sort_keys=True).encode())[:8]
    out = _p(cfg, "processed_dir") / ds
    meta_path = out / "index_meta.json"
    if resume and _done(meta_path, h):
        print(f"[{ds}] index: up to date, skipped")
        return json.loads(meta_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["datasets"][ds]

    if df is None:
        raw = io.find_raw_file(_p(cfg, "raw_dir"), dcfg["raw_glob"])
        source_file, raw_sha256 = raw.name, raw_sha256 or sha256_file(raw)
        print(f"[{ds}] reading {raw}")
        df = io.read_raw(raw, n_rows_hint=n_rows_hint)
    mem = [("loaded", rss_gb())]
    aud = audit_mod.audit_frame(df, ds, source_file or "in-memory", raw_sha256 or "n/a")
    write_json(_p(cfg, "audit_dir") / f"{ds}.json", aud)
    if aud["schema_check"]["missing"]:
        print(f"[{ds}] WARNING columns missing vs expected NF-v3 schema: {aud['schema_check']['missing']}")
    if schema.TIME_START not in df:
        raise RuntimeError(f"{ds}: no {schema.TIME_START}; cannot sort or group")

    mem.append(("audited", rss_gb()))
    df = sort_by_time(df)
    mem.append(("sorted", rss_gb()))
    group = hour_groups(df[schema.TIME_START].to_numpy(np.float64), cfg["window"]["block_ms"])
    codes, classes = class_codes(df[schema.LABEL_MULTI])
    drops = schema.drop_map(cfg["keep_ttl"])
    hash_cols = [c for c in df.columns if c not in drops and c != "row_id"
                 and pd.api.types.is_numeric_dtype(df[c])]
    hash_arrays = [df[c].to_numpy() for c in hash_cols]  # views, no copy
    wi, wstats = make_windows(group, codes,
                              lambda a, b: np.stack([x[a:b] for x in hash_arrays], 1).astype(np.float32),
                              cfg["window"]["T"], cfg["window"]["stride"])
    del hash_arrays
    mem.append(("windowed", rss_gb()))
    wi, dstats = dedupe(wi)
    wi_fam, fam_classes, fam_info = None, None, None
    if dcfg.get("class_map"):
        fcodes, fam_classes, fam_info = family_codes(codes, classes, ROOT / dcfg["class_map"])
        wi_fam, fam_info["majority_ties"] = relabel(wi, fcodes)
        del fcodes
    n_benign_windows = int((wi.y_bin == 0).sum())

    interim = _p(cfg, "interim_dir") / ds
    interim.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in (schema.LABEL_MULTI, "IPV4_SRC_ADDR", "IPV4_DST_ADDR") if c in df],
            inplace=True)
    df["group"] = group
    df["cls"] = codes
    _write_parquet(df, interim / "sorted.parquet")
    mem.append(("parquet_written", rss_gb()))
    print(f"[{ds}] memory (GB): {mem}", flush=True)

    info = {"dataset": ds, "name": dcfg["name"], "cfg_hash": h, "classes": classes,
            "flows": int(len(df)), "hash_columns": hash_cols, "windowing": wstats, "dedupe": dstats,
            "benign_windows_after_dedupe": n_benign_windows, "schemes": {}}
    min_benign = dcfg.get("loaco_min_benign_windows")
    loaco_allowed = "loaco" in dcfg["role"] or (min_benign is not None and n_benign_windows >= min_benign)
    info["loaco_allowed"] = bool(loaco_allowed)

    np.savez(out / "windows.npz", **wi.to_npz_dict())
    if dcfg.get("target_only"):
        tsets, tinfo, tlog = target_only_sets(wi, classes, dcfg["target_only"], cfg["data_seed"])
        np.savez(out / "target_sets.npz", **tsets)
        info.update(target_only=tinfo, verification=tlog)
        mem.append(("done", rss_gb()))
        info["memory_trace_gb"] = mem
        info["peak_rss_gb_process"] = peak_rss_gb()
        info["provenance"] = provenance(config=cfg, data_sha256={ds: raw_sha256})
        write_json(meta_path, info)
        print(f"[{ds}] index (target only): {len(wi)} windows; test {tinfo['test']}; pool {tinfo['anchor_pool']}")
        return info
    if wi_fam is not None:
        np.savez(out / "windows_family.npz", y_cls=wi_fam.y_cls, class_mask=wi_fam.class_mask,
                 purity=wi_fam.purity, classes=np.array(fam_classes))
        info["family"] = {"classes": fam_classes, **fam_info}
    for scheme in cfg["splits"]:
        sd = out / scheme
        sd.mkdir(exist_ok=True)
        split, sinfo = splits.assign_split(wi, scheme, cfg["data_seed"])
        comp = splits.composition(wi, split, classes)
        sets, set_info = splits.make_sets(wi, split, cfg["balance"], cfg["data_seed"])
        if loaco_allowed:
            folds, finfo = splits.loaco_folds(wi, split, classes, cfg["balance"], cfg["data_seed"])
        else:
            folds, finfo = {}, {"skipped": f"benign windows {n_benign_windows} < {min_benign}"}
        vlog = verify.verify_splits(wi, split, df["row_id"].to_numpy())
        vlog += verify.verify_sets(wi, split, sets, set_info, cfg["balance"])
        vlog += verify.verify_loaco(wi, split, folds, classes)
        extra = {}
        if wi_fam is not None:
            ffolds, ffinfo = splits.loaco_folds(wi_fam, split, fam_classes, cfg["balance"], cfg["data_seed"])
            vlog += [{**v, "check": "[family] " + v["check"]} for v in verify.verify_loaco(wi_fam, split, ffolds, fam_classes)]
            _save_folds(sd / "loaco_family.npz", ffolds)
            extra["loaco_family"] = ffinfo
            extra["composition_family"] = splits.composition(wi_fam, split, fam_classes)
        nn_cfg = dcfg.get("natural_novelty")
        if nn_cfg and scheme == "temporal_gap":
            found = comp["test_classes_absent_from_train"]
            if sorted(found) != sorted(nn_cfg):
                raise RuntimeError(f"{ds}: natural_novelty classes {sorted(found)} differ from the amendment's {sorted(nn_cfg)}")
            nn_sets, nn_info = splits.natural_novelty_sets(wi, split, classes, nn_cfg, cfg["balance"], cfg["data_seed"])
            vlog += verify.verify_natural_novelty(wi, split, nn_sets, classes, nn_cfg)
            np.savez(sd / "natural_novelty.npz", **nn_sets)
            extra["natural_novelty"] = nn_info

        train_groups = np.array(sinfo["groups"]["train"])
        train_rows = np.isin(group, train_groups)
        cl = Cleaner(keep_ttl=cfg["keep_ttl"]).fit(df, cfg["corr_sample_rows"], cfg["data_seed"], rows=train_rows)
        write_json(sd / "cleaner_fit.json", cl.to_dict())
        np.save(sd / "split.npy", split)
        np.savez(sd / "sets.npz", **sets)
        _save_folds(sd / "loaco.npz", folds)
        info["schemes"][scheme] = {"split": sinfo, "composition": comp, "sets": set_info,
                                   "loaco": finfo, "verification": vlog,
                                   "cleaner_fit_rows": int(train_rows.sum()),
                                   "cleaner_kept": cl.features, "cleaner_dropped": cl.dropped, **extra}
    mem.append(("done", rss_gb()))
    info["memory_trace_gb"] = mem
    info["peak_rss_gb_process"] = peak_rss_gb()
    info["provenance"] = provenance(config=cfg, data_sha256={ds: raw_sha256})
    write_json(meta_path, info)
    print(f"[{ds}] index: {wstats['windows']} windows, {dstats['duplicates_removed']} duplicates removed")
    return info


def target_only_sets(wi: WindowIndex, classes: list[str], tcfg: dict, seed: int) -> tuple[dict, dict, list]:
    """AMENDMENT_01 A1.4: re-anchoring pool (benign windows of the first k groups) and a
    group-disjoint test set (all remaining benign + attack at 1/ratio, equal per class)."""
    rng = np.random.default_rng(seed)
    groups = np.unique(wi.group)  # ids are in time order
    first = groups[:tcfg["anchor_groups_first"]]
    in_first = np.isin(wi.group, first)
    pool = np.flatnonzero(in_first & (wi.y_bin == 0))
    sizes = sorted(tcfg["anchor_sizes"])
    if len(pool) < sizes[-1]:
        raise RuntimeError(f"re-anchoring pool has {len(pool)} benign windows < {sizes[-1]}")
    perm = rng.permutation(pool)
    sets = {"anchor_pool": np.sort(pool)}
    for k in sizes:
        sets[f"anchor_{k}"] = np.sort(perm[:k])  # nested: anchor_100 is a subset of anchor_500
    rest = ~in_first
    benign = np.flatnonzero(rest & (wi.y_bin == 0))
    n_att = len(benign) // tcfg["test_ratio"]
    att_classes = [c for c in range(1, len(classes)) if ((wi.y_cls == c) & rest).any()]
    avail = {c: np.flatnonzero(rest & (wi.y_cls == c)) for c in att_classes}
    quota = {c: 0 for c in att_classes}
    left, open_c = n_att, list(att_classes)
    while left > 0 and open_c:  # equal allocation; a class short of windows passes its share on
        share = max(left // len(open_c), 1)
        for c in list(open_c):
            take = min(share, len(avail[c]) - quota[c], left)
            quota[c] += take
            left -= take
            if quota[c] >= len(avail[c]):
                open_c.remove(c)
            if left == 0:
                break
    attack = np.concatenate([rng.choice(avail[c], quota[c], replace=False) for c in att_classes if quota[c]])
    sets["test"] = np.sort(np.r_[benign, attack]).astype(np.int64)
    log = []
    verify._check(len(np.intersect1d(wi.group[sets["anchor_pool"]], wi.group[sets["test"]])) == 0,
                  "D4: re-anchoring pool and test share no group (hence no flow)", log)
    for a, b in zip(sizes[:-1], sizes[1:]):
        verify._check(set(sets[f"anchor_{a}"]) <= set(sets[f"anchor_{b}"]), f"D4: anchor_{a} nested in anchor_{b}", log)
    verify._check(bool(np.all(wi.y_bin[sets["anchor_pool"]] == 0)), "D4: re-anchoring pool is benign only", log)
    verify._check(len(np.unique(sets["test"])) == len(sets["test"]), "D4: no test window drawn twice", log)
    info = {"anchor_groups": first.tolist(), "anchor_pool": int(len(pool)), "anchor_sizes": sizes,
            "test": {"benign": int(len(benign)), "attack": int(len(attack)),
                     "attack_by_class": {classes[c]: int(quota[c]) for c in att_classes},
                     "available_by_class": {classes[c]: int(len(avail[c])) for c in att_classes},
                     "ratio": f"{len(benign)}:{len(attack)}"},
            "windows_total": int(len(wi))}
    return sets, info, log


def stage_transfer_target_only(src: str, tgt: str, cfg: dict, common: list[str], resume: bool = False) -> dict:
    """Source cleaner applied to a target-only dataset (D4): test + re-anchoring windows."""
    out = _p(cfg, "processed_dir") / "transfer" / f"{src}__to__{tgt}"
    meta_path = out / "meta.json"
    src_meta = json.loads((_p(cfg, "processed_dir") / src / "archive_meta.json").read_text())
    h = src_meta["cfg_hash"]
    if resume and _done(meta_path, h):
        print(f"[{src}->{tgt}] transfer: up to date, skipped")
        return json.loads(meta_path.read_text())
    import pyarrow.parquet as pq
    parquet = _p(cfg, "interim_dir") / tgt / "sorted.parquet"
    missing = [c for c in common if c not in pq.read_schema(parquet).names]
    if missing:
        raise RuntimeError(f"{tgt}: frozen common features missing from the target: {missing}")
    tbase = _p(cfg, "processed_dir") / tgt
    tinfo = json.loads((tbase / "index_meta.json").read_text())
    wi = WindowIndex.from_npz(np.load(tbase / "windows.npz"))
    sets = dict(np.load(tbase / "target_sets.npz"))
    meta = {"source": src, "target": tgt, "cfg_hash": h, "target_only": True, "schemes": {}}
    for scheme in cfg["splits"]:
        src_cleaner_path = _p(cfg, "processed_dir") / src / scheme / "archive" / "cleaner.json"
        cl = Cleaner.from_dict(json.loads(src_cleaner_path.read_text()))
        if cl.features != common:
            raise RuntimeError(f"{src}/{scheme} cleaner features differ from the frozen common list")
        meta["schemes"][scheme] = _write_archive(
            out / scheme, parquet, tinfo["flows"], cl, wi, np.full(len(wi), -1, np.int8), sets, {},
            {"source_cleaner_sha256": sha256_file(src_cleaner_path), "missing_features": missing,
             "note": "target-only (AMENDMENT_01 A1.4): source cleaner; nothing fitted on the target"})
    meta["peak_rss_gb_process"] = peak_rss_gb()
    meta["provenance"] = provenance(config=cfg)
    write_json(meta_path, meta)
    print(f"[{src}->{tgt}] target-only transfer package written")
    return meta


# ------------------------------------------------------------------ finalize
def _compact(wi: WindowIndex, ref: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows needed by windows `ref`; returns (row positions, start_compact for all windows; -1 if absent)."""
    diff = np.zeros(n_rows + 1, np.int32)
    np.add.at(diff, wi.start[ref], 1)
    np.add.at(diff, wi.start[ref] + wi.T, -1)
    need = np.cumsum(diff[:-1]) > 0
    rows = np.flatnonzero(need)
    pos = np.cumsum(need) - 1
    start_c = np.full(len(wi), -1, np.int64)
    start_c[ref] = pos[wi.start[ref]]
    return rows, start_c


def _referenced(sets: dict, folds: dict, extra_sets: list[dict] = ()) -> np.ndarray:
    parts = list(sets.values()) + [i for f in folds.values() for i in f.values()]
    parts += [i for d in extra_sets for i in d.values()]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def common_features(cfg: dict, ds_list: list[str]) -> list[str]:
    cls = {}
    for ds in ds_list:
        for scheme in cfg["splits"]:
            p = _p(cfg, "processed_dir") / ds / scheme / "cleaner_fit.json"
            cls[f"{ds}/{scheme}"] = Cleaner.from_dict(json.loads(p.read_text()))
    return common_feature_list(cls)


def _write_archive(out: Path, parquet: Path, n_rows: int, cl: Cleaner, wi: WindowIndex,
                   split: np.ndarray, sets: dict, folds: dict, extra: dict,
                   extra_files: dict | None = None, copy_files: dict | None = None) -> dict:
    """extra_files: {file name: {key: window index array}} saved beside sets.npz and materialised too.
    copy_files: {file name: {key: array}} saved as-is (e.g. family labels), not window indices."""
    out.mkdir(parents=True, exist_ok=True)
    extra_files = extra_files or {}
    copy_files = copy_files or {}
    ref = _referenced(sets, folds, list(extra_files.values()))
    rows, start_c = _compact(wi, ref, n_rows)
    tab = _read_rows(parquet, cl.features + ["row_id"], rows)
    flows, tdiag = cl.transform(tab)
    vlog = verify.verify_flows(flows)
    np.save(out / "flows.npy", flows)
    np.save(out / "row_id.npy", tab["row_id"].to_numpy())
    del tab
    np.savez(out / "windows.npz", **wi.to_npz_dict(), start_compact=start_c, split=split)
    np.savez(out / "sets.npz", **sets)
    _save_folds(out / "loaco.npz", folds)
    write_json(out / "cleaner.json", cl.to_dict())
    for name, arrs in {**extra_files, **copy_files}.items():
        np.savez(out / name, **arrs)
    files = {f: sha256_file(out / f) for f in ("flows.npy", "row_id.npy", "windows.npz", "sets.npz",
                                                "loaco.npz", "cleaner.json", *extra_files, *copy_files)}
    return {"rows_materialised": int(len(rows)), "windows_referenced": int(len(ref)),
            "n_features": len(cl.features), "transform": tdiag, "verification": vlog,
            "sha256": files, **extra}


def stage_finalize(ds: str, cfg: dict, common: list[str], resume: bool = False) -> dict:
    h = cfg_hash(cfg) + sha256_bytes(json.dumps(common).encode())[:8]
    base = _p(cfg, "processed_dir") / ds
    meta_path = base / "archive_meta.json"
    if resume and _done(meta_path, h):
        print(f"[{ds}] finalize: up to date, skipped")
        return json.loads(meta_path.read_text())
    parquet = _p(cfg, "interim_dir") / ds / "sorted.parquet"
    n_rows = json.loads((base / "index_meta.json").read_text())["flows"]
    wi = WindowIndex.from_npz(np.load(base / "windows.npz"))
    meta = {"dataset": ds, "cfg_hash": h, "common_features": common, "schemes": {}}
    for scheme in cfg["splits"]:
        sd = base / scheme
        cl = Cleaner.from_dict(json.loads((sd / "cleaner_fit.json").read_text())).restrict(common)
        sets = dict(np.load(sd / "sets.npz"))
        folds = load_folds(sd / "loaco.npz")
        extra_files = {}
        if (sd / "loaco_family.npz").exists():
            extra_files["loaco_family.npz"] = dict(np.load(sd / "loaco_family.npz"))
        if (sd / "natural_novelty.npz").exists():
            extra_files["natural_novelty.npz"] = dict(np.load(sd / "natural_novelty.npz"))
        copy_files = {}
        if (base / "windows_family.npz").exists():
            copy_files["windows_family.npz"] = {k: v for k, v in np.load(base / "windows_family.npz").items()}
        meta["schemes"][scheme] = _write_archive(sd / "archive", parquet, n_rows, cl, wi, np.load(sd / "split.npy"),
                                                 sets, folds, {}, extra_files, copy_files)
    meta["peak_rss_gb_process"] = peak_rss_gb()
    meta["provenance"] = provenance(config=cfg)
    write_json(meta_path, meta)
    print(f"[{ds}] finalize: archives written")
    return meta


def stage_transfer(src: str, tgt: str, cfg: dict, resume: bool = False) -> dict:
    out = _p(cfg, "processed_dir") / "transfer" / f"{src}__to__{tgt}"
    meta_path = out / "meta.json"
    src_meta = json.loads((_p(cfg, "processed_dir") / src / "archive_meta.json").read_text())
    h = src_meta["cfg_hash"]
    if resume and _done(meta_path, h):
        print(f"[{src}->{tgt}] transfer: up to date, skipped")
        return json.loads(meta_path.read_text())
    parquet = _p(cfg, "interim_dir") / tgt / "sorted.parquet"
    tbase = _p(cfg, "processed_dir") / tgt
    n_rows = json.loads((tbase / "index_meta.json").read_text())["flows"]
    wi = WindowIndex.from_npz(np.load(tbase / "windows.npz"))
    meta = {"source": src, "target": tgt, "cfg_hash": h, "schemes": {}}
    for scheme in cfg["splits"]:
        src_cleaner_path = _p(cfg, "processed_dir") / src / scheme / "archive" / "cleaner.json"
        cl = Cleaner.from_dict(json.loads(src_cleaner_path.read_text()))
        sets = dict(np.load(tbase / scheme / "sets.npz"))
        meta["schemes"][scheme] = _write_archive(
            out / scheme, parquet, n_rows, cl, wi, np.load(tbase / scheme / "split.npy"), sets, {},
            {"source_cleaner_sha256": sha256_file(src_cleaner_path),
             "note": "target rebuilt with the source cleaner; no statistic fitted on the target"})
    meta["peak_rss_gb_process"] = peak_rss_gb()
    meta["provenance"] = provenance(config=cfg)
    write_json(meta_path, meta)
    print(f"[{src}->{tgt}] transfer package written")
    return meta


# --------------------------------------------------------------- archive I/O
class Archive:
    """Read-only view of one prepared archive (in-domain or transfer)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.flows = np.load(self.path / "flows.npy", mmap_mode="r")
        z = np.load(self.path / "windows.npz")
        self.wi = WindowIndex.from_npz(z)
        self.start_c = z["start_compact"]
        self.split = z["split"]
        self.sets = dict(np.load(self.path / "sets.npz"))
        self.folds = load_folds(self.path / "loaco.npz")
        self.cleaner = json.loads((self.path / "cleaner.json").read_text())
        p = self.path / "loaco_family.npz"
        self.folds_family = load_folds(p) if p.exists() else {}
        p = self.path / "natural_novelty.npz"
        self.natural_novelty = dict(np.load(p)) if p.exists() else {}
        p = self.path / "windows_family.npz"
        self.family = dict(np.load(p)) if p.exists() else None

    @property
    def n_features(self) -> int:
        return self.flows.shape[1]

    def windows(self, idx: np.ndarray, labelling: str = "file") -> dict:
        """labelling='family' returns family class codes (D1) in y_cls."""
        s = self.start_c[idx]
        if (s < 0).any():
            raise KeyError("window not materialised in this archive")
        X = np.asarray(self.flows[s[:, None] + np.arange(self.wi.T)[None, :]])
        if labelling == "family":
            if self.family is None:
                raise KeyError("no family labelling in this archive")
            y_cls, pur = self.family["y_cls"][idx], self.family["purity"][idx]
        else:
            y_cls, pur = self.wi.y_cls[idx], self.wi.purity[idx]
        return {"X": X, "y_bin": self.wi.y_bin[idx].astype(np.int64),
                "y_cls": y_cls.astype(np.int64), "purity": pur, "idx": idx}
