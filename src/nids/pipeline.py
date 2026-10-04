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
from .windowing import WindowIndex, class_codes, dedupe, hour_groups, make_windows, sort_by_time


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


def load_folds(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for k in z.files:
        c, part = k.rsplit("/", 1)
        out.setdefault(c, {})[part] = z[k]
    return out


# --------------------------------------------------------------------- index
def stage_index(ds: str, cfg: dict, df: pd.DataFrame | None = None, resume: bool = False,
                raw_sha256: str | None = None, source_file: str | None = None) -> dict:
    h = cfg_hash(cfg)
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
        df = io.read_raw(raw)
    aud = audit_mod.audit_frame(df, ds, source_file or "in-memory", raw_sha256 or "n/a")
    write_json(_p(cfg, "audit_dir") / f"{ds}.json", aud)
    if aud["schema_check"]["missing"]:
        print(f"[{ds}] WARNING columns missing vs expected NF-v3 schema: {aud['schema_check']['missing']}")
    if schema.TIME_START not in df:
        raise RuntimeError(f"{ds}: no {schema.TIME_START}; cannot sort or group")

    df = sort_by_time(df)
    group = hour_groups(df[schema.TIME_START].to_numpy(np.float64), cfg["window"]["block_ms"])
    codes, classes = class_codes(df[schema.LABEL_MULTI])
    drops = schema.drop_map(cfg["keep_ttl"])
    hash_cols = [c for c in df.columns if c not in drops and c != "row_id"
                 and pd.api.types.is_numeric_dtype(df[c])]
    raw_feat = df[hash_cols].to_numpy(np.float32)
    wi, wstats = make_windows(group, codes, raw_feat, cfg["window"]["T"], cfg["window"]["stride"])
    del raw_feat
    wi, dstats = dedupe(wi)
    n_benign_windows = int((wi.y_bin == 0).sum())

    interim = _p(cfg, "interim_dir") / ds
    interim.mkdir(parents=True, exist_ok=True)
    df.drop(columns=[c for c in (schema.LABEL_MULTI, "IPV4_SRC_ADDR", "IPV4_DST_ADDR") if c in df],
            inplace=True)
    df["group"] = group
    df["cls"] = codes
    df.to_parquet(interim / "sorted.parquet", index=False)

    info = {"dataset": ds, "name": dcfg["name"], "cfg_hash": h, "classes": classes,
            "flows": int(len(df)), "hash_columns": hash_cols, "windowing": wstats, "dedupe": dstats,
            "benign_windows_after_dedupe": n_benign_windows, "schemes": {}}
    min_benign = dcfg.get("loaco_min_benign_windows")
    loaco_allowed = "loaco" in dcfg["role"] or (min_benign is not None and n_benign_windows >= min_benign)
    info["loaco_allowed"] = bool(loaco_allowed)

    np.savez(out / "windows.npz", **wi.to_npz_dict())
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
                                   "cleaner_kept": cl.features, "cleaner_dropped": cl.dropped}
    info["provenance"] = provenance(config=cfg, data_sha256={ds: raw_sha256})
    write_json(meta_path, info)
    print(f"[{ds}] index: {wstats['windows']} windows, {dstats['duplicates_removed']} duplicates removed")
    return info


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


def _referenced(sets: dict, folds: dict) -> np.ndarray:
    parts = list(sets.values()) + [i for f in folds.values() for i in f.values()]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)


def common_features(cfg: dict, ds_list: list[str]) -> list[str]:
    cls = {}
    for ds in ds_list:
        for scheme in cfg["splits"]:
            p = _p(cfg, "processed_dir") / ds / scheme / "cleaner_fit.json"
            cls[f"{ds}/{scheme}"] = Cleaner.from_dict(json.loads(p.read_text()))
    return common_feature_list(cls)


def _write_archive(out: Path, flows_tab: pd.DataFrame, cl: Cleaner, wi: WindowIndex,
                   split: np.ndarray, sets: dict, folds: dict, extra: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ref = _referenced(sets, folds)
    rows, start_c = _compact(wi, ref, len(flows_tab))
    flows, tdiag = cl.transform(flows_tab.iloc[rows])
    vlog = verify.verify_flows(flows)
    np.save(out / "flows.npy", flows)
    np.save(out / "row_id.npy", flows_tab["row_id"].to_numpy()[rows])
    np.savez(out / "windows.npz", **wi.to_npz_dict(), start_compact=start_c, split=split)
    np.savez(out / "sets.npz", **sets)
    _save_folds(out / "loaco.npz", folds)
    write_json(out / "cleaner.json", cl.to_dict())
    files = {f: sha256_file(out / f) for f in ("flows.npy", "row_id.npy", "windows.npz", "sets.npz",
                                                "loaco.npz", "cleaner.json")}
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
    tab = pd.read_parquet(_p(cfg, "interim_dir") / ds / "sorted.parquet")
    wi = WindowIndex.from_npz(np.load(base / "windows.npz"))
    meta = {"dataset": ds, "cfg_hash": h, "common_features": common, "schemes": {}}
    for scheme in cfg["splits"]:
        sd = base / scheme
        cl = Cleaner.from_dict(json.loads((sd / "cleaner_fit.json").read_text())).restrict(common)
        sets = dict(np.load(sd / "sets.npz"))
        folds = load_folds(sd / "loaco.npz")
        meta["schemes"][scheme] = _write_archive(sd / "archive", tab, cl, wi, np.load(sd / "split.npy"),
                                                 sets, folds, {})
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
    tab = pd.read_parquet(_p(cfg, "interim_dir") / tgt / "sorted.parquet")
    tbase = _p(cfg, "processed_dir") / tgt
    wi = WindowIndex.from_npz(np.load(tbase / "windows.npz"))
    meta = {"source": src, "target": tgt, "cfg_hash": h, "schemes": {}}
    for scheme in cfg["splits"]:
        src_cleaner_path = _p(cfg, "processed_dir") / src / scheme / "archive" / "cleaner.json"
        cl = Cleaner.from_dict(json.loads(src_cleaner_path.read_text()))
        sets = dict(np.load(tbase / scheme / "sets.npz"))
        meta["schemes"][scheme] = _write_archive(
            out / scheme, tab, cl, wi, np.load(tbase / scheme / "split.npy"), sets, {},
            {"source_cleaner_sha256": sha256_file(src_cleaner_path),
             "note": "target rebuilt with the source cleaner; no statistic fitted on the target"})
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

    @property
    def n_features(self) -> int:
        return self.flows.shape[1]

    def windows(self, idx: np.ndarray) -> dict:
        s = self.start_c[idx]
        if (s < 0).any():
            raise KeyError("window not materialised in this archive")
        X = np.asarray(self.flows[s[:, None] + np.arange(self.wi.T)[None, :]])
        return {"X": X, "y_bin": self.wi.y_bin[idx].astype(np.int64),
                "y_cls": self.wi.y_cls[idx].astype(np.int64), "purity": self.wi.purity[idx], "idx": idx}
