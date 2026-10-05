"""Data preparation: cleaner, windows, splits, LOACO, verification, transfer."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from nids import pipeline, schema, splits, verify
from nids.preprocess import Cleaner, common_feature_list
from nids.provenance import ROOT
from nids.synthetic import generate
from nids.windowing import WindowIndex, class_codes, dedupe, hour_groups, make_windows, sort_by_time


@pytest.fixture(scope="module")
def df():
    return generate(20_000, hours=10, seed=3)


def test_expected_schema_has_53_features():
    assert len(schema.EXPECTED_FEATURES) == 53
    assert len(set(schema.EXPECTED_FEATURES)) == 53


def test_cleaner_drops_and_never_uses_ttl_or_ids(df):
    cl = Cleaner().fit(df)
    for c in list(schema.DROP_IDENTIFIERS) + list(schema.DROP_TTL) + [schema.TIME_START, schema.LABEL_BIN]:
        assert c not in cl.features
    assert cl.dropped["DURATION_IN"].startswith("abs(corr)")
    assert "IN_BYTES" in cl.log1p
    X, diag = cl.transform(df)
    assert np.isfinite(X).all() and X.dtype == np.float32
    assert diag["nonfinite_imputed"] > 0  # injected inf/NaN were imputed
    assert np.allclose(X.mean(0), 0, atol=1e-4) and np.allclose(X.std(0), 1, atol=1e-3)


def test_ttl_ablation_keeps_ttl(df):
    assert "MIN_TTL" in Cleaner(keep_ttl=True).fit(df).features


def test_cleaner_statistics_come_from_train_only(df):
    tr, te = df.iloc[:10_000], df.iloc[10_000:].copy()
    cl = Cleaner().fit(tr)
    te["IN_BYTES"] *= 1000  # target shift must not move the fitted statistics
    before = json.dumps(cl.to_dict(), sort_keys=True)
    cl.transform(te)
    assert json.dumps(cl.to_dict(), sort_keys=True) == before
    assert cl.fit_rows == 10_000


def test_cleaner_roundtrip_and_restrict(df):
    cl = Cleaner().fit(df)
    cl2 = Cleaner.from_dict(json.loads(json.dumps(cl.to_dict())))
    assert np.array_equal(cl.transform(df)[0], cl2.transform(df)[0])
    common = cl.features[:5]
    cl2.restrict(common)
    assert cl2.features == common and cl2.transform(df)[0].shape[1] == 5
    assert common_feature_list({"a": cl, "b": cl2}) == common


def test_windows_never_cross_groups_and_labels(df):
    d = sort_by_time(df)
    g = hour_groups(d[schema.TIME_START].to_numpy())
    codes, classes = class_codes(d[schema.LABEL_MULTI])
    assert classes[0] == schema.BENIGN
    wi, st = make_windows(g, codes, np.zeros((len(d), 1)), T=32, stride=16)
    assert np.all(g[wi.start] == g[wi.start + 31])
    for i in np.random.default_rng(0).choice(len(wi), 200, replace=False):
        c = codes[wi.start[i]:wi.start[i] + 32]
        assert wi.y_bin[i] == int((c > 0).any())
        if wi.y_bin[i]:
            assert wi.y_cls[i] == np.bincount(c[c > 0]).argmax()
        assert wi.purity[i] == pytest.approx((c == wi.y_cls[i]).mean())
        for k in range(len(classes)):
            assert bool(int(wi.class_mask[i]) >> k & 1) == bool((c == k).any())


def test_dedupe_removes_identical_windows():
    g = np.zeros(64, np.int32)
    codes = np.zeros(64, np.int16)
    feats = np.tile(np.arange(16, dtype=np.float32)[:, None], (4, 1))  # period 16 -> repeated windows
    wi, _ = make_windows(g, codes, feats, T=16, stride=16)
    out, st = dedupe(wi)
    assert len(wi) == 4 and len(out) == 1 and st["duplicates_removed"] == 3
    assert out.start[0] == 0  # earliest kept


def _wi(df):
    d = sort_by_time(df)
    g = hour_groups(d[schema.TIME_START].to_numpy())
    codes, classes = class_codes(d[schema.LABEL_MULTI])
    feats = d[["IN_BYTES", "OUT_BYTES", "IN_PKTS"]].to_numpy(np.float32)
    wi, _ = make_windows(g, codes, feats)
    wi, _ = dedupe(wi)
    return d, wi, classes


CFG = dict(train_ratio=1, eval_ratio=5, train_cap=1000, val_cap=300, test_cap=300, natural_cap=500, loaco_min_train=20)


@pytest.mark.parametrize("scheme", ["temporal_gap", "grouped_random"])
def test_splits_sets_and_loaco_verify(df, scheme):
    d, wi, classes = _wi(df)
    split, info = splits.assign_split(wi, scheme, seed=0)
    if scheme == "temporal_gap":
        tr, va, te = (wi.group[split == s] for s in (splits.TRAIN, splits.VAL, splits.TEST))
        assert tr.max() < va.min() and va.max() < te.min()
        assert len(info["groups"]["gap"]) == 1
    verify.verify_splits(wi, split, d["row_id"].to_numpy())
    sets, sinfo = splits.make_sets(wi, split, CFG, seed=0)
    verify.verify_sets(wi, split, sets, sinfo, CFG)
    folds, finfo = splits.loaco_folds(wi, split, classes, CFG, seed=0)
    assert folds
    verify.verify_loaco(wi, split, folds, classes)


def test_verification_fails_closed(df):
    d, wi, classes = _wi(df)
    split, _ = splits.assign_split(wi, "temporal_gap")
    bad = split.copy()
    i = np.flatnonzero(split == splits.TEST)[0]
    bad[np.flatnonzero(wi.group == wi.group[i])[0]] = splits.TRAIN  # leak one test window into train
    with pytest.raises(verify.VerificationError):
        verify.verify_splits(wi, bad, d["row_id"].to_numpy())
    sets, sinfo = splits.make_sets(wi, split, CFG)
    sets["train"] = np.r_[sets["train"], sets["test"][:1]]
    with pytest.raises(verify.VerificationError):
        verify.verify_sets(wi, split, sets, sinfo, CFG)


def test_balance_exact_ratio_without_replacement():
    rng = np.random.default_rng(0)
    sel, info = splits.balance(np.arange(100), np.arange(100, 130), 5, 3000, rng)
    assert info["attack"] == 20 and info["benign"] == 100 and len(np.unique(sel)) == 120


def test_end_to_end_pipeline_and_transfer(tmp_path):
    cfg = yaml.safe_load(open(ROOT / "configs/data.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg.update(raw_dir=str(tmp_path / "raw"), interim_dir=str(tmp_path / "interim"),
               processed_dir=str(tmp_path / "proc"), audit_dir=str(tmp_path / "audit"), corr_sample_rows=5000)
    cfg["balance"].update(CFG)
    for d in cfg["datasets"].values():  # real-data-only options (AMENDMENT_01) are tested separately
        for k in ("class_map", "natural_novelty", "target_only"):
            d.pop(k, None)
    frames = {"D1": generate(20_000, hours=10, seed=1), "D2": generate(15_000, hours=9, seed=2, shift=0.5)}
    for ds, f in frames.items():
        pipeline.stage_index(ds, cfg, df=f)
    common = pipeline.common_features(cfg, list(frames))
    for ds in frames:
        pipeline.stage_finalize(ds, cfg, common)
    pipeline.stage_transfer("D1", "D2", cfg)
    assert json.loads((tmp_path / "audit" / "D1.json").read_text())["rows"] == 20_000

    arc = pipeline.Archive(tmp_path / "proc" / "D1" / "temporal_gap" / "archive")
    w = arc.windows(arc.sets["train"])
    assert w["X"].shape[1:] == (32, len(common)) and set(np.unique(w["y_bin"])) == {0, 1}
    raw_sorted = pd.read_parquet(tmp_path / "interim" / "D1" / "sorted.parquet")
    cl = Cleaner.from_dict(arc.cleaner)
    i = arc.sets["test"][0]
    s = arc.wi.start[i]
    expect, _ = cl.transform(raw_sorted.iloc[s:s + 32])
    assert np.array_equal(arc.windows(np.array([i]))["X"][0], expect)

    # transfer: target windows transformed with the SOURCE cleaner, bit for bit
    tarc = pipeline.Archive(tmp_path / "proc" / "transfer" / "D1__to__D2" / "temporal_gap")
    assert tarc.cleaner == arc.cleaner
    t_sorted = pd.read_parquet(tmp_path / "interim" / "D2" / "sorted.parquet")
    j = tarc.sets["test"][0]
    s = tarc.wi.start[j]
    assert np.array_equal(tarc.windows(np.array([j]))["X"][0], cl.transform(t_sorted.iloc[s:s + 32])[0])
    own = json.loads((tmp_path / "proc" / "D2" / "temporal_gap" / "archive" / "cleaner.json").read_text())
    assert own["mean"] != arc.cleaner["mean"]


def test_chunked_reader_and_inplace_sort_match_reference(tmp_path):
    from nids import io
    df = generate(5_000, hours=3, seed=9)
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)  # unsorted, like UNSW-v3
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    ref = pd.read_csv(p)
    a = io.read_raw(p, chunksize=700)
    b = io.read_raw(p, chunksize=700, n_rows_hint=5_000)
    for got in (a, b):
        assert list(got.columns) == list(ref.columns)
        assert np.allclose(got["IN_BYTES"].to_numpy(), ref["IN_BYTES"].to_numpy(dtype=np.float32), equal_nan=True)
        assert np.array_equal(got[schema.TIME_START].to_numpy(), ref[schema.TIME_START].to_numpy())
        assert (got[schema.LABEL_MULTI].astype(str).to_numpy() == ref[schema.LABEL_MULTI].astype(str).to_numpy()).all()
    s = sort_by_time(b)
    order = np.argsort(ref[schema.TIME_START].to_numpy(), kind="stable")
    assert np.array_equal(s["row_id"].to_numpy(), order)
    assert (s[schema.LABEL_MULTI].astype(str).to_numpy() == ref[schema.LABEL_MULTI].astype(str).to_numpy()[order]).all()
    assert np.allclose(s["IN_BYTES"].to_numpy(), ref["IN_BYTES"].to_numpy(dtype=np.float32)[order])


def _amend_cfg(tmp_path):
    cfg = copy.deepcopy(yaml.safe_load(open(ROOT / "configs/data.yaml")))
    cfg.update(raw_dir=str(tmp_path / "raw"), interim_dir=str(tmp_path / "interim"),
               processed_dir=str(tmp_path / "proc"), audit_dir=str(tmp_path / "audit"), corr_sample_rows=5000)
    cfg["balance"].update(CFG)
    for d in cfg["datasets"].values():
        for k in ("class_map", "natural_novelty", "target_only"):
            d.pop(k, None)
    return cfg


def test_family_labelling_natural_novelty_and_archive(tmp_path):
    """AMENDMENT_01 A1.2/A1.3 on synthetic data: family folds and natural-novelty sets."""
    cmap = tmp_path / "map.yaml"
    cmap.write_text("families:\n  Benign: [Benign]\n  Volumetric: [dos, slow]\n  Beacon: [beacon]\n  Scan: [scan, late]\n")
    df = generate(30_000, hours=12, seed=4)
    lab = df[schema.LABEL_MULTI].astype(str).to_numpy()
    late = (lab == "scan") & (np.arange(len(df)) > int(0.85 * len(df)))
    lab[(lab == "scan") & ~late] = "dos"  # 'late' appears only at the end -> test-only under temporal_gap
    lab[late] = "late"
    df[schema.LABEL_MULTI] = pd.Categorical(lab)
    cfg = _amend_cfg(tmp_path)
    cfg["datasets"]["D1"]["class_map"] = str(cmap)
    cfg["datasets"]["D1"]["natural_novelty"] = ["late"]
    info = pipeline.stage_index("D1", cfg, df=df)
    assert info["family"]["classes"] == ["Benign", "Beacon", "Scan", "Volumetric"]
    tg = info["schemes"]["temporal_gap"]
    assert tg["natural_novelty"]["classes"] == ["late"] and all(v["ok"] for v in tg["verification"])
    assert "loaco_family" in tg
    pipeline.stage_finalize("D1", cfg, pipeline.common_features(cfg, ["D1"]))
    arc = pipeline.Archive(tmp_path / "proc" / "D1" / "temporal_gap" / "archive")
    t = arc.windows(arc.natural_novelty["test/late"])
    assert set(np.unique(t["y_cls"])) <= {0, info["classes"].index("late")}
    fam = arc.windows(arc.sets["train"], labelling="family")
    assert fam["y_cls"].max() <= 3 and np.array_equal(fam["y_bin"], arc.windows(arc.sets["train"])["y_bin"])
    some = next(iter(arc.folds_family.values()))
    arc.windows(some["test"], labelling="family")  # family fold windows are materialised


def test_natural_novelty_mismatch_fails_closed(tmp_path):
    cfg = _amend_cfg(tmp_path)
    cfg["datasets"]["D1"]["natural_novelty"] = ["dos"]  # present in train -> must refuse
    with pytest.raises(RuntimeError, match="natural_novelty"):
        pipeline.stage_index("D1", cfg, df=generate(20_000, hours=10, seed=1))


def test_target_only_sets_are_group_disjoint(tmp_path):
    cfg = _amend_cfg(tmp_path)
    cfg["datasets"]["D4"]["target_only"] = {"anchor_groups_first": 3, "anchor_sizes": [10, 40],
                                             "test_ratio": 5, "attack_allocation": "equal"}
    info = pipeline.stage_index("D4", cfg, df=generate(20_000, hours=10, seed=6))
    t = info["target_only"]
    assert all(v["ok"] for v in info["verification"])
    z = np.load(tmp_path / "proc" / "D4" / "target_sets.npz")
    wi = WindowIndex.from_npz(np.load(tmp_path / "proc" / "D4" / "windows.npz"))
    assert not set(wi.group[z["anchor_pool"]]) & set(wi.group[z["test"]])
    assert set(z["anchor_10"]) <= set(z["anchor_40"])
    assert t["test"]["attack"] == t["test"]["benign"] // 5
    q = list(t["test"]["attack_by_class"].values())
    assert max(q) - min(q) <= 1 or min(t["test"]["available_by_class"].values()) < max(q)
