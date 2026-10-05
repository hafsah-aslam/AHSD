"""Fail-closed validators. Any failure raises `VerificationError`."""

from __future__ import annotations

import numpy as np

from .splits import TEST, TRAIN, VAL, SPLIT_NAMES
from .windowing import WindowIndex


class VerificationError(AssertionError):
    pass


def _check(cond: bool, msg: str, log: list):
    log.append({"check": msg, "ok": bool(cond)})
    if not cond:
        raise VerificationError(msg)


def window_rows(wi: WindowIndex, idx: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(0, np.int64)
    return np.unique((wi.start[idx][:, None] + np.arange(wi.T)[None, :]).ravel())


def verify_splits(wi: WindowIndex, split: np.ndarray, row_id: np.ndarray) -> list:
    log = []
    _check(len(np.unique(wi.sha)) == len(wi.sha), "window SHA-256 globally unique after dedupe", log)
    used = {s: np.flatnonzero(split == s) for s in (TRAIN, VAL, TEST)}
    for a, b in ((TRAIN, VAL), (TRAIN, TEST), (VAL, TEST)):
        na, nb = SPLIT_NAMES[a], SPLIT_NAMES[b]
        _check(len(np.intersect1d(wi.sha[used[a]], wi.sha[used[b]])) == 0, f"no shared window hash {na}/{nb}", log)
        _check(len(np.intersect1d(wi.group[used[a]], wi.group[used[b]])) == 0, f"group-disjoint {na}/{nb}", log)
        ra = row_id[window_rows(wi, used[a])]
        rb = row_id[window_rows(wi, used[b])]
        _check(len(np.intersect1d(ra, rb)) == 0, f"row-ID-disjoint {na}/{nb}", log)
    return log


def verify_sets(wi: WindowIndex, split: np.ndarray, sets: dict, info: dict, cfg: dict) -> list:
    log = []
    want = {"train": TRAIN, "val": VAL, "test": TEST, "test_natural": TEST}
    for name, idx in sets.items():
        _check(len(np.unique(idx)) == len(idx), f"{name}: no window drawn twice", log)
        _check(bool(np.all(split[idx] == want[name])), f"{name}: all windows from split {SPLIT_NAMES[want[name]]}", log)
    for name, ratio, cap in (("train", cfg["train_ratio"], cfg["train_cap"]),
                             ("val", cfg["eval_ratio"], cfg["val_cap"]),
                             ("test", cfg["eval_ratio"], cfg["test_cap"])):
        idx = sets[name]
        nb, na = int((wi.y_bin[idx] == 0).sum()), int((wi.y_bin[idx] == 1).sum())
        _check(len(idx) <= cap, f"{name}: size {len(idx)} <= cap {cap}", log)
        _check(nb == round(na * ratio), f"{name}: benign:attack = {ratio}:1 exactly ({nb}:{na})", log)
    return log


def verify_loaco(wi: WindowIndex, split: np.ndarray, folds: dict, classes: list[str]) -> list:
    log = []
    for name, f in folds.items():
        c = classes.index(name)
        bit = np.uint64(1) << np.uint64(c)
        for part in ("train", "val"):
            _check(bool(np.all((wi.class_mask[f[part]] & bit) == 0)), f"LOACO {name}: {part} has no flow of the held-out class", log)
        t = f["test"]
        _check(bool(np.all(split[t] == TEST)), f"LOACO {name}: test from test split", log)
        _check(bool(np.all((wi.y_cls[t] == 0) | (wi.y_cls[t] == c))), f"LOACO {name}: test is benign + held-out only", log)
        _check(bool(np.all(split[f["train"]] == TRAIN)), f"LOACO {name}: train from train split", log)
    return log


def verify_flows(flows: np.ndarray) -> list:
    log = []
    _check(bool(np.isfinite(flows).all()), "cleaned flows finite", log)
    _check(flows.dtype == np.float32, "cleaned flows float32", log)
    return log


def verify_natural_novelty(wi: WindowIndex, split: np.ndarray, sets: dict, classes: list[str],
                           nn_classes: list[str]) -> list:
    log = []
    bits = np.uint64(0)
    for c in nn_classes:
        bits |= np.uint64(1) << np.uint64(classes.index(c))
    for part, s in (("train", TRAIN), ("val", VAL)):
        idx = sets[part]
        _check(bool(np.all(split[idx] == s)), f"natural_novelty: {part} from split {SPLIT_NAMES[s]}", log)
        _check(bool(np.all((wi.class_mask[idx] & bits) == 0)), f"natural_novelty: {part} has no flow of any NN class", log)
    tr_classes = set(np.unique(wi.y_cls[split == TRAIN]).tolist())
    for c in nn_classes:
        k = classes.index(c)
        _check(k not in tr_classes, f"natural_novelty {c}: absent from the train split", log)
        t = sets[f"test/{c}"]
        _check(bool(np.all(split[t] == TEST)), f"natural_novelty {c}: test from test split", log)
        _check(bool(np.all((wi.y_cls[t] == 0) | (wi.y_cls[t] == k))), f"natural_novelty {c}: test is benign + {c} only", log)
        _check(int((wi.y_cls[t] == k).sum()) > 0, f"natural_novelty {c}: test holds windows of the class", log)
    return log
