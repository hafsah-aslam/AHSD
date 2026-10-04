"""Group-level splits, balancing, natural-prior test sets, and LOACO folds.

Splits assign whole 1-hour groups, so no flow and no window crosses a split.
Fractions are measured in windows (groups differ a lot in volume).
"""

from __future__ import annotations

import numpy as np

from .windowing import WindowIndex

TRAIN, GAP, VAL, TEST = 0, 1, 2, 3
SPLIT_NAMES = {TRAIN: "train", GAP: "gap", VAL: "val", TEST: "test"}


def _group_sizes(wi: WindowIndex) -> tuple[np.ndarray, np.ndarray]:
    g, n = np.unique(wi.group, return_counts=True)
    return g, n


def _assign_cumulative(groups_in_order: np.ndarray, sizes: np.ndarray, gap: bool,
                       frac=(0.6, 0.2, 0.2)) -> dict[int, int]:
    total = sizes.sum()
    out, cum, i = {}, 0, 0
    G = len(groups_in_order)
    while i < G and cum < frac[0] * total:
        out[int(groups_in_order[i])] = TRAIN
        cum += sizes[i]
        i += 1
    if gap and i < G:
        out[int(groups_in_order[i])] = GAP
        cum += sizes[i]
        i += 1
    while i < G and cum < (frac[0] + frac[1]) * total:
        out[int(groups_in_order[i])] = VAL
        cum += sizes[i]
        i += 1
    while i < G:
        out[int(groups_in_order[i])] = TEST
        i += 1
    return out


def assign_split(wi: WindowIndex, scheme: str, seed: int = 0) -> tuple[np.ndarray, dict]:
    g, n = _group_sizes(wi)
    if scheme == "temporal_gap":
        order = np.argsort(g)  # group ids are already in time order
        gmap = _assign_cumulative(g[order], n[order], gap=True)
    elif scheme == "grouped_random":
        order = np.random.default_rng(seed).permutation(len(g))
        gmap = _assign_cumulative(g[order], n[order], gap=False)
    else:
        raise ValueError(scheme)
    lut = np.full(int(wi.group.max()) + 1 if len(wi) else 1, -1, dtype=np.int8)
    for k, v in gmap.items():
        lut[k] = v
    split = lut[wi.group]
    info = {"scheme": scheme, "seed": seed,
            "groups": {SPLIT_NAMES[s]: sorted(k for k, v in gmap.items() if v == s) for s in SPLIT_NAMES},
            "windows": {SPLIT_NAMES[s]: int((split == s).sum()) for s in SPLIT_NAMES}}
    for s in (TRAIN, VAL, TEST):
        if (split == s).sum() == 0:
            raise RuntimeError(f"{scheme}: split {SPLIT_NAMES[s]} is empty (too few groups)")
    return split, info


def composition(wi: WindowIndex, split: np.ndarray, classes: list[str]) -> dict:
    """Per-class window counts (majority label) per split, plus classes absent from train."""
    table = {}
    for s in (TRAIN, VAL, TEST):
        m = split == s
        table[SPLIT_NAMES[s]] = {classes[c]: int((wi.y_cls[m] == c).sum()) for c in range(len(classes))}
    absent = [c for c in classes[1:] if table["test"][c] > 0 and table["train"][c] == 0]
    absent_val = [c for c in classes[1:] if table["val"][c] > 0 and table["train"][c] == 0]
    return {"windows_by_majority_class": table,
            "test_classes_absent_from_train": absent,
            "val_classes_absent_from_train": absent_val}


def balance(idx_benign: np.ndarray, idx_attack: np.ndarray, ratio_benign_per_attack: float,
            cap: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Exact ratio benign:attack = r:1 without replacement, at most `cap` windows."""
    r = ratio_benign_per_attack
    n_a = int(min(len(idx_attack), np.floor(cap / (1 + r)), np.floor(len(idx_benign) / r)))
    n_b = int(round(n_a * r))
    sel_a = rng.choice(idx_attack, n_a, replace=False) if n_a else np.zeros(0, np.int64)
    sel_b = rng.choice(idx_benign, n_b, replace=False) if n_b else np.zeros(0, np.int64)
    sel = np.sort(np.r_[sel_b, sel_a]).astype(np.int64)
    return sel, {"benign": n_b, "attack": n_a, "available_benign": int(len(idx_benign)),
                 "available_attack": int(len(idx_attack)), "target_ratio": r, "cap": cap}


def natural_prior(idx: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= cap:
        return np.sort(idx)
    return np.sort(rng.choice(idx, cap, replace=False))


def make_sets(wi: WindowIndex, split: np.ndarray, cfg: dict, seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    sets, info = {}, {}
    for name, s, r, cap in (("train", TRAIN, cfg["train_ratio"], cfg["train_cap"]),
                            ("val", VAL, cfg["eval_ratio"], cfg["val_cap"]),
                            ("test", TEST, cfg["eval_ratio"], cfg["test_cap"])):
        m = split == s
        sets[name], info[name] = balance(np.flatnonzero(m & (wi.y_bin == 0)),
                                         np.flatnonzero(m & (wi.y_bin == 1)), r, cap, rng)
    sets["test_natural"] = natural_prior(np.flatnonzero(split == TEST), cfg["natural_cap"], rng)
    info["test_natural"] = {"n": int(len(sets["test_natural"])),
                            "attack_fraction": float(wi.y_bin[sets["test_natural"]].mean()) if len(sets["test_natural"]) else None}
    return sets, info


def loaco_folds(wi: WindowIndex, split: np.ndarray, classes: list[str], cfg: dict,
                seed: int = 0) -> tuple[dict, dict]:
    """Leave-one-attack-class-out folds.

    Held-out class c: train/val exclude every window that contains any flow of
    class c (not only majority-c windows). Test = benign + windows whose
    majority class is c; windows that merely contain c are excluded from the
    fold test set.
    """
    rng = np.random.default_rng(seed)
    folds, info = {}, {}
    for c in range(1, len(classes)):
        name = classes[c]
        n_train_c = int(((split == TRAIN) & (wi.y_cls == c)).sum())
        rec = {"class_index": c, "train_windows_majority": n_train_c}
        if n_train_c < cfg["loaco_min_train"]:
            rec.update(eligible=False, reason=f"< {cfg['loaco_min_train']} training windows")
            info[name] = rec
            continue
        has_c = (wi.class_mask & (np.uint64(1) << np.uint64(c))) != 0
        f = {}
        for nm, s, r, cap in (("train", TRAIN, cfg["train_ratio"], cfg["train_cap"]),
                              ("val", VAL, cfg["eval_ratio"], cfg["val_cap"])):
            m = (split == s) & ~has_c
            f[nm], rec[nm] = balance(np.flatnonzero(m & (wi.y_bin == 0)),
                                     np.flatnonzero(m & (wi.y_bin == 1)), r, cap, rng)
        m = split == TEST
        f["test"], rec["test"] = balance(np.flatnonzero(m & (wi.y_bin == 0)),
                                         np.flatnonzero(m & (wi.y_cls == c)),
                                         cfg["eval_ratio"], cfg["test_cap"], rng)
        rec["eligible"] = True
        rec["feasible"] = rec["test"]["attack"] > 0 and rec["train"]["attack"] > 0
        if not rec["feasible"]:
            rec["reason"] = "held-out class absent from the test split" if rec["test"]["attack"] == 0 \
                else "no remaining attack windows in train"
        folds[name] = f
        info[name] = rec
    return folds, info
