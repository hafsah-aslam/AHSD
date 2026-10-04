"""Time ordering, 1-hour grouping, and fixed-length flow windows.

Windows never cross a group. A window is an attack window if it holds at least
one attack flow; its multiclass label is the majority class among its attack
flows (ties -> lowest class index, counted). Purity is the fraction of the
window's flows that carry the window's label.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema

HOUR_MS = 3_600_000.0


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Stable sort by flow start; the raw row number is kept as `row_id`."""
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    t = df[schema.TIME_START].to_numpy()
    if np.all(t[1:] >= t[:-1]):
        return df.reset_index(drop=True)
    order = np.argsort(t, kind="stable")  # stable: ties keep raw row order
    # reorder column by column: peak memory is one extra column, not a full frame copy
    for c in list(df.columns):
        col = df[c]
        df[c] = col.array.take(order) if isinstance(col.dtype, pd.CategoricalDtype) else col.to_numpy()[order]
    return df.reset_index(drop=True)


def hour_groups(t_ms: np.ndarray, block_ms: float = HOUR_MS) -> np.ndarray:
    """Contiguous blocks of `block_ms`; empty blocks are skipped, ids are 0..G-1 in time order."""
    if np.isnan(t_ms).any():
        raise ValueError("NaN flow start timestamps; cannot group")
    blk = np.floor((t_ms - t_ms.min()) / block_ms).astype(np.int64)
    _, gid = np.unique(blk, return_inverse=True)
    return gid.astype(np.int32)


def class_codes(attack: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Benign is code 0; attack classes follow in sorted order."""
    cat = attack.astype(str).astype("category") if not isinstance(attack.dtype, pd.CategoricalDtype) else attack
    cats = [str(c) for c in cat.cat.categories]
    is_b = [c.lower() == schema.BENIGN.lower() for c in cats]
    names = sorted(c for c, b in zip(cats, is_b) if not b)
    classes = [schema.BENIGN] + names
    lut = np.array([0 if b else classes.index(c) for c, b in zip(cats, is_b)], dtype=np.int16)
    codes = cat.cat.codes.to_numpy()
    if (codes < 0).any():
        raise ValueError("missing attack labels")
    return lut[codes], classes


@dataclass
class WindowIndex:
    start: np.ndarray        # int64 position in the time-sorted flow array
    group: np.ndarray        # int32
    y_bin: np.ndarray        # int8
    y_cls: np.ndarray        # int16 (0 = benign)
    purity: np.ndarray       # float32
    class_mask: np.ndarray   # uint64 bit c set if any flow of class c is in the window
    n_attack: np.ndarray     # int16 attack flows in the window
    sha: np.ndarray          # S32 raw-feature hash
    T: int
    stride: int

    def __len__(self):
        return len(self.start)

    def subset(self, idx) -> "WindowIndex":
        f = {k: getattr(self, k)[idx] for k in ("start", "group", "y_bin", "y_cls", "purity",
                                                 "class_mask", "n_attack", "sha")}
        return WindowIndex(**f, T=self.T, stride=self.stride)

    def to_npz_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("start", "group", "y_bin", "y_cls", "purity",
                                            "class_mask", "n_attack", "sha")}
        d["T"] = np.array(self.T)
        d["stride"] = np.array(self.stride)
        return d

    @classmethod
    def from_npz(cls, z) -> "WindowIndex":
        return cls(**{k: z[k] for k in ("start", "group", "y_bin", "y_cls", "purity",
                                          "class_mask", "n_attack", "sha")},
                   T=int(z["T"]), stride=int(z["stride"]))


def make_windows(group: np.ndarray, codes: np.ndarray, raw_features,
                 T: int = 32, stride: int = 16) -> tuple[WindowIndex, dict]:
    """Build every window. Raw features are hashed for global dedupe.

    `raw_features` is an (n, F) array, or a callable (a, b) -> float32 array of
    rows a:b, so a 27M-row dataset is hashed one group at a time.
    """
    n = len(group)
    rows_fn = raw_features if callable(raw_features) else (lambda a, b: raw_features[a:b])
    if len(codes) != n or (not callable(raw_features) and len(raw_features) != n):
        raise ValueError("length mismatch")
    if np.any(np.diff(group) < 0):
        raise ValueError("groups must be non-decreasing in time order")
    bounds = np.flatnonzero(np.diff(group)) + 1
    g_start = np.r_[0, bounds]
    g_end = np.r_[bounds, n]
    starts = []
    dropped_tail = 0
    for a, b in zip(g_start, g_end):
        m = b - a
        if m < T:
            dropped_tail += m
            continue
        s = np.arange(a, b - T + 1, stride, dtype=np.int64)
        starts.append(s)
        dropped_tail += int(b - (s[-1] + T))
    start = np.concatenate(starts) if starts else np.zeros(0, np.int64)
    W = len(start)
    n_cls = int(codes.max()) + 1 if n else 1
    if n_cls > 64:
        raise ValueError("more than 64 classes; class_mask is uint64")

    win_codes = codes[start[:, None] + np.arange(T)[None, :]] if W else np.zeros((0, T), np.int16)
    counts = np.zeros((W, n_cls), dtype=np.int16)
    for c in range(n_cls):
        counts[:, c] = (win_codes == c).sum(axis=1)
    n_attack = (T - counts[:, 0]).astype(np.int16)
    y_bin = (n_attack > 0).astype(np.int8)
    att = counts[:, 1:]
    maj = np.where(y_bin == 1, att.argmax(axis=1) + 1, 0).astype(np.int16) if n_cls > 1 else np.zeros(W, np.int16)
    ties = int(((att == att.max(axis=1, keepdims=True)).sum(axis=1) > 1)[y_bin == 1].sum()) if n_cls > 1 else 0
    purity = counts[np.arange(W), maj].astype(np.float32) / T
    mask = np.zeros(W, dtype=np.uint64)
    for c in range(n_cls):
        mask |= (counts[:, c] > 0).astype(np.uint64) << np.uint64(c)

    sha = np.empty(W, dtype="S32")
    win_group_end = np.searchsorted(start, g_end, side="left")
    i = 0
    for a, b, iend in zip(g_start, g_end, win_group_end):
        if i >= iend:
            continue
        raw = np.ascontiguousarray(rows_fn(a, b), dtype=np.float32)
        row_bytes = raw.shape[1] * 4
        buf = raw.view(np.uint8).reshape(-1)
        for k in range(i, iend):
            s = start[k] - a
            sha[k] = hashlib.sha256(buf[s * row_bytes:(s + T) * row_bytes]).digest()
        i = iend

    wi = WindowIndex(start=start, group=group[start].astype(np.int32), y_bin=y_bin, y_cls=maj,
                     purity=purity, class_mask=mask, n_attack=n_attack, sha=sha, T=T, stride=stride)
    stats = {"windows": W, "flows_not_in_any_window": int(dropped_tail),
             "majority_ties": ties, "groups": int(len(g_start)),
             "groups_too_short": int(((g_end - g_start) < T).sum())}
    return wi, stats


def dedupe(wi: WindowIndex) -> tuple[WindowIndex, dict]:
    """Keep the earliest window of each raw-feature hash; drop later copies globally."""
    _, first = np.unique(wi.sha, return_index=True)
    keep = np.sort(first)
    # label conflicts among duplicates: same hash, different binary label
    order = np.argsort(wi.sha, kind="stable")
    s_sorted, y_sorted = wi.sha[order], wi.y_bin[order]
    same = s_sorted[1:] == s_sorted[:-1]
    conflicts = int((same & (y_sorted[1:] != y_sorted[:-1])).sum())
    out = wi.subset(keep)
    return out, {"windows_before": int(len(wi)), "windows_after": int(len(out)),
                 "duplicates_removed": int(len(wi) - len(out)),
                 "duplicate_label_conflicts": conflicts}


def materialize(flows: np.ndarray, start: np.ndarray, T: int) -> np.ndarray:
    """(n_windows, T, F) view-copy of cleaned flows."""
    return flows[start[:, None] + np.arange(T)[None, :]]
