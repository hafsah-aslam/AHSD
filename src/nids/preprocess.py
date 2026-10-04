"""Feature cleaning fitted on training rows only.

Order of operations (identical for every dataset):
  ±inf -> NaN -> train median imputation -> log1p (non-negative heavy-tailed
  counts) -> drop zero-variance -> drop one of each |corr| >= 0.999999 pair
  -> standardise with train mean/std.

`Cleaner.to_dict()` holds every fitted statistic, so a transfer target is
rebuilt with the source's medians, mean/std and feature order and nothing is
ever fitted on the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

CORR_THRESHOLD = 0.999999


@dataclass
class Cleaner:
    candidate_features: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)  # final order
    median: dict[str, float] = field(default_factory=dict)
    log1p: list[str] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)  # col -> reason
    keep_ttl: bool = False
    fit_rows: int = 0
    corr_sample_rows: int = 0

    # ------------------------------------------------------------------ fit
    def fit(self, df_train: pd.DataFrame, corr_sample: int = 1_000_000, seed: int = 0,
            rows: np.ndarray | None = None) -> "Cleaner":
        """Fit on `df_train`, or on its boolean `rows` mask (avoids copying the frame)."""
        drops = schema.drop_map(self.keep_ttl)
        cols = [c for c in df_train.columns if c not in drops]
        self.dropped = {c: r for c, r in drops.items() if c in df_train.columns}
        non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df_train[c])]
        for c in non_numeric:
            self.dropped[c] = "non-numeric column"
        cols = [c for c in cols if c not in non_numeric]
        self.candidate_features = list(cols)
        mask = np.ones(len(df_train), bool) if rows is None else np.asarray(rows, bool)
        self.fit_rows = int(mask.sum())

        # Column by column keeps peak memory near one float64 column (20M-row datasets).
        rng = np.random.default_rng(seed)
        n = self.fit_rows
        samp = np.sort(rng.choice(n, corr_sample, replace=False)) if n > corr_sample else np.arange(n)
        self.corr_sample_rows = int(len(samp))
        kept, S_cols, self.median, self.log1p = [], [], {}, []
        self.mean, self.std = {}, {}
        for c in cols:
            x = df_train[c].to_numpy(dtype=np.float64)[mask]
            x[~np.isfinite(x)] = np.nan
            m = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
            self.median[c] = m
            x[np.isnan(x)] = m
            if schema.is_heavy_tailed(c) and n and x.min() >= 0:
                self.log1p.append(c)
                x = np.log1p(x)
            sd = float(x.std()) if n else 0.0
            if not np.isfinite(sd) or sd <= 1e-12:
                self.dropped[c] = "zero variance on train"
                continue
            kept.append(c)
            self.mean[c], self.std[c] = float(x.mean()), sd
            S_cols.append(x[samp])
        cols = kept
        S = np.stack(S_cols, axis=1) if S_cols else np.zeros((0, 0))
        C = np.corrcoef(S, rowvar=False) if len(cols) > 1 else np.ones((1, 1))
        drop_j = set()
        for i in range(len(cols)):
            if i in drop_j:
                continue
            for j in range(i + 1, len(cols)):
                if j not in drop_j and abs(C[i, j]) >= CORR_THRESHOLD:
                    drop_j.add(j)
                    self.dropped[cols[j]] = f"abs(corr) >= {CORR_THRESHOLD} with {cols[i]} on train"
        cols = [c for j, c in enumerate(cols) if j not in drop_j]
        self.mean = {c: self.mean[c] for c in cols}
        self.std = {c: self.std[c] for c in cols}
        self.features = cols
        return self

    def restrict(self, common: list[str]) -> "Cleaner":
        """Keep only `common` features (in that order); per-feature stats are unchanged."""
        missing = [c for c in common if c not in self.features]
        if missing:
            raise ValueError(f"common features not kept by this cleaner: {missing}")
        for c in self.features:
            if c not in common:
                self.dropped[c] = "not in the cross-dataset common feature list"
        self.features = list(common)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, chunk: int = 1_000_000) -> tuple[np.ndarray, dict]:
        """Return float32 (n, F) and a dict of transform-time diagnostics."""
        out = np.empty((len(df), len(self.features)), dtype=np.float32)
        diag = {"nonfinite_imputed": 0, "log1p_negative_clipped": 0}
        for a in range(0, len(df), chunk):
            out[a:a + chunk], d = self._transform(df.iloc[a:a + chunk])
            for k in diag:
                diag[k] += d[k]
        return out, diag

    def _transform(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"input lacks fitted features: {missing}")
        X = df[self.features].to_numpy(dtype=np.float64, copy=True)
        nonfinite = int((~np.isfinite(X)).sum())
        X[~np.isfinite(X)] = np.nan
        med = np.array([self.median[c] for c in self.features])
        X = _impute(X, med)
        neg_clipped = 0
        for j, c in enumerate(self.features):
            if c in self.log1p:
                neg = X[:, j] < 0
                neg_clipped += int(neg.sum())
                X[neg, j] = 0.0
                X[:, j] = np.log1p(X[:, j])
        mu = np.array([self.mean[c] for c in self.features])
        sd = np.array([self.std[c] for c in self.features])
        X = (X - mu) / sd
        return X.astype(np.float32), {"nonfinite_imputed": nonfinite, "log1p_negative_clipped": neg_clipped}

    # ---------------------------------------------------------- serialise
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "Cleaner":
        return cls(**d)


def _impute(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    nan_r, nan_c = np.where(np.isnan(X))
    X[nan_r, nan_c] = med[nan_c]
    return X


def common_feature_list(cleaners: dict[str, Cleaner]) -> list[str]:
    """Intersection of kept features, in canonical schema order."""
    sets = [set(c.features) for c in cleaners.values()]
    inter = set.intersection(*sets) if sets else set()
    order = schema.EXPECTED_FEATURES + sorted(x for x in inter if x not in schema.EXPECTED_FEATURES)
    return [c for c in order if c in inter]
