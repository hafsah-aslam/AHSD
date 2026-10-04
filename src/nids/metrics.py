"""Metrics and statistics (spec §9). Scores are always higher = attack."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             roc_auc_score, roc_curve)

T_975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def d_prime(y: np.ndarray, s: np.ndarray) -> float:
    """(μ_attack − μ_benign) / sqrt((σ²_a + σ²_b)/2); orientation fixed (higher = attack)."""
    a, b = s[y == 1], s[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2 + 1e-24))


def tpr_at_fpr(y, s, fpr_target: float) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_val_fpr(s_val_benign: np.ndarray, fpr: float) -> float:
    return float(np.quantile(s_val_benign, 1 - fpr))


def classification(y, pred) -> dict:
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else 0.0}


def detection(y, s) -> dict:
    return {"auc": auc(y, s), "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
            "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01), "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
            "d_prime": d_prime(y, s)}


def mean_ci(x) -> dict:
    """Mean ± Student-t 95% CI (t = 2.776 for n = 5); never 1.96."""
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(x[0]), "ci95": float("nan"), "n": 1}
    t = T_975.get(n, stats.t.ppf(0.975, n - 1))
    return {"mean": float(x.mean()), "ci95": float(t * x.std(ddof=1) / np.sqrt(n)), "n": n}


def spearman_bootstrap(pred, meas, n_boot: int = 1000, seed: int = 0) -> dict:
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    ok = np.isfinite(pred) & np.isfinite(meas)
    pred, meas = pred[ok], meas[ok]
    n = len(pred)
    if n < 3:
        return {"rho": float("nan"), "ci95": [float("nan")] * 2, "n": n, "p": float("nan")}
    rho, p = stats.spearmanr(pred, meas)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(pred[i])) > 1 and len(np.unique(meas[i])) > 1:
            boots.append(stats.spearmanr(pred[i], meas[i])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return {"rho": float(rho), "p": float(p), "ci95": [float(lo), float(hi)], "n": n, "n_boot_valid": len(boots)}


def group_bootstrap(metric_fn, y, s, groups, n_boot: int = 1000, seed: int = 0) -> dict:
    """Resample whole groups (1-hour blocks) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    by = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([by[g] for g in rng.choice(ug, len(ug), replace=True)])
        v = metric_fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return {"point": float(metric_fn(y, s)), "ci95": [float(lo), float(hi)], "n_valid": len(vals)}


# ------------------------------------------------------------------ DeLong
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N)
    out[J] = T
    return out


def delong(y, s1, s2) -> dict:
    """DeLong test for two correlated AUCs on the same windows (Sun & Xu 2014 fast form)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([s1, s2])
    k = 2
    tx = np.array([_midrank(p[pos]) for p in preds])
    ty = np.array([_midrank(p[neg]) for p in preds])
    tz = np.array([_midrank(np.r_[p[pos], p[neg]]) for p in preds])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return {"auc1": float(aucs[0]), "auc2": float(aucs[1]), "z": float(z), "p": float(p)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
