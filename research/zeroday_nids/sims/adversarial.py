"""Adversarial falsification of the Panel e-Detection mathematics (05_revised_framework.md).

Synthetic only. Each experiment is built to BREAK a stated guarantee or approximation, not to show a
favourable case. Seeded; every number in the report is printed by this file.

Usage: python adversarial.py [--quick]       -> results_adversarial.json
"""
from __future__ import annotations

import json
import sys
from math import erf, erfc, log, sqrt
from pathlib import Path

import numpy as np

QUICK = "--quick" in sys.argv
OUT = Path(__file__).resolve().parent / "results_adversarial.json"


# ============================================================================ shared primitives
def mix_cal(p):
    """Mixture calibrator  int_0^1 k p^(k-1) dk  (tuning-free, valid)."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(p)
    return (1 - p + p * lp) / (p * lp ** 2)


def Phi(x):
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(x) / sqrt(2)))


def Phi_inv(u):
    lo, hi = -12.0, 12.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + erf(mid / sqrt(2))) < u:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def binormal_slope(p, delta):
    """ROC'(u) for N(0,1) vs N(delta,1):  exp(delta z - delta^2/2), z = Phi^{-1}(1-u)."""
    z = np.array([Phi_inv(1 - x) for x in np.atleast_1d(np.clip(p, 1e-12, 1 - 1e-12))])
    return np.exp(delta * z - delta ** 2 / 2)


def ecusum(e, A):
    """Alarm indices of e-CUSUM with reset."""
    M, al = 1.0, []
    for i, x in enumerate(e):
        M = max(M, 1.0) * x
        if M >= A:
            al.append(i)
            M = 1.0
    return al


def first_alarm(e, A):
    M = 1.0
    for i, x in enumerate(e):
        M = max(M, 1.0) * x
        if M >= A:
            return i
    return None


def ar1(rng, n_series, T, rho):
    x = np.empty((n_series, T))
    x[:, 0] = rng.normal(size=n_series)
    a = sqrt(1 - rho ** 2)
    z = rng.normal(size=(n_series, T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + a * z[:, t]
    return x


def dd_pvalues(s, rng, g, L, k, ref_stride, t_start, idx=None):
    """Double-differenced, delayed-reference conformal p-values.
    Residual r = s - median over entities. Reference for (v,t): r[v, t-g-L : t-g : ref_stride].
    Evaluated at t = t_start, t_start+k, ...; returns (times, P[N, n_eval])."""
    r = s - np.median(s, axis=0, keepdims=True)
    N, T = s.shape
    rows = range(N) if idx is None else idx
    times = np.arange(t_start, T, k)
    P = np.empty((len(list(rows)), len(times)))
    rows = list(rows)
    for j, t in enumerate(times):
        ref = r[rows, t - g - L:t - g:ref_stride]
        gt = (ref > r[rows, t][:, None]).sum(1)
        P[:, j] = (gt + rng.uniform(size=len(rows))) / (ref.shape[1] + 1)
    return times, P


def fa_rate(P, A):
    tot = sum(len(ecusum(mix_cal(row), A)) for row in P)
    return 1000.0 * tot / P.size


# ============================================================================ A. M2/M3 validity
def beta_ar1_bound(rho, k):
    """beta(k) <= sqrt(I/2), I = -1/2 log(1 - rho^(2k)) (Markov + Jensen + Pinsker)."""
    return sqrt(-0.25 * log(1 - rho ** (2 * k)))


def exp_A(A=200.0):
    out = {"A": A, "bound_per_1000": 1000.0 / A}
    N, T = 40, 9000 if not QUICK else 5000
    rows = []
    for rho in (0.6, 0.9):
        k = int(np.ceil(log(0.01) / log(rho)))
        rng = np.random.default_rng(11)
        eps = ar1(rng, N, T, rho)
        alpha = rng.normal(0, 1, N)
        mu = 0.6 * np.sin(2 * np.pi * np.arange(T) / 500)
        s = mu + alpha[:, None] + eps
        L = 60 * k
        _, P_theorem = dd_pvalues(s, rng, g=k, L=L, k=k, ref_stride=k, t_start=L + k)
        _, P_dense = dd_pvalues(s, rng, g=k, L=L, k=1, ref_stride=1, t_start=L + k)
        m = L // k
        beta = beta_ar1_bound(rho, k)
        rows.append({"rho": rho, "stride_k": k, "m_ref": m, "beta_k_bound": beta,
                     "FA_theorem_design(strided ref+eval)": fa_rate(P_theorem, A),
                     "FA_dense_ref_dense_eval": fa_rate(P_dense, A)})
    out["validity_design"] = rows

    # Counterexample A1: idiosyncratic random walk (non-stationary, not mixing)
    rng = np.random.default_rng(12)
    rw = []
    for sig in (0.0, 0.01, 0.03):
        eps = ar1(rng, N, T, 0.6) + np.cumsum(rng.normal(0, sig, (N, T)), axis=1)
        s = rng.normal(0, 1, N)[:, None] + eps
        res = {"rw_step_sd": sig}
        for g in (6, 600):
            _, P = dd_pvalues(s, rng, g=g, L=360, k=9, ref_stride=9, t_start=g + 360)
            res[f"FA_gap{g}"] = fa_rate(P, A)
        rw.append(res)
    out["counterexample_random_walk"] = rw

    # Counterexample A2: benign regime switching with long dwell (two levels 1.5 sd apart)
    rng = np.random.default_rng(13)
    sw = []
    for dwell in (50, 500, 3000):
        state = np.zeros((N, T))
        st = rng.integers(0, 2, N)
        for t in range(T):
            flip = rng.uniform(size=N) < 1.0 / dwell
            st = np.where(flip, 1 - st, st)
            state[:, t] = st
        s = 3.0 * state + ar1(rng, N, T, 0.6)
        _, P = dd_pvalues(s, rng, g=9, L=360, k=9, ref_stride=9, t_start=400)
        sw.append({"mean_dwell": dwell, "FA": fa_rate(P, A)})
    out["counterexample_regime_switching"] = sw

    # Counterexample A3: majority of a role compromised (median breakdown)
    rng = np.random.default_rng(14)
    maj = []
    for frac in (0.2, 0.45, 0.6, 0.8):
        s = ar1(rng, N, 5000, 0.6) + rng.normal(0, 1, N)[:, None]
        n_att = int(frac * N)
        s[:n_att, 3000:] += 2.0
        times, P = dd_pvalues(s, rng, g=600, L=360, k=9, ref_stride=9, t_start=1000)
        after = times >= 3000
        att_det = np.mean([len([a for a in ecusum(mix_cal(P[v, after]), A)]) > 0 for v in range(n_att)])
        ben_fa = np.mean([len([a for a in ecusum(mix_cal(P[v, after]), A)]) > 0 for v in range(n_att, N)])
        maj.append({"fraction_of_role_attacked": frac, "attacked_hosts_detected": float(att_det),
                    "benign_hosts_alarming": float(ben_fa)})
    out["counterexample_role_majority"] = maj
    return out


# ============================================================================ B. M4 detectability
def kl_mc(sample_logratio):
    return float(np.mean(sample_logratio))


def exp_B(A=1000.0):
    rng = np.random.default_rng(21)
    out = {"A": A}
    n = 200000 if not QUICK else 50000
    # B1: equal AUC = 0.75, different shapes
    d_bn = sqrt(2) * Phi_inv(0.75)
    x0 = rng.normal(size=n)
    shapes = {}
    # (i) binormal
    x1 = rng.normal(d_bn, 1, n)
    lr = lambda x: d_bn * x - d_bn ** 2 / 2
    shapes["binormal"] = {"Psi": kl_mc(lr(x1)), "AUC": float(np.mean(x1[:20000, None] > x0[None, :20000]))}
    # (ii) partially separable: with prob pi shifted by D=4, else benign  (AUC ~ 0.75)
    D = 4.0
    auc_sep = float(Phi(D / sqrt(2)))
    pi = (0.75 - 0.5) / (auc_sep - 0.5)
    lab = rng.uniform(size=n) < pi
    x1m = np.where(lab, rng.normal(D, 1, n), rng.normal(0, 1, n))
    lrm = lambda x: np.log(pi * np.exp(D * x - D ** 2 / 2) + (1 - pi))
    shapes["partially_separable"] = {"Psi": kl_mc(lrm(x1m)), "pi": pi,
                                     "AUC": float(np.mean(x1m[:20000, None] > x0[None, :20000]))}
    # (iii) wide-variance attack N(0, 2.6^2) vs N(0,1) one-sided upper tail via |x| score (AUC ~0.75)
    sig = 2.6
    xa = np.abs(rng.normal(0, sig, n)); xb = np.abs(x0)
    lrw = lambda x: -np.log(sig) + 0.5 * x ** 2 * (1 - 1 / sig ** 2)   # density ratio of |N(0,s)| vs |N(0,1)|
    shapes["variance_inflation"] = {"Psi": kl_mc(lrw(xa)), "AUC": float(np.mean(xa[:20000, None] > xb[None, :20000]))}
    # TPR at 1% FPR for each
    thr = np.quantile(x0, 0.99); thr_abs = np.quantile(xb, 0.99)
    shapes["binormal"]["TPR@1%"] = float(np.mean(x1 > thr))
    shapes["partially_separable"]["TPR@1%"] = float(np.mean(x1m > thr))
    shapes["variance_inflation"]["TPR@1%"] = float(np.mean(xa > thr_abs))
    # Psi_min(AUC): I-projection = truncated exponential density on [0,1] with mean 1-AUC
    def psi_min(auc):
        m = 1 - auc
        lo, hi = 1e-6, 200.0                               # density c*exp(-lam u); mean decreasing in lam
        for _ in range(200):
            lam = 0.5 * (lo + hi)
            mean = 1 / lam - np.exp(-lam) / (1 - np.exp(-lam))
            lo, hi = (lam, hi) if mean > m else (lo, lam)
        lam = 0.5 * (lo + hi); c = lam / (1 - np.exp(-lam))
        return float(np.log(c) - lam * m)                  # E[log(c e^{-lam p})]
    def kl_bin(a, b):
        return a * log(a / b) + (1 - a) * log((1 - a) / (1 - b))
    for k, v in shapes.items():
        v["Psi_min_from_AUC_0.75"] = psi_min(0.75)
        v["lower_bound_from_TPR@1%"] = kl_bin(v["TPR@1%"], 0.01) if v["TPR@1%"] > 0.01 else 0.0
        v["pred_delay_logA/Psi"] = log(A) / v["Psi"]
    out["equal_AUC_different_Psi"] = shapes

    # B2: calibrator mismatch: D = Psi - KL(P1 || Q) for the mixture calibrator Q
    rows = []
    for delta in (0.25, 0.5, 1.0):
        x = rng.normal(delta, 1, n)
        p = 1 - Phi(x)
        Psi = kl_mc(delta * x - delta ** 2 / 2)
        D_mix = float(np.mean(np.log(mix_cal(p))))
        # KL(P1 || Q) = E_P1[log ROC'(p) - log q(p)]
        kl_pq = float(np.mean((delta * x - delta ** 2 / 2) - np.log(mix_cal(p))))
        rows.append({"delta": delta, "Psi": Psi, "KL(P1||Q_mix)": kl_pq, "Psi-KL": Psi - kl_pq, "D_mix_direct": D_mix})
    out["calibrator_mismatch"] = rows

    # B3: duty-cycle evasion (attack active in a fraction f of windows) and hedged calibrator
    delta = 1.0
    reps = 300 if not QUICK else 80
    duty = []
    for f in (0.3, 0.5, 0.7, 1.0):
        res = {"f": f, "drift_theory_oracle": (2 * f - 1) * delta ** 2 / 2}
        for name, cal in (("oracle_ROC_slope", lambda p: binormal_slope(p, delta)),
                          ("hedged_lambda=f", lambda p, f=f: f * binormal_slope(p, delta) + (1 - f)),
                          ("hedged_lambda=0.3", lambda p: 0.3 * binormal_slope(p, delta) + 0.7)):
            dl, drift = [], []
            for r in range(reps):
                on = rng.uniform(size=1500) < f
                x = rng.normal(0, 1, 1500) + delta * on
                e = cal(1 - Phi(x))
                drift.append(np.mean(np.log(e)))
                fa = first_alarm(e, A)
                dl.append(fa if fa is not None else 1500)
            res[f"{name}_drift"] = float(np.mean(drift))
            res[f"{name}_median_delay"] = float(np.median(dl))
            res[f"{name}_detected_within_1500"] = float(np.mean(np.array(dl) < 1500))
        duty.append(res)
    out["duty_cycle_random"] = duty
    per = []
    for on, off in ((3, 7), (5, 5), (7, 3)):
        f = on / (on + off)
        sched = np.tile(np.r_[np.ones(on), np.zeros(off)], 1500 // (on + off) + 1)[:1500].astype(bool)
        res = {"on": on, "off": off, "f": f, "drift_theory_oracle": (2 * f - 1) * delta ** 2 / 2}
        for name, cal in (("oracle_ROC_slope", lambda p: binormal_slope(p, delta)),
                          ("hedged_lambda=f", lambda p, f=f: f * binormal_slope(p, delta) + (1 - f))):
            dl = []
            for r in range(reps):
                x = rng.normal(0, 1, 1500) + delta * sched
                fa = first_alarm(cal(1 - Phi(x)), A)
                dl.append(fa if fa is not None else 1500)
            res[f"{name}_median_delay"] = float(np.median(dl))
            res[f"{name}_detected_within_1500"] = float(np.mean(np.array(dl) < 1500))
        per.append(res)
    out["duty_cycle_periodic"] = per
    return out


# ============================================================================ C. slow / persistent attacks
def exp_C(A=200.0):
    """Entity-level references: single gap (short / long), frozen anchor, and multi-scale merge.
    Multi-scale merge = average of mixture e-values from each reference (valid under any dependence)."""
    rng = np.random.default_rng(31)
    N, T = 30, 12000 if not QUICK else 8000
    k = 9
    out = {"A": A}

    def run(s, att_rows, t_on):
        r = s - np.median(s, axis=0, keepdims=True)
        times = np.arange(3400, s.shape[1], k)
        refs = {"gap20": (20, 360), "gap1500": (1500, 360)}
        res = {}
        Pall = {}
        for name, (g, L) in refs.items():
            P = np.empty((s.shape[0], len(times)))
            for j, t in enumerate(times):
                ref = r[:, t - g - L:t - g:k]
                P[:, j] = ((ref > r[:, t][:, None]).sum(1) + rng.uniform(size=s.shape[0])) / (ref.shape[1] + 1)
            Pall[name] = P
        anchor = r[:, 1000:3000:k]
        P = np.empty((s.shape[0], len(times)))
        for j, t in enumerate(times):
            P[:, j] = ((anchor > r[:, t][:, None]).sum(1) + rng.uniform(size=s.shape[0])) / (anchor.shape[1] + 1)
        Pall["anchor"] = P
        E_multi = (mix_cal(Pall["gap20"]) + mix_cal(Pall["gap1500"]) + mix_cal(Pall["anchor"])) / 3
        for name in ("gap20", "gap1500", "anchor"):
            E = mix_cal(Pall[name])
            res[name] = summarize(E, times, att_rows, t_on)
        res["multiscale_avg"] = summarize(E_multi, times, att_rows, t_on)
        return res

    def summarize(E, times, att_rows, t_on):
        det, delay, alarms_late = [], [], []
        for v in att_rows:
            al = [times[i] for i in ecusum(E[v], A)]
            post = [a for a in al if a >= t_on]
            det.append(len(post) > 0)
            delay.append((post[0] - t_on) / k if post else None)
            alarms_late.append(len([a for a in post if a >= t_on + 4000]))
        ben = [v for v in range(E.shape[0]) if v not in att_rows]
        fa = 1000.0 * sum(len(ecusum(E[v], A)) for v in ben) / (len(ben) * E.shape[1])
        d = [x for x in delay if x is not None]
        return {"detect_frac": float(np.mean(det)), "median_delay_windows": float(np.median(d)) if d else None,
                "alarms_after_+4000_per_attacked_host": float(np.mean(alarms_late)), "benign_FA_per_1000": fa}

    base = ar1(rng, N, T, 0.6) + rng.normal(0, 1, N)[:, None]
    att = list(range(5))
    t_on = 4000
    # C1 persistent step attack of moderate size lasting to the end
    s = base.copy(); s[att, t_on:] += 0.8
    out["persistent_step_0.8sd"] = run(s, att, t_on)
    # C2 slow ramp: +1.5 sd reached after 6000 steps, then held
    s = base.copy()
    ramp = np.clip((np.arange(T) - t_on) / 6000, 0, 1) * 1.5
    s[att] += ramp
    out["slow_ramp_1.5sd_over_6000"] = run(s, att, t_on)
    # C3 trade-off counterexample: benign idiosyncratic slow drift (no attack) -> anchor false alarms
    s = base + np.cumsum(rng.normal(0, 0.01, (N, T)), axis=1)
    out["benign_idiosyncratic_drift_no_attack"] = run(s, [], t_on)
    return out


# ============================================================================ D. M7 coordination
def coord(B, lag=0):
    if lag == 0:
        k = B.sum(0)
        return float(np.sum(k * (k - 1) / 2))
    tot = 0.0
    for l in range(-lag, lag + 1):
        Bs = np.roll(B, l, axis=1)
        tot += float(np.sum(np.triu(B @ Bs.T, 1)))
    return tot


def null_pvalue(B, rng, mode, period=None, n_perm=200, lag=0, hist=None):
    obs = coord(B, lag)
    N, T = B.shape
    if mode == "historical":
        return (1 + np.sum(np.array(hist) >= obs)) / (len(hist) + 1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        if mode == "circular":
            sh = rng.integers(0, T, N)
        else:                                             # phase-preserving: shifts by multiples of period
            sh = period * rng.integers(0, T // period, N)
        null[i] = coord(np.stack([np.roll(B[v], sh[v]) for v in range(N)]), lag)
    return (1 + np.sum(null >= obs)) / (n_perm + 1)


def exp_D(reps=None):
    reps = reps or (120 if not QUICK else 40)
    N, T, P, q = 12, 960, 96, 0.05
    scen = ["independent", "software_update_subset", "scheduled_backup_subset", "outage_all", "opchange_all",
            "bots_new_beacon", "bots_jittered_beacon"]
    res = {s: {"circular": 0, "phase": 0, "historical": 0, "circular_lag3": 0, "circular_bin4": 0} for s in scen}
    for rep in range(reps):
        rng = np.random.default_rng(4000 + rep)
        for sc in scen:
            # 6 "days" of history + current day for the historical null; days are length T
            days = []
            for d in range(7):
                X = ar1(rng, N, T, rng.uniform(0.3, 0.9))
                if sc == "scheduled_backup_subset":                # every day, hosts 0-3, same phase each period
                    for t0 in range(10, T, P):
                        X[:4, t0:t0 + 3] += 3.0
                days.append(X)
            X = days[-1]
            if sc == "software_update_subset":                     # one-off: 5 hosts updated together today
                t0 = rng.integers(100, T - 100); X[:5, t0:t0 + 30] += 2.5
            if sc == "outage_all":
                t0 = rng.integers(100, T - 100); X[:, t0:t0 + 40] += 3.0
            if sc == "opchange_all":
                X[:, T // 2:] += 2.0
            if sc == "bots_new_beacon":                            # 4 bots start beaconing today (period 37)
                for t0 in range(5, T, 37):
                    X[:4, t0] += 3.0
            if sc == "bots_jittered_beacon":
                for t0 in range(5, T, 37):
                    for v in range(4):
                        tj = min(T - 1, t0 + rng.integers(0, 4)); X[v, tj] += 3.0
            def excee(Y):
                R = Y - np.median(Y, axis=0, keepdims=True)
                thr = np.quantile(R, 1 - q, axis=1, keepdims=True)
                return (R > thr).astype(float)
            B = excee(X)
            hist = [coord(excee(Y)) for Y in days[:-1]]
            res[sc]["circular"] += null_pvalue(B, rng, "circular") <= 0.05
            res[sc]["phase"] += null_pvalue(B, rng, "phase", period=P) <= 0.05
            res[sc]["historical"] += null_pvalue(B, rng, "historical", hist=hist) <= 1 / 7 + 1e-9
            res[sc]["circular_lag3"] += null_pvalue(B, rng, "circular", lag=3, n_perm=100) <= 0.05
            Bb = B.reshape(N, T // 4, 4).max(2)                     # time-binned exceedances (bin = 4)
            res[sc]["circular_bin4"] += null_pvalue(Bb, rng, "circular") <= 0.05
    for sc in scen:
        for k in res[sc]:
            res[sc][k] = res[sc][k] / reps
    res["note"] = ("rejection rates; historical null has only 6 past days so its smallest attainable p is 1/7 "
                   "(test at p <= 1/7, i.e. level 0.143) — power and size are not comparable to the others")
    return res


# ============================================================================ E. M9 identification
def exp_E(seed=51):
    rng = np.random.default_rng(seed)
    d, n_cal, n_test = 24, 40, 200
    def family(support, mags):
        prof = np.zeros(d); prof[support] = mags; return prof
    known = {f"atk{k}": family(rng.choice(d, 4, replace=False), rng.uniform(1, 3, 4)) for k in range(5)}
    known["benign_change_A"] = family(rng.choice(d, 3, replace=False), rng.uniform(1, 2, 3))
    known["benign_change_B"] = family(rng.choice(d, 3, replace=False), rng.uniform(1, 2, 3))
    def draw(prof, n, noise=0.35):
        X = np.maximum(prof[None, :] + rng.normal(0, noise, (n, d)), 0)
        return X / (X.sum(1, keepdims=True) + 1e-12)
    cal = {k: draw(v, n_cal) for k, v in known.items()}
    def knn_score(x, C, k=5, exclude_self=False):
        dd = np.sort(np.linalg.norm(C - x, axis=1))
        return float(np.mean(dd[1:k + 1] if exclude_self else dd[:k]))
    cal_scores = {k: np.array([knn_score(C[i], C, exclude_self=True) for i in range(len(C))]) for k, C in cal.items()}
    def support(x, m=4):
        return set(np.argsort(-x)[:m].tolist())
    modal = {k: support(v) for k, v in known.items()}
    cal_jac = {k: np.array([len(support(C[i]) & modal[k]) / len(support(C[i]) | modal[k]) for i in range(len(C))])
               for k, C in cal.items()}
    alpha = 0.1
    def decide(x):
        pis = {k: (1 + np.sum(cal_scores[k] >= knn_score(x, cal[k]))) / (n_cal + 1) for k in known}
        G = [k for k, p in pis.items() if p > alpha]
        if len(G) == 1:
            return "name:" + G[0]
        if len(G) > 1:
            return "ambiguous"
        jac = {k: len(support(x) & modal[k]) / len(support(x) | modal[k]) for k in known}
        pj = {k: (1 + np.sum(cal_jac[k] <= jac[k])) / (n_cal + 1) for k in known}
        V = [k for k, p in pj.items() if p > alpha]
        return ("variant:" + V[0]) if len(V) == 1 else ("ambiguous" if V else "unfamiliar")
    def tally(X):
        out = {}
        for x in X:
            r = decide(x); key = r.split(":")[0]
            out[key] = out.get(key, 0) + 1
        return {k: v / len(X) for k, v in out.items()}
    res = {}
    # known family test episodes (coverage of the true name)
    correct = 0; tot = 0
    for k, v in known.items():
        for x in draw(v, n_test // 5):
            tot += 1; correct += decide(x) == "name:" + k
    res["known_correct_name_rate"] = correct / tot
    res["known_target_>=1-alpha"] = 1 - alpha
    # variant: same support as atk0, magnitudes permuted and doubled
    sup0 = sorted(modal["atk0"])
    var_prof = np.zeros(d); var_prof[sup0] = rng.permutation(known["atk0"][sup0]) * 2.0
    res["variant_of_atk0"] = tally(draw(var_prof, n_test))
    # genuinely new family: disjoint support
    used = set().union(*modal.values())
    free = [j for j in range(d) if j not in used]
    new_prof = family(rng.choice(free, min(4, len(free)), replace=False), rng.uniform(1, 3, min(4, len(free))))
    res["new_family_disjoint_support"] = tally(draw(new_prof, n_test))
    # adversarial new family sharing 3 of 4 features with atk1
    s1 = sorted(modal["atk1"])
    adv_prof = family(s1[:3] + [free[0] if free else s1[3]], rng.uniform(1, 3, 4))
    res["new_family_overlapping_atk1"] = tally(draw(adv_prof, n_test))
    # unseen benign change (not in library): indistinguishable from a new attack family by construction
    ben_new = family(rng.choice(d, 3, replace=False), rng.uniform(1, 2, 3))
    res["unseen_benign_change"] = tally(draw(ben_new, n_test))
    return res


# ============================================================================ F. M11 contamination
def exp_F(A=200.0):
    """1-D entity with a normality pool used to score it (a stand-in for the base detector's notion of
    normal). Policies: naive (pool = most recent W windows), lagged+purge (pool = windows older than g_u,
    windows within B of any alarm removed), lagged+purge+anchor (also an e-CUSUM against a frozen pool)."""
    rng = np.random.default_rng(61)
    reps = 60 if not QUICK else 20
    T, W, gu, Bp = 6000, 600, 600, 900
    out = {"A": A}
    # F1 boiling frog: slow ramp reaching +3 sd over 4000 steps
    def simulate(x, policy, Q=0):
        """Returns alarms, set of indices ever admitted to the normality pool."""
        ok = np.ones(len(x), bool)
        admitted = np.zeros(len(x), bool)
        M, Ma, alarms = 1.0, 1.0, []
        anchor = x[200:1000]
        last_pool = np.arange(200, 1000)                     # vetted warm-up pool
        for t in range(1000, len(x), 3):
            if policy == "naive":
                pool_idx = np.arange(t - W, t)
            else:
                lo = max(0, t - gu - W); hi = t - gu
                pool_idx = np.arange(lo, hi)[ok[lo:hi]]
                if len(pool_idx) < 50:
                    pool_idx = last_pool                     # freeze the last vetted pool; never fall back to unvetted data
                else:
                    last_pool = pool_idx
            admitted[pool_idx] = True
            pool = x[pool_idx]
            p = (np.sum(pool > x[t]) + rng.uniform()) / (len(pool) + 1)
            M = max(M, 1.0) * float(mix_cal(np.array([p]))[0])
            hit = M >= A
            if "anchor" in policy:
                pa = (np.sum(anchor > x[t]) + rng.uniform()) / (len(anchor) + 1)
                Ma = max(Ma, 1.0) * float(mix_cal(np.array([pa]))[0])
                hit = hit or Ma >= A
            if hit:
                alarms.append(t); M = Ma = 1.0
                ok[max(0, t - Bp):min(len(x), t + 1 + Q)] = False      # retroactive purge + quarantine
        return alarms, admitted, ok

    res = {}
    for policy, Q in (("naive", 0), ("lagged+purge", 0), ("lagged+purge+quarantine", 10 ** 6),
                      ("lagged+purge+anchor", 0)):
        det, contam = [], []
        for r in range(reps):
            x = ar1(rng, 1, T, 0.6)[0]
            t0 = 1500
            x[t0:] += np.clip((np.arange(T - t0)) / 4000, 0, 1) * 3.0
            alarms, admitted, ok = simulate(x, policy, Q=Q)
            det.append(any(a >= t0 for a in alarms))
            if policy == "naive":
                idx = np.arange(T - W, T)
            else:
                idx = np.arange(T - gu - W, T - gu); idx = idx[ok[idx]]
            contam.append(float(np.mean(idx >= t0)) if len(idx) else 0.0)
        res[policy] = {"ramp_detected": float(np.mean(det)), "final_pool_attack_fraction": float(np.mean(contam))}
    out["boiling_frog_ramp_3sd_over_4000"] = res
    # F2 contamination bound check for step attacks of length H: N_c vs sum_j P(tau > g_u + j)
    H, delta = 1500, 0.7
    cres = {}
    for label, Q in (("purge_only", 0), ("purge+quarantine", 10 ** 6)):
        taus, Ncs = [], []
        for r in range(reps * 2):
            x = ar1(rng, 1, T, 0.6)[0]
            t0 = 2500
            x[t0:t0 + H] += delta
            alarms, admitted, ok = simulate(x, "lagged+purge", Q=Q)
            post = [a for a in alarms if a >= t0]
            taus.append((post[0] - t0) if post else np.inf)
            Ncs.append(int(np.sum(admitted[t0:t0 + H])))           # episode windows that ever entered the pool
        taus = np.array(taus)
        bound = float(sum(np.mean(taus > gu + j) for j in range(H)))
        cres[label] = {"mean_contaminating_windows": float(np.mean(Ncs)), "bound_sum_P(tau>g_u+j)": bound,
                       "P(tau>g_u)": float(np.mean(taus > gu)), "P(no detection)": float(np.mean(np.isinf(taus)))}
    out["contamination_step_attack"] = {"H": H, "delta": delta, "g_u": gu, **cres}
    return out


def main():
    if "--only-F" in sys.argv:
        res = json.loads(OUT.read_text()); res["F_contamination"] = exp_F()
    else:
        res = {"A_validity": exp_A(), "B_detectability": exp_B(), "C_slow_persistent": exp_C(),
               "D_coordination": exp_D(), "E_identification": exp_E(), "F_contamination": exp_F()}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
