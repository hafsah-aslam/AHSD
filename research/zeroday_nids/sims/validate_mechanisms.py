"""Synthetic Monte Carlo checks of the mathematical claims in 04_mathematical_framework.md.

These are NOT NIDS experiments. They test whether the proposed statistics behave as derived
(validity / false-alarm control, delay, size and power) on a controlled panel model of anomaly scores:

    s[v,t] = mu[t] + alpha[v] + eps[v,t]          (panel decomposition, eq. M1)
    mu[t]  : network-wide (common-mode) drift      eps: AR(1) per entity, coefficient rho
    alpha  : persistent entity offset (heterogeneity)

Usage:  python validate_mechanisms.py            -> prints a report and writes results.json
All randomness is seeded; numbers in the report are produced by this file only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results.json"
KAPPA, CAP = 0.2, 50.0          # capped power calibrator e = min(kappa * p^(kappa-1), CAP)


# ----------------------------------------------------------------------------- data model
def panel(rng, N=60, T=4000, rho=0.6, sd_alpha=1.0, drift=True, drift_scale=1.0):
    t = np.arange(T)
    mu = drift_scale * (0.6 * np.sin(2 * np.pi * t / 500) + np.cumsum(rng.normal(0, 0.02, T))) if drift else np.zeros(T)
    alpha = rng.normal(0, sd_alpha, N)
    eps = np.empty((N, T))
    z = rng.normal(size=(N, T))
    eps[:, 0] = z[:, 0]
    a = np.sqrt(1 - rho ** 2)
    for k in range(1, T):
        eps[:, k] = rho * eps[:, k - 1] + a * z[:, k]
    return mu, alpha, eps


def calibrate(p):
    return np.minimum(KAPPA * np.power(np.maximum(p, 1e-12), KAPPA - 1.0), CAP)


def calibrate_mix(p):
    """Mixture calibrator e(p) = int_0^1 k p^(k-1) dk = (1 - p + p ln p) / (p (ln p)^2); valid, no tuning."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    lp = np.log(p)
    return (1 - p + p * lp) / (p * lp ** 2)


def conformal_p(x, ref, rng):
    """Randomised conformal p-value of x against reference sample ref (larger = more anomalous)."""
    gt = np.sum(ref > x)
    eq = np.sum(ref == x)
    return (gt + rng.uniform() * (eq + 1)) / (len(ref) + 1)


def e_cusum_alarms(e, A):
    """e-CUSUM: M_t = max(M_{t-1},1)*e_t, alarm when M_t >= A, then reset. Returns alarm times."""
    M, alarms = 1.0, []
    for t, x in enumerate(e):
        M = max(M, 1.0) * x
        if M >= A:
            alarms.append(t)
            M = 1.0
    return alarms


# ----------------------------------------------------------------------------- p-value constructions
def p_train_ref(s, rng, W0=500):
    """(A) reference = all entities' scores in the first W0 steps (classic 'training-calibrated')."""
    ref = np.sort(s[:, :W0].ravel())
    N, T = s.shape
    P = np.full((N, T), np.nan)
    for t in range(W0, T):
        gt = len(ref) - np.searchsorted(ref, s[:, t], side="right")
        P[:, t] = (gt + rng.uniform(size=N)) / (len(ref) + 1)
    return P


def p_peer(s, rng, W0=500):
    """(B) reference = contemporaneous peers only (cross-sectional rank)."""
    N, T = s.shape
    P = np.full((N, T), np.nan)
    for t in range(W0, T):
        col = s[:, t]
        r = (col[None, :] > col[:, None]).sum(1)          # number of peers above
        P[:, t] = (r + rng.uniform(size=N)) / N
    return P


def p_double_diff(s, rng, W0=500, L=400, gap=20, stride=1):
    """(C) double-differenced residual: remove common mode by cross-sectional median, then calibrate
    each entity against its OWN past residuals [t-gap-L, t-gap). stride>1 evaluates every stride-th step."""
    r = s - np.median(s, axis=0, keepdims=True)
    N, T = s.shape
    P = np.full((N, T), np.nan)
    for t in range(W0, T, stride):
        ref = r[:, t - gap - L:t - gap]
        gt = (ref > r[:, [t]]).sum(1)
        P[:, t] = (gt + rng.uniform(size=N)) / (L + 1)
    return P, r


def gated_dd_alarms(s, rng, A, W0=500, L=400, gap=20, stride=6, G=5.0, calib=None):
    """(C+) evidence-gated double difference. Each entity's reference = its own past residuals at
    evaluated times whose e-CUSUM state was below G (windows under suspicion are not admitted).
    Returns per-entity lists of (alarm times) and (evaluated times, p-values)."""
    calib = calib or calibrate_mix
    r = s - np.median(s, axis=0, keepdims=True)
    N, T = s.shape
    out = []
    for v in range(N):
        admitted = np.zeros(T, bool)
        admitted[:W0] = True                       # warm-up assumed clean
        M, alarms, ts, ps = 1.0, [], [], []
        for t in range(W0, T, stride):
            lo, hi = t - gap - L, t - gap
            ref = r[v, lo:hi][admitted[lo:hi]]
            if len(ref) < 50:
                ref = r[v, lo:hi]
            p = (np.sum(ref > r[v, t]) + rng.uniform()) / (len(ref) + 1)
            M = max(M, 1.0) * float(calib(np.array([p]))[0])
            ts.append(t); ps.append(p)
            if M < G:
                admitted[t:t + stride] = True
            if M >= A:
                alarms.append(t); M = 1.0
        out.append((alarms, np.array(ts), np.array(ps)))
    return out


def p_network(s, rng, W0=500, L=400, gap=20, stride=1):
    """(N) network-wide component: cross-sectional median mu_hat[t], calibrated on its own past."""
    m = np.median(s, axis=0)
    T = len(m)
    P = np.full(T, np.nan)
    for t in range(W0, T, stride):
        ref = m[t - gap - L:t - gap]
        P[t] = (np.sum(ref > m[t]) + rng.uniform()) / (L + 1)
    return P


def alarm_rate(P, A):
    """False alarms per entity per 1000 evaluated steps under the null."""
    tot, steps = 0, 0
    for row in P:
        e = calibrate(row[~np.isnan(row)])
        tot += len(e_cusum_alarms(e, A))
        steps += len(e)
    return 1000.0 * tot / steps


def uniformity_gap(P):
    """sup_u |F_hat(u) - u| of pooled p-values (marginal validity diagnostic)."""
    x = np.sort(P[~np.isnan(P)])
    u = np.arange(1, len(x) + 1) / len(x)
    return float(np.max(np.abs(u - x)))


# ----------------------------------------------------------------------------- experiments
def exp_validity(seed=0, A=200.0):
    """H-V1/H-V2: which constructions keep the time-to-false-alarm >= A (rate <= 1000/A per 1000 steps)?"""
    out = {"A": A, "bound_per_1000": 1000.0 / A, "rows": []}
    for rho in (0.0, 0.6, 0.9):
        rng = np.random.default_rng(seed)
        mu, alpha, eps = panel(rng, rho=rho)
        s = mu[None, :] + alpha[:, None] + eps
        row = {"rho": rho}
        row["train_ref"] = alarm_rate(p_train_ref(s, rng), A)
        row["peer"] = alarm_rate(p_peer(s, rng), A)
        Pdd, _ = p_double_diff(s, rng)
        row["double_diff"] = alarm_rate(Pdd, A)
        stride = 1 if rho == 0 else int(np.ceil(np.log(0.05) / np.log(rho)))   # rho^k <= 0.05
        Pst, _ = p_double_diff(s, rng, stride=stride)
        row["double_diff_strided"] = alarm_rate(Pst, A)
        row["stride"] = stride
        row["unif_gap_dd"] = uniformity_gap(Pdd)
        g = gated_dd_alarms(s, rng, A, stride=stride)
        row["gated_dd_mix"] = 1000.0 * sum(len(a) for a, _, _ in g) / sum(len(ts) for _, ts, _ in g)
        out["rows"].append(row)
    rng = np.random.default_rng(seed + 7)
    mu, alpha, eps = panel(rng, rho=0.6)
    mu = mu + 0.0008 * np.arange(len(mu))          # benign upward ramp (e.g. growing load), +3.2 sd over the run
    s = mu[None, :] + alpha[:, None] + eps
    Pst, _ = p_double_diff(s, rng, stride=6)
    Pdl, _ = p_double_diff(s, rng, W0=1100, gap=700, stride=6)
    out["benign_ramp_rho0.6"] = {"train_ref": alarm_rate(p_train_ref(s, rng), A),
                                 "double_diff_strided": alarm_rate(Pst, A),
                                 "double_diff_strided_delayed_ref_gap700": alarm_rate(Pdl, A)}
    return out


def exp_delay(seed=1, A=200.0, rho=0.6, n_att=6, dur=600, gaps=(20, 700)):
    """H-D1/H-D2: entity-local attack eps += delta for `dur` steps on n_att of 60 entities.
    Compares the reference lag `gap` (short = self-masking, long = delayed reference), reports e-CUSUM
    detection within the attack, median delay, D = mean log e during the attack, the first-order
    prediction log(A)/D, and the per-window detection rate at p <= 0.01 (the HoNE-style operating point)."""
    rows = []
    stride = int(np.ceil(np.log(0.05) / np.log(rho)))
    for gap in gaps:
        for delta in (0.25, 0.5, 1.0):
            rng = np.random.default_rng(seed)
            mu, alpha, eps = panel(rng, rho=rho)
            s = mu[None, :] + alpha[:, None] + eps
            t0 = 2500
            s[:n_att, t0:t0 + dur] += delta
            P, _ = p_double_diff(s, rng, W0=gap + 400, gap=gap, stride=stride)
            det, delays, logs, per_window = 0, [], [], []
            for v in range(n_att):
                idx = np.where(~np.isnan(P[v]))[0]
                idx = idx[idx >= t0 - 200]
                e = calibrate_mix(P[v, idx])
                al = [idx[k] for k in e_cusum_alarms(e, A)]
                hit = [a for a in al if t0 <= a < t0 + dur]
                if hit:
                    det += 1
                    delays.append((hit[0] - t0) / stride)
                in_att = (idx >= t0) & (idx < t0 + dur)
                logs.extend(np.log(e[in_att]))
                per_window.append(np.mean(P[v, idx[in_att]] <= 0.01))
            D = float(np.mean(logs))
            rows.append({"gap": gap, "delta": delta, "stride": stride, "windows_in_attack": int(dur / stride),
                         "ecusum_detect_frac": det / n_att,
                         "median_delay_windows": float(np.median(delays)) if delays else None,
                         "D_mean_log_e": D,
                         "pred_delay_logA_over_D": float(np.log(A) / D) if D > 0 else None,
                         "per_window_TPR_at_p<=0.01": float(np.mean(per_window))})
    return rows


def exp_network(seed=2, A=200.0, rho=0.6, jump=1.0):
    """H-N1: a network-wide shift (all entities, +jump) is invisible to the entity-level double
    difference (by construction) but detected by an e-CUSUM on the common-mode component."""
    rng = np.random.default_rng(seed)
    mu, alpha, eps = panel(rng, rho=rho, drift=False)
    s = mu[None, :] + alpha[:, None] + eps
    t1 = 3000
    s[:, t1:t1 + 300] += jump
    stride = int(np.ceil(np.log(0.05) / np.log(rho)))
    Pn = p_network(s, rng, stride=stride)
    idx = np.where(~np.isnan(Pn))[0]
    al = [idx[k] for k in e_cusum_alarms(calibrate(Pn[idx]), A)]
    pre = [a for a in al if a < t1]
    hit = [a for a in al if t1 <= a < t1 + 300]
    Pdd, _ = p_double_diff(s, rng, stride=stride)
    ent_hits = 0
    for v in range(s.shape[0]):
        ii = np.where(~np.isnan(Pdd[v]))[0]
        a2 = [ii[k] for k in e_cusum_alarms(calibrate(Pdd[v, ii]), A)]
        ent_hits += any(t1 <= a < t1 + 300 for a in a2)
    return {"network_false_alarms_before_shift": len(pre), "network_detected": bool(hit),
            "network_delay_windows": (hit[0] - t1) / stride if hit else None,
            "entities_alarming_during_shift_double_diff": ent_hits, "N": s.shape[0]}


def coord_stat(B):
    """Coordination statistic: total pairwise co-exceedances sum_t sum_{u<w} b_u b_w."""
    k = B.sum(0)
    return float(np.sum(k * (k - 1) / 2))


def coordination_test(B, rng, n_perm=300):
    """Circular-shift null: shift each entity's exceedance series by an independent random offset.
    Preserves each entity's autocorrelation and exceedance rate; destroys cross-entity alignment."""
    obs = coord_stat(B)
    N, T = B.shape
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = rng.integers(0, T, N)
        null[i] = coord_stat(np.stack([np.roll(B[v], sh[v]) for v in range(N)]))
    return (1 + np.sum(null >= obs)) / (n_perm + 1)


def binomial_naive_test(B):
    """Naive null ignoring temporal dependence: co-exceedances ~ Poisson(sum over pairs of q_u q_w T)."""
    from math import erf, sqrt
    q = B.mean(1)
    T = B.shape[1]
    lam = T * (np.sum(q) ** 2 - np.sum(q ** 2)) / 2
    z = (coord_stat(B) - lam) / np.sqrt(lam)
    return 0.5 * (1 - erf(z / sqrt(2)))


def exp_coordination(n_rep=200, N=12, T=1500, q=0.05, common=0.0, rho_lo=0.0, rho_hi=0.95):
    """H-C1 size and H-C2 power. Null: independent entities with heterogeneous AR(1) bursts of
    exceedances. Alternative: k entities share weak synchronous beacons (each beacon raises the
    exceedance probability to 0.5 for one step, every ~40 steps)."""
    res = {"size_circular": 0, "size_naive": 0, "power_circular": 0, "power_naive": 0, "n_rep": n_rep}
    for rep in range(n_rep):
        rng = np.random.default_rng(1000 + rep)
        rhos = rng.uniform(rho_lo, rho_hi, N)
        X = np.empty((N, T))
        X[:, 0] = rng.normal(size=N)
        for t in range(1, T):
            X[:, t] = rhos * X[:, t - 1] + np.sqrt(1 - rhos ** 2) * rng.normal(size=N)
        if common > 0:                                           # shared common-mode shock, then removed
            c = np.convolve(rng.normal(size=T), np.ones(20) / 20, mode="same") * 4
            X = X + common * c[None, :]
            X = X - np.median(X, axis=0, keepdims=True)
        thr = np.quantile(X, 1 - q, axis=1, keepdims=True)
        B0 = (X > thr).astype(float)
        res["size_circular"] += coordination_test(B0, rng) <= 0.05
        res["size_naive"] += binomial_naive_test(B0) <= 0.05
        B1 = B0.copy()
        beacon_t = np.arange(rng.integers(0, 40), T, 40)
        for v in range(4):                                       # 4 coordinated entities
            fire = beacon_t[rng.uniform(size=len(beacon_t)) < 0.5]
            B1[v, fire] = 1.0
        res["power_circular"] += coordination_test(B1, rng) <= 0.05
        res["power_naive"] += binomial_naive_test(B1) <= 0.05
    for k in list(res):
        if k != "n_rep":
            res[k] = res[k] / n_rep
    return res


def exp_ebh(n_rep=300, N=50, n_alt=10, rho_common=0.5, alpha=0.1):
    """e-BH (Wang & Ramdas) FDR under cross-entity dependence: entity e-values are products of
    calibrated p-values over 30 windows; nulls share a common Gaussian factor (equicorrelation)."""
    from math import erfc, sqrt
    sf = np.vectorize(lambda z: 0.5 * erfc(z / sqrt(2)))
    fdp, power = [], []
    for rep in range(n_rep):
        rng = np.random.default_rng(5000 + rep)
        W = 30
        common = rng.normal(size=W)
        Z = np.sqrt(rho_common) * common[None, :] + np.sqrt(1 - rho_common) * rng.normal(size=(N, W))
        Z[:n_alt] += 0.8
        P = sf(Z)                       # each entry is a valid p-value marginally
        E = np.prod(calibrate_mix(P), axis=1)    # valid: windows independent within an entity
        order = np.argsort(-E)
        k_star = 0
        for k in range(1, N + 1):
            if E[order[k - 1]] >= N / (alpha * k):
                k_star = k
        rej = set(order[:k_star].tolist())
        fd = len([r for r in rej if r >= n_alt])
        fdp.append(fd / max(len(rej), 1))
        power.append(len([r for r in rej if r < n_alt]) / n_alt)
    return {"alpha": alpha, "FDR": float(np.mean(fdp)), "power": float(np.mean(power)),
            "rho_common": rho_common, "N": N, "n_alt": n_alt}


def main():
    res = {"validity": exp_validity(), "delay": exp_delay(), "network": exp_network(),
           "coordination": exp_coordination(),
           "coordination_common_shock_removed": exp_coordination(common=1.0),
           "coordination_high_persistence": exp_coordination(rho_lo=0.95, rho_hi=0.995),
           "ebh": exp_ebh()}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
