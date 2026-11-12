"""Theorem 1a check: online (full-past, smoothed) conformal p-values on the strided double-differenced
residual are exactly i.i.d. uniform under exchangeability (Vovk, Gammerman & Shafer 2005). Under
beta-mixing the strided sequence is within (n-1)beta(k) TV of i.i.d. (Berbee coupling), so
P(false alarm within n evaluations) <= n/A + (n-1) beta(k), with no deflation.
Compares with the delayed sliding reference (Theorem 1b, approximate) on null and attack panels.
Also: Cramer root theta* for the periodic duty-cycle case of adversarial.py.
Synthetic only. Usage: python online_conformal.py"""
import json
from math import exp, log, sqrt
from pathlib import Path
import numpy as np

def mix_cal(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); lp = np.log(p)
    return (1 - p + p * lp) / (p * lp ** 2)

def ecusum(e, A):
    M, al = 1.0, []
    for i, x in enumerate(e):
        M = max(M, 1.0) * x
        if M >= A:
            al.append(i); M = 1.0
    return al

def ar1(rng, n, T, rho):
    x = np.empty((n, T)); x[:, 0] = rng.normal(size=n); a = sqrt(1 - rho ** 2)
    z = rng.normal(size=(n, T))
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + a * z[:, t]
    return x

def online_p(seq, rng, warm=50):
    """Smoothed online conformal p-values: p_j = (#{i<=j: x_i > x_j} + U * #{i<=j: x_i == x_j}) / j."""
    out = np.full(len(seq), np.nan)
    for j in range(len(seq)):
        past = seq[:j + 1]
        gt = np.sum(past > seq[j]); eq = np.sum(past == seq[j])
        out[j] = (gt + rng.uniform() * eq) / (j + 1)
    out[:warm] = np.nan                     # warm-up: not used for alarms (still part of the bag)
    return out

def run(rho, A=200.0, N=40, T=12000, attack=None, seed=0):
    rng = np.random.default_rng(seed)
    k = int(np.ceil(log(0.01) / log(rho)))
    s = ar1(rng, N, T, rho) + rng.normal(0, 1, N)[:, None] + 0.6 * np.sin(2 * np.pi * np.arange(T) / 500)
    if attack:
        t0, delta, n_att = attack
        s[:n_att, t0:] += delta
    r = s - np.median(s, axis=0, keepdims=True)
    times = np.arange(0, T, k)
    res = {"rho": rho, "stride": k, "n_eval": len(times)}
    fa, det, delay = 0, [], []
    for v in range(N):
        p = online_p(r[v, times], rng)
        idx = np.where(~np.isnan(p))[0]
        al = [times[idx[i]] for i in ecusum(mix_cal(p[idx]), A)]
        if attack and v < attack[2]:
            post = [a for a in al if a >= attack[0]]
            det.append(bool(post)); delay.append((post[0] - attack[0]) / k if post else None)
            fa += len([a for a in al if a < attack[0]])
        else:
            fa += len(al)
    res["FA_per_1000_evals"] = 1000.0 * fa / (N * len(times))
    if attack:
        d = [x for x in delay if x is not None]
        res["detect_frac"] = float(np.mean(det)); res["median_delay_windows"] = float(np.median(d)) if d else None
    return res

def cramer_root(f=0.3, delta=1.0):
    g = lambda th: f * exp(th * (th + 1) * delta ** 2 / 2) + (1 - f) * exp(th * (th - 1) * delta ** 2 / 2) - 1
    lo, hi = 1e-6, 1.0 - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

if __name__ == "__main__":
    out = {"null": [run(0.6), run(0.9)],
           "step_attack_0.8sd_5hosts_t0=6000": run(0.6, attack=(6000, 0.8, 5), seed=3),
           "step_attack_0.5sd_5hosts_t0=6000": run(0.6, attack=(6000, 0.5, 5), seed=4),
           "cramer_root_periodic_f0.3_delta1": cramer_root()}
    Path(__file__).with_name("results_online.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
