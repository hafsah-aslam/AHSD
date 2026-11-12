"""Phase 6 synthetic falsification suite: covert mission capacity (contribution A) and the
peer-studentized variance-time scan (contribution B). Synthetic data only; no HoNE code or data.

Run:  python capacity.py [--quick]   ->  results_capacity.json

PRE-REGISTERED SUPERIORITY CRITERIA for PS-VTS (fixed before the first run; see 06_*.md section 8):
  S1 validity:   nominal false-alarm probability over the monitoring horizon <= alpha + 2 MC s.e. in every
                 benign scenario that satisfies its stated assumption (peer exchangeability).
  S2 capacity:   at matched *empirical* false-alarm rate (every baseline re-calibrated on the same benign
                 scenario, i.e. given its best possible threshold), the median covert capacity (critical
                 cumulative progress) of PS-VTS is <= 0.75 x that of EVERY baseline for long attacks
                 (L >= 256) in scenarios B0 and B1, and not more than 1.10 x for short attacks (L = 32).
  If either criterion fails the result is reported as "no superiority" for that condition.
"""
import json, math, sys
import numpy as np

QUICK = "--quick" in sys.argv
ALPHA = 0.05

# ----------------------------------------------------------------------------- utilities
def fgn(rng, n_series, T, H):
    """Exact fractional Gaussian noise (unit variance) by Davies-Harte circulant embedding."""
    k = np.arange(T + 1)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([g, g[-2:0:-1]])                       # length 2T
    lam = np.fft.fft(c).real
    lam[lam < 0] = 0.0
    M = len(c)
    Z = rng.standard_normal((n_series, M)) + 1j * rng.standard_normal((n_series, M))
    X = np.fft.fft(np.sqrt(lam / M) * Z, axis=1)
    return X.real[:, :T]

def fgn_cov(T, H):
    k = np.arange(T)
    g = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    return g[np.abs(k[:, None] - k[None, :])]

def norm_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

def norm_isf(p):
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_sf(mid) > p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

def _betacf(a, b, x):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300); h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1.0) < 1e-14: break
    return h

def betainc(a, b, x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b

def t_sf(x, df):
    if x <= 0: return 1.0 - t_sf(-x, df)
    return 0.5 * betainc(df / 2.0, 0.5, df / (df + x * x))

def t_isf(p, df):
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_sf(mid, df) > p: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

# ----------------------------------------------------------------------------- E1: exact capacity
def exp_exact(rng):
    """Check G*(alpha,beta) = (z_{1-alpha} + z_beta) * SD_benign(1'X) against the Neyman-Pearson oracle,
    for several benign covariances; and that a non-optimal (constant) shape of equal progress is more
    detectable. Also checks that practical detectors cannot beat the converse under the optimal shape."""
    T = 256 if QUICK else 512
    reps = 4000 if QUICK else 20000
    out = {}
    covs = {
        "iid": np.eye(T),
        "ar1_rho0.8": 0.8 ** np.abs(np.subtract.outer(np.arange(T), np.arange(T))) ,
        "fgn_H0.7": fgn_cov(T, 0.7),
        "fgn_H0.9": fgn_cov(T, 0.9),
        "local_level_q0.01": np.eye(T) + 0.01 * np.minimum.outer(np.arange(1, T + 1), np.arange(1, T + 1)),
    }
    for name, S in covs.items():
        L = np.linalg.cholesky(S + 1e-10 * np.eye(T))
        one = np.ones(T)
        sd_prog = math.sqrt(one @ S @ one)
        res = {"SD_progress": sd_prog}
        for beta in (0.1, 0.5, 0.9):
            zb = norm_isf(1 - beta)                         # Phi^{-1}(beta)
            za = norm_isf(ALPHA)
            G = (za + zb) * sd_prog
            a_opt = G * (S @ one) / (one @ S @ one)         # covariance-shaped optimal attack
            a_con = np.full(T, G / T)                       # constant spread, same progress
            Sinv_a_opt = np.linalg.solve(S, a_opt)
            Sinv_a_con = np.linalg.solve(S, a_con)
            Z = rng.standard_normal((reps, T)) @ L.T       # benign draws
            # NP oracle statistic for each attack, threshold at level alpha under H0 (exact Gaussian)
            det = {}
            for lab, a, w in (("optimal_shape", a_opt, Sinv_a_opt), ("constant_shape", a_con, Sinv_a_con)):
                sd_stat = math.sqrt(a @ w)
                thr = za * sd_stat
                det[lab] = float(np.mean((Z + a) @ w > thr))
            # practical detector: one-sided test on the total sum (knows nothing about shape)
            thr_sum = za * sd_prog
            det["sum_test_optimal_shape"] = float(np.mean((Z + a_opt).sum(1) > thr_sum))
            res[f"beta={beta}"] = {"G*": G, "predicted_detection": beta, **det}
        out[name] = res
    # scaling table (analytic): G*(T) for fGn = (z_a + z_b) T^H
    out["scaling_G*_over_iid"] = {f"H={H}": {str(Tn): float(Tn ** H / Tn ** 0.5) for Tn in (64, 1024, 10080, 100000)}
                                  for H in (0.5, 0.7, 0.8, 0.9)}
    return out

# ----------------------------------------------------------------------------- panel generator
def make_panel(rng, n, Ttot, scen, He=0.75, Hc=0.9, se=0.5, sc=1.0, heavy=False, hetero=False):
    c = sc * fgn(rng, 1, Ttot, Hc)[0]
    if heavy:
        # heavy-tailed idiosyncratic noise: fGn-correlated Student-t(3)-like via variance mixing per block
        E = fgn(rng, n, Ttot, He)
        mix = np.sqrt(3.0 / rng.chisquare(3, size=(n, Ttot // 16 + 1))).repeat(16, axis=1)[:, :Ttot]
        E = E * mix / math.sqrt(3.0)
    else:
        E = fgn(rng, n, Ttot, He)
    E = se * E
    if hetero:
        E[0] *= 2.0                                        # target host twice as variable as its peers
    X = c[None, :] + E + rng.normal(0, 1, size=(n, 1))     # host offsets
    t = np.arange(Ttot)
    if scen in ("B1", "B2"):
        mod = 1.0 + 0.6 * np.sin(2 * np.pi * t / 512.0)    # common diurnal variance modulation
        X = (X - X.mean(1, keepdims=True)) * mod[None, :] + X.mean(1, keepdims=True)
        t0 = rng.integers(Ttot // 3, Ttot - 200)
        X[:, t0:] += 3.0                                   # common benign level shift (outage / change)
    if scen == "B2":
        X[0] += np.cumsum(rng.normal(0, 0.02, Ttot))       # idiosyncratic benign drift on target (violates exchangeability)
    return X

# ----------------------------------------------------------------------------- detectors
SCALES = [16, 32, 64, 128, 256, 512, 1024]

def windows(Nmon, s):
    step = max(1, s // 4)
    ends = np.arange(s, Nmon + 1, step)
    return ends

def n_tests(Nmon):
    return sum(len(windows(Nmon, s)) for s in SCALES)

class Ctx:
    def __init__(self, Nref, Nmon, n):
        self.Nref, self.Nmon, self.n = Nref, Nmon, n
        self.M = n_tests(Nmon)
        self.t_thr = t_isf(ALPHA / self.M, n - 2)
        self.z_thr = norm_isf(ALPHA / self.M)

def win_sums(x, s, ends):
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return cs[ends] - cs[ends - s]

def overlap(onset, L, s, ends):
    lo = np.maximum(ends - s, onset); hi = np.minimum(ends, onset + L)
    return np.clip(hi - lo, 0, None)

def psvts_scores(X, ctx):
    """Peer-studentized variance-time scan for host 0. Returns per-window (base, scale) such that the
    studentized statistic is (base + a*overlap)/scale and the alarm rule is stat >= t_thr."""
    Xm = X[:, ctx.Nref:]
    out = []
    for s in SCALES:
        ends = windows(ctx.Nmon, s)
        S = np.stack([win_sums(Xm[u], s, ends) for u in range(X.shape[0])])
        others = S[1:]
        mu, sd = others.mean(0), others.std(0, ddof=1)
        scale = sd * math.sqrt(1 + 1 / (X.shape[0] - 1))
        out.append((s, ends, S[0] - mu, scale))
    return out

def own_vts_prep(x, ctx):
    """Own-history VTS: estimate variance-time curve V(s)=V1 s^{2H} from the reference by aggregated
    variance at scales 1..64, extrapolate; Gaussian Bonferroni threshold."""
    ref = x[:ctx.Nref]
    mu = ref.mean()
    ms, vs = [], []
    for m in (1, 2, 4, 8, 16, 32, 64):
        nb = len(ref) // m
        b = ref[:nb * m].reshape(nb, m).sum(1)
        ms.append(m); vs.append(b.var(ddof=1))
    slope, icpt = np.polyfit(np.log(ms), np.log(vs), 1)
    Hhat = min(max(slope / 2, 0.5), 0.99)
    V1 = math.exp(icpt)
    xm = x[ctx.Nref:]
    out = []
    for s in SCALES:
        ends = windows(ctx.Nmon, s)
        S = win_sums(xm - mu, s, ends)
        out.append((s, ends, S, math.sqrt(V1) * s ** Hhat * np.ones_like(S)))
    return out, Hhat

def crit_from_linear(prep, onset, L, thr, shape="step"):
    """Minimal step amplitude a such that some window overlapping the attack satisfies
    (base + a*ov)/scale >= thr. Windows with no overlap that already alarm are false alarms (ignored)."""
    best = np.inf
    for s, ends, base, scale in prep:
        if shape == "step":
            ov = overlap(onset, L, s, ends).astype(float)
        else:                                               # ramp a*(t-onset)/L over [onset, onset+L), then held
            ov = ramp_overlap(onset, L, s, ends)
        m = ov > 0
        if not np.any(m): continue
        a = (thr * scale[m] - base[m]) / ov[m]
        a = np.maximum(a, 0.0)
        best = min(best, float(a.min()))
    return best

def ramp_overlap(onset, L, s, ends):
    prof = np.zeros(ends.max() + 1)
    t = np.arange(len(prof))
    seg = (t >= onset) & (t < onset + L)
    prof[seg] = (t[seg] - onset + 1) / L
    cs = np.concatenate([[0.0], np.cumsum(prof)])
    return cs[ends] - cs[ends - s]

def max_stat_linear(prep, thr):
    return max(float(np.max(base / scale)) for s, ends, base, scale in prep) / thr

# sequential baselines (bisection on amplitude)
def cusum_alarm(z, k, h):
    S = 0.0
    for t, v in enumerate(z):
        S = max(0.0, S + v - k)
        if S >= h: return t
    return -1

def cusum_stat_path(z, k):
    S, out = 0.0, np.empty(len(z))
    for t, v in enumerate(z):
        S = max(0.0, S + v - k); out[t] = S
    return out

def ewma_path(x, ctx, lam_base=0.005, lam=0.05):
    ref = x[:ctx.Nref]
    m, sd = ref.mean(), ref.std()
    e = 0.0
    out = np.empty(ctx.Nmon)
    for i, v in enumerate(x[ctx.Nref:]):
        r = (v - m) / sd
        e = (1 - lam) * e + lam * r
        out[i] = e / math.sqrt(lam / (2 - lam))
        m = (1 - lam_base) * m + lam_base * v              # adaptive baseline (absorbs slow changes)
    return out

def mix_cal(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return (1 - p + p * np.log(p)) / (p * np.log(p) ** 2)

def conformal_path(x, ctx, rng):
    """Online full-past smoothed conformal p-values + mixture-calibrated e-CUSUM (Theorem 1a design,
    stride 1). Returns log W path over the monitoring period."""
    ref = list(np.sort(x[:ctx.Nref]))
    import bisect
    W, out = 1.0, np.empty(ctx.Nmon)
    for i, v in enumerate(x[ctx.Nref:]):
        n = len(ref)
        gt = n - bisect.bisect_right(ref, v)
        p = (gt + rng.uniform()) / (n + 1)
        W = max(W, 1.0) * float(mix_cal(np.array([p]))[0])
        out[i] = math.log(W)
        bisect.insort(ref, v)
    return out

def seq_detect(path_fn, x, onset_abs, thr):
    path = path_fn(x)
    idx = np.where(path >= thr)[0]
    return len(idx) > 0 and idx[0] + 0 >= onset_abs   # first alarm at/after onset (monitoring-relative)

def bisect_amp(make_x, detect, hi0=8.0, iters=12):
    """Smallest amplitude in [0, hi] at which detect(make_x(a)) is True (assumes monotone)."""
    if not detect(make_x(hi0)): return np.inf
    lo, hi = 0.0, hi0
    if detect(make_x(0.0)): return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if detect(make_x(mid)): hi = mid
        else: lo = mid
    return hi

# ----------------------------------------------------------------------------- E2-E5: detector comparison
def residual_vs_median(X):
    return X[0] - np.median(X[1:], axis=0)

def run_comparison(rng, scen, n=20, reps_null=None, reps_att=None, Ls=(32, 256, 1024), shapes=("step",),
                   heavy=False, hetero=False, coord_frac=0.0, include_seq=True):
    reps_null = reps_null or (60 if QUICK else 200)
    reps_att = reps_att or (30 if QUICK else 100)
    Nref, Nmon = 1024, 2048
    ctx = Ctx(Nref, Nmon, n)
    Ttot = Nref + Nmon
    # ---- calibration on benign runs: record each detector's max normalised statistic per run
    stats = {k: [] for k in ("PS-VTS", "own-VTS", "median-resid+VTS", "CUSUM-iid", "EWMA-adaptive",
                             "conformal-eCUSUM", "median-resid+CUSUM")}
    k_cusum = 0.25
    for r in range(reps_null):
        X = make_panel(rng, n, Ttot, scen, heavy=heavy, hetero=hetero)
        stats["PS-VTS"].append(max_stat_linear(psvts_scores(X, ctx), ctx.t_thr))
        prep, _ = own_vts_prep(X[0], ctx)
        stats["own-VTS"].append(max_stat_linear(prep, ctx.z_thr))
        prep, _ = own_vts_prep(residual_vs_median(X), ctx)
        stats["median-resid+VTS"].append(max_stat_linear(prep, ctx.z_thr))
        if include_seq:
            for lab, x in (("CUSUM-iid", X[0]), ("median-resid+CUSUM", residual_vs_median(X))):
                ref = x[:Nref]; z = (x[Nref:] - ref.mean()) / ref.std()
                stats[lab].append(float(cusum_stat_path(z, k_cusum).max()))
            stats["EWMA-adaptive"].append(float(ewma_path(X[0], ctx).max()))
            stats["conformal-eCUSUM"].append(float(conformal_path(X[0], ctx, rng).max()))
    # nominal thresholds (i.i.d.-calibrated, the way each method would be set if its assumption held)
    nominal = {"PS-VTS": 1.0, "own-VTS": 1.0, "median-resid+VTS": 1.0,
               "CUSUM-iid": NOMINAL["cusum_h"], "median-resid+CUSUM": NOMINAL["cusum_h"],
               "EWMA-adaptive": NOMINAL["ewma_h"], "conformal-eCUSUM": math.log(Nmon / ALPHA)}
    res = {"scenario": scen, "n_peers": n, "heavy": heavy, "hetero": hetero, "coord_frac": coord_frac,
           "nominal_FA": {}, "oracle_threshold": {}}
    for kname, v in stats.items():
        if not v: continue
        v = np.array(v)
        res["nominal_FA"][kname] = float(np.mean(v >= nominal[kname]))
        res["oracle_threshold"][kname] = float(np.quantile(v, 1 - ALPHA))   # matched empirical FA = alpha
    # ---- attacks: critical cumulative progress G = a_crit * (progress per unit amplitude)
    cap = {}
    for shape in shapes:
        for L in Ls:
            key = f"{shape}_L{L}"
            crit = {k: [] for k in res["oracle_threshold"]}
            for r in range(reps_att):
                X = make_panel(rng, n, Ttot, scen, heavy=heavy, hetero=hetero)
                onset = int(rng.integers(0, Nmon - L))
                prog_per_amp = L if shape == "step" else (L + 1) / 2.0 + 0.0
                # linear detectors: exact critical amplitude (oracle thresholds)
                thr = res["oracle_threshold"]
                if coord_frac > 0:
                    kc = max(1, int(round(coord_frac * n)))      # attacker also shifts kc-1 peers identically
                    def mk(a, X=X, onset=onset):
                        Y = X.copy()
                        prof = np.zeros(Ttot); prof[Nref + onset:Nref + onset + L] = 1.0
                        Y[:kc] += a * prof
                        return Y
                    crit["PS-VTS"].append(bisect_amp(mk, lambda Y: detect_linear(psvts_scores(Y, ctx), ctx.t_thr * thr["PS-VTS"], onset, L)) * prog_per_amp)
                    prep_fn = lambda Y: own_vts_prep(residual_vs_median(Y), ctx)[0]
                    crit["median-resid+VTS"].append(bisect_amp(mk, lambda Y: detect_linear(prep_fn(Y), ctx.z_thr * thr["median-resid+VTS"], onset, L)) * prog_per_amp)
                    continue
                crit["PS-VTS"].append(crit_from_linear(psvts_scores(X, ctx), onset, L, ctx.t_thr * thr["PS-VTS"], shape) * prog_per_amp)
                prep, _ = own_vts_prep(X[0], ctx)
                crit["own-VTS"].append(crit_from_linear(prep, onset, L, ctx.z_thr * thr["own-VTS"], shape) * prog_per_amp)
                # median residual: attack on host 0 passes 1:1 into the residual (median of peers unaffected)
                prep, _ = own_vts_prep(residual_vs_median(X), ctx)
                crit["median-resid+VTS"].append(crit_from_linear(prep, onset, L, ctx.z_thr * thr["median-resid+VTS"], shape) * prog_per_amp)
                if include_seq and shape == "step":
                    prof = np.zeros(Ttot); prof[Nref + onset:Nref + onset + L] = 1.0
                    for lab, base in (("CUSUM-iid", X[0]), ("median-resid+CUSUM", residual_vs_median(X))):
                        ref = base[:Nref]; m0, s0 = ref.mean(), ref.std()
                        h = thr[lab]
                        def det(a, base=base):
                            z = (base[Nref:] + a * prof[Nref:] - m0) / s0
                            p = cusum_stat_path(z, k_cusum); idx = np.where(p >= h)[0]
                            return len(idx) > 0 and idx[0] >= onset
                        crit[lab].append(bisect_amp(lambda a: a, det) * prog_per_amp)
                    hE = thr["EWMA-adaptive"]
                    def detE(a):
                        p = ewma_path(X[0] + a * prof, ctx); idx = np.where(p >= hE)[0]
                        return len(idx) > 0 and idx[0] >= onset
                    crit["EWMA-adaptive"].append(bisect_amp(lambda a: a, detE) * prog_per_amp)
                    if not QUICK or r < 10:
                        hC = thr["conformal-eCUSUM"]
                        seed = int(rng.integers(1 << 30))
                        def detC(a):
                            p = conformal_path(X[0] + a * prof, ctx, np.random.default_rng(seed)); idx = np.where(p >= hC)[0]
                            return len(idx) > 0 and idx[0] >= onset
                        crit["conformal-eCUSUM"].append(bisect_amp(lambda a: a, detC, iters=8) * prog_per_amp)
            cap[key] = {k: (float(np.median(v)) if len(v) else None) for k, v in crit.items()}
            # oracle bound for the idiosyncratic component only (what a perfect panel method faces):
            # G* = (z_alpha + z_0.5) * se * L^He  for a known onset & shape; scan pays extra
            if shape == "step":
                cap[key]["oracle_idio_G*"] = norm_isf(ALPHA) * 0.5 * L ** 0.75
                cap[key]["oracle_total_G*"] = norm_isf(ALPHA) * math.sqrt(0.25 * L ** 1.5 + 1.0 * L ** 1.8)
    res["median_critical_progress"] = cap
    return res

def detect_linear(prep, thr, onset, L):
    for s, ends, base, scale in prep:
        ov = overlap(onset, L, s, ends)
        m = ov > 0
        if np.any((base[m] / scale[m]) >= thr): return True
    return False

NOMINAL = {}

def calibrate_nominal(rng):
    """i.i.d. N(0,1)-calibrated thresholds for CUSUM (k=0.25) and EWMA over Nmon=2048 at FA=alpha."""
    Nmon = 2048; reps = 300 if QUICK else 1000
    cm, em = [], []
    class C: Nref = 1024; Nmon = 2048
    for _ in range(reps):
        z = rng.standard_normal(1024 + Nmon)
        cm.append(cusum_stat_path(z[1024:], 0.25).max())
        em.append(ewma_path(z, C).max())
    NOMINAL["cusum_h"] = float(np.quantile(cm, 1 - ALPHA))
    NOMINAL["ewma_h"] = float(np.quantile(em, 1 - ALPHA))
    return dict(NOMINAL)

# ----------------------------------------------------------------------------- calibration frontier
def exp_frontier(rng):
    """Truth: fGn H=0.8 (unit var). Detector: scan calibrated assuming H_det. Measure FA and capacity
    (median critical progress of a step of length 512 at the *nominal* threshold)."""
    Nmon = 2048; reps = 150 if QUICK else 500
    class C: pass
    ctx = C(); ctx.Nmon = Nmon
    M = n_tests(Nmon); z = norm_isf(ALPHA / M)
    out = {}
    X = fgn(rng, reps, Nmon, 0.8)
    for Hd in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        fa, crit = 0, []
        for r in range(reps):
            prep = []
            for s in SCALES:
                ends = windows(Nmon, s)
                prep.append((s, ends, win_sums(X[r], s, ends), s ** Hd * np.ones(len(ends))))
            fa += max_stat_linear(prep, z) >= 1.0
            onset = int(rng.integers(0, Nmon - 512))
            crit.append(crit_from_linear(prep, onset, 512, z) * 512)
        out[f"H_det={Hd}"] = {"FA": fa / reps, "median_capacity_L512": float(np.median(crit)),
                              "predicted_capacity_scale_512^Hdet": 512 ** Hd}
    out["truth_H"] = 0.8
    return out


# ----------------------------------------------------------------------------- A3: unknown offset + finite reference
def exp_nuisance(rng):
    """Capacity when the host offset is unknown and only a finite own-history reference of length Nref is
    available. Attack supported on the monitoring window W (length s) only. The maximal-invariant (UMPI)
    oracle has non-centrality d^2 = a' P a, P = S^-1 - S^-1 1 1' S^-1 / (1' S^-1 1); maximising w'a over
    a supported on W with d <= d0 gives G* = d0 * sqrt(w' (P_WW)^-1 w). Compared with the known-offset
    value d0 * SD(w'X) = d0 * s^H. Monte Carlo check of the UMPI detection rate at one point."""
    Nref = 256
    za = norm_isf(ALPHA)
    out = {"Nref": Nref}
    for H in (0.5, 0.8):
        rows = {}
        for sw in (16, 64, 256, 1024, 2048):
            T = Nref + sw
            S = fgn_cov(T, H) if H != 0.5 else np.eye(T)
            Si = np.linalg.inv(S)
            one = np.ones(T)
            P = Si - np.outer(Si @ one, Si @ one) / (one @ Si @ one)
            Pw = P[Nref:, Nref:]
            w = np.ones(sw)
            G_inv = za * math.sqrt(w @ np.linalg.solve(Pw, w))
            G_known = za * sw ** H
            rows[str(sw)] = {"G*_unknown_offset": G_inv, "G*_known_offset": G_known, "ratio": G_inv / G_known}
            if sw == 256:
                a_w = np.linalg.solve(Pw, w); a_w *= G_inv / (w @ a_w)
                a = np.concatenate([np.zeros(Nref), a_w])
                Lc = np.linalg.cholesky(S + 1e-10 * np.eye(T))
                reps = 4000 if QUICK else 20000
                Z = rng.standard_normal((reps, T)) @ Lc.T + rng.normal(0, 5, size=(reps, 1))   # random offset
                stat = (Z + a) @ (P @ a)
                thr = za * math.sqrt(a @ P @ a)
                rows[str(sw)]["MC_detection_at_G*(pred 0.5)"] = float(np.mean(stat > thr))
                rows[str(sw)]["MC_FA(pred 0.05)"] = float(np.mean(Z @ (P @ a) > thr))
        out[f"H={H}"] = rows
    return out

def main():
    rng = np.random.default_rng(20261112)
    out = {"quick": QUICK, "alpha": ALPHA}
    out["E1_exact_capacity"] = exp_exact(rng)
    print("E1 done", flush=True)
    out["A3_nuisance"] = exp_nuisance(rng)
    print("A3 done", flush=True)
    out["nominal_thresholds"] = calibrate_nominal(rng)
    out["E3_frontier"] = exp_frontier(rng)
    print("E3 done", flush=True)
    out["E4_comparison"] = {}
    for scen in ("B0", "B1", "B2"):
        out["E4_comparison"][scen] = run_comparison(rng, scen, shapes=("step", "ramp") if scen != "B2" else ("step",))
        print("E4", scen, "done", flush=True)
    out["E5_falsification"] = {
        "heavy_tailed": run_comparison(rng, "B0", heavy=True, Ls=(256,), include_seq=False),
        "hetero_target": run_comparison(rng, "B0", hetero=True, Ls=(256,), include_seq=False),
        "small_role_n5": run_comparison(rng, "B0", n=5, Ls=(256,), include_seq=False),
        "coordinated_10pct": run_comparison(rng, "B0", Ls=(256,), include_seq=False, coord_frac=0.10),
        "coordinated_30pct": run_comparison(rng, "B0", Ls=(256,), include_seq=False, coord_frac=0.30),
        "coordinated_50pct": run_comparison(rng, "B0", Ls=(256,), include_seq=False, coord_frac=0.50),
    }
    print("E5 done", flush=True)
    with open("results_capacity.json" if not QUICK else "results_capacity_quick.json", "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
