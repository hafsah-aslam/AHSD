"""Check of Proposition 5 (04_mathematical_framework.md): for a p-value p = 1 - F0(score),
(i)  g(p) = ROC'(p) is a valid e-value calibrator: E_H0[g(p)] = 1;
(ii) under H1, E_H1[log g(p)] = KL(H1 || H0 on scores) =: Psi (growth-rate optimal);
(iii) e-CUSUM delay with g is approx log(A)/Psi, faster than the tuning-free mixture calibrator;
(iv) binormal map AUC -> Psi = (Phi^{-1}(AUC))^2 used to turn HoNE AUCs into predicted delays.
Binormal model: benign scores N(0,1), attack scores N(delta,1). Synthetic only.
Usage: python validate_roc_calibrator.py"""
import json
from math import erf, sqrt, log
from pathlib import Path
import numpy as np

Phi = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


def Phi_inv(u):                       # bisection; adequate for this check
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + erf(mid / sqrt(2))) < u else (lo, mid)
    return (lo + hi) / 2


def roc_slope(p, delta):
    """ROC(u) = 1 - Phi(Phi^{-1}(1-u) - delta)  =>  ROC'(u) = exp(delta*z - delta^2/2), z = Phi^{-1}(1-u)."""
    z = np.array([Phi_inv(1 - x) for x in np.atleast_1d(p)])
    return np.exp(delta * z - delta ** 2 / 2)


def mix(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); lp = np.log(p)
    return (1 - p + p * lp) / (p * lp ** 2)


def ecusum_delay(e, A):
    M = 1.0
    for t, x in enumerate(e):
        M = max(M, 1.0) * x
        if M >= A:
            return t + 1
    return None


def main(A=1000.0, n=4000, reps=300):
    rng = np.random.default_rng(0)
    out = {"A": A, "rows": []}
    for delta in (0.25, 0.5, 1.0, 1.5):
        p0 = 1 - Phi(rng.normal(size=n))
        p1 = 1 - Phi(rng.normal(delta, 1, size=n))
        g0, g1 = roc_slope(p0, delta), roc_slope(p1, delta)
        auc = float(Phi(delta / sqrt(2)))
        d_roc, d_mix = [], []
        for r in range(reps):
            x = 1 - Phi(rng.normal(delta, 1, size=3000))
            d_roc.append(ecusum_delay(roc_slope(x[:600], delta), A) or 600)
            d_mix.append(ecusum_delay(mix(x), A) or 3000)
        out["rows"].append({"delta": delta, "AUC": auc, "E_H0_g": float(g0.mean()),
                            "Psi_hat": float(np.log(g1).mean()), "Psi_theory": delta ** 2 / 2,
                            "Psi_from_AUC": Phi_inv(auc) ** 2,
                            "pred_delay_logA_over_Psi": log(A) / (delta ** 2 / 2),
                            "median_delay_roc_calibrator": float(np.median(d_roc)),
                            "D_mix": float(np.log(mix(p1)).mean()),
                            "median_delay_mixture": float(np.median(d_mix))})
    # HoNE-implied predictions (binormal assumption), from stored per-class best AUCs
    hone = {"D1 Infiltration (best 0.60)": 0.60, "D1 Bot (best 0.69)": 0.69, "D3 scanning (best 0.63)": 0.63,
            "rolling family AUC 0.75": 0.75, "D3 Backdoor natural (best 0.98)": 0.98}
    out["hone_predicted_delay_windows_A1000"] = {k: round(log(A) / max(Phi_inv(v) ** 2, 1e-9), 1) for k, v in hone.items()}
    Path(__file__).with_name("results_roc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
