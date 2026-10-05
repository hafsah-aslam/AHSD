#!/usr/bin/env python
"""P1b report and gate G1 (AMENDMENT_02), from results JSON only.

Writes P1B_REPORT.md, GATE_G1_DECISION.md, results/p1b/summary.json and report/p1b/*.
Fails closed if an experiment mixes code commits or ran from a dirty tree.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_p1 as R1  # noqa: E402
from nids import metrics  # noqa: E402
from nids.predictors import EXPECTED_SIGN  # noqa: E402
from nids.provenance import ROOT, provenance, rq_sha256, write_json  # noqa: E402

N_BOOT, BOOT_SEED = 2000, 0
WITHIN = {"D1_family_gr": "D1 families", "D2_gr": "D2 grouped_random", "D3_gr": "D3 grouped_random"}
PREDS = ["D_pred", "wasserstein", "mahalanobis", "stationarity"]


def _load(pattern: str) -> list[dict]:
    out = []
    for p in sorted(ROOT.glob(pattern)):
        r = json.loads(p.read_text())
        if not r.get("skipped") and not r.get("smoke"):
            out.append(r)
    return out


def _one_commit(runs: list[dict], name: str) -> str:
    cs = {r["provenance"]["code_commit"] for r in runs}
    if len(cs) != 1 or any(r["provenance"]["code_dirty"] for r in runs):
        raise SystemExit(f"{name}: runs from {len(cs)} code commits or a dirty tree: {sorted(cs)}")
    return cs.pop()


def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(stats.spearmanr(x, y)[0]) if len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else float("nan")


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


# ------------------------------------------------------------------ A2.2
def within_dataset(points: list[dict]) -> dict:
    by = {ev: [p for p in points if p["eval_id"] == ev] for ev in WITHIN}
    rng = np.random.default_rng(BOOT_SEED)
    out = {}
    for k in PREDS:
        sign = EXPECTED_SIGN[k]
        per = {ev: sign * _rho([p[k] for p in pts], [p["auc_stress"]["mean"] for p in pts]) for ev, pts in by.items()}
        boots = []
        for _ in range(N_BOOT):
            vals = []
            for pts in by.values():
                i = rng.integers(0, len(pts), len(pts))
                vals.append(sign * _rho([pts[j][k] for j in i], [pts[j]["auc_stress"]["mean"] for j in i]))
            if all(np.isfinite(vals)):
                boots.append(np.mean(vals))
        lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
        out[k] = {"per_dataset_oriented_rho": per, "mean_oriented_rho": float(np.mean(list(per.values()))),
                  "ci95": [float(lo), float(hi)], "n_boot_valid": len(boots),
                  "n_per_dataset": {ev: len(p) for ev, p in by.items()}}
    return out


# ------------------------------------------------------------------ A2.3
def s2dir_analysis(runs: list[dict], points: list[dict]) -> dict:
    by_fold = defaultdict(list)
    for r in runs:
        by_fold[(r["eval_id"], r["held_out"])].append(r)
    rows = []
    for (ev, cls), rs in sorted(by_fold.items()):
        q = [r["s2dir"]["Q_benign_halves"] for r in rs]
        norm = float(np.std(q, ddof=1)) if len(q) > 1 else float("nan")
        vals = [r["s2dir"]["raw"] / norm for r in rs]
        p = next(x for x in points if x["eval_id"] == ev and x["class"] == cls)
        rows.append({"eval_id": ev, "class": cls, "seeds": sorted(r["seed"] for r in rs),
                     "s2dir": float(np.mean(vals)), "s2dir_seed_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
                     "normaliser_std_Qb": norm, "auc_stress": p["auc_stress"]["mean"], "mahalanobis": p["mahalanobis"],
                     "D_pred": p["D_pred"],
                     "repro_max_abs_dev": max(max(r["reproduction"]["abs_dev"].values()) for r in rs)})
    out = {"rows": rows}
    for name, sel in (("pooled", list(WITHIN)), *[(ev, [ev]) for ev in WITHIN]):
        rr = [r for r in rows if r["eval_id"] in sel]
        out[name] = {k: metrics.spearman_bootstrap([r[k] for r in rr], [r["auc_stress"] for r in rr], n_boot=N_BOOT)
                     for k in ("s2dir", "mahalanobis", "D_pred")}
    return out


# ------------------------------------------------------------------ A2.5
def d3_nn(runs: list[dict], p1_runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[r["held_out"]].append(r)
    p1 = {(r["held_out"], r["seed"]): r for r in p1_runs}
    rows = []
    for cls, rs in sorted(by.items()):
        aucs = [r["measured"]["stress"]["auc"] for r in sorted(rs, key=lambda r: r["seed"])]
        repro = [abs(r["measured"]["stress"]["auc"] - p1[(cls, r["seed"])]["measured"]["stress"]["auc"])
                 for r in rs if (cls, r["seed"]) in p1]
        rows.append({"class": cls, "seeds": sorted(r["seed"] for r in rs), **metrics.mean_ci(aucs),
                     "aucs": aucs, "frac_below_0.5": float(np.mean(np.array(aucs) < 0.5)),
                     "p1_seed_repro_max_abs_dev": max(repro) if repro else None})
    return {"rows": rows}


# ------------------------------------------------------------------ A2.6 / G1
def g1(runs: list[dict]) -> dict:
    by = defaultdict(list)
    for r in runs:
        by[(r["dataset"], r["model"]["id"])].append(r)
    gammas = sorted({r["model"]["gamma"] for r in runs if r["model"]["variant"] == "frozen"})
    out = {"datasets": {}, "gammas": gammas}
    for ds in sorted({r["dataset"] for r in runs}):
        m = lambda mid, key: [x["test"][key] for x in by[(ds, mid)]]  # noqa: E731
        sweep = []
        for g in gammas:
            mid = f"frozen_g{g:g}"
            sweep.append({"gamma": g, "d_prime": metrics.mean_ci(m(mid, "d_prime_stress")),
                          "macro_f1": metrics.mean_ci(m(mid, "macro_f1_pct")), "n_seeds": len(by[(ds, mid)])})
        rho_a = _rho([s["gamma"] for s in sweep], [s["d_prime"]["mean"] for s in sweep])
        s1_ids = sorted({mid for (d, mid) in by if d == ds and mid.startswith("S1_")})
        s1_val = {mid: float(np.mean([x["best_val_macro_f1"] for x in by[(ds, mid)]])) for mid in s1_ids}

        def key(mid):
            spec = by[(ds, mid)][0]["model"]
            return (-s1_val[mid], spec["tau_mult"], spec["kappa"])
        sel = min(s1_ids, key=key) if s1_ids else None
        fixed = "frozen_g0"
        d_fix, f_fix = np.mean(m(fixed, "d_prime_stress")), np.mean(m(fixed, "macro_f1_pct"))
        d_s1, f_s1 = (np.mean(m(sel, "d_prime_stress")), np.mean(m(sel, "macro_f1_pct"))) if sel else (np.nan, np.nan)
        crit_a = bool(rho_a <= -0.8)
        crit_b = bool(d_s1 >= 0.8 * d_fix and (f_fix - f_s1) < 1.0)
        ad = by[(ds, "adaptive")]
        # descriptive per-class collapse
        coll = defaultdict(lambda: defaultdict(list))
        for g in gammas:
            for x in by[(ds, f"frozen_g{g:g}")]:
                for c, v in x["per_class"].items():
                    coll[c][g].append(v["d_prime_stress"])
        stat0 = defaultdict(list)
        for x in by[(ds, fixed)]:
            for c, v in x["per_class"].items():
                stat0[c].append(v["stationarity"])
        collapse = []
        for c in sorted(coll):
            d0 = np.mean(coll[c][0]) if coll[c][0] else np.nan
            collapse.append({"class": c, "stationarity_g0": float(np.mean(stat0[c])) if stat0[c] else np.nan,
                             "d0": float(d0),
                             "ratio": {f"{g:g}": float(np.mean(coll[c][g]) / d0) if coll[c][g] and d0 else np.nan
                                       for g in gammas}})
        out["datasets"][ds] = {
            "sweep": sweep, "rho_gamma_dprime": rho_a, "criterion_a": crit_a,
            "s1_grid_mean_val_macro_f1": s1_val, "s1_selected": sel,
            "s1_selected_test_d_prime": metrics.mean_ci(m(sel, "d_prime_stress")) if sel else None,
            "s1_selected_test_macro_f1": metrics.mean_ci(m(sel, "macro_f1_pct")) if sel else None,
            "fixed_test_d_prime": metrics.mean_ci(m(fixed, "d_prime_stress")),
            "fixed_test_macro_f1": metrics.mean_ci(m(fixed, "macro_f1_pct")),
            "s1_dprime_ratio": float(d_s1 / d_fix) if d_fix else np.nan, "s1_macro_f1_drop": float(f_fix - f_s1),
            "criterion_b": crit_b, "both": bool(crit_a and crit_b),
            "adaptive": {"gamma_final": metrics.mean_ci([x["gamma_final"] for x in ad]),
                         "gamma_by_epoch_mean": np.mean([x["gamma_by_epoch"] for x in ad], axis=0).tolist() if ad else [],
                         "test_d_prime": metrics.mean_ci([x["test"]["d_prime_stress"] for x in ad]),
                         "test_macro_f1": metrics.mean_ci([x["test"]["macro_f1_pct"] for x in ad])},
            "collapse": collapse,
            "collapse_rho_stationarity_vs_ratio_g0.2": _rho([c["stationarity_g0"] for c in collapse],
                                                           [c["ratio"].get("0.2", np.nan) for c in collapse])
                                                      if len(collapse) > 2 else float("nan"),
        }
    n_both = sum(v["both"] for v in out["datasets"].values())
    out["datasets_passing"] = n_both
    out["outcome"] = "GO" if n_both >= 2 else "STOP"
    return out


def sweep_figure(G: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"D1": ("#2a78d6", "o"), "D2": ("#eb6834", "s"), "D3": ("#1baf7a", "^")}
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for ds, v in G["datasets"].items():
        col, mk = style.get(ds, ("#6b6a63", "o"))
        x = [s["gamma"] for s in v["sweep"]]
        y = [s["d_prime"]["mean"] for s in v["sweep"]]
        e = [s["d_prime"]["ci95"] if np.isfinite(s["d_prime"]["ci95"]) else 0 for s in v["sweep"]]
        ax.errorbar(x, y, yerr=e, fmt=f"-{mk}", color=col, ms=6, mec="white", mew=1.2, lw=2, capsize=0,
                    label=f"{ds} (ρ = {v['rho_gamma_dprime']:.2f})")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xlabel("frozen γ (symlog)", fontsize=9)
    ax.set_ylabel("test d′ (stress)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    p1_runs, _ = R1.load(ROOT / "results/p1")
    points = R1.aggregate(p1_runs)
    s2r = _load("results/p1b/s2dir/*/*/seed*.json")
    nnr = _load("results/p1b/d3_nn/D3_nn/*/seed*.json")
    g1r = _load("results/p1b/g1/*/*/seed*.json")
    commits = {"P1b-S2dir": _one_commit(s2r, "S2-dir") if s2r else None,
               "P1b-D3nn": _one_commit(nnr, "D3 natural_novelty") if nnr else None,
               "P1b-G1": _one_commit(g1r, "G1") if g1r else None}
    p1_summary = json.loads((ROOT / "results/p1/summary.json").read_text())
    gate1 = json.loads((ROOT / "results/p1/gate.json").read_text())
    S = {"within_dataset": within_dataset(points),
         "pooled_rho_p1": {k: p1_summary["correlations"]["b_pooled"][k] for k in PREDS},
         "s2dir": s2dir_analysis(s2r, points) if s2r else None,
         "d3_nn": d3_nn(nnr, [r for r in p1_runs if r["eval_id"] == "D3_nn"]) if nnr else None,
         "g1": g1(g1r) if g1r else None,
         "code_commits": commits, "n_runs": {"s2dir": len(s2r), "d3_nn": len(nnr), "g1": len(g1r)},
         "rq_sha256": rq_sha256(),
         "rq_sha256_in_runs": sorted({r["provenance"]["rq_sha256"] for r in s2r + nnr + g1r}),
         "p1_gate": {"outcome": gate1["outcome"], "primary": gate1["primary_oriented"], "raw": gate1["raw_sign_variant"]},
         "provenance": provenance()}
    write_json(ROOT / "results/p1b/summary.json", S)
    if S["g1"]:
        sweep_figure(S["g1"], ROOT / "report/p1b/g1_gamma_sweep.pdf")
    (ROOT / "P1B_REPORT.md").write_text(render(S))
    if S["g1"]:
        (ROOT / "GATE_G1_DECISION.md").write_text(render_gate(S))
    print(f"G1: {S['g1']['outcome'] if S['g1'] else 'no runs'}; wrote P1B_REPORT.md")


def render_gate(S) -> str:
    G = S["g1"]
    L = ["# GATE_G1_DECISION", "", f"**Outcome: {G['outcome']}** ({G['datasets_passing']} of "
         f"{len(G['datasets'])} datasets pass both criteria; GO needs ≥ 2).", "",
         "Rule (AMENDMENT_02 A2.6, frozen before any P1b run): GO iff on ≥ 2 of 3 datasets both "
         "(a) Spearman(γ, mean test d′) ≤ −0.8 over the 7 frozen γ and (b) S1 test d′ ≥ 0.8 × fixed d′ "
         "and S1 test macro-F1 drop vs fixed < 1.0 point (means over seeds). d′ on the stress score.", "",
         "| Dataset | ρ(γ, d′) | (a) | S1 selected (val) | S1 d′ / fixed d′ | macro-F1 drop (pts) | (b) | both |",
         "|---|---|---|---|---|---|---|---|"]
    for ds, v in G["datasets"].items():
        L.append(f"| {ds} | {_f(v['rho_gamma_dprime'])} | {'pass' if v['criterion_a'] else 'fail'} | "
                 f"{v['s1_selected']} | {_f(v['s1_dprime_ratio'])} | {_f(v['s1_macro_f1_drop'], 2)} | "
                 f"{'pass' if v['criterion_b'] else 'fail'} | {'yes' if v['both'] else 'no'} |")
    L += ["", f"G1 code commit: `{S['code_commits']['P1b-G1']}`; RQ SHA-256 in runs: "
          + ", ".join(f"`{h}`" for h in S["rq_sha256_in_runs"]) + ".",
          "Generated by `scripts/report_p1b.py` from `results/p1b/g1` JSON.", ""]
    return "\n".join(L)


def render(S) -> str:
    L = ["# P1b report", "",
         "Generated by `scripts/report_p1b.py` from results JSON. Scope: AMENDMENT_02 (RESEARCH_QUESTIONS.md). "
         "P2 not started.", ""]
    g = S["p1_gate"]
    L += ["## A2.1 S2 status", "",
          f"S2 is a secondary analysis (original rule, 0.3 ≤ ρ < 0.6; pooled oriented ρ = {_f(g['primary']['rho_S2'])}). "
          f"P1 gate: **{g['outcome']}** — oriented ρ_W {_f(g['primary']['rho_W'])}, difference CI "
          f"[{_f(g['primary']['diff_ci95'][0])}, {_f(g['primary']['diff_ci95'][1])}]; raw-sign ρ_W {_f(g['raw']['rho_W'])}, "
          f"difference CI [{_f(g['raw']['diff_ci95'][0])}, {_f(g['raw']['diff_ci95'][1])}].", ""]
    W = S["within_dataset"]
    L += ["## A2.2 Within-dataset ρ (secondary, not gating)", "",
          "Oriented Spearman ρ (predictor vs measured LOACO stress AUC) per dataset, their mean, and a stratified "
          f"bootstrap 95% CI of the mean ({N_BOOT} resamples); pooled ρ from P1 alongside.", "",
          "| Predictor | " + " | ".join(f"{v} (n={W['D_pred']['n_per_dataset'][k]})" for k, v in WITHIN.items())
          + " | mean of within-dataset ρ | 95% CI | pooled oriented ρ (P1) |",
          "|---|" + "---|" * (len(WITHIN) + 3)]
    for k in PREDS:
        w, pr = W[k], S["pooled_rho_p1"][k]
        L.append(f"| {k} | " + " | ".join(_f(w["per_dataset_oriented_rho"][ev]) for ev in WITHIN)
                 + f" | {_f(w['mean_oriented_rho'])} | [{_f(w['ci95'][0])}, {_f(w['ci95'][1])}] | {_f(pr['oriented_rho'])} |")
    L.append("")
    if S["s2dir"]:
        sd = S["s2dir"]
        md = max(r["repro_max_abs_dev"] for r in sd["rows"])
        L += ["## A2.3 S2-dir — EXPLORATORY (never gating)", "",
              f"Computed on re-executed P1 runs ({S['n_runs']['s2dir']} runs); every re-execution reproduced the recorded "
              f"P1 stress AUC, probability AUC and D_pred (max |Δ| = {md:.2e}, tolerance 1e-6). Measured AUC = P1 value.", "",
              "| Set | S2-dir ρ [95% CI] | Mahalanobis ρ [95% CI] | S2 D_pred ρ [95% CI] | n |", "|---|---|---|---|---|"]
        for name in ("pooled", *WITHIN):
            v = sd[name]
            cell = lambda x: f"{_f(x['rho'])} [{_f(x['ci95'][0])}, {_f(x['ci95'][1])}]"  # noqa: E731
            L.append(f"| {WITHIN.get(name, name)} | {cell(v['s2dir'])} | {cell(v['mahalanobis'])} | {cell(v['D_pred'])} | "
                     f"{v['s2dir']['n']} |")
        L += ["", "Raw Spearman ρ (no sign orientation was pre-registered for S2-dir; Mahalanobis expected +).", "",
              "| Eval | Class | S2-dir (seed sd) | normaliser std(Q_b) | AUC stress (P1) | Mahalanobis |", "|---|---|---|---|---|---|"]
        for r in sd["rows"]:
            L.append(f"| {r['eval_id']} | {r['class']} | {_f(r['s2dir'], 2)} ({_f(r['s2dir_seed_sd'], 2)}) | "
                     f"{r['normaliser_std_Qb']:.3g} | {_f(r['auc_stress'])} | {_f(r['mahalanobis'], 2)} |")
        L.append("")
    L += ["## A2.4 D1 natural_novelty — DROPPED", "",
          "Infeasible as specified in A1.3: every D1 temporal_gap validation window (33,826) contains at least one "
          "Infilteration flow, so removing windows with natural-novelty flows leaves validation empty. See PILOT_REPORT.md.", ""]
    if S["d3_nn"]:
        L += ["## A2.5 D3 natural_novelty — 8 seeds", "",
              f"All 8 seeds on one code commit (`{S['code_commits']['P1b-D3nn']}`). Stress AUC; Student-t 95% CI (n = 8).", "",
              "| Class | mean AUC | 95% CI (±) | fraction of seeds < 0.5 | per-seed AUC | max |Δ| vs P1 seeds 17/23/42 |",
              "|---|---|---|---|---|---|"]
        for r in S["d3_nn"]["rows"]:
            L.append(f"| {r['class']} | {_f(r['mean'])} | {_f(r['ci95'])} | {r['frac_below_0.5']:.3f} | "
                     + ", ".join(f"{a:.3f}" for a in r["aucs"]) + f" | {('%.2e' % r['p1_seed_repro_max_abs_dev']) if r['p1_seed_repro_max_abs_dev'] is not None else '—'} |")
        L.append("")
    if S["g1"]:
        G = S["g1"]
        L += [f"## A2.6 Gate G1 — **{G['outcome']}**", "", "See `GATE_G1_DECISION.md`. Figure: `report/p1b/g1_gamma_sweep.pdf`.", ""]
        for ds, v in G["datasets"].items():
            L += [f"### {ds}", "", f"Spearman(γ, mean test d′) = {_f(v['rho_gamma_dprime'])} → (a) "
                  f"{'pass' if v['criterion_a'] else 'fail'}.", "",
                  "| γ | test d′ (mean ± t-CI) | test macro-F1 % (mean ± t-CI) | seeds |", "|---|---|---|---|"]
            for s in v["sweep"]:
                L.append(f"| {s['gamma']:g} | {_f(s['d_prime']['mean'])} ± {_f(s['d_prime']['ci95'])} | "
                         f"{_f(s['macro_f1']['mean'], 2)} ± {_f(s['macro_f1']['ci95'], 2)} | {s['n_seeds']} |")
            L += ["", "S1 grid (mean best validation macro-F1): " + ", ".join(
                f"{k} {_f(x, 4)}" for k, x in sorted(v["s1_grid_mean_val_macro_f1"].items())) + f". Selected: **{v['s1_selected']}**.",
                  f"S1 test d′ {_f(v['s1_selected_test_d_prime']['mean'])} vs fixed {_f(v['fixed_test_d_prime']['mean'])} "
                  f"(ratio {_f(v['s1_dprime_ratio'])}); test macro-F1 {_f(v['s1_selected_test_macro_f1']['mean'], 2)} vs "
                  f"{_f(v['fixed_test_macro_f1']['mean'], 2)} (drop {_f(v['s1_macro_f1_drop'], 2)} pts) → (b) "
                  f"{'pass' if v['criterion_b'] else 'fail'}.",
                  f"Adaptive γ (descriptive): final {_f(v['adaptive']['gamma_final']['mean'], 4)} ± "
                  f"{_f(v['adaptive']['gamma_final']['ci95'], 4)}; per-epoch mean "
                  + ", ".join(f"{x:.4f}" for x in v["adaptive"]["gamma_by_epoch_mean"])
                  + f"; test d′ {_f(v['adaptive']['test_d_prime']['mean'])}.", "",
                  "Per-class collapse ratio d′_γ / d′_0 (descriptive) vs stationarity index (γ = 0 model):", "",
                  "| Class | stationarity | d′_0 | " + " | ".join(f"γ={g:g}" for g in G["gammas"][1:]) + " |",
                  "|---|---|---|" + "---|" * (len(G["gammas"]) - 1)]
            for c in v["collapse"]:
                L.append(f"| {c['class']} | {_f(c['stationarity_g0'])} | {_f(c['d0'], 2)} | "
                         + " | ".join(_f(c["ratio"][f"{g:g}"], 2) for g in G["gammas"][1:]) + " |")
            L += ["", f"Spearman(stationarity, ratio at γ = 0.2) = {_f(v['collapse_rho_stationarity_vs_ratio_g0.2'])} "
                  f"(n = {len(v['collapse'])}; descriptive).", ""]
    L += ["## Provenance", "",
          "| Experiment | code_commit | runs |", "|---|---|---|"]
    for k, c in S["code_commits"].items():
        L.append(f"| {k} | `{c}` | {S['n_runs'][{'P1b-S2dir': 's2dir', 'P1b-D3nn': 'd3_nn', 'P1b-G1': 'g1'}[k]]} |")
    L += ["", f"Each experiment ran from one clean code commit (checked; the generator refuses otherwise). "
          f"RESEARCH_QUESTIONS.md SHA-256 in runs: " + ", ".join(f"`{h}`" for h in S["rq_sha256_in_runs"])
          + f" (AMENDMENT_02; `docs/AMENDMENTS.json`). Report generated at `{S['provenance']['code_commit']}`.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
