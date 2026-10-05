#!/usr/bin/env python
"""Build the P1 pilot report from results JSON only (no hand-typed numbers).

AMENDMENT_01 A1.5: Spearman rho (bootstrap 95% CI) for
  (a) D1 families, (b) pooled = D1 families + D2 + D3 (grouped_random), (c) D1 sub-labels;
the P1 decision rule applies to (b). Per-class table with predicted / observed inversion,
D2 temporal_gap vs grouped_random, natural_novelty.

Outputs: <results_dir>/summary.json, <report_dir>/*.pdf|tex, PILOT_REPORT.md (smoke: under smoke_out/).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import metrics  # noqa: E402
from nids.predictors import EXPECTED_SIGN  # noqa: E402
from nids.provenance import ROOT, provenance, rq_sha256, write_json  # noqa: E402

# Categorical slots 1-3 of the reference palette (light mode), one per dataset,
# with a distinct marker each so identity is never colour alone.
DS_STYLE = {"D1": ("#2a78d6", "o"), "D2": ("#eb6834", "s"), "D3": ("#1baf7a", "^")}
PRED_ORDER = ["D_pred", "wasserstein", "mahalanobis", "stationarity"]
PRED_EXTRA = ["D_pred_val_benign", "D_pred_per_channel"]
RHO_SETS = {
    "a_families": ("(a) D1 families", {"families"}),
    "b_pooled": ("(b) pooled: D1 families + D2 + D3 (grouped_random)", {"families", "pooled"}),
    "c_sublabels": ("(c) D1 sub-labels (variant novelty)", {"sublabels"}),
}
DECISION_SET = "b_pooled"


def load(results_dir: Path):
    runs, skipped = [], []
    for p in sorted(results_dir.rglob("*.json")):
        if p.name == "summary.json":
            continue
        r = json.loads(p.read_text())
        (skipped if r.get("skipped") else runs).append(r)
    return runs, skipped


def aggregate(runs):
    by = defaultdict(list)
    for r in runs:
        by[(r["eval_id"], r["held_out"])].append(r)
    points = []
    for (ev, cls), rs in sorted(by.items()):
        r0 = rs[0]
        pt = {"eval_id": ev, "dataset": r0["dataset"], "scheme": r0["scheme"], "kind": r0["kind"],
              "labelling": r0["labelling"], "rho_set": r0["rho_set"], "class": cls,
              "seeds": sorted(r["seed"] for r in rs), "n_test": r0["n_test"]}
        for k in ("stress", "prob", "s3"):
            pt[f"auc_{k}"] = metrics.mean_ci([r["measured"][k]["auc"] for r in rs])
        pt["d_prime_stress"] = metrics.mean_ci([r["measured"]["stress"]["d_prime"] for r in rs])
        for k in PRED_ORDER + PRED_EXTRA:
            pt[k] = float(np.mean([r["predictors"][k] for r in rs]))
        pt["D_pred_seed_sd"] = float(np.std([r["predictors"]["D_pred"] for r in rs], ddof=1)) if len(rs) > 1 else float("nan")
        pt["ir_tail_ratio"] = float(np.mean([r["s2"]["ir_tail_ratio"] for r in rs]))
        pt["alpha_mean"] = float(np.mean([r["s2"]["alpha_mean"] for r in rs]))
        pt["beta_mean"] = float(np.mean([r["s2"]["beta_mean"] for r in rs]))
        pt["val_macro_f1"] = float(np.mean([r["training"]["best_val_macro_f1"] for r in rs]))
        pt["inversion_predicted"] = bool(pt["D_pred"] < 0)
        pt["inversion_observed"] = bool(pt["auc_stress"]["mean"] < 0.5)
        points.append(pt)
    return points


def correlate(points, n_boot=1000):
    auc = np.array([p["auc_stress"]["mean"] for p in points])
    out = {}
    for k in PRED_ORDER + PRED_EXTRA:
        sp = metrics.spearman_bootstrap([p[k] for p in points], auc, n_boot=n_boot)
        sign = EXPECTED_SIGN.get(k, 1)
        sp.update(expected_sign=sign, oriented_rho=sign * sp["rho"],
                  oriented_ci95=sorted(sign * v for v in sp["ci95"]))
        out[k] = sp
    return out


def inversion_table(points):
    pi = np.array([p["inversion_predicted"] for p in points], bool)
    oi = np.array([p["inversion_observed"] for p in points], bool)
    tab = [[int((pi & oi).sum()), int((pi & ~oi).sum())], [int((~pi & oi).sum()), int((~pi & ~oi).sum())]]
    return {"table_rows_pred_yes_no_cols_obs_yes_no": tab, "n": len(points),
            "accuracy": float((pi == oi).mean()) if len(points) else float("nan"),
            "fisher_p": float(fisher_exact(tab)[1]) if len(points) else float("nan")}


def decision(corr):
    rho, w = corr["D_pred"]["oriented_rho"], corr["wasserstein"]["oriented_rho"]
    if not np.isfinite(rho):
        return "undetermined (too few points)"
    beats = np.isfinite(w) and rho > w
    if rho >= 0.6:
        return "proceed: rho >= 0.6 and above Wasserstein" if beats else \
            "rho >= 0.6 but NOT above Wasserstein: report to the authors"
    if rho >= 0.3:
        return "S2 becomes a secondary analysis (0.3 <= rho < 0.6)"
    return "rho < 0.3: report to the authors before continuing"


def scatter(points, corr, title, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axhline(0.5, color="#b0afa8", lw=1, ls="--", zorder=0)
    ax.axvline(0.0, color="#b0afa8", lw=1, ls="--", zorder=0)
    for ds in sorted({p["dataset"] for p in points}):
        col, mk = DS_STYLE.get(ds, ("#6b6a63", "o"))
        pts = [p for p in points if p["dataset"] == ds]
        x = [p["D_pred"] for p in pts]
        y = [p["auc_stress"]["mean"] for p in pts]
        e = [p["auc_stress"]["ci95"] if np.isfinite(p["auc_stress"]["ci95"]) else 0 for p in pts]
        ax.errorbar(x, y, yerr=e, fmt=mk, ms=7, color=col, mec="white", mew=1.5, elinewidth=1,
                    capsize=0, label=ds, zorder=3)
        for i, (p, xi, yi) in enumerate(zip(pts, x, y)):
            ax.annotate(p["class"], (xi, yi), xytext=(4, 3 if i % 2 == 0 else -9), textcoords="offset points",
                        fontsize=6.5, color="#3d3c36")
    ax.set_xscale("symlog", linthresh=1.0)
    r = corr["D_pred"]
    ax.set_title(f"{title}\nSpearman ρ = {r['rho']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}], n = {r['n']}",
                 fontsize=8.5, color="#3d3c36")
    ax.set_xlabel("Predicted detectability $D_{pred}$ (S2, symlog)", fontsize=9)
    ax.set_ylabel("Measured LOACO AUC (stress)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def latex_table(corrs, path: Path):
    names = {"D_pred": "S2 $D_{pred}$", "wasserstein": "Wasserstein", "mahalanobis": "Mahalanobis",
             "stationarity": "Stationarity", "D_pred_val_benign": "S2 (val-benign ref.)",
             "D_pred_per_channel": "S2 (per-channel $H$)"}
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Set & Predictor & sign & $\rho$ & oriented $\rho$ & 95\% CI (oriented) \\", r"\midrule"]
    for sk, corr in corrs.items():
        for k, nm in names.items():
            c = corr[k]
            L.append(f"{sk.split('_')[0]} & {nm} & {'+' if c['expected_sign'] > 0 else '$-$'} & {c['rho']:.3f} & "
                     f"{c['oriented_rho']:.3f} & [{c['oriented_ci95'][0]:.3f}, {c['oriented_ci95'][1]:.3f}] \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def _wall_hours(runs) -> float:
    """LOACO: one training per run. natural_novelty: one training per (eval, seed); its class
    records carry the elapsed time since that training began, so take the max per seed."""
    tot = sum(r.get("wall_seconds", 0) for r in runs if r["kind"] == "loaco")
    nn = defaultdict(float)
    for r in runs:
        if r["kind"] == "natural_novelty":
            k = (r["eval_id"], r["seed"])
            nn[k] = max(nn[k], r.get("wall_seconds", 0))
    return (tot + sum(nn.values())) / 3600


def _f(v, nd=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def render_md(S, smoke: bool) -> str:
    L = ["# P1 pilot report", ""]
    if smoke:
        L += ["> **SMOKE RUN on synthetic NF-v3-schema data. These numbers mean nothing.**", ""]
    L += [f"Generated by `scripts/report_p1.py` from `{S['results_dir']}` JSON. RQ SHA-256 in force: "
          f"`{S['rq_sha256']}` (runs: {', '.join('`' + h[:12] + '…`' for h in S['rq_sha256_in_runs'])}). "
          f"Git: `{S['provenance']['git_hash']}`.", "",
          "Scope (AMENDMENT_01): AHSD-fixed, seeds " + ", ".join(map(str, S["seeds"])) +
          "; LOACO on grouped_random (D1 families + sub-labels, D2, D3), D2 also on temporal_gap; "
          "natural_novelty on temporal_gap (D1, D3). Measured AUC = stress score (higher = attack). "
          "Seed CI: Student-t, n = number of seeds.", "",
          f"Runs: {S['n_runs']} (class × seed); points: {S['n_points']}; skipped folds: {len(S['skipped'])}.", ""]
    dec = S["correlations"][DECISION_SET]
    L += ["## Decision (rule applied to set (b), oriented ρ)", "",
          f"**{S['decision']}**: S2 oriented ρ = {_f(dec['D_pred']['oriented_rho'])} "
          f"[{_f(dec['D_pred']['oriented_ci95'][0])}, {_f(dec['D_pred']['oriented_ci95'][1])}], "
          f"Wasserstein oriented ρ = {_f(dec['wasserstein']['oriented_rho'])}, n = {dec['D_pred']['n']}.", ""]
    L += ["## RQ1: Spearman ρ, predicted vs measured AUC", ""]
    for sk, (title, _) in RHO_SETS.items():
        c = S["correlations"][sk]
        L += [f"### {title}", "", "| Predictor | expected sign | ρ | oriented ρ | 95% CI (oriented) | p | n |",
              "|---|---|---|---|---|---|---|"]
        for k, v in c.items():
            L.append(f"| {k} | {'+' if v['expected_sign'] > 0 else '−'} | {_f(v['rho'])} | {_f(v['oriented_rho'])} | "
                     f"[{_f(v['oriented_ci95'][0])}, {_f(v['oriented_ci95'][1])}] | {_f(v['p'], 3)} | {v['n']} |")
        L.append("")
    L += ["## RQ2: inversion (predicted D_pred < 0, observed AUC < 0.5)", ""]
    for sk, inv in S["inversion"].items():
        t = inv["table_rows_pred_yes_no_cols_obs_yes_no"]
        L += [f"- {sk}: predicted&observed {t[0][0]}, predicted only {t[0][1]}, observed only {t[1][0]}, "
              f"neither {t[1][1]}; accuracy {_f(inv['accuracy'])}, Fisher p {_f(inv['fisher_p'], 3)}, n {inv['n']}"]
    L += ["", "## Per-class table", "",
          "| Eval | Class | n test (b/c) | D_pred (seed sd) | AUC stress (± t-CI) | inversion predicted | inversion observed | "
          "AUC prob | AUC S3 | Wasserstein | Mahalanobis | Stationarity |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in S["points"]:
        a = p["auc_stress"]
        L.append(f"| {p['eval_id']} | {p['class']} | {p['n_test']['benign']}/{p['n_test']['held_out']} | "
                 f"{_f(p['D_pred'], 2)} ({_f(p['D_pred_seed_sd'], 2)}) | {_f(a['mean'])} ± {_f(a['ci95'])} | "
                 f"{'yes' if p['inversion_predicted'] else 'no'} | {'yes' if p['inversion_observed'] else 'no'} | "
                 f"{_f(p['auc_prob']['mean'])} | {_f(p['auc_s3']['mean'])} | {_f(p['wasserstein'])} | "
                 f"{_f(p['mahalanobis'], 2)} | {_f(p['stationarity'])} |")
    L += ["", "## D2: temporal_gap vs grouped_random (A1.1)", "",
          "| Class | AUC stress grouped_random | AUC stress temporal_gap | difference (gr − tg) | D_pred gr | D_pred tg |",
          "|---|---|---|---|---|---|"]
    for r in S["d2_split_difference"]:
        L.append(f"| {r['class']} | {_f(r['auc_gr'])} | {_f(r['auc_tg'])} | {_f(r['diff'])} | {_f(r['dpred_gr'], 2)} | "
                 f"{_f(r['dpred_tg'], 2)} |")
    L += ["", "## natural_novelty (temporal_gap; A1.3)", "",
          "Stress, supervised probability and S3 only; the §7B benign-only baselines run in P3.", "",
          "| Eval | Class | n test (b/c) | AUC stress | AUC prob | AUC S3 | D_pred | inversion predicted | inversion observed |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in [q for q in S["points"] if q["kind"] == "natural_novelty"]:
        L.append(f"| {p['eval_id']} | {p['class']} | {p['n_test']['benign']}/{p['n_test']['held_out']} | "
                 f"{_f(p['auc_stress']['mean'])} ± {_f(p['auc_stress']['ci95'])} | {_f(p['auc_prob']['mean'])} | "
                 f"{_f(p['auc_s3']['mean'])} | {_f(p['D_pred'], 2)} | {'yes' if p['inversion_predicted'] else 'no'} | "
                 f"{'yes' if p['inversion_observed'] else 'no'} |")
    L += ["", "## Diagnostics", "",
          f"- Linearised gates (mean over points): α {_f(S['diag']['alpha'])}, β {_f(S['diag']['beta'])}; "
          f"impulse-response energy beyond 32 steps (ir_tail_ratio): median {S['diag']['ir_tail_median']:.2e}, "
          f"max {S['diag']['ir_tail_max']:.2e}.",
          f"- Best validation macro-F1 per training: median {_f(S['diag']['val_f1_median'])}, "
          f"min {_f(S['diag']['val_f1_min'])}.",
          f"- Wall time: {S['diag']['wall_hours']:.2f} h of training+scoring over {S['n_runs']} runs.", ""]
    if S["skipped"]:
        L += ["## Skipped folds", ""] + [f"- {s['eval_id']} / {s['held_out']}: "
                                         f"{s['fold_info'].get('reason', '')} (train windows "
                                         f"{s['fold_info'].get('train_windows_majority')})" for s in S["skipped"]]
    L += ["", "Figures: `report/p1/rq1_scatter_<set>.pdf`. Table: `report/p1/rq1_predictors.tex`.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_pilot.yaml")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.smoke:
        cfg.update(cfg["smoke"])
    rdir, odir = ROOT / cfg["results_dir"], ROOT / cfg["report_dir"]
    runs, skipped = load(rdir)
    if not runs:
        raise SystemExit(f"no results in {rdir}")
    points = aggregate(runs)
    corrs, invs = {}, {}
    for sk, (title, members) in RHO_SETS.items():
        pts = [p for p in points if p["rho_set"] in members]
        corrs[sk] = correlate(pts, n_boot=cfg["n_boot"])
        invs[sk] = inversion_table(pts)
        if pts:
            scatter(pts, corrs[sk], title, odir / f"rq1_scatter_{sk}.pdf")
    invs["all_points"] = inversion_table(points)
    gr = {p["class"]: p for p in points if p["eval_id"].startswith("D2_gr")}
    tg = {p["class"]: p for p in points if p["eval_id"].startswith("D2_tg")}
    d2diff = [{"class": c, "auc_gr": gr[c]["auc_stress"]["mean"] if c in gr else float("nan"),
               "auc_tg": tg[c]["auc_stress"]["mean"] if c in tg else float("nan"),
               "diff": (gr[c]["auc_stress"]["mean"] - tg[c]["auc_stress"]["mean"]) if c in gr and c in tg else float("nan"),
               "dpred_gr": gr[c]["D_pred"] if c in gr else float("nan"),
               "dpred_tg": tg[c]["D_pred"] if c in tg else float("nan")} for c in sorted(set(gr) | set(tg))]
    tails = [p["ir_tail_ratio"] for p in points]
    S = {"results_dir": cfg["results_dir"], "seeds": cfg["seeds"], "n_runs": len(runs), "n_points": len(points),
         "points": points, "correlations": corrs, "inversion": invs,
         "decision_set": DECISION_SET, "decision": decision(corrs[DECISION_SET]),
         "d2_split_difference": d2diff,
         "diag": {"alpha": float(np.mean([p["alpha_mean"] for p in points])),
                  "beta": float(np.mean([p["beta_mean"] for p in points])),
                  "ir_tail_median": float(np.median(tails)), "ir_tail_max": float(np.max(tails)),
                  "val_f1_median": float(np.median([r["training"]["best_val_macro_f1"] for r in runs])),
                  "val_f1_min": float(np.min([r["training"]["best_val_macro_f1"] for r in runs])),
                  "wall_hours": _wall_hours(runs)},
         "skipped": [{"eval_id": s["eval_id"], "held_out": s["held_out"], "fold_info": s["fold_info"]} for s in skipped],
         "rq_sha256": rq_sha256(), "rq_sha256_in_runs": sorted({r["provenance"]["rq_sha256"] for r in runs}),
         "provenance": provenance(config=cfg), "smoke": a.smoke}
    if S["rq_sha256_in_runs"] != [S["rq_sha256"]]:
        S["warning"] = "runs were produced under a different RESEARCH_QUESTIONS.md"
    write_json(rdir / "summary.json", S)
    latex_table(corrs, odir / "rq1_predictors.tex")
    md = ROOT / ("smoke_out/PILOT_REPORT.md" if a.smoke else "PILOT_REPORT.md")
    md.write_text(render_md(S, a.smoke))
    print(f"decision ({DECISION_SET}): {S['decision']}\nwrote {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
