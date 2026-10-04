#!/usr/bin/env python
"""Build the P1 pilot report from results JSON only (no hand-typed numbers).

Outputs: <results_dir>/summary.json, <report_dir>/rq1_predictors.tex,
<report_dir>/rq1_scatter.pdf, and PILOT_REPORT.md (smoke: under smoke_out/).
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

# Categorical slots 1-4 of the reference palette (light mode), one per dataset,
# with a distinct marker per dataset so identity is never colour alone.
DS_STYLE = {"D1": ("#2a78d6", "o"), "D2": ("#eb6834", "s"), "D3": ("#1baf7a", "^"), "D4": ("#eda100", "D")}
PRED_ORDER = ["D_pred", "wasserstein", "mahalanobis", "stationarity"]


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
        by[(r["dataset"], r["held_out"])].append(r)
    points = []
    for (ds, cls), rs in sorted(by.items()):
        pt = {"dataset": ds, "class": cls, "seeds": sorted(r["seed"] for r in rs)}
        for k in ("stress", "prob", "s3"):
            pt[f"auc_{k}"] = metrics.mean_ci([r["measured"][k]["auc"] for r in rs])
        for k in PRED_ORDER + ["D_pred_val_benign", "D_pred_per_channel"]:
            pt[k] = float(np.mean([r["predictors"][k] for r in rs]))
        points.append(pt)
    return points


def correlate(points, n_boot=1000):
    auc = np.array([p["auc_stress"]["mean"] for p in points])
    out = {}
    for k in PRED_ORDER + ["D_pred_val_benign", "D_pred_per_channel"]:
        sp = metrics.spearman_bootstrap([p[k] for p in points], auc, n_boot=n_boot)
        sign = EXPECTED_SIGN.get(k, EXPECTED_SIGN["D_pred"] if k.startswith("D_pred") else 1)
        sp["expected_sign"] = sign
        sp["oriented_rho"] = sign * sp["rho"]
        sp["oriented_ci95"] = sorted(sign * v for v in sp["ci95"])
        out[k] = sp
    return out


def inversion_table(points):
    pred_inv = np.array([p["D_pred"] < 0 for p in points])
    obs_inv = np.array([p["auc_stress"]["mean"] < 0.5 for p in points])
    tab = [[int((pred_inv & obs_inv).sum()), int((pred_inv & ~obs_inv).sum())],
           [int((~pred_inv & obs_inv).sum()), int((~pred_inv & ~obs_inv).sum())]]
    p = float(fisher_exact(tab)[1]) if len(points) else float("nan")
    return {"rows_predicted_inversion_yes_no": tab, "cols_observed_inversion_yes_no": True,
            "accuracy": float((pred_inv == obs_inv).mean()) if len(points) else float("nan"),
            "fisher_p": p, "n": len(points)}


def decision(corr):
    rho = corr["D_pred"]["oriented_rho"]
    beats = rho > corr["wasserstein"]["oriented_rho"]
    if not np.isfinite(rho):
        return "undetermined (too few points)"
    if rho >= 0.6 and beats:
        return "proceed: rho >= 0.6 and above Wasserstein"
    if rho >= 0.6:
        return "rho >= 0.6 but NOT above Wasserstein: report to the authors"
    if rho >= 0.3:
        return "S2 becomes a secondary analysis (0.3 <= rho < 0.6)"
    return "rho < 0.3: report to the authors before continuing"


def scatter(points, corr, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
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
            ax.annotate(p["class"], (xi, yi), xytext=(4, 3 if i % 2 == 0 else -9), textcoords="offset points", fontsize=7,
                        color="#3d3c36")
    ax.set_xscale("symlog", linthresh=1.0)
    r = corr["D_pred"]
    ax.set_title(f"Spearman ρ = {r['rho']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}], n = {r['n']}",
                 fontsize=9, color="#3d3c36")
    ax.set_xlabel("Predicted detectability $D_{pred}$ (S2)", fontsize=9)
    ax.set_ylabel("Measured LOACO AUC (stress)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def latex_table(corr, path: Path):
    names = {"D_pred": "S2 $D_{pred}$", "wasserstein": "Wasserstein", "mahalanobis": "Mahalanobis",
             "stationarity": "Stationarity", "D_pred_val_benign": "S2 (val-benign ref.)",
             "D_pred_per_channel": "S2 (per-channel $H$)"}
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Predictor & sign & $\rho$ & oriented $\rho$ & 95\% CI (oriented) \\", r"\midrule"]
    for k, nm in names.items():
        c = corr[k]
        L.append(f"{nm} & {'+' if c['expected_sign'] > 0 else '$-$'} & {c['rho']:.3f} & "
                 f"{c['oriented_rho']:.3f} & [{c['oriented_ci95'][0]:.3f}, {c['oriented_ci95'][1]:.3f}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


def render_md(summary, smoke: bool) -> str:
    c, inv = summary["correlations"], summary["inversion"]
    L = ["# P1 pilot report", ""]
    if smoke:
        L += ["> **SMOKE RUN on synthetic NF-v3-schema data, 1 seed, 2 epochs. These numbers mean nothing.**", ""]
    L += [f"RQ file SHA-256: `{summary['rq_sha256']}`  ", f"Git: `{summary['provenance']['git_hash']}`", "",
          f"Points (class × dataset): {summary['n_points']}; skipped folds: {len(summary['skipped'])}", "",
          "## RQ1: does S2 forecast per-class LOACO AUC?", "",
          "| Predictor | expected sign | ρ | oriented ρ | 95% CI (oriented) | n |", "|---|---|---|---|---|---|"]
    for k, v in c.items():
        L.append(f"| {k} | {'+' if v['expected_sign'] > 0 else '−'} | {v['rho']:.3f} | {v['oriented_rho']:.3f} | "
                 f"[{v['oriented_ci95'][0]:.3f}, {v['oriented_ci95'][1]:.3f}] | {v['n']} |")
    L += ["", f"**Decision rule outcome:** {summary['decision']}", "",
          "## RQ2: predicted vs observed inversion (AUC < 0.5)", "",
          "| | observed inverted | observed not inverted |", "|---|---|---|",
          f"| predicted inverted (D_pred < 0) | {inv['rows_predicted_inversion_yes_no'][0][0]} | {inv['rows_predicted_inversion_yes_no'][0][1]} |",
          f"| predicted not inverted | {inv['rows_predicted_inversion_yes_no'][1][0]} | {inv['rows_predicted_inversion_yes_no'][1][1]} |",
          "", f"Accuracy {inv['accuracy']:.3f}, Fisher exact p = {inv['fisher_p']:.3g}.", "",
          "## Per-class points", "",
          "| Dataset | Class | AUC stress (mean ± CI) | AUC prob | D_pred | Wasserstein | Mahalanobis | Stationarity | seeds |",
          "|---|---|---|---|---|---|---|---|---|"]
    for p in summary["points"]:
        a, b = p["auc_stress"], p["auc_prob"]
        L.append(f"| {p['dataset']} | {p['class']} | {a['mean']:.3f} ± {a['ci95']:.3f} | {b['mean']:.3f} | "
                 f"{p['D_pred']:.3f} | {p['wasserstein']:.3f} | {p['mahalanobis']:.3f} | {p['stationarity']:.3f} | "
                 f"{len(p['seeds'])} |")
    if summary["skipped"]:
        L += ["", "## Skipped folds", ""] + [f"- {s['dataset']} / {s['held_out']}: "
                                             f"{s['fold_info'].get('reason', s['fold_info'])}" for s in summary["skipped"]]
    L += ["", "Figure: `rq1_scatter.pdf`. Table: `rq1_predictors.tex`. CI with n ≤ 3 seeds uses Student-t.", ""]
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
    corr = correlate(points, n_boot=cfg["n_boot"])
    summary = {"n_points": len(points), "points": points, "correlations": corr,
               "inversion": inversion_table(points), "decision": decision(corr),
               "skipped": [{"dataset": s["dataset"], "held_out": s["held_out"], "fold_info": s["fold_info"]}
                           for s in skipped],
               "rq_sha256": rq_sha256(), "rq_sha256_in_runs": sorted({r["provenance"]["rq_sha256"] for r in runs}),
               "provenance": provenance(config=cfg), "smoke": a.smoke}
    if summary["rq_sha256_in_runs"] != [summary["rq_sha256"]]:
        summary["warning"] = "runs were produced under a different RESEARCH_QUESTIONS.md"
    write_json(rdir / "summary.json", summary)
    scatter(points, corr, odir / "rq1_scatter.pdf")
    latex_table(corr, odir / "rq1_predictors.tex")
    md = ROOT / ("smoke_out/PILOT_REPORT.md" if a.smoke else "PILOT_REPORT.md")
    md.write_text(render_md(summary, a.smoke))
    print(f"decision: {summary['decision']}\nwrote {md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
