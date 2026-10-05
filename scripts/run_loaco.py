#!/usr/bin/env python
"""P1 pilot (AMENDMENT_01 scope): AHSD-fixed, LOACO and natural_novelty, S2 + competing predictors.

Evaluations are listed in the config. Each writes one JSON per (held-out class,
seed) under results/p1/<eval id>/. Nothing is aggregated here;
scripts/report_p1.py builds the tables, figures and PILOT_REPORT.md.

  python scripts/run_loaco.py --config configs/p1_pilot.yaml [--resume] [--only D2_gr ...]
  python scripts/run_loaco.py --config configs/p1_pilot.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import metrics, predictors, scores, spectral  # noqa: E402
from nids.models.ahsd import AHSD  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.preflight import preflight  # noqa: E402
from nids.provenance import ROOT, provenance, require_clean_tree, write_json  # noqa: E402
from nids.train import fit, predict, set_seed  # noqa: E402


def s2_block(m, pr_train_benign, z_by: dict, T: int) -> dict:
    """Linearise with mean gates over benign training windows; predict per-window in-band energy."""
    a_ch, b_ch, g = pr_train_benign["alpha"], pr_train_benign["beta"], pr_train_benign["gamma_t"]
    a, b = float(a_ch.mean()), float(b_ch.mean())
    E0 = m.E0.numpy()
    ir = spectral.ahsd_impulse_response(a, b, g, T)
    g2 = spectral.gain2(spectral.transfer_function(ir, T))
    g2_ch = spectral.gain2(spectral.transfer_function(spectral.ahsd_impulse_response(a_ch, b_ch, g, T), T))
    out = {"alpha_mean": a, "beta_mean": b, "gamma_mean": g, "H_gain2": g2.tolist(),
           "ir_tail_ratio": spectral.ir_tail_ratio(a, b, g, T)}
    for name, z in z_by.items():
        P = spectral.window_power(z - E0[None, None, :])
        out[f"P_{name}"] = P.mean(0).tolist()
        out[f"S_{name}"] = spectral.in_band_energy(g2, P)
        out[f"Sch_{name}"] = spectral.in_band_energy(g2_ch, P)
    return out


def train_model(arc: Archive, tr: dict, va: dict, seed: int, cfg: dict, log):
    torch.set_num_threads(cfg.get("threads", 4))
    # Seed BEFORE building the model, so the initial weights depend on this run's seed only.
    # (P1, code f557bcd, built the model before fit() seeded: its initial weights came from the
    # torch RNG state left by the previous run in the process. Fixed after P1; see P1B_REPORT.md.)
    set_seed(seed)
    m = AHSD(arc.n_features, D=cfg["D"], variant="fixed")
    tinfo = fit(m, tr, va, seed=seed, epochs=cfg["epochs"], log=log)
    return m, tinfo


def score_class(m, arc: Archive, tr: dict, va: dict, te: dict, c: int, seed: int, cfg: dict,
                cache: dict) -> dict:
    """Measured detectability of class c (test = benign + c) and every predictor, on one trained model."""
    if "sv" not in cache:
        cache["sv"] = predict(m, va["X"], states=True)
        cache["trb"] = predict(m, tr["X"][tr["y_bin"] == 0], states=True)
        cache["mah"] = predictors.BenignMahalanobis(tr["X"][tr["y_bin"] == 0])
    sv, trb = cache["sv"], cache["trb"]
    st = predict(m, te["X"], states=True)
    y = te["y_bin"]
    is_c, is_b = te["y_cls"] == c, te["y_cls"] == 0
    if not np.all(is_c | is_b):
        raise AssertionError("test set holds windows other than benign and the held-out class")
    s2 = s2_block(m, trb, {"class": st["z"][is_c], "benign": st["z"][is_b],
                          "val_benign": sv["z"][va["y_bin"] == 0]}, arc.wi.T)
    D_pred = spectral.detectability(s2["S_class"], s2["S_benign"])
    D_pred_val = spectral.detectability(s2["S_class"], s2["S_val_benign"])
    D_pred_ch = spectral.detectability(s2["Sch_class"], s2["Sch_benign"])
    for k in ("S_class", "S_benign", "S_val_benign", "Sch_class", "Sch_benign", "Sch_val_benign"):
        s2[k] = {"mean": float(np.mean(s2[k])), "std": float(np.std(s2[k], ddof=1)), "n": int(len(s2[k]))}
    comp = {
        "wasserstein": predictors.wasserstein_to_train_attacks(te["X"][is_c], tr["X"][tr["y_bin"] == 1], seed=seed),
        "mahalanobis": float(cache["mah"](te["X"][is_c]).mean()),
        "stationarity": float(predictors.stationarity_index(st["z"][is_c]).mean()),
    }
    vb = va["y_bin"] == 0
    s3 = scores.s3_score(st["stress"], st["delta_norm"], sv["stress"][vb], sv["delta_norm"][vb])
    thr = metrics.threshold_at_val_fpr(sv["stress"][vb], 0.01)
    return {
        "n_test": {"benign": int(is_b.sum()), "held_out": int(is_c.sum())},
        "measured": {
            "stress": metrics.detection(y, st["stress"]),
            "prob": metrics.detection(y, st["prob"]),
            "s3": metrics.detection(y, s3),
            "stress_group_bootstrap_auc": metrics.group_bootstrap(
                metrics.auc, y, st["stress"], arc.wi.group[te["idx"]], n_boot=cfg["n_boot"], seed=seed),
            "stress_tpr_at_val_1pct_threshold": float((st["stress"][y == 1] > thr).mean()),
        },
        "predictors": {"D_pred": D_pred, "D_pred_val_benign": D_pred_val, "D_pred_per_channel": D_pred_ch, **comp},
        "s2": s2,
        "gamma_final": float(m.gamma().detach()),
    }


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def run_evaluation(ev: dict, cfg: dict, resume: bool, smoke: bool):
    proc = ROOT / cfg["processed_dir"]
    out_root = ROOT / cfg["results_dir"] / ev["id"]
    ds, scheme, lab = ev["dataset"], ev["scheme"], ev.get("labelling", "file")
    arc = Archive(proc / ds / scheme / "archive")
    meta = json.loads((proc / ds / "index_meta.json").read_text())
    data_sha = json.loads((proc / ds / "archive_meta.json").read_text())["schemes"][scheme]["sha256"]
    smeta = meta["schemes"][scheme]
    base = {"eval_id": ev["id"], "dataset": ds, "scheme": scheme, "kind": ev["kind"], "labelling": lab,
            "rho_set": ev["rho_set"]}

    def done(path):
        return resume and path.exists()

    def save(path, res):
        res["provenance"] = provenance(config={**cfg, "evaluation": ev}, data_sha256=data_sha)
        res["smoke"] = smoke
        write_json(path, res)

    if ev["kind"] == "loaco":
        classes = meta["family"]["classes"] if lab == "family" else meta["classes"]
        finfo = smeta["loaco_family"] if lab == "family" else smeta["loaco"]
        folds = arc.folds_family if lab == "family" else arc.folds
        for cls, rec in finfo.items():
            if not rec.get("eligible") or not rec.get("feasible"):
                save(out_root / f"{_safe(cls)}__skipped.json", {**base, "held_out": cls, "skipped": True, "fold_info": rec})
                continue
            f = folds[cls]
            for seed in cfg["seeds"]:
                path = out_root / _safe(cls) / f"seed{seed}.json"
                if done(path):
                    continue
                t0 = time.time()
                print(f"[{ev['id']}] LOACO {cls} seed {seed}", flush=True)
                tr, va, te = (arc.windows(f[k], labelling=lab) for k in ("train", "val", "test"))
                m, tinfo = train_model(arc, tr, va, seed, cfg, log=lambda s: print(s, flush=True))
                res = score_class(m, arc, tr, va, te, classes.index(cls), seed, cfg, {})
                save(path, {**base, "held_out": cls, "seed": seed, **res, "training": tinfo,
                            "fold_info": rec, "wall_seconds": time.time() - t0})
    elif ev["kind"] == "natural_novelty":
        classes = meta["classes"]
        nn = arc.natural_novelty
        nn_info = smeta["natural_novelty"]
        nn_classes = nn_info["classes"]
        empty = [k for k in ("train", "val") if nn_info[k]["benign"] == 0 or nn_info[k]["attack"] == 0]
        if empty:  # fail closed: no checkpoint selection possible (AMENDMENT_01 A1.3 as written)
            reason = (f"{' and '.join(empty)} empty after removing windows with any flow of "
                      f"{', '.join(nn_classes)} ({ {k: nn_info[k] for k in ('train', 'val')} })")
            for c in nn_classes:
                save(out_root / f"{_safe(c)}__skipped.json",
                     {**base, "held_out": c, "skipped": True, "fold_info": {"reason": reason}})
            print(f"[{ev['id']}] natural_novelty infeasible: {reason}", flush=True)
            return
        for seed in cfg["seeds"]:
            paths = {c: out_root / _safe(c) / f"seed{seed}.json" for c in nn_classes}
            if all(done(p) for p in paths.values()):
                continue
            t0 = time.time()
            print(f"[{ev['id']}] natural_novelty seed {seed} (one model, {len(nn_classes)} classes)", flush=True)
            tr, va = arc.windows(nn["train"]), arc.windows(nn["val"])
            m, tinfo = train_model(arc, tr, va, seed, cfg, log=lambda s: print(s, flush=True))
            cache = {}
            for c in nn_classes:
                te = arc.windows(nn[f"test/{c}"])
                res = score_class(m, arc, tr, va, te, classes.index(c), seed, cfg, cache)
                save(paths[c], {**base, "held_out": c, "seed": seed, **res, "training": tinfo,
                                "set_info": smeta["natural_novelty"][f"test/{c}"],
                                "wall_seconds": time.time() - t0})
    else:
        raise ValueError(ev["kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_pilot.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", help="evaluation ids to run")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.smoke:
        cfg.update(cfg["smoke"])
    else:
        require_clean_tree("P1")
        preflight("P1 " + a.config)
    for ev in cfg["evaluations"]:
        if a.only and ev["id"] not in a.only:
            continue
        run_evaluation(ev, cfg, a.resume, a.smoke)


if __name__ == "__main__":
    main()
