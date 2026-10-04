#!/usr/bin/env python
"""P1 pilot: AHSD-fixed LOACO + S2 predictor + competing predictors (spec §8 P1).

One JSON per (dataset, held-out class, seed) under results/p1/. Nothing is
aggregated here; scripts/report_p1.py builds tables, figures and the report.

  python scripts/run_loaco.py --config configs/p1_pilot.yaml [--resume]
  python scripts/run_loaco.py --config configs/p1_pilot.yaml --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids import metrics, predictors, scores, spectral  # noqa: E402
from nids.models.ahsd import AHSD  # noqa: E402
from nids.pipeline import Archive  # noqa: E402
from nids.provenance import ROOT, provenance, sha256_file, write_json  # noqa: E402
from nids.train import fit, predict  # noqa: E402


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


def run_fold(arc: Archive, cls: str, seed: int, cfg: dict, log) -> dict:
    f = arc.folds[cls]
    tr, va, te = arc.windows(f["train"]), arc.windows(f["val"]), arc.windows(f["test"])
    c = arc_classes(arc).index(cls)
    torch.set_num_threads(cfg.get("threads", 4))
    m = AHSD(arc.n_features, D=cfg["D"], variant="fixed")
    tinfo = fit(m, tr, va, seed=seed, epochs=cfg["epochs"], log=log)

    st = predict(m, te["X"], states=True)
    sv = predict(m, va["X"], states=True)
    trb = predict(m, tr["X"][tr["y_bin"] == 0], states=True)
    y = te["y_bin"]
    is_c, is_b = te["y_cls"] == c, te["y_cls"] == 0
    assert np.all(is_c | is_b)

    T = arc.wi.T
    s2 = s2_block(m, trb, {"class": st["z"][is_c], "benign": st["z"][is_b],
                          "val_benign": sv["z"][va["y_bin"] == 0]}, T)
    D_pred = spectral.detectability(s2["S_class"], s2["S_benign"])
    D_pred_val = spectral.detectability(s2["S_class"], s2["S_val_benign"])
    D_pred_ch = spectral.detectability(s2["Sch_class"], s2["Sch_benign"])
    for k in ("S_class", "S_benign", "S_val_benign", "Sch_class", "Sch_benign", "Sch_val_benign"):
        s2[k] = {"mean": float(np.mean(s2[k])), "std": float(np.std(s2[k], ddof=1)), "n": int(len(s2[k]))}

    mah = predictors.BenignMahalanobis(tr["X"][tr["y_bin"] == 0])
    comp = {
        "wasserstein": predictors.wasserstein_to_train_attacks(te["X"][is_c], tr["X"][tr["y_bin"] == 1], seed=seed),
        "mahalanobis": float(mah(te["X"][is_c]).mean()),
        "stationarity": float(predictors.stationarity_index(st["z"][is_c]).mean()),
    }
    vb = va["y_bin"] == 0
    s3 = scores.s3_score(st["stress"], st["delta_norm"], sv["stress"][vb], sv["delta_norm"][vb])
    thr = metrics.threshold_at_val_fpr(sv["stress"][vb], 0.01)
    return {
        "dataset": cfg["_ds"], "scheme": cfg["scheme"], "held_out": cls, "seed": seed,
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
        "training": tinfo,
        "gamma_final": float(m.gamma().detach()),
    }


def arc_classes(arc: Archive) -> list[str]:
    meta = json.loads((arc.path.parents[1] / "index_meta.json").read_text())
    return meta["classes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p1_pilot.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / a.config))
    if a.smoke:
        cfg.update(cfg["smoke"])
    proc = ROOT / cfg["processed_dir"]
    out_root = ROOT / cfg["results_dir"]
    for ds in cfg["datasets"]:
        arc = Archive(proc / ds / cfg["scheme"] / "archive")
        meta = json.loads((proc / ds / "index_meta.json").read_text())
        finfo = meta["schemes"][cfg["scheme"]]["loaco"]
        data_sha = json.loads((proc / ds / "archive_meta.json").read_text())["schemes"][cfg["scheme"]]["sha256"]
        for cls, rec in finfo.items():
            if not isinstance(rec, dict) or not rec.get("eligible") or not rec.get("feasible"):
                write_json(out_root / ds / f"{_safe(cls)}__skipped.json",
                           {"dataset": ds, "held_out": cls, "skipped": True, "fold_info": rec,
                            "provenance": provenance(config=cfg)})
                continue
            for seed in cfg["seeds"]:
                path = out_root / ds / _safe(cls) / f"seed{seed}.json"
                if a.resume and path.exists():
                    continue
                print(f"[{ds}] LOACO {cls} seed {seed}")
                res = run_fold(arc, cls, seed, {**cfg, "_ds": ds}, log=print)
                res["provenance"] = provenance(config=cfg, data_sha256=data_sha)
                res["smoke"] = bool(a.smoke)
                write_json(path, res)


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


if __name__ == "__main__":
    main()
