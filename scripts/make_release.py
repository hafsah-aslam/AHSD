#!/usr/bin/env python
"""Release package (AMENDMENT_06 outputs): splits, file manifest, reporting checklist.

Writes release/:
  splits/<name>.npz   per archive used by the final run: a window table (start position in the
                      time-sorted flow array, raw row id of the first flow, 1-h/10-min group) and every
                      set / LOACO fold as int32 indices into that table; plus group -> split codes.
  splits/MANIFEST.json  SHA-256 of every split file and of the source archive files it was read from.
  FILES.json          SHA-256 of the code, configs, amendments and reports at this commit.
  CHECKLIST.md        reporting checklist, filled from JSON (amendments, manifest, final-run summary).
  RELEASE.md          contents and reproduction steps.
Reads data/processed (never writes there). Run from a clean tree after the final report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nids.pipeline import load_folds  # noqa: E402
from nids.provenance import ROOT, provenance, sha256_file, write_json  # noqa: E402
from nids.windowing import WindowIndex  # noqa: E402

PROC = ROOT / "data/processed"
OUT = ROOT / "release"
SPLIT_NAMES = {0: "train", 1: "gap", 2: "val", 3: "test"}


def archives():
    """(release name, archive dir, sets file, fold files, split-from-windows)"""
    A = []
    for ds in ("D1", "D2", "D3", "D5"):
        for sch in ("temporal_gap", "grouped_random"):
            d = PROC / ds / sch / "archive"
            A.append((f"{ds}_{sch}", d, d / "sets.npz",
                      {k: d / f for k, f in (("loaco", "loaco.npz"), ("loaco_family", "loaco_family.npz")) if (d / f).exists()}))
    A.append(("D1_natural_novelty_p1c", PROC / "D1/temporal_gap/p1c", PROC / "D1/temporal_gap/p1c/sets.npz", {}))
    A.append(("D3_natural_novelty_p1c", PROC / "D3/temporal_gap/p1c", PROC / "D3/temporal_gap/p1c/sets.npz", {}))
    A.append(("D5_natural_novelty_p1d", PROC / "D5/temporal_gap/p1d", PROC / "D5/temporal_gap/p1d/sets.npz", {}))
    d = PROC / "D5/purged_block/archive"
    A.append(("D5_purged_block", d, d / "sets.npz", {"loaco": d / "loaco.npz"}))
    for p in sorted((PROC / "transfer").glob("*__to__*")):
        d = p / "temporal_gap"
        A.append((f"transfer_{p.name}_temporal_gap", d, d / "sets.npz", {}))
    return A


def export(name, d, sets_path, fold_paths):
    z = np.load(d / "windows.npz")
    wi = WindowIndex.from_npz(z)
    start_c, split = z["start_compact"], z["split"]
    row_id = np.load(d / "row_id.npy", mmap_mode="r")
    sets = {k: v for k, v in np.load(sets_path).items()}
    folds = {k: load_folds(p) for k, p in fold_paths.items()}
    ref = [v for v in sets.values()] + [a for fs in folds.values() for f in fs.values() for a in f.values()]
    used = np.unique(np.concatenate(ref)) if ref else np.zeros(0, np.int64)
    if (start_c[used] < 0).any():
        raise SystemExit(f"{name}: referenced window not materialised")
    pos = {int(w): i for i, w in enumerate(used)}
    lut = np.full(len(wi), -1, np.int64)
    lut[used] = np.arange(len(used))
    out = {"window_start": wi.start[used].astype(np.int64), "window_first_row_id": np.asarray(row_id)[start_c[used]].astype(np.int64),
           "window_group": wi.group[used].astype(np.int32), "window_y_bin": wi.y_bin[used], "window_y_cls": wi.y_cls[used],
           "T": np.array(wi.T), "stride": np.array(wi.stride)}
    for k, v in sets.items():
        out[f"set__{k}"] = lut[v].astype(np.int32)
    for fk, fs in folds.items():
        for cls, f in fs.items():
            for part, idx in f.items():
                out[f"{fk}__{cls}__{part}"] = lut[idx].astype(np.int32)
    if (split >= 0).any():
        g, first = np.unique(wi.group, return_index=True)
        out["group_ids"], out["group_split"] = g.astype(np.int32), split[first].astype(np.int8)
    assert len(pos) == len(used)
    path = OUT / "splits" / f"{name}.npz"
    np.savez_compressed(path, **out)
    src = {f: sha256_file(d / f) for f in ("windows.npz", "row_id.npy") if (d / f).exists()}
    src[sets_path.name] = sha256_file(sets_path)
    for k, p in fold_paths.items():
        src[p.name] = sha256_file(p)
    return {"file": f"splits/{name}.npz", "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "windows": int(len(used)), "sets": sorted(sets), "folds": {k: sorted(v) for k, v in folds.items()},
            "source_dir": str(d.relative_to(ROOT)), "source_sha256": src}


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def checklist(summary, amend, manifest, files_commit):
    am = amend["amendments"]
    L = ["# Reporting checklist", "", "Filled by `scripts/make_release.py` from JSON. Each item names its evidence.", "",
         "| Item | Status | Evidence |", "|---|---|---|"]

    def row(item, status, ev):
        L.append(f"| {item} | {status} | {ev} |")

    row("Research questions frozen before any run", "yes", "`RESEARCH_QUESTIONS.md`; SHA-256 recorded in every result")
    row("Amendments dated and hashed before the runs they govern", f"{len(am)} amendments",
        "; ".join(f"{a['id']} ({a['date']}) `{a['new_sha256'][:12]}…`" for a in am) + " — `docs/AMENDMENTS.json`")
    row("Gate verdicts kept as recorded", "yes (P1, G1, G2, G3 = STOP)", "`NEGATIVE_RESULTS.md`, `GATE_*.md`")
    row("Negative results reported", "yes", "`NEGATIVE_RESULTS.md`")
    row("Single code commit for every paper number", f"`{summary['code_commit']}`", "`results/final/summary.json` (fails closed otherwise)")
    row("Clean tree + preflight seeding test before runs", "yes", "`nids/provenance.require_clean_tree`, `nids/preflight.py`")
    row("Seeds", "17, 23, 42, 101, 202 (n = 5)", "every result JSON")
    row("Confidence intervals", "Student-t, t = 2.776 (n = 5); never 1.96", "`nids/metrics.mean_ci`")
    row("Seed std reported for every detector × evaluation", "yes", "`RESULTS_SUMMARY.md` R1–R3, R7; `report/final/tables/T_seed_std.tex`")
    row("Score orientation fixed (higher = attack), never flipped", "yes", "`nids/detectors.py`, AMENDMENT_03/06")
    row("Detector families separated (no unmatched pooling)", "yes", "AMENDMENT_06 A6.2")
    row("Post-hoc additions labelled", "P(attack) marked † (added post-G3)", "AMENDMENT_06 A6.2")
    row("No random flow-level splits", "yes: 1-h groups (D5 purged_block: 10-min blocks)", "`nids/splits.py`, `release/splits`")
    row("Group and row disjointness across splits verified (fail closed)", "yes", "`nids/verify.py`; `verification` in each meta JSON")
    row("Global window dedupe (SHA-256)", "yes", "`nids/windowing.dedupe`; `index_meta.json`")
    row("No statistic fitted on test or target data", "yes", "cleaner fitted on train rows; transfer targets rebuilt with the source cleaner")
    row("Test split never used for selection", "yes", "checkpoint on validation; thresholds fixed at 0.5 for P(attack)/XGBoost")
    row("Thin-validation caveat", "reported", "D5 train-tail validation 26 attack windows; checkpoint sensitivity R6/F6")
    row("Protocol limitations", "reported", "A5.2 fallback; D5 purged_block single fold; D5 NN 2 classes; AHSD-only saturation table")
    row("Raw data unmodified; SHA-256 recorded", f"{len(manifest['datasets'])} datasets", "`data/MANIFEST.json`")
    row("No Kaggle or unofficial mirrors", "yes", "`data/MANIFEST.json` URLs")
    row("Splits released", f"{len(json.loads((OUT / 'splits/MANIFEST.json').read_text())['files'])} files", "`release/splits/MANIFEST.json`")
    row("Tables and figures generated from JSON only", "yes", "`scripts/report_final.py` → `report/final/`")
    row("Pending author decisions listed, not guessed", "yes", "`DECISIONS_PENDING.md` (incl. C1–C6 text, item 8)")
    row("Release commit", f"`{files_commit}`", "`release/FILES.json`")
    return "\n".join(L) + "\n"


def main():
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("dirty tree; commit first")
    (OUT / "splits").mkdir(parents=True, exist_ok=True)
    files = []
    for name, d, sp, fp in archives():
        files.append(export(name, d, sp, fp))
        print(f"{name}: {files[-1]['windows']} windows, {files[-1]['bytes'] / 1e6:.2f} MB", flush=True)
    write_json(OUT / "splits/MANIFEST.json", {
        "files": files,
        "reconstruction": ("Rebuild data/interim/<ds>/sorted.parquet with scripts/download.py + scripts/prepare.py from the raw "
                           "files listed in data/MANIFEST.json (SHA-256 checked). Flows are stably sorted by "
                           "FLOW_START_MILLISECONDS (ties keep raw row order; row_id = raw row number). A window is the T = 32 "
                           "consecutive sorted flows starting at window_start; window_first_row_id checks the alignment. "
                           "Sets and folds are int32 indices into the window table. group_split: 0 train, 1 gap, 2 val, 3 test."),
        "provenance": provenance()})
    head = git("rev-parse", "HEAD")
    tracked = [p for p in git("ls-files").splitlines()
               if p.startswith(("src/", "scripts/", "configs/", "tests/", "docs/")) or p in (
                   "RESEARCH_QUESTIONS.md", "DECISIONS_PENDING.md", "RESULTS_SUMMARY.md", "NEGATIVE_RESULTS.md",
                   "PREPROCESS_LOG.md", "README.md", "pyproject.toml", "requirements.txt", "data/MANIFEST.json")]
    write_json(OUT / "FILES.json", {"commit": head, "files": {p: sha256_file(ROOT / p) for p in sorted(tracked)}})
    summary = json.loads((ROOT / "results/final/summary.json").read_text())
    amend = json.loads((ROOT / "docs/AMENDMENTS.json").read_text())
    manifest = json.loads((ROOT / "data/MANIFEST.json").read_text())
    (OUT / "CHECKLIST.md").write_text(checklist(summary, amend, manifest, head))
    (OUT / "RELEASE.md").write_text("\n".join([
        "# Release package", "",
        "Pre-registered benchmark of novelty detection protocols for sequence NIDS (AMENDMENT_06). Generated by "
        "`scripts/make_release.py`.", "",
        "## Contents", "",
        "- Code: `src/nids/`, `scripts/` (entry points below); tests: `tests/`.",
        "- Configs: `configs/` (`data.yaml`, class maps).",
        "- Pre-registration and amendments: `RESEARCH_QUESTIONS.md`, `docs/AMENDMENTS.json` (dated SHA-256 chain).",
        "- Splits: `release/splits/*.npz` + `release/splits/MANIFEST.json`.",
        "- Results: `results/final/**` (one JSON per evaluation × unit × seed), `results/final/summary.json`.",
        "- Report: `RESULTS_SUMMARY.md`, `NEGATIVE_RESULTS.md`, `report/final/tables/*.tex`, `report/final/figures/*.pdf`.",
        "- Checklist: `release/CHECKLIST.md`; file hashes: `release/FILES.json`.", "",
        "## Reproduce", "",
        "```", "python scripts/download.py          # official UQ / Lycos sources; SHA-256 checked",
        "python scripts/d5_build_source.py   # D5 (Lycos2017) labelling, authors' labelling.py",
        "python scripts/prepare.py           # audit, clean, window, split, archives, transfer packages",
        "python scripts/p1c_build.py; python scripts/p1d_build.py; python scripts/d5_purged_block.py",
        "python scripts/final_run.py         # refuses a dirty tree; preflight seeding test",
        "python scripts/report_final.py      # tables, figures, summaries from JSON",
        "python scripts/make_release.py", "```", "",
        f"Release commit: `{head}`. Final-run code commit: `{summary['code_commit']}`.", ""]))
    print(f"release: {len(files)} split files, {sum(f['bytes'] for f in files) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
