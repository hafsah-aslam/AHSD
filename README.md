# Normality as a Filter: spectral detectability and self-inflicted frog-boiling in sequence NIDS

Research code for the TDSC/TIFS project. The questions are pre-registered in
[`RESEARCH_QUESTIONS.md`](RESEARCH_QUESTIONS.md). Its SHA-256 is stored in
every result JSON.

## Status

| Phase | State |
|---|---|
| P0 code: download/manifest, audit, preprocessing, windowing, splits, LOACO folds, transfer packages, fail-closed validators, unit tests, `--smoke` | **done, tested on synthetic NF-v3-schema data** |
| P0 data: D1–D4 download, audits, archives | **blocked**: the UQ hosts are not reachable from the build environment (see below) |
| P1 code: AHSD-fixed LOACO runner, S2 predictor, competing predictors, report generator | done, smoke-tested on synthetic data |
| P1 run on D1 + D2 | waiting on data |
| P2–P8 | not started (S1 model, S1 bound, S3 score and the statistics helpers exist; baselines and other backbones do not) |

No result on real data exists. The numbers under `smoke_out/` come from
synthetic data and mean nothing. That directory is git-ignored.

## Getting the data

```bash
pip install -r requirements.txt
# Either fill verified URLs into configs/download.yaml and run:
python scripts/download.py
# or download the four v3 files from https://staff.itee.uq.edu.au/marius/NIDS_datasets/
# into data/raw/ (one file each; .csv or .zip) and register them:
python scripts/download.py --register      # writes data/MANIFEST.json (URL, SHA-256, size, rows, columns, date)
```

Raw files are never modified. A file whose SHA-256 changes after registration
is refused.

## Pipeline

```bash
python scripts/prepare.py --resume         # audit/<ds>.json, PREPROCESS_LOG.md, data/processed/...
python scripts/run_loaco.py --resume       # P1: results/p1/<ds>/<class>/seed<k>.json
python scripts/report_p1.py                # results/p1/summary.json, report/p1/*.pdf|tex, PILOT_REPORT.md
python -m pytest                           # unit tests (S2 math within 5% on sinusoids, S1 bound, validators)
```

Each entry point also takes `--smoke`, which runs end to end on synthetic
data in about 10 s.

### What `prepare.py` does (spec §4)

1. **Audit.** Rows, columns, dtypes, NaN/inf counts, class counts, timestamp
   range and a schema check, written to `audit/<ds>.json`.
2. **Sort and group.** Sort by `FLOW_START_MILLISECONDS` (stable, ties broken
   by raw row number), then cut into contiguous 1-hour groups.
3. **Window.** T = 32, stride 16, never crossing a group. Binary label = any
   attack flow. Multiclass label = majority attack class. Each window also
   stores its purity and a class-presence bitmask.
4. **Dedupe.** SHA-256 of each window's raw feature bytes. The earliest copy
   is kept globally.
5. **Split.** `temporal_gap` (60 / one discarded gap group / 20 / 20, by time)
   and `grouped_random` (60/20/20 over shuffled groups). Fractions are measured
   in windows.
6. **Balance.** Train 1:1, val/test 5:1, caps 14k/3k/3k, no replacement, exact
   ratios. Plus a natural-prior test set.
7. **LOACO folds.** One per class with ≥ 200 training windows. The fold's
   train and val sets exclude every window holding *any* flow of the held-out
   class.
8. **Cleaner, fitted on train-group rows only.** inf → NaN → train median;
   log1p on non-negative heavy-tailed counts; drop zero-variance columns; drop
   one of each |corr| ≥ 0.999999 pair; standardise. The feature list is the
   intersection across all datasets.
9. **Archive.** Compacted float32 flows plus the window index. Each archive
   file gets a SHA-256.
10. **Transfer packages.** Every target is rebuilt with the *source* cleaner,
    and nothing is fitted on the target.
11. **Verify (fail-closed).** Window hashes are unique. Groups, row IDs and
    hashes are disjoint across splits. Sets come from the right split with
    exact ratios. LOACO train holds no held-out flow. Flows are finite.

## Layout

```
src/nids/      schema, io, audit, preprocess, windowing, splits, verify, pipeline,
               models/ahsd.py (fixed/adaptive/frozen/no_stress/S1), spectral.py (S2),
               theory.py (S1 bound), scores.py (S3), predictors.py, metrics.py, train.py
scripts/       download.py, prepare.py, run_loaco.py, report_p1.py
configs/       data.yaml, download.yaml, p1_pilot.yaml
docs/          S1_BOUND.md
tests/         pytest suite
```

## Implementation choices not fixed by the spec (flag before P1 if you disagree)

- **Split fractions.** 60/20/20 is counted in windows, not in groups. Groups
  vary a lot in volume.
- **Majority ties.** Ties for a window's majority class go to the lowest class
  index. They are counted in the log.
- **E₀ in AHSD.** During training E₀ is an EMA of benign embeddings. After
  training it is set to the exact benign training mean. The head saw the EMA
  value.
- **S1 τ unit.** The median per-step benign stress `mean_d|h_t − E_{t−1}|` is
  measured before training, re-measured once after epoch 1, then frozen. The
  base γ of S1 is learned in [0, 0.2], as in the adaptive variant.
- **S2 benign reference.** Primary `D_pred` uses benign *test* windows, as the
  spec says ("embedded test windows"). Two variants are also stored:
  benign-validation reference, and per-channel H(ω).
- **Training task.** The head is binary (benign vs attack), with
  checkpointing on validation macro-F1.
- **5% S2 test.** Where the true steady-state energy is ~0 (DC under γ > 0,
  which a 32-step truncated H cannot null exactly), the test also allows an
  absolute tolerance of 1e-3 × input energy. `ir_tail_ratio` is saved with
  every S2 result as a truncation diagnostic.
