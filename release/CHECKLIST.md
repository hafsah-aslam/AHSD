# Reporting checklist

Filled by `scripts/make_release.py` from JSON. Each item names its evidence.

| Item | Status | Evidence |
|---|---|---|
| Research questions frozen before any run | yes | `RESEARCH_QUESTIONS.md`; SHA-256 recorded in every result |
| Amendments dated and hashed before the runs they govern | 6 amendments | AMENDMENT_01 (2026-10-05) `83f3ba055431…`; AMENDMENT_02 (2026-10-05) `fb0b207e5f82…`; AMENDMENT_03 (2026-10-05) `dc2efcae3986…`; AMENDMENT_04 (2026-10-05) `6a474852e58a…`; AMENDMENT_05 (2026-10-09) `5ebbbcffe106…`; AMENDMENT_06 (2026-10-09) `d8eaf3399c18…` — `docs/AMENDMENTS.json` |
| Gate verdicts kept as recorded | yes (P1, G1, G2, G3 = STOP) | `NEGATIVE_RESULTS.md`, `GATE_*.md` |
| Negative results reported | yes | `NEGATIVE_RESULTS.md` |
| Single code commit for every paper number | `e6421ea9c81ee7c9e238205eb913905dc67163a1` | `results/final/summary.json` (fails closed otherwise) |
| Clean tree + preflight seeding test before runs | yes | `nids/provenance.require_clean_tree`, `nids/preflight.py` |
| Seeds | 17, 23, 42, 101, 202 (n = 5) | every result JSON |
| Confidence intervals | Student-t, t = 2.776 (n = 5); never 1.96 | `nids/metrics.mean_ci` |
| Seed std reported for every detector × evaluation | yes | `RESULTS_SUMMARY.md` R1–R3, R7; `report/final/tables/T_seed_std.tex` |
| Score orientation fixed (higher = attack), never flipped | yes | `nids/detectors.py`, AMENDMENT_03/06 |
| Detector families separated (no unmatched pooling) | yes | AMENDMENT_06 A6.2 |
| Post-hoc additions labelled | P(attack) marked † (added post-G3) | AMENDMENT_06 A6.2 |
| No random flow-level splits | yes: 1-h groups (D5 purged_block: 10-min blocks) | `nids/splits.py`, `release/splits` |
| Group and row disjointness across splits verified (fail closed) | yes | `nids/verify.py`; `verification` in each meta JSON |
| Global window dedupe (SHA-256) | yes | `nids/windowing.dedupe`; `index_meta.json` |
| No statistic fitted on test or target data | yes | cleaner fitted on train rows; transfer targets rebuilt with the source cleaner |
| Test split never used for selection | yes | checkpoint on validation; thresholds fixed at 0.5 for P(attack)/XGBoost |
| Thin-validation caveat | reported | D5 train-tail validation 26 attack windows; checkpoint sensitivity R6/F6 |
| Protocol limitations | reported | A5.2 fallback; D5 purged_block single fold; D5 NN 2 classes; AHSD-only saturation table |
| Raw data unmodified; SHA-256 recorded | 5 datasets | `data/MANIFEST.json` |
| No Kaggle or unofficial mirrors | yes | `data/MANIFEST.json` URLs |
| Splits released | 21 files | `release/splits/MANIFEST.json` |
| Tables and figures generated from JSON only | yes | `scripts/report_final.py` → `report/final/` |
| Pending author decisions listed, not guessed | yes | `DECISIONS_PENDING.md` (incl. C1–C6 text, item 8) |
| Release commit | `dfc3fb8a4a014d6db83a548fadc6b313c03f3bea` | `release/FILES.json` |
