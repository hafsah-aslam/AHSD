# Decisions pending (authors)

Items that need an author decision. Nothing here has been resolved by choice.
Numbers come from `results/p1/summary.json` and `results/p1/gate.json`.

## 1. P1 gate = STOP (2026-10-05)
- `GATE_DECISION.md`: pooled ρ_S2 = 0.326 (< 0.6). The 95% CI of ρ_S2 − ρ_W
  includes 0 under both the oriented and the raw-sign Wasserstein variant.
- Per the frozen gate rule, P2 has not started, and `scripts/run_all.py` has
  not been built or run.
- Decision needed: stop, revise the plan, or proceed under a new pre-registered
  rule.
- The RQ file's own decision rule, applied to set (b), gives
  "S2 becomes a secondary analysis (0.3 ≤ ρ < 0.6)". This differs from the
  gate's STOP; both are reported.

## 2. D1 natural_novelty is infeasible as written (AMENDMENT_01 A1.3)
- Every `temporal_gap` validation window of D1 (33,826) contains at least one
  Infilteration flow. Excluding them leaves the validation set empty, so no
  checkpoint can be selected.
- D1_nn was recorded as skipped (fail closed).
- Decision needed: a validation protocol for D1 natural_novelty, or drop it.

## 3. Observations relevant to items 1–2 (facts only, no proposal adopted)
- Within D1 families, S2's ρ = 0.886 (95% CI [0.455, 1.000], n = 6).
- Pooled with D2 and D3, S2's ρ = 0.326 (95% CI [−0.248, 0.719], n = 20).
- D2 classes have D_pred near 0 (−0.14 to 0.12) but stress AUC of 0.886–0.971.
- Mahalanobis has the highest pooled ρ: 0.611 (95% CI [0.161, 0.869]).
- RQ2:
  - No LOACO class inverted (0 of 20 pooled points).
  - S2 predicted inversion for 3 D2 `grouped_random` classes and D3 scanning,
    and for all 8 D2 `temporal_gap` classes. None inverted.
  - All 5 D3 natural_novelty classes have stress AUC < 0.5. Their seed CIs are
    wide (up to ±0.54); S2 predicted inversion for 1 of the 5 (mitm).

## 4. Gate G1 = STOP (AMENDMENT_02 A2.6; 2026-10-05)
- `GATE_G1_DECISION.md`: 0 of 3 datasets pass both criteria.
  - (a) fails everywhere: ρ(γ, mean test d′) = -0.143 / -0.143 / 0.000 (D1 / D2 / D3).
  - (b) passes everywhere: S1 d′ ratio 1.108 / 1.145 / 0.954; macro-F1 drop -1.36 / -0.06 / 0.04 points.
- Facts:
  - Test d′ is not monotone in γ: it peaks at γ = 0.02 / 0.02 / 0.02 and is lowest at γ = 0.2 on all three datasets.
  - Learned adaptive γ stays near its initial value of 0.05 (final 0.0484 / 0.0479 / 0.0473).
- P2 has not started. Decision needed on the plan.

## 5. S2-dir normaliser for D1 families / Bot
- The first P1 training (Bot, seed 17) cannot be reproduced, so Bot's normaliser std(Q_b) uses 2 seeds instead of 3.
- Its S2-dir value (3689.1) is far larger than the other folds'. The normaliser divides by a 2-sample std of 0.000207.
- Spearman ρ is rank-based, so the extreme value is bounded in effect, but the normalisation itself is fragile with n ≤ 3 seeds.
- S2-dir is EXPLORATORY; no action taken.

## 6. Gate G2 = STOP (AMENDMENT_03 A3.4; 2026-10-05)
- `GATE_G2_DECISION.md`: criteria (a) 4/9 (needs 5), (b) 5/9 (pass), (c) 0/9.
- PLAN_P2_v2.md was not drafted (only for GO). P2 has not started.
- Facts from `P1C_REPORT.md` (no interpretation adopted):
  - On D3, the four benign-only detectors (IF, OCSVM, PCA, AE) reach mean AUC
    0.70–0.99 on every natural-novelty class.
  - On D3, the five backbone-based scores (AHSD stress, MSP, Energy,
    Mahalanobis, kNN) range from 0.13 to 0.67, with wide seed CIs.
  - On D1, every detector lies between 0.38 and 0.69.
  - Re-anchoring with N = 500 raises no detector's AUC by ≥ 0.15 on any class.
  - On D3, re-anchoring lowers the backbone-based scores. For example, AHSD
    stress Backdoor goes 0.67 → 0.29, and kNN xss 0.49 → 0.03.
  - Benign drift d′ (test vs train-period benign) is large on D3 for the
    backbone-based scores (AHSD 2.69, Mahalanobis 2.63, kNN 2.73) and small on
    D1 (≤ 0.19 for all detectors).
- Decision needed on the plan.

## 7. P1d stopped before any model run: D5 LOACO infeasible (AMENDMENT_04)
- Status: `P1D_REPORT.md`, `GATE_G3_DECISION.md` (G3 not evaluated).
- D5 (Lycos2017) P0 is complete and validated:
  - 1,837,498 flows; label counts identical to the authors' labelling log;
  - 114,783 windows in 42 one-hour groups; 73 own-space features.
- **D5 LOACO (`grouped_random`) has 0 feasible folds.** Every class with
  ≥ 200 training windows is absent from the test split, because each attack
  occupies a few hours of a 5-day capture.
- **D5 natural_novelty is feasible.**
  - Classes: ddos and portscan (6,639 and 1,335 test windows).
  - Train-tail validation is thin: 130 benign + 26 attack windows
    (22 heartbleed, 4 dos_slowloris).
- Nothing else in A4.5 was run.
- Decision needed: how to obtain the D5 LOACO arm, or how G3 should treat its
  absence. Options, none adopted:
  1. A D5-specific LOACO split. For example, a shorter block than 1 hour,
     or a day-stratified group split. Either is a new protocol and needs an
     amendment before any run.
  2. Gate (b) on D3 only, with D5 contributing (a) only. This changes the
     frozen G3.
  3. The Wilkie et al. protocol (flow-level 50/50 split, with SQLi and
     Heartbleed held out). This conflicts with the spec's "no random
     flow-level splits", and Heartbleed (22) and SQLi (23 windows) are below
     the 200-window LOACO threshold.

## 8. C1–C6 contribution text (AMENDMENT_06 A6.1) — RESOLVED 2026-10-09
- The authors' decision says the paper is the pre-registered benchmark
  "C1–C6 in my plan". No C1–C6 list exists in this repository or in the
  original project spec.
- `RESULTS_SUMMARY.md` is to be organised by C1–C6. Until the text is
  supplied, it is organised by evaluation, with a C1–C6 mapping left for the
  authors. No contribution wording has been invented.
- **Resolved (2026-10-09):** the authors supplied C1–C6. `RESULTS_SUMMARY.md` is organised by
  them, with automatic evidence checks; the per-evaluation results (R1–R7) are kept as an appendix.
  No number changed.

## 9. Saturation table scope (AMENDMENT_06) — RESOLVED 2026-10-09
- "Supervised backbones vs XGBoost": the only supervised neural backbone
  implemented is AHSD (fixed). The other spec §5 backbones were never built,
  because P2 never started.
- The table compares AHSD P(attack) with XGBoost and states this limitation.
- **Resolved (2026-10-09):** no new backbones, no further runs. The section is renamed
  "Temporal-split degradation: AHSD vs XGBoost", and the single-backbone scope is stated as a
  limitation.

## 10. Contribution wording vs evidence (C2, C4, C5; found 2026-10-09) — RESOLVED 2026-10-10
The automatic evidence checks in `RESULTS_SUMMARY.md` flag the following. No number was changed and
no claim was reworded.
- **C4 — not supported as worded for P(attack).**
  - MSP and Energy are below 0.5 in all 4 named evaluations.
  - P(attack) is not below 0.5 in any of them (mean AUC 0.999 / 0.752 / 0.853 / 0.575).
  - P(attack) falls below 0.5 on single classes only: D3 LOACO ddos and scanning, and D5 natural
    novelty portscan.
  - The family mean is below 0.5 because of MSP and Energy.
  - Decision needed: the wording, e.g. restrict C4 to MSP and Energy.
- **C2 — qualifications.**
  - XGBoost (the supervised reference) is higher than the leading novelty family in LOACO D2,
    natural novelty D3 and natural novelty D5.
  - On natural novelty D1 the representation family is 0.623 [0.562, 0.692], so "near chance
    for all families" does not hold for it.
- **C5 — holds on D3 only, with one backbone.**
  - D1: both models drop.
  - D2: neither drops.
  - D5: temporal_gap scores higher than grouped_random for both, and grouped_random favours
    P(attack) (0.831 vs 0.705).
- **Resolved (2026-10-10):** the authors adopted revised wording for C2, C4 and C5
  (C5 renamed "Temporal and cross-dataset generalization"). No number changed. The automatic evidence
  checks in `RESULTS_SUMMARY.md` test every sub-claim against the JSON; all six (C1–C6) read "holds"
  (also in `results/final/summary.json`, `contribution_checks`).

## 11. 100 GB dataset — RESOLVED 2026-10-09
Not joining this paper. No amendment, no run.
