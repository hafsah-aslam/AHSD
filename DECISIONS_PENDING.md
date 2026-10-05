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
