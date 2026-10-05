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
