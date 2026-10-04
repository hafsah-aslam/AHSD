# Research Questions (pre-registration)

Project: *Normality as a Filter: Spectral Detectability and Self-Inflicted
Frog-Boiling in Sequence Intrusion Detectors.*

This file is frozen before any experiment on real data runs. Its SHA-256 is
written into every result JSON (`provenance.rq_sha256`). Any later edit changes
the hash, so results produced under a different version are detectable. Every RQ
is reported whatever the answer; negative results are findings.

## Conventions fixed before any run

- **Score orientation.** Every detection score is oriented so that a higher
  value means "more attack". d′ and AUC use that orientation, and it is never
  flipped after results are seen. AUC < 0.5 is reported as an inversion.
- **Primary LOACO score (RQ1, RQ2).** The AUC of the post-hoc stress score
  `stress = mean_{t,d} |h_t − E_t|` of AHSD-fixed (γ = 0), on test windows of
  benign plus the held-out class. The supervised head's attack probability is
  also reported, as a secondary score.
- **Primary split.** `temporal_gap`. `grouped_random` is reported as a
  robustness check.
- **Predictor sign (RQ1).** Spearman ρ between each predictor and the measured
  LOACO AUC. The expected sign is declared here and the comparison uses
  ρ multiplied by that sign ("oriented ρ"):
  | Predictor | Expected sign |
  |---|---|
  | S2 `D_pred` | + |
  | Mahalanobis distance from benign | + |
  | Wasserstein distance to the training attacks (Sarhan et al. 2021) | − |
  | Stationarity index `mean‖z_t − z_{t−1}‖ / std(z)` | + |
  Raw signed ρ and |ρ| are also reported.
- **CI.** Bootstrap over (class × dataset) points, 1,000 resamples, 95%
  percentile interval.

## Questions

| RQ | Question | Test |
|---|---|---|
| RQ1 | Does the spectral predictor forecast per-class LOACO AUC? | Spearman ρ (predicted vs measured) across all classes × datasets, compared against Wasserstein, Mahalanobis, and stationarity predictors |
| RQ2 | Does it predict which classes invert (AUC < 0.5)? | Sign of the predicted in-band energy difference (`D_pred < 0` ⇒ inversion) vs observed inversion; reported as a 2×2 table with accuracy and Fisher's exact p |
| RQ3 | Does γ > 0 absorb persistent attacks without an attacker? | d′ vs γ sweep; per-class collapse vs class stationarity |
| RQ4 | Does the gated anchored equilibrium (S1) prevent absorption? | d′ and macro-F1 vs fixed/adaptive; empirical vs theoretical boil-rate bound |
| RQ5 | Does a two-sided score (S3) recover inverted classes? | AUC on predicted-inversion classes |
| RQ6 | Where does transfer break, and does re-anchoring fix it? | Layer probes, CKA, head/encoder retraining, E₀ re-estimation |
| RQ7 | Do backbone timescales predict backbone blind spots? | RQ1 repeated per backbone using each one's own H(ω) |

## P1 decision rule (fixed)

- ρ ≥ 0.6 and oriented ρ above the Wasserstein predictor's: the TDSC/TIFS paper proceeds.
- 0.3 ≤ ρ < 0.6: S2 becomes a secondary analysis.
- ρ < 0.3: report to the authors before continuing.

## Answers

None yet. Answers are filled in from `results/` JSON only, after the runs.
