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

---

## AMENDMENT_01 (2026-10-05, before any model was trained on real data)

- **Supersedes:** RESEARCH_QUESTIONS.md with SHA-256
  `c2c9bd1be22c0bf25f8e4c7ff50c91cd226d2028742b4d8da622c4fe72e6943e`
  (commit 60f2b42).
- **New SHA-256:** a file cannot contain its own hash. The post-amendment hash
  is in `docs/AMENDMENTS.json`, next to the old one. Every result JSON stores
  the hash in force when it was produced (`provenance.rq_sha256`).
- **Why:** the P0 audit of the real data (`P0_REPORT.md`) showed the
  following. Under `temporal_gap`, D1 and D3 have no feasible
  leave-one-attack-class-out (LOACO) fold, because each attack class is
  recorded on its own day(s). D4 cannot be split in time without an empty
  validation split. D1's file carries 14 sub-labels where UQ reports 6
  families.
- **Basis:** no detector has been trained or scored on real data before this
  amendment. Its only inputs were audits, window and split statistics, and
  fold-feasibility counts.

### A1.1 LOACO protocol
- LOACO uses `grouped_random` (60/20/20 over shuffled 1-hour groups) for D1,
  D2 and D3.
- `temporal_gap` stays the primary split for in-distribution evaluation and
  for transfer.
- D2 LOACO runs under both `temporal_gap` and `grouped_random`. The
  per-class difference is reported.

### A1.2 D1 class granularity
- The primary D1 LOACO classes are the six UQ attack families: Bot,
  BruteForce, DDoS, DoS, Infiltration, Web Attacks.
- The 14 file sub-labels are a secondary "variant novelty" analysis. A
  held-out sub-label's sibling sub-labels stay in training.
- The mapping is `configs/d1_class_map.yaml`. It is validated against the UQ
  page by `scripts/validate_class_map.py`: each family's sub-label counts sum
  exactly to the UQ figure.

### A1.3 natural_novelty (new evaluation)
- Split: `temporal_gap`.
- Classes: those present in the test split but absent from the train split.
  - D1: Bot, Infilteration.
  - D3: Backdoor, mitm, password, ransomware, xss (file labels).
  - The pipeline recomputes these lists and stops if they differ.
- Training and validation sets: the `temporal_gap` sets, after removing every
  window that contains any flow of a natural-novelty class.
- Test set, per class: benign plus windows whose majority class is that
  class, 5:1, cap 3,000, from the `temporal_gap` test split.
- Scores: AHSD stress, plus all §7B baselines (the benign-only detectors of
  §7: Isolation Forest, OCSVM, LOF, PCA reconstruction, Autoencoder, Deep
  SVDD, KitNET).
  - P1 (AHSD-fixed only) reports stress, the supervised probability and S3.
  - The §7B baselines run in P3.

### A1.4 D4 (transfer target only)
- D4 has no train/validation/test split, no LOACO, and is not used for the
  common feature list.
- Feature list: the frozen 41-feature list from D1–D3.
- Packages: each D4 transfer package is built from D4 windows using the
  source's medians, μ and σ. It stops if any of the 41 features is missing
  in D4.
- **Re-anchoring pool:** the benign windows of the first four 1-hour groups,
  in time order (groups 0–3, 522 benign windows). The S4 re-anchoring subsets
  of 100 and 500 windows are nested random draws from this pool (seed 0).
- **Test set:** all windows from the remaining groups.
  - Benign: all of them (793).
  - Attack: one fifth as many (5:1), split equally across DDoS, DoS,
    Reconnaissance and Theft, with any shortfall in one class redistributed
    to the others.
  - The pool and the test set share no group, and therefore no flow.
  - This replaces "test = all 1,315 benign": that is incompatible with a
    re-anchoring set disjoint from the test set.

### A1.5 P1 reporting and decision rule
- Spearman ρ with a 95% bootstrap CI is reported separately for:
  - (a) D1 families only;
  - (b) the pooled set: D1 families + D2 (`grouped_random`) + D3 (`grouped_random`);
  - (c) D1 sub-labels.
- D2 (`temporal_gap`) and natural_novelty points are reported in the
  per-class table. They are not part of (a)–(c).
- The P1 decision rule in this file applies to set (b), using oriented ρ.
- Inversion is predicted when D_pred < 0 and observed when the stress AUC is
  below 0.5. Both are tabulated per class.

### A1.6 Dataset discrepancies (no relabelling)
- File labels are kept as they are.
- `results/dataset_discrepancies.json` records two discrepancies for the
  paper:
  - the D2 Backdoor/Shellcode/Analysis count permutation relative to the UQ
    page;
  - the 36-row D4 Theft gap (1,615 in the file vs 1,651 on the page).

---

## AMENDMENT_02 (2026-10-05, after P1 and its gate, before any P1b run)

- **Supersedes:** RESEARCH_QUESTIONS.md with SHA-256
  `83f3ba055431860ef62e45e11c920fe6f08a533b81acdd06431a9c9a58e3ad04`
  (AMENDMENT_01, commit 32333f6).
- **New SHA-256:** recorded in `docs/AMENDMENTS.json`.
- **Context:**
  - P1 results exist: `PILOT_REPORT.md`.
  - The P1 gate (`GATE_DECISION.md`) returned STOP.
  - The authors accepted STOP for P2 under the plan in force.
- **Status:** this amendment is written and hashed before any P1b run. It does
  not change any P1 result or definition.

### A2.1 S2 status
- Under the original rule (0.3 ≤ ρ < 0.6), S2 is a secondary analysis.
- Pooled oriented ρ = 0.326.
- The P1 gate outcome (STOP) and both Wasserstein readings (oriented and raw
  sign) stay in the record (`GATE_DECISION.md`, `results/p1/gate.json`).

### A2.2 Secondary analysis (declared now, not gating)
- Compute the Spearman ρ between predictor and measured LOACO stress AUC
  within each dataset:
  - D1 families;
  - D2 `grouped_random`;
  - D3 `grouped_random`.
- Report the mean of the three within-dataset ρ for S2, Wasserstein,
  Mahalanobis and stationarity, alongside the pooled ρ.
- Signs are oriented as pre-registered (Wasserstein −).
- **Uncertainty:** stratified bootstrap, resampling classes within each dataset
  (2000 resamples, seed 0), with a percentile 95% CI.

### A2.3 S2-dir (EXPLORATORY; never used in any gate)
Definition (frozen by the authors):
- Whiten embedded windows with the benign training covariance:
  u = W (z − E₀), W = Σ_benign^(−1/2), shrinkage 0.1.
- Compute per-channel spectra P_{c,d}(ω) of u.
- S2dir(c) = Σ_d Σ_ω |H(ω)|² · (P_{c,d}(ω) − P_{benign,d}(ω))², normalised by
  the benign seed-to-seed std of the same quantity.
- Report its ρ against measured AUC, with a bootstrap CI, alongside
  Mahalanobis.

Operational choices added by the implementer (the definition above leaves
them open):
- **Covariance:** Σ_benign is the covariance of (z − E₀) over all time steps of
  the fold's benign training windows. Shrinkage: Σ_s = 0.9·Σ + 0.1·(tr Σ / D)·I.
  W = Σ_s^(−1/2) via eigendecomposition.
- **Filter:** H is the same linearised AHSD filter as S2 (mean gates over
  benign training windows; 32-step impulse response).
- **Power:** one-sided and normalised as in S2, but not averaged over
  channels.
- **Benign reference:** the fold's benign test windows (as in primary S2).
- **Normaliser** (the "same quantity" on benign data):
  - per seed, split the fold's benign test windows into two random halves
    (seed-specific split);
  - compute Q_b = Σ_d Σ_ω |H|²(P_{half A,d} − P_{half B,d})²;
  - the normaliser is the std (ddof = 1) of Q_b across the fold's 3 seeds;
  - S2dir is reported per seed and averaged over seeds.
- **Embeddings without retraining:** P1 stored no checkpoints or embeddings,
  so S2-dir is computed by re-executing each P1 LOACO run deterministically
  (same code path, seed, data and thread count).
  - The re-executed run must reproduce the recorded P1 stress AUC, probability
    AUC and D_pred to within 1e-6.
  - Any mismatch fails closed, and S2-dir is not reported for that run.
  - Re-executed runs produce no new measured results; the measured AUC stays
    the P1 value.
  - Scope: the pooled-set evaluations, D1 families, D2 `grouped_random` and
    D3 `grouped_random` (60 runs).

### A2.4 D1 natural_novelty: DROPPED
- It is infeasible as specified in A1.3.
- Every D1 `temporal_gap` validation window (33,826 windows) contains at
  least one Infilteration flow. Excluding windows with natural-novelty flows
  therefore empties the validation set, and no checkpoint can be selected.
- This is documented in `PILOT_REPORT.md`.

### A2.5 D3 natural_novelty: 8 seeds
- Seeds: {17, 23, 42} plus {101, 202, 303, 404, 505}.
- Report per class:
  - mean stress AUC;
  - Student-t 95% CI (n = 8);
  - the fraction of seeds with AUC < 0.5.
- No interpretation until all 8 seeds exist.
- **Implementation:** all 8 seeds run on one P1b code commit, so the
  reported quantity has a single code version. The 3 original seeds are
  re-run, and their reproduction of the P1 values is checked and reported.

### A2.6 Gate G1 (frozen). Experiment P1b-G1
- **Data:** D1, D2, D3; `grouped_random`; in-distribution (no LOACO).
  - The balanced train/val/test sets of the archive: train 1:1, val/test 5:1.
- **Models:**
  - AHSD frozen_γ, γ ∈ {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2};
  - AHSD adaptive (γ learned in [0, 0.2], logged every epoch);
  - S1 gated-anchored, τ ∈ {0.5, 1, 2} × median benign validation stress,
    κ ∈ {0.01, 0.05}.
- **S1 selection:** on validation macro-F1 only. Per dataset, choose the
  (τ, κ) with the highest mean (over seeds) best-validation macro-F1. Ties go
  to the smaller τ, then the smaller κ. Then measure test d′.
- **Seeds:** {17, 23, 42}.
- **GO** if both of the following hold on at least 2 of the 3 datasets:
  - (a) Spearman(γ, mean test d′) ≤ −0.8 across the 7 frozen γ;
  - (b) S1 test d′ ≥ 0.8 × fixed (γ = 0) test d′, AND the S1 test macro-F1
    drop versus fixed is < 1.0 point (means over seeds).
  Otherwise STOP and report.
- **Descriptive (not gating):**
  - per-class collapse ratio d′_γ / d′_0 against the class stationarity index;
  - the learned adaptive γ.

Operational choices added by the implementer:
- **d′:** computed on the post-hoc stress score (higher = attack, never
  flipped), on the test set. Mean = mean over seeds.
- **Macro-F1:** from the supervised head (argmax), on the test set, in
  percentage points.
- **"fixed (γ = 0)":** the frozen_γ = 0 runs. This is the same computation as
  AHSD-fixed.
- **S1 τ unit:** the median per-step stress mean_d |h_t − E_{t−1}| on benign
  validation windows, measured with the gate open.
  - Measured before training and re-measured once after epoch 1, then
    frozen.
  - This replaces the earlier benign-training measurement, for S1 only.
- **Per-class collapse:** class-vs-benign test d′ for each attack class with
  at least 20 test windows.
  - The stationarity index is computed on the γ = 0 model's embeddings of
    that class (mean over seeds).

---

## AMENDMENT_03 (2026-10-05, after gate G1, before any P1c run)

- **Supersedes:** RESEARCH_QUESTIONS.md with SHA-256
  `fb0b207e5f82edb4fb66eb0d4cbbfd454eb02838213bac9741d11ca00d24840c`
  (AMENDMENT_02, commit 35be633).
- **New SHA-256:** recorded in `docs/AMENDMENTS.json`.
- **Context:** both gates returned STOP (`GATE_DECISION.md` and
  `GATE_G1_DECISION.md`), and the authors accepted both. P2 is not started.
- **What this amendment adds:** pilot P1c. It is written and hashed before any
  P1c run.

### A3.1 Status
- G1 answered RQ3/RQ4 negatively. Test d′ does not collapse monotonically
  with γ: it peaks at γ = 0.02 on D1, D2 and D3. The learned adaptive γ
  stays ≈ 0.048.
- S1 is dropped.
- S2-dir stays exploratory and secondary.

### A3.2 D1 natural_novelty made feasible ("train-tail validation")
- **Validation:** the last 20% (by time) of the `temporal_gap` TRAIN period,
  before the gap.
  - Whole 1-hour groups are used. The tail is the train-period groups that
    start after 80% of the period's windows, counting cumulatively in time
    order.
- **Training:** the first 80% of the train period.
- **Test:** unchanged (the `temporal_gap` test period), except for the
  re-anchoring removal in A3.3(iii).
- **Exclusions:** windows with any natural-novelty flow are excluded from
  training and validation.
- **Feasibility counts:** P0 data, computed before this amendment with no
  model.
  - D1 tail: 146,904 benign and 3,767 attack windows. The tail's attacks are
    only Brute_Force_-Web, Brute_Force_-XSS and SQL_Injection, none of which
    occur in the training head.
  - D3 tail: 26,627 benign and 119,972 attack windows (ddos only).
- **Scope:** D2 is excluded (it has no test-only classes).
- **Implementer's note:** train-tail validation applies to D3 as well as D1,
  so both datasets share one protocol. It also makes A3.3(ii)'s
  "train-period benign val windows" well defined.

### A3.3 P1c experiments
- **Data:** D1 and D3 natural_novelty.
  - D1 classes: Bot, Infilteration.
  - D3 classes: Backdoor, mitm, password, ransomware, xss.
- **Seeds:** {17, 23, 42, 101, 202}. Seeding is fixed: models are seeded
  before construction, and the preflight test runs first.

**(i) Detectors**, all on identical windows (9 detectors):
1. AHSD-fixed stress.
2–5. Post-hoc scores on the same trained AHSD backbone:
   - 2. MSP;
   - 3. Energy;
   - 4. Mahalanobis on penultimate features;
   - 5. kNN on penultimate features.
6–9. Benign-only detectors:
   - 6. Isolation Forest;
   - 7. OCSVM;
   - 8. PCA reconstruction;
   - 9. Autoencoder.

Per class, report mean AUC, the Student-t 95% CI, and the fraction of seeds
with AUC < 0.5.

**(ii) Drift diagnostic**, per detector:
- d′(test-period benign vs train-period benign validation windows);
- d′(held-out attack vs train-period benign validation windows).

**(iii) Re-anchoring.**
- Refit only the benign reference, using N ∈ {100, 500} benign windows from
  the first test-period groups:
  - the AHSD E₀;
  - the Mahalanobis and kNN benign statistics;
  - the IF, OCSVM, PCA and AE models, refit on benign windows.
- Remove those groups from the test set for every condition, with and
  without re-anchoring.
- No attack labels are used, and the supervised backbone is not retrained.

### A3.4 Gate G2 (frozen). GO only if all three hold
- **(a)** ≥ 5 of 9 detectors have mean AUC < 0.5 on ≥ 3 natural-novelty
  classes (D1 and D3 pooled: 7 classes).
- **(b)** ≥ 5 of 9 detectors have median-over-classes
  d′(test benign vs train benign) ≥ 0.5.
- **(c)** For ≥ 5 of 9 detectors, re-anchoring with N = 500 raises mean AUC
  by ≥ 0.15 over no re-anchoring, AND to above 0.5, on at least half of the
  classes (≥ 4 of 7).
- Every criterion is reported per detector and per dataset, whatever the
  outcome.

### Operational choices added by the implementer
These are fixed here, before any P1c run, because A3.3 and A3.4 leave them
open.

**Score orientation.** Every score is higher = more anomalous (attack), and
is never flipped.

| Detector | Score |
|---|---|
| AHSD stress | stress |
| MSP | 1 − max softmax probability |
| Energy | −logsumexp(logits) |
| Mahalanobis | distance to the benign mean |
| kNN | distance to the k-th nearest benign reference |
| IF | −score_samples |
| OCSVM | −decision_function |
| PCA, AE | mean squared reconstruction error |

**Penultimate features.** The output of the AHSD head's hidden layer
(Linear → GELU, D = 128).

**Benign reference** (the thing re-anchoring refits):
- **AHSD stress:** E₀ = mean embedding of the reference windows.
- **Mahalanobis:** benign mean and Ledoit-Wolf covariance of penultimate
  features.
- **kNN:** a k = 10 search over L2-normalised penultimate features of the
  reference windows.
- **IF, OCSVM, PCA, AE:** refit on the reference windows only (not
  augmented).
- **Without re-anchoring,** the reference is the benign windows of the
  training set.
- **MSP and Energy** have no benign reference. Re-anchoring does not apply
  to them, so they count as failing (c).

**Benign-only detector inputs.** Windows flattened to 32 × 41 = 1,312
cleaned features, with these settings:

| Detector | Settings |
|---|---|
| IF | 200 trees, random_state = seed |
| OCSVM | RBF, ν = 0.1, γ = "scale"; deterministic, so identical across seeds |
| PCA | components explaining 95% of the variance (at most N − 1 when refit on N windows); deterministic |
| AE | MLP 1312–256–64–256–1312 (ReLU), Adam lr 1e-3, 30 epochs, batch 128, MSE, seed = run seed |

**Sets** (data seed 0; identical for every detector and seed):
- **Train:** balanced 1:1 from the train head, cap 14,000.
- **Validation:** 5:1 from the train tail, cap 3,000.
- **Anchors:**
  - anchor_N = the first N benign windows of the test period, in time order
    (so anchor_100 ⊂ anchor_500);
  - every group containing an anchor_500 window is removed from the test
    pool.
- **Test:**
  - one shared benign sample of 2,500 windows from the remaining test
    benign;
  - per class, up to 500 windows whose majority class is that class.
  - So the benign:attack ratio is 5:1 when 500 attack windows exist.
- **Drift baseline:** "Train-period benign" means the benign validation
  windows (train tail).

**Aggregation and gate details.**
- "Mean AUC" is the mean over seeds per class.
- For (b), a class's d′(test benign vs train benign) is its dataset's value,
  because the test benign sample is shared within a dataset. The median is
  over the 7 classes.
- d′ uses test benign (or the attack) as the positive class, with higher
  score = more anomalous.
