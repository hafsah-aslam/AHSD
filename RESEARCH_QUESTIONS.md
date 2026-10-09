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

---

## AMENDMENT_04 (2026-10-05, after gate G2, before any P1d data processing or run)

- **Supersedes:** RESEARCH_QUESTIONS.md with SHA-256
  `dc2efcae3986b9cc076374c9324e6b2cf740503ffe1ab7bd92b2031406cdf5ec`
  (AMENDMENT_03, commit 8a6dc3c).
- **New SHA-256:** recorded in `docs/AMENDMENTS.json`.
- **Context:** all three gates returned STOP (P1, G1, G2). The authors
  accepted them.
- **What this amendment fixes:** P1d is the final confirmatory pilot. Its gate
  (G3) fixes the paper's direction, with no further pivots.
- **Inputs used before this amendment:**
  - the Lycos2017 archive was downloaded, with its SHA-256 recorded below;
  - the archive listing, the authors' README and `labelling.py` code, the
    feature documentation and one CSV header were read;
  - the Wilkie et al. paper (arXiv 2601.09902) was read.
  - No flow data was parsed or processed.

### A4.1 Status
- G2 was negative. Drift is not the inversion mechanism, and re-anchoring is
  dropped.
- The P1c observation is EXPLORATORY and is not used as confirmatory
  evidence. That observation: on D3 natural novelty, the benign-only
  detectors beat the scores computed from the supervised AHSD model.

### A4.2 Confirmatory hypothesis H-REV (frozen)
Under natural temporal novelty, benign-only detectors outperform
supervised-model-based scores, AND this ordering reverses (or vanishes) under
LOACO on the same dataset.

### A4.3 Fresh dataset D5 = Lycos2017 (corrected CIC-IDS2017)
- **Source:** https://lycos-ids.univ-lemans.fr/ (download page).
  - It links to `https://maupiti-git.univ-lemans.fr/lycos/lycos-ids2017/archive/master.zip`,
    with no form or login.
  - Archive SHA-256:
    `7457f65587fe609fbe983672c12577bb9a07996e6920f8e4a183b8a24a51cf1e`
    (532,484,362 bytes).
- **Processing:** the same P0 pipeline as before:
  - manifest with SHA-256, and an audit;
  - identifier and timestamp drops;
  - train-only fitting;
  - 1-hour groups, windows T = 32 with stride 16, dedupe;
  - fail-closed validators.
- **Feature space:** D5 is processed in its own feature space (no transfer
  in P1d).
- **Feasibility report before any model run:**
  - the natural_novelty classes, i.e. those in the `temporal_gap` test split
    but absent from its train split;
  - whether the train-tail validation has attack windows.
  If either is infeasible: STOP and report.

### A4.4 Detectors (the 9 from P1c, plus 1)
1. AHSD stress.
2–5. MSP, Energy, Mahalanobis and kNN, all on the supervised AHSD model.
6–9. IF, OCSVM, PCA and AE (benign-only).
10. Wilkie et al. 2026 contrastive zero-day loss (CLAD; IEEE TNSM,
    doi 10.1109/TNSM.2026.3652529), implemented from the paper.

Settings and score directions follow AMENDMENT_03.

### A4.5 Evaluations, seeds {17, 23, 42, 101, 202}
- D5 natural_novelty (train-tail validation).
- D5 LOACO (`grouped_random`).
- D3 LOACO (`grouped_random`), with all 10 detectors (not yet run).
- D3 natural_novelty: rerun only the new Wilkie detector.

### A4.6 Gate G3 (frozen)
Defined per dataset and evaluation:
- B = the mean over classes of (the mean AUC of the 4 benign-only detectors).
- S = the mean over classes of (the mean AUC of the 5 supervised-backbone
  scores).

GO only if ALL of the following hold:
- **(a)** D5 natural_novelty: B − S ≥ 0.15, AND its paired-bootstrap 95% CI
  is > 0.
- **(b)** D5 LOACO: B − S ≤ 0.05 (a reversal, or no benign-only advantage).
- **(c)** The same (a)/(b) pattern holds on D3, using the natural_novelty
  results from P1c and the new LOACO.

Report the Wilkie detector separately against B and S (not gating). Report
every number either way.

### Operational choices added by the implementer
These are fixed here, before any P1d processing or run.

**D5 construction.**
- **Labels:** produced by running the authors' `labelling.py`, unmodified
  (LYCOS-IDS2017 `master`), on their LycoSTand CSVs from the same archive.
  It labels flows by timestamp windows and addresses.
- **Adapter** (logged in `PREPROCESS_LOG.md`):
  - `timestamp` (µs) becomes `FLOW_START_MILLISECONDS` (÷ 1000; used only
    for sorting and grouping);
  - `label` becomes `Attack`, and `Label` = (`label` ≠ "benign");
  - `src_addr`, `dst_addr`, `src_port` and `dst_port` take the NF identifier
    names, so the identifier drop applies;
  - `flow_id` is removed (identifier).
- **Derived file:** written beside the untouched archive under
  `data/raw/lycos/derived/`, with its own SHA-256 and the SHA-256s of
  `labelling.py` and the input archive.
- **Heavy-tailed rule for D5** (lower-case LycoSTand names): log1p is applied
  to non-negative features whose name contains any of `len`, `cnt`, `tot`,
  `bytes`, `per_s`, `iat`, `duration`, `active`, `idle`, `bulk`, `subflow`,
  `win`, `var`, `std`, `mean`, `max`, `min`.
  - The `flag_*`, `fwd_flag_*` and `bwd_flag_*` counts, `ip_prot` and
    `down_up_ratio` are not logged.
  - The NF rule (upper-case names) is unchanged.
- **Splits:**
  - D5 has no TTL columns.
  - `temporal_gap` and `grouped_random` splits are built as for D1–D3.
  - LOACO folds come from `grouped_random` (≥ 200 training windows, and the
    class must be present in the test split).

**Natural-novelty classes for D5.**
- These are all classes present (as a window's majority class) in the
  `temporal_gap` test split and absent from its train split.
- A class needs ≥ 20 test windows to be evaluated. Others are listed but not
  evaluated.
- Validation is train-tail validation (A3.2).
- Re-anchoring is dropped, so the D5 test pool is the whole test period.
- D3 natural_novelty (Wilkie rerun, and the B/S values for (c)) uses the
  P1c package unchanged, with its anchor groups removed, so that every
  detector is compared on identical windows.

**LOACO for D3 and D5.**
- Use the existing `grouped_random` folds:
  - training excludes every window holding any flow of the held-out class;
  - test = benign + held-out class, 5:1, cap 3,000.
- Per fold and seed:
  - the supervised AHSD model is trained on the fold's training set;
  - the benign-only detectors and the AHSD/Mahalanobis/kNN benign references
    use that set's benign windows;
  - Wilkie is trained on the fold's training set.

**Wilkie / CLAD, as implemented from the paper.**
- **Encoder φ:** a linear projection to d_model, then L blocks of
  [Linear(d_model→d_model), ReLU], then a linear head to f_o, L2-normalised
  onto the unit sphere.
- **Loss (Eq. 7):** for each benign anchor i in the batch:
  - the mean over the other benign samples p of d(z_i, z_p)²;
  - plus the mean over the malicious samples n of (1 − d(z_i, z_n))²;
  - with d(z, z′) = (1 − z·z′)/2;
  - averaged over the batch's benign anchors.
- **Score (Eqs. 8–9):** s(x) = −z·μ, with μ the L2-normalised sum of the
  embeddings of the benign training windows.
- **Unspecified in the paper, fixed here:**
  - Input: the flattened window (T × F), the same windows as every other
    detector.
  - Architecture: d_model = 256, L = 2, f_o = 64, no dropout.
  - Optimiser: AdamW, lr 1e-3, weight decay 1e-4, batch 256.
  - Batches are drawn from the 1:1-balanced training set, which stands in
    for "weighted class balancing".
  - Epochs: 15, the shared budget of spec §5, instead of the paper's 200. A
    linear warm-up over the first 10% of steps (the paper's 20/200 ratio)
    is followed by cosine annealing.
  - There is no hyperparameter search (spec §5: no per-model tuning), where
    the paper ran a 200-iteration random search.
  - Checkpoint: the epoch with the best validation AUROC of s(x) (benign vs
    attack). The paper has no classifier head, so validation macro-F1 does
    not apply.
  - Seed: the run seed, set before model construction.

**G3 statistics.**
- **Per-class AUC** is the mean over seeds.
- **B and S** are computed from the class means of each detector group,
  using all evaluated classes of that dataset and evaluation.
- **Paired bootstrap for (a)** (2,000 resamples, bootstrap seed 0):
  - resample classes with replacement, and independently resample seeds with
    replacement;
  - per resample, compute the mean over the resampled classes of [the mean
    over the resampled seeds of (the mean benign-only AUC minus the mean
    supervised AUC)];
  - the 95% CI is the 2.5–97.5 percentile interval.
- **For (a):** the point estimate must be ≥ 0.15 and the CI lower bound
  > 0.
- **For (b):** the point estimate must be ≤ 0.05.
- **(c):** applies the same two tests to D3, using P1c's no-re-anchoring
  natural_novelty AUCs (code commit 702fc4d) and the new D3 LOACO.
- **Wilkie** is reported separately as its AUC minus B and its AUC minus S,
  per evaluation (not gating).

---

## AMENDMENT_05 (2026-10-09, before any purged_block feasibility count or P1d model run)

- **Supersedes:** RESEARCH_QUESTIONS.md with SHA-256
  `6a474852e58aeab6f71ccc98d25204a172a025879262214b352e78a816dbc05c`
  (AMENDMENT_04, commit abcf500).
- **New SHA-256:** recorded in `docs/AMENDMENTS.json`.
- **Context:**
  - P1d stopped before any model run, because D5 LOACO under
    `grouped_random` had 0 feasible folds (`P1D_REPORT.md`,
    `DECISIONS_PENDING.md` item 7).
  - The authors adopted a D5-specific LOACO protocol and rejected options 2
    and 3. Option 2 survives only as the pre-declared fallback in A5.2.
  - No 10-minute-block count of any kind has been computed.

### A5.1 D5 LOACO protocol "purged_block"
- **Groups:** 10-minute blocks instead of 1-hour blocks, for D5 only.
- **Split:** day-stratified. Within each capture day, assign blocks 60/20/20
  to train/val/test by a seeded shuffle.
- **Purge:**
  - remove from train and val every block adjacent (±1 block, same day) to a
    test block;
  - remove from train every block adjacent to a val block;
  - log the purged block counts.
- **LOACO fold thresholds:** unchanged, plus one new condition. The held-out
  class needs:
  - ≥ 200 windows in the training pool before removal;
  - ≥ 50 test windows (new).
- **Everything else** follows the unchanged P0 rules: dedupe, disjointness,
  fail-closed checks.

### A5.2 Pre-declared fallback (decided before any count)
- If purged_block yields < 3 feasible D5 LOACO folds:
  - G3 criterion (b) is evaluated on D3 LOACO only;
  - D5 contributes criterion (a) only;
  - this is recorded as a protocol limitation in the paper.

### A5.3 D5 natural_novelty: thin validation (130 benign + 26 attack)
- **Primary:** checkpoint by validation macro-F1, as frozen.
- **Pre-declared sensitivity analysis:** the final-epoch checkpoint.
- Both are reported; only the primary enters the gate.

### A5.4 G3
Criteria (a), (b) and (c) are otherwise unchanged.

### Operational choices added by the implementer
These are fixed here, before any count or run.

**Blocks.**
- Block index = floor((FLOW_START_MILLISECONDS − first D5 flow start) / 600,000).
- Empty blocks hold no flows and are skipped. Group ids are the non-empty
  blocks in time order.
- Windows (T = 32, stride 16) are rebuilt within 10-minute blocks, and never
  cross one.
- Dedupe uses the same raw-feature hashes and the same columns as the D5
  index (`hash_columns` in `data/processed/D5/index_meta.json`).

**Capture day.**
- A capture day is a maximal run of non-empty blocks in which consecutive
  blocks are less than 6 hours apart (a gap of ≥ 6 h starts a new day).
- The script fails closed unless exactly 5 days (Monday–Friday) are found.

**Per-day split** (data seed 0; one RNG over the days in time order):
- shuffle the day's non-empty blocks;
- n_train = round(0.6 n), n_val = round(0.2 n), n_test = n − n_train − n_val.

**Adjacency and purge.**
- "Adjacent" means a block index differing by exactly 1 within the same day.
  A neighbouring empty block is not a block, so it purges nothing.
- Purge order:
  1. train or val blocks adjacent to a test block are removed;
  2. then train blocks adjacent to a remaining val block are removed.
- Purged blocks are used nowhere.
- Logged: purged counts per day and per rule.

**Folds.**
- These rules are unchanged: the `grouped_random` fold mechanics, the
  training and validation exclusion of every window holding any flow of the
  held-out class, and the 1:1 / 5:1 balance with caps 14,000 / 3,000 / 3,000.
- "≥ 200 windows in the training pool before removal" means ≥ 200 train-split
  windows whose majority class is the held-out class.
- "≥ 50 test windows" means ≥ 50 test-split windows with that majority class.

**Features.**
- The purged_block cleaner is fitted on the purged_block train rows only.
- It is restricted to the frozen D5 own feature list of 73 features
  (`data/processed/D5/own_features.json`).
- The script fails closed if any of those features is dropped (for example,
  for zero variance) on the purged_block train split.

**A5.2 application.**
- `scripts/d5_purged_block.py` writes the feasible-fold count and the A5.2
  decision to `results/p1d/a52_decision.json` before any model run.
- G3 reads that file.
- Under the fallback, D5 LOACO is not run.

**A5.3 scope.**
- The final-epoch sensitivity re-trains the models that have a checkpoint
  choice:
  - the supervised AHSD backbone, so all 5 supervised scores;
  - CLAD.
  Their primary checkpoints stay as frozen: AHSD by best validation macro-F1,
  CLAD by best validation AUROC (AMENDMENT_04).
- The benign-only detectors have no checkpoint choice. Their primary scores
  are reused, and they are seed-deterministic.
- Sensitivity B − S = the primary benign-only B minus the final-epoch S.
  This is reported only and never enters the gate.
