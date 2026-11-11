# Phase 1 — Examination of HoNE

Scope: the HoNE manuscript (`paper/hone_user/main.tex`, appendix), code (`src/nids`, `scripts`),
configurations (`configs/`), pre-registration (`RESEARCH_QUESTIONS.md`, seven amendments), negative
results (`NEGATIVE_RESULTS.md`) and the stored final results (`results/final_v2/summary.json`, 210 runs).
Raw datasets are not present in this container (`data/` holds manifests only); every number below is
read from stored result files, not recomputed. Nothing in HoNE was modified.

## 1. What HoNE is

An *evaluation* study. It asks whether leave-one-attack-class-out (LOACO) evidence predicts detection of
naturally emerging attacks (classes that first appear after the training period), on the same detectors,
data, backbones and seeds.

| Item | Setting in HoNE |
|---|---|
| Datasets | NF-CIC-IDS2018-v3 (D1), NF-UNSW-NB15-v3 (D2), NF-ToN-IoT-v3 (D3), NF-BoT-IoT-v3 (D4, transfer only), Lycos2017 (D5) |
| Unit of analysis | **window of T=32 consecutive flows in network-wide start-time order**, stride 16, inside 1-hour blocks; window = attack if *any* flow is an attack; 41 common features (addresses, ports, DNS IDs, TTL dropped) |
| Protocols | LOACO (grouped random), natural novelty (temporal gap, train-tail validation), purged blocks (D5), 4 rolling cut-points (45 class–cut units, 19 unique classes), in-distribution grouped vs temporal, zero-shot transfer |
| Detectors (12) | Confidence: MSP, Energy, P(attack); representation: AHSD stress, Mahalanobis, kNN; benign-only: IF, OCSVM, PCA, AE; contrastive: CLAD; reference: XGBoost(-s). Second backbone: GRU |
| Backbone | AHSD (fixed benign reference E, gated recurrence); GRU control |
| Metrics | ROC-AUC, PR-AUC, TPR@1%/0.1% FPR (test-set oracle threshold), seed s.d., paired and cluster bootstraps, protocol-validity measures (ρ, ΔAUC, winner retention, regret) |

## 2. What HoNE established

1. **LOACO is not a reliable proxy for natural novelty** at detector level: best detector kept in 1/12
   comparisons; median ρ = 0.55; 75% of 138 detector cells lower under natural novelty; only the leading
   *family* survives (12/12).
2. **Temporal LOACO is infeasible on day-scheduled captures** (D1, D3, D5).
3. **Natural novelty raises seed variability** of every stochastic supervised score (up to 17× on AHSD,
   2.6× on GRU representation); benign-only detectors stay stable (s.d. < 0.018).
4. **Confidence scores invert in binary detectors**: MSP = σ(|Δ|) measures certainty, not class; confident
   detection of a novel attack reads as normal. P(attack) and representation distance are valid on average.
5. **Temporal shift alone does not explain the gap** (dataset- and backbone-specific); zero-shot transfer
   is near chance (median AUC 0.500 over 108 cells).
6. **AUC overstates operational detection**: over rolling cuts no family exceeds 18% TPR@1% FPR even with
   an oracle threshold; 184 of 333 detector–class points with AUC ≥ 0.75 catch < 20%.
7. **Four pre-registered hypotheses failed** (spectral detectability predictor, adaptive-equilibrium
   absorption, benign re-anchoring, benign-only superiority confined to natural novelty).

## 3. What HoNE did *not* establish

- It proposes **no detector**; it shows that existing ones and their evaluation are inadequate.
- It does not explain *why* specific classes are undetectable (it shows *that* they are).
- It does not test entity-level, relational (graph) or multi-window evidence; every score is per window.
- It does not test thresholds set on past traffic (deployable operating points); TPR@1% is an oracle bound.
- It does not separate benign drift from malicious change; it measures their combined effect.
- It does not address unknown-attack *identification* (grouping, naming, abstention) or continual learning.
- Evidence covers four NetFlow-v3 datasets and Lycos2017; natural-novelty classes are few (19 unique).

## 4. Weaknesses exposed (evidence from stored results)

### W1. An undetectable core of attack classes
Best AUC over **all 12 detectors** stays below 0.70 for: D1 Infiltration (0.57–0.60, every protocol),
D1 Bot (0.66–0.70), D1 Brute-Force-Web (0.68), D3 scanning (0.63), D5 bot (0.61–0.63),
D5 web SQL injection (0.65). These are low-rate, application-layer or multi-stage behaviours.
**Interpretation (hypothesis, to be tested):** the information needed is not in a single window of
network-wide flows; it lives in *one entity's behaviour over time* or in *relations between entities*.

### W2. The unit of analysis dilutes the signal
Windows are 32 flows in global arrival order, so one window mixes many unrelated hosts, and one attack flow
makes a window "attack". A low-rate attack contributes ~1 of 32 flows. The recurrent backbones therefore
model arrival order of unrelated flows rather than a behavioural sequence. This is a design choice of HoNE
inherited from common practice; it is a plausible root cause of W1 and of the low TPR at fixed FPR (W5).

### W3. Supervised representations are anchored to the attacks they saw, not to benign behaviour
- Re-estimating the benign reference on target traffic (P1c re-anchoring) *destroyed* supervised
  representation scores (e.g. Mahalanobis D3 password 0.588 → 0.091; kNN xss 0.493 → 0.031), while
  benign-only detectors barely moved (IF xss 0.958 → 0.906).
- P(attack) can invert on a novel class (D5 portscan 0.157) — the model confidently places the new attack on
  the benign side of a boundary learned from other attacks.
- Under natural novelty, seed variability of supervised scores rises up to 17×.

### W4. No single scoring principle is reliable; families fail on different classes
Across 75 class units the best supervised and best benign-only scores correlate only 0.35; benign-only
wins by > 0.15 AUC on 18 units, supervised on 4. Examples: D1 DDoS (LOACO) supervised ≈ 0.5–0.6 vs PCA 0.978;
D5 portscan P(attack) 0.157 vs Mahalanobis 0.980; D3 natural novelty supervised representation ≈ chance vs
IF ≈ 0.95–0.99. A detector cannot know, per input, which reference is valid.

### W5. Separation at the tail is poor even when AUC is moderate
Rolling-cut family TPR@1% FPR: confidence 0.044, representation 0.125, benign-only 0.092, CLAD 0.101,
XGBoost 0.183. Novel attacks overlap the benign score tail; per-window thresholding cannot reach useful
detection at operator false-alarm rates.

### W6. Evaluation-side constraints for any new detector
Natural-novelty classes are few and dataset-specific; day-scheduled captures confound "new class" with
"new day"; benign drift and attack onset coincide in time. Any claimed improvement must survive these.

## 5. Opportunities: from evaluation to detection

These are *questions*, not yet proposals; Phase 2–3 will test them against the literature.

| # | Weakness | Candidate question | What kind of contribution it would need |
|---|---|---|---|
| O1 | W1, W2 | Does an **entity-centred** behavioural representation (per host / per host-pair, over time) recover the undetectable core? | representation (not necessarily a new network) |
| O2 | W5 | Can **evidence accumulation** over an entity's successive windows (sequential change detection) turn weak per-window separation into high detection at fixed FPR, with a threshold fixed on past benign traffic? | decision mechanism + statistical bound |
| O3 | W3 | Can a representation be **anchored to benign behaviour** so that unseen attacks cannot be mapped onto the benign side (no "confident inversion")? | learning objective |
| O4 | W4 | Can the detector **select or weight its reference per input** (attack-relative vs benign-relative) using a measurable validity signal, rather than fixing one family? | decision function |
| O5 | W3, W6 | Can **benign drift** be separated from **malicious change** by what changes (population-wide vs entity-local, gradual vs abrupt, role-consistent vs role-violating)? | representation + test statistic |
| O6 | — | Can detected unknowns be **grouped and named with abstention** (open-set identification with calibrated "insufficient evidence")? | decision function + evaluation |
| O7 | W6 | Can claims be validated on more natural-novelty events (additional time-stamped datasets) with deployable (past-traffic) thresholds? | evaluation (extends HoNE) |

Constraint carried forward: HoNE dropped addresses and ports to prevent identity leakage. Any entity-centred
method must use addresses only to *group* flows (or derive anonymous roles), never as features, and must be
tested for identity memorisation.

## 6. Assets reusable without modification

- Leakage-controlled loaders, splits (temporal gap, purged blocks, rolling cuts), integrity checks
  (`src/nids/splits.py`, `verify.py`, `scripts/rolling_build.py`).
- Detector baselines and the 12-detector panel; bootstrap and protocol-validity analysis
  (`scripts/report_agreement.py`, `scripts/cluster_bootstrap.py`).
- Pre-registration and amendment workflow.
New work will live in `research/zeroday_nids/` and a new package, leaving HoNE code and results unchanged.
