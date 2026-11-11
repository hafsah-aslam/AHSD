# Phase 3 — Research gap matrix, ranked opportunities, and candidate direction

Inputs: Phase 1 (`01_hone_examination.md`), 64 literature records (`02_literature/records.md`).
Status: **for review — no implementation before the direction is approved.**

## 1. Gap matrix (closest and most-cited work)

Legend: ✓ yes · ✗ no · ~ partly · ? not determined from what was read. "Novelty eval." = attacks evaluated
are new in *time* (temporal / natural), not only artificially held out. "Unit" = what one score is about.
"Seq." = evidence accumulated over time per entity with a stated decision rule. "FA guar." = false-alarm
control with a stated statistical guarantee. "Thr. w/o test" = operating threshold fixed without test labels.

| Work | Novelty eval. | Held-out families | Unit | Per-entity state | Seq. | FA guar. | Thr. w/o test | Drift vs attack | Unknown ident. | Fixed-FPR metric |
|---|---|---|---|---|---|---|---|---|---|---|
| CLAD (Z1) | ✗ | ✓ | flow | ✗ | ✗ | ✗ | ? | ✗ | ~ (OpenAUC) | ✗ |
| ThreatFormer (Z2, PP) | ✓ chrono | ✓ | window | ✗ | ✗ | ✗ | ? | ✗ | ✗ | ✓ R@1% |
| Sakr et al. (Z9) | ✗ | ✓ LOAO | flow | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Zoppi (Z6), Sarhan (Z7) | ✗ | ✓ | flow | ✗ | ✗ | ✗ | ? | ✗ | ✗ | ✗ |
| Zhang OCN (O1) | ✗ | ✓ | flow | ✗ | ✗ | ✗ | ~ | ✗ | ✓ cluster + learn | ✗ |
| Transcendent (O5) | ✓ (malware time) | — | sample | ✗ | ✗ | ~ conformal | ✓ | ~ reject drifting | ✗ | ✗ |
| E-GraphSAGE (G1) | ✗ | ✗ | flow (edge) | ~ graph | ✗ | ✗ | ? | ✗ | ✗ | ✗ |
| Anomal-E (G2) | ✗ | ✗ | flow (edge) | ~ graph | ✗ | ✗ | ? | ✗ | ✗ | ✗ |
| GraphIDS (G4) | ✗ random | ✗ | flow | ✗ | ✗ | ✗ | ✗ (val labels) | ✗ | ✗ | ✗ |
| HyperVision (G5) | ? | ~ unknown attacks | flow graph | ~ | ✗ | ✗ | ? | ✗ | ✗ | ✗ |
| NetVigil (G3) | ~ benign change | — | graph | ~ | ✗ | ✗ | ? | ~ robust to benign change | ✗ | ? |
| Euler (T1) | ✓ (LANL time) | — | edge/host | ✓ | ✗ | ✗ | ? | ✗ | ✗ | ? |
| TGN-SVDD (T2, PP) | ✓ temporal | ✗ | event | ✓ node memory | ✗ | ✗ | ✓ (train pct.) | ✗ | ✗ | ✗ |
| CCSTGN (T4) | ✓ temporal | ✗ | event | ✓ channel state | ✗ | ✗ | ✓ (val FPR) | ✗ | ✗ | ✓ |
| Kitsune (Q1) | ~ | — | packet | ✓ damped stats | ✗ | ✗ | ✓ | ✗ | ✗ | ? |
| Whisper (Q2) | ? | — | flow | ~ | ✗ | ✗ | ? | ✗ | ✗ | ? |
| OWAD (C5) | ✓ long-term | — | sample | ✗ | ✗ | ~ shift test | ✓ | ✓ normality shift | ✗ | ? |
| CADE (A1) | ✓ | ✓ | sample | ✗ | ✗ | ✗ | ✓ | ~ drift samples | ~ explain | ✗ |
| FIRCE (A10, PP) | ~ drift captures | ✗ | flow / chunk | ✗ | ✗ chunk EMA | ✗ (no theorem) | ✓ | ~ | ✗ | ✗ |
| Youssef (Q6, PP) | ✓ stream | ✗ | flow (stream) | ✗ | ~ stream BOCPD | ✗ | ✓ | ✗ | ✗ | ✗ |
| O-RAN conformal (A11) | ? | ? | sample | ✗ | ✗ | ✓ conformal FA | ✓ | ✗ | ✗ | ✓ |
| Classic CUSUM/SR NIDS (A12) | ✓ | — | aggregate counts | ~ per channel | ✓ | ✓ parametric ARL | ✓ | ✗ | ✗ | — |
| Peer/role methods (G7, patents) | ? | — | host | ✓ | ~ heuristic index | ✗ | ✓ | ~ peers | ✗ | ✗ |
| E-detectors (A8) | (theory) | — | stream | — | ✓ | ✓ nonparam. ARL | ✓ | — | — | — |
| **HoNE (E12)** | ✓ natural + rolling | ✓ | window | ✗ | ✗ | ✗ | ✗ (oracle TPR@1%) | ✗ | ✗ | ✓ (oracle) |

### What the matrix shows
- **Solved reasonably well:** in-distribution detection of volumetric attacks (DoS/DDoS, scans in random
  splits); supervised classification of *known* families; throughput at line rate (HyperVision, Whisper).
- **Partially addressed:** temporal evaluation (TGN-SVDD, CCSTGN, ThreatFormer, Euler); per-entity state
  (TGN memory, Kitsune, CCSTGN); drift detection (OWAD, CADE, FIRCE); conformal / rejection (Transcendent,
  FIRCE, O-RAN); open-set identification (Zhang OCN, OSR benchmarks).
- **Addressed with weak protocols:** most zero-day and graph NIDS (random splits or class hold-out, thresholds
  on labelled validation or test, no fixed-FPR metric) — GraphIDS, E-GraphSAGE, Anomal-E, CLAD, Zoppi, OSR.
- **Unresolved (no row has ✓ in all of these):** per-entity evidence accumulation **with** a stated
  false-alarm guarantee **on top of learned detectors**, evaluated under **natural novelty** with thresholds
  fixed on past traffic; separating **benign drift from malicious change**; the **undetectable core**
  (Infiltration, Bot, web/SQL injection, scanning) — missed by HoNE's 12 detectors, by TGN-SVDD (Bot,
  Infiltration) and by GraphIDS (Infiltration).
- **Assumptions existing models depend on:** stationarity of benign traffic between training and test;
  one score per flow/window suffices; a threshold from (labelled) validation transfers to deployment;
  IP identity as node identity (graph methods; shortcut acknowledged by TGN-SVDD).
- **Distinguishing unusual-but-legitimate from malicious:** no reviewed method has a mechanism beyond
  "deviation from training benign"; peer-based ideas exist only as heuristics/patents.

## 2. Ranked research opportunities

Scores 1–5 (5 best). Criteria per directive: Orig. originality · Sig. scientific significance ·
Feas. feasibility · Perf. potential improvement · Math. mathematical contribution · Test. testability ·
Pub. publication potential. Total /35.

| Rank | Opportunity | Orig. | Sig. | Feas. | Perf. | Math. | Test. | Pub. | Total |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **Peer-calibrated sequential detection**: per-entity e-processes from conformal p-values calibrated against *contemporaneous peers*, with nonparametric time-to-false-alarm control; separates population-wide (benign) drift from entity-local (malicious) change | 4 | 5 | 4 | 4 | 4 | 5 | 5 | **31** |
| 2 | Entity-centred behavioural representation replacing network-wide windows (test of W2 as the cause of the undetectable core) | 3 | 4 | 4 | 4 | 2 | 5 | 4 | 26 |
| 3 | Valid combination of heterogeneous detectors (supervised + benign-relative) by merging conformal e-values instead of learned weights (fixes W4 without choosing a family) | 3 | 4 | 5 | 3 | 3 | 5 | 3 | 26 |
| 4 | Deployable natural-novelty benchmark: thresholds from past traffic, more time-stamped datasets, event-level delay metrics (extends HoNE) | 2 | 4 | 5 | — | 1 | 5 | 4 | 21+ |
| 5 | Open-set identification with abstention: conformal prediction sets over known families + clustering of alerting entities' trajectories; empty/large set = "insufficient evidence" | 3 | 4 | 4 | 3 | 2 | 4 | 3 | 23 |
| 6 | Benign-support-constrained supervised objective (no unseen region scored as benign; removes confident inversion, W3) | 2 | 3 | 4 | 3 | 3 | 4 | 3 | 22 |
| 7 | Identity-free relational roles for graph NIDS (roles inferred from structure, never IPs; tests TGN-SVDD's shortcut concern) | 3 | 3 | 3 | 3 | 2 | 4 | 3 | 21 |
| 8 | Spiking / neuromorphic NIDS | 1 | 2 | 3 | 2 | 1 | 3 | 2 | 14 |

Why SNNs rank last: none of the reviewed SNN work evaluates natural novelty, energy evidence is simulated,
and nothing in HoNE's failure analysis points to a limitation that spiking computation would address. They
remain an option for *deploying* a validated method (rank 1's per-entity e-process is cheap enough for it),
not a research direction on their own.

Directions 1, 2, 3 and 5 compose into one system (§5); 4 is the evaluation that any of them must pass.

## 3. Candidate direction 1 — precise formulation (draft for approval)

### 3.1 Objects
- Entities v ∈ V(t): hosts (source address), optionally host pairs. Addresses are used **only to group**
  flows; they are never features and never persist across datasets.
- For entity v, the t-th window x_{v,t} ∈ ℝ^{T×d} = its next T flows (entity-centred, opportunity 2).
- Any base detector s: ℝ^{T×d} → ℝ (higher = more anomalous); e.g. a benign-only AE/IF, P(attack),
  Mahalanobis. The framework is detector-agnostic.
- Peer calibration set C_{v,t}: scores of windows of *other* entities in the same role, from the recent past
  window [t−Δ, t), size n. Roles from behaviour clustering on past benign-labelled periods (opportunity 7).

### 3.2 Peer-relative conformal p-value
p_{v,t} = (1 + #{c ∈ C_{v,t} : c ≥ s(x_{v,t})}) / (n + 1).

**Null hypothesis H0(v,t):** s(x_{v,t}) is exchangeable with the scores in C_{v,t}
("v currently behaves like its contemporaneous peers"). Under H0, P(p_{v,t} ≤ α) ≤ α (standard conformal
validity; Bates et al. 2023).

**Why peers rather than training data:** benign drift that moves the whole population (new software
version, diurnal load, new service) shifts C_{v,t} and s(x_{v,t}) together, so H0 still holds and no alarm
accumulates. An attack on or from v makes v deviate from peers, so H0 fails. This turns "benign drift vs
malicious change" into a testable statement: *common-mode vs entity-local deviation*.

### 3.3 Per-entity e-process and alarm
Convert p to an e-value with a calibrator, e.g. e = κ p^{κ−1}, κ ∈ (0,1) (E[e] ≤ 1 if p is super-uniform).
e-CUSUM: M_{v,t} = max(M_{v,t−1}, 1) · e_{v,t}, M_{v,0} = 1; alarm when M_{v,t} ≥ A.

**Claim to establish (from Shin, Ramdas & Rinaldo, adapted):** if, under H0, the sequence (e_{v,t})_t is
such that each e_{v,t} has conditional expectation ≤ 1 given the past, then the expected time to a false
alarm for entity v is ≥ A, without parametric assumptions. Across |V| entities, the network-wide
false-alarm rate is ≤ |V|/A per window step (union bound), so an operator sets A from an alarm budget
(e.g. "≤ 1 false alarm per day").

**Detection delay (to derive):** under an attack that persists on v and gives E[log e_{v,t}] = D > 0,
expected delay ≈ (log A)/D windows. D depends on the base detector's separation *in the tail*, which links
directly to HoNE's finding: a detector with AUC 0.75 and only 9–18% TPR@1% FPR per window can still alarm
within log(A)/D windows if the attack persists, while a per-window threshold cannot.

### 3.4 Assumptions, failure conditions, cost
- **Conditional validity is the critical assumption.** Successive windows of one entity are dependent;
  peers may not be exchangeable with v (unique roles, servers). Validity must therefore be shown
  empirically (alarm rate on benign-only periods) and, where possible, by construction (strided windows,
  conditional p-values within role). This is the main theoretical risk.
- **Fails by design** for population-wide attacks (DDoS hitting all peers, worms) — these move peers too.
  Mitigation: run the same e-process against a training-time reference in parallel and merge the two
  e-values (opportunity 3); volumetric attacks are already detected well by existing scores.
- **Fails** for one-shot attacks (a single flow), for entities with no peers, and against mimicry that
  matches peer score distributions (an adversary who imitates peers; to be studied, not assumed away).
- **Cost:** per window O(n) to rank against C (O(log n) with a sorted buffer); state per entity O(1).
  Memory O(|V| + n·|roles|). Compatible with line-rate operation; the base detector dominates cost.

### 3.5 Experiments that could falsify the central claims
1. **Validity:** on benign-only test periods (HoNE temporal splits), empirical time-to-false-alarm ≥ A
   for each A in a grid. If it fails, the conditional-validity assumption is wrong for network data.
2. **Drift separation:** inject benign common-mode shifts (time-of-day, HoNE's temporal-gap drift) — the
   peer-calibrated version must keep its false-alarm rate while a training-calibrated version inflates it.
3. **Accumulation helps where HoNE failed:** on natural-novelty and rolling classes, detection rate at
   fixed alarm budget and median delay, versus the same base detector thresholded per window (HoNE's
   TPR@1% with a *past-traffic* threshold). Pre-registered prediction: gains concentrate on persistent,
   low-rate classes (Bot, Infiltration, brute force) and are absent for one-shot classes.
4. **Ablations:** peers vs training reference vs both; calibrator κ; e-CUSUM vs e-SR; window length T;
   entity = host vs host pair; base detector family.

## 4. Novelty audit — direction 1 against the closest work (full texts read where marked F)

| Closest work | What it does | What it does not do | Overlap risk |
|---|---|---|---|
| Classic CUSUM/SR intrusion detection (A12) | quickest detection on traffic counts, parametric, often per channel | learned detectors; nonparametric guarantee; peer-relative null | **High in spirit** — sequential detection in NIDS is decades old; novelty must be in the null, the guarantee and the evaluation |
| E-detectors (A8) | general nonparametric e-CUSUM/e-SR with ARL control | network entities, peer calibration, multi-entity budget | **Method source** — we adapt, not invent, the e-detector |
| Conformal test martingales (Vovk; Cherubin 2018, A9) | exchangeability testing of one stream; feature selection for AD | peer-relative calibration across entities; NIDS evaluation | Medium — must cite and differentiate precisely |
| FIRCE (A10, F) | per-flow conformal p-values; chunk drift via EMA | per-entity accumulation; ARL; theorem | Low–medium |
| Youssef (Q6, F) | stream-level BOCPD score; evaluation audit | per-entity; conformal; guarantee | Low |
| TGN-SVDD (T2, F), CCSTGN (T4, F), Euler (T1) | per-entity *representation* (memory) with per-event threshold | sequential test, FA guarantee, peer null; held-out families | Low for the test, high for representation — direction 2 must be positioned as complementary |
| Kitsune (Q1) | per-channel damped statistics as *features* | decision rule with guarantee | Low |
| Peer/role methods (G7, patents) | peers as behavioural baseline, heuristic scores/indices | statistical validity; learned base detectors | **Medium** — "peer-based anomaly detection" exists; the claimed novelty is the conformal peer null with a guarantee |
| OWAD (C5), CADE (A1) | detect / explain normality shift; drifting samples | entity-local vs common-mode distinction; sequential alarms | Medium on the drift question |
| O-RAN conformal FA control (A11) | conformal false-alarm control per sample | sequential, per-entity, peers | Low–medium |

**Honest assessment.** No single ingredient is new: conformal p-values, e-detectors, per-entity state and
peer groups all exist. What did not turn up in this search is their combination into a *peer-relative
null* with nonparametric time-to-false-alarm control, applied on top of learned NIDS detectors and
evaluated under natural novelty with past-traffic thresholds. Before any novelty claim, a targeted search
is still required for: "conformal" + "peer group"/"population" + "anomaly"; "e-process" + "intrusion";
"multi-stream quickest change detection with exchangeable streams" (statistics literature — the most likely
place for prior art on the peer-relative null). The contribution should be framed as **a principled
detection-and-decision framework with a falsifiable validity claim**, not as a new neural architecture.

## 5. How the opportunities compose (system sketch, not yet an architecture)

```
flows ─► group by entity (address used for grouping only) ─► entity windows x_{v,t}
        ─► base detectors (benign-only, supervised P(attack), representation distance)   [existing]
        ─► peer-relative conformal p-values  +  training-reference p-values             [dir. 1, 3]
        ─► merged e-value per entity ─► e-CUSUM state M_{v,t} ─► alarm at A (alarm budget) [dir. 1]
        ─► alarmed entities: conformal prediction set over known families, or "unknown";
           unknown alarms clustered by behaviour profile; analyst-verified clusters added    [dir. 5]
```
Continual learning enters only through verified alarms (added to known families) and benign-only periods
(peer calibration sets refresh automatically), which avoids learning malicious traffic as benign without
labels.

## 6. Decision requested

1. Approve **direction 1** (with 2 and 3 as components, 5 as an extension, 4 as the evaluation standard),
   or choose another from §2.
2. Before implementation I would: (a) complete the targeted prior-art search in §4; (b) read the remaining
   full texts listed in `02_literature/records.md` §13; (c) write the formal validity statement and delay
   bound with explicit assumptions; (d) pre-register predictions 1–4 of §3.5 in the HoNE style.
3. Data: raw NetFlow-v3 and Lycos files must be downloaded again into the container (HoNE's
   `scripts/download.py`); addresses are present in the raw schema and are used only for grouping.
