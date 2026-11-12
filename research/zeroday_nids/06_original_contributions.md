# Phase 6: Search for original zero-day NIDS contributions

**Status: awaiting review.**

This phase is limited to research discovery, mathematics, prior-art investigation and synthetic falsification.

- HoNE is untouched.
- No real dataset was used.
- No NIDS was implemented.
- Synthetic results are reported as **simulation evidence, not experimental validation**.

**Materials**
- Prior-art matrix: `02_literature/phase6_prior_art_matrix.md` (records P1–P41).
- Earlier records: `02_literature/records.md` (64 records), plus the works cited in docs 04 and 05.
- Simulations: `sims/capacity.py` → `sims/results_capacity.json`, seed 20261112, about 8 minutes on 4 cores.

**Tags**, as in doc 05:
- **[T]** proved here;
- **[K]** known result;
- **[P]** short derivation;
- **[S]** simulation only;
- **[O]** open.

## Summary (deliverable K, in brief)

1. **One contribution survives: a theory of detectability floors.** It has a defender-side guarantee and was checked adversarially. I classify it **Potentially original**:
   - The core inequality is elementary. It follows from Neyman–Pearson and Cauchy–Schwarz.
   - The i.i.d. square-root case is already published in quickest change detection (Ramtin, Nain & Towsley 2025/26; P7, P8).
   - What appears to be new:
     - (i) the exact finite-horizon floor for general benign covariance, together with the minimax optimality of the shape-agnostic cumulative test;
     - (ii) the T^H law under long-range dependence (LRD), which contrasts with the covariance-invariance of covert information (P3);
     - (iii) a linear floor for any finite own-history reference. This formally settles the "validity versus absorption" tension left open in doc 05 §1.4 — it is a fundamental trade-off;
     - (iv) the reduction of the floor obtained from a peer residual.
2. **The practical candidate, PS-VTS, failed its pre-registered superiority criterion** (§6). It is withdrawn as a contribution:
   - Its validity held: false-alarm rate 0.005–0.015 against 0.05 in every benign scenario.
   - Its detection floor was 2–7× *higher* (worse) than an existing combination (peer-median residual + variance-time scan).
   - It collapsed once 30% of the peer group was compromised.
3. **No second or third contribution meets the strict standard.** As the directive requires, I do not force a selection. §9 states what further work would be needed. The most promising open problem is *valid* calibration of the near-floor-optimal residual scan under LRD. That detector reaches the floor up to the unavoidable scan price, but at nominal thresholds its false-alarm rate was 0.12–0.46.

---

## A. Literature and prior-art review

The phase-6 matrix (`02_literature/phase6_prior_art_matrix.md`) adds 41 records in five groups:
- covert-communication square-root laws (P1–P6);
- covert or stealthy change detection and CPS attacks (P7–P14);
- NIDS comparators (P15–P21);
- statistics of LRD, scans and robust testing (P22–P35);
- other representations: Koopman, TDA, SNN, Hurst features (P36–P41).

Each record states:
- the problem;
- the formulation;
- the assumptions;
- the guarantee;
- the evidence;
- the limitations;
- the exact difference from the candidates here.

**Read at full-text level** (via HTML; the parts read are stated in the matrix):
- P3, Bouette–Luzzi–Wang;
- P7 and P8, Ramtin–Nain–Towsley;
- P10, Bai–Pasqualetti–Gupta.

The other records are at abstract level (A) or are foundational works described from knowledge (K). Those must be re-read before any citation.

**Most important finding.** The closest prior art was found *after* the main idea was formed. It narrowed the claim considerably:
- **Ramtin et al.** already prove the √γ cumulative-damage law for i.i.d. or Brownian pre-change data with a fixed reference.
- **Bouette et al.** prove that for covert *information*, an arbitrary noise covariance leaves the square-root law unchanged ("L=1 irrespectively of … Σ_n", read verbatim).

The contribution below exists only in the gap between these two results.

## B. Research-gap analysis: fundamental questions

| Question | What the literature does | What is unanswered (evidence) |
|---|---|---|
| Q1. What cumulative anomalous activity is *guaranteed* detectable by any NIDS, given realistic benign dependence? | Detector-specific evaluations (AUC, TPR@FPR). √ laws for i.i.d. data (P1, P6, P7). Hurst exponents used as detection *features* (P40). | No detector-independent floor for LRD traffic. LRD is the empirical norm (P22). P24 shows LRD and mean changes are hard to separate, but it is not converted into a floor. |
| Q2. Is the validity/absorption tension in doc 05 §1.4 fundamental? | Self-starting / Phase-I SPC and poisoning bounds for specific detectors (P13). | No detector-independent statement linking reference length to the minimum detectable sustained rate. |
| Q3. How much does peer (cross-sectional) information lower the floor? | Peer-group analysis and synthetic control are used heuristically (P20, P21). | No quantification in terms of a floor. |
| Q4. Can benign synchronised change be separated from propagation? | Edge-reversal and shuffle tests (P30); Hopper (P17); identifiability limits (P31). | NIDS-specific validity under rollouts is untested. Doc 05 found one-off subset rollouts irreducible by timing alone. |
| Q5. Which representation maximises detectability per unit of attack effect? | Feature engineering; Fisher information in detection theory. | The floor gives a selection criterion: minimise the benign SD of the monitored functional. |
| Q6. Can unseen attacks be separated from unseen benign change? | Open-set conformal (doc 05 §5). | Proven unidentifiable without side information (doc 05 L5.1). No new mechanism found. |

## C. Candidate innovations (12)

Each card lists the 14 required fields in compressed form:
1. problem;
2. limitation of existing work;
3. new idea;
4. formulation;
5. mechanism;
6. theoretical advantage;
7. practical advantage;
8. assumptions;
9. failure conditions;
10. complexity;
11. closest work;
12. why it is not a reformulation;
13. how it could be disproved;
14. validation.

**C1. Detectability floors under benign dependence** (selected; §5)
1. The minimum cumulative deviation any detector can reliably detect.
2. Floors exist only for i.i.d. data (P7, P8) or for information functionals (P3).
3. The floor equals a constant times the benign SD *of the monitored functional itself*.
4. G* = (z_{1−α} + z_β)·SD₀(wᵀX).
5. Monitor cumulative, mission-aligned functionals at multiple scales.
6. An exact minimax value and saddle point.
7. A computable, benign-only security metric per entity and feature.
8. Gaussian benign process (locally, LAN); additive anomaly.
9. Non-additive (substitutive) anomalies; heavy tails with infinite variance.
10. O(T²) to compute SD₀ exactly, or O(T) from a variance-time curve.
11. P7, P8, P3, P10.
12. A new functional class (linear displacement) with general covariance, a finite-reference law and a panel law.
13. Find a detector that beats G* under Gaussian Σ, or show that P8's unread sections already contain the general-Σ result.
14. Monte Carlo of the NP oracle at G* (§7, E1).

**C2. PS-VTS: peer-studentised variance-time scan** (failed; §6)
1. A detector valid under arbitrary temporal dependence.
2. Mixing-based validity (doc 05 Theorem 1a) fails under LRD.
3. Studentise each window sum against peers at the same time; exact t-law; Bonferroni over windows.
4. T = (S₀ − S̄_peers)/(s_peers·√(1 + 1/(n−1))) ~ t_{n−2}.
5. Multi-scale scan.
6. FA ≤ α with no temporal assumption.
7. Robust to LRD and to common non-stationarity.
8. Peers exchangeable and cross-sectionally Gaussian.
9. Coordinated compromise; heterogeneous peers; small roles.
10. O(n·N·log N).
11. P21, P20, P26.
12. The exact cross-sectional calibration under LRD is the claimed difference.
13. Its floor must beat every baseline by at least 25% (pre-registered).
14. E4/E5. **Result: failed.**

**C3. Propagation-direction cascade test**
1. Separate lateral propagation from benign rollouts.
2. Timing-only nulls cannot (doc 05 L4.1).
3. Test whether anomaly onsets follow *initiator-directed* contacts (infected → target) or pull contacts (host → update server).
4. A statistic counting onsets preceded by an outgoing contact from an already-anomalous host; an edge-direction-reversal null.
5. Permutation over edge direction.
6. Valid under direction exchangeability.
7. Addresses software-rollout false alarms.
8. Contact direction is observable.
9. P2P / multi-tier rollouts; attacker-initiated pull.
10. O(edges).
11. **P30 edge-reversal test**, P17, P18.
12. Only the domain mapping is new. Classified **Adaptation**.
13. Benign P2P rollout rejects the null.
14. Synthetic rollouts against chain propagation (not run).

**C4. Exact conformal novelty e-process for new-destination discovery**
1. Calibrated novelty-rate monitoring.
2. Heuristic "first-seen" alerts.
3. Under exchangeability, P(next value is new) equals the expected singleton fraction. This gives an exact conformal p-value for novelty.
4. p = (singleton count)/(n + 1).
5. e-CUSUM.
6. Exact under exchangeability.
7. Applies to scan, spread and exfiltration-to-new-endpoint missions.
8. Exchangeable benign draws.
9. Benign novelty drift (new services) breaks it.
10. O(1) amortised.
11. P29; Good–Turing; conformal methods.
12. A known identity used as a non-conformity score. **Adaptation**.
13. Benign growth in novelty rate causes false alarms.
14. Not run.

**C5. Unbalanced-OT decomposition: benign transport versus mass creation**
1. Separate deformation of benign behaviour from newly created behaviour.
2. Detectors measure distance to a single reference.
3. Decompose the shift into transport cost plus created mass.
4. min_π [⟨C, π⟩ + λ·KL(π1 ‖ p) + λ·KL(πᵀ1 ‖ q)].
5. Alarm on created mass.
6. Tolerates bounded transport.
7. Robust to benign deformation.
8. A metric in which benign drift is small.
9. Benign novelty also creates mass.
10. O(n² per iteration) with Sinkhorn.
11. P34, P28.
12. Combination of known tools. **Combination**.
13. Benign software change produces large created mass.
14. Not run.

**C6. Causal-mechanism invariance (ICP-style)**
1. Shift-robust detection.
2. Marginal detectors flag benign shifts.
3. Benign shifts act on root causes; attacks change mechanisms.
4. Test the invariance of P(Y | pa(Y)) across environments.
5. Per-mechanism e-tests.
6. Invariance to root-cause shifts.
7. Fewer false alarms after workload change.
8. A known causal graph and causal sufficiency.
9. Benign protocol upgrades change mechanisms.
10. Graph learning is expensive.
11. P33.
12. Direct application. **Adaptation**.
13. Benign upgrades violate invariance.
14. Not run.

**C7. Koopman new-mode detection**
1. Periodic command-and-control activity robust to amplitude shifts.
2. Periodograms on single flows.
3. Detect new eigenvalues rather than amplitude changes.
4. DMD spectrum novelty.
5. Spectral e-test.
6. Invariance to amplitude.
7. Detection of beacons.
8. Approximately linear lifted dynamics.
9. Jittered timing (doc 05 §4).
10. O(r³).
11. P36.
12. Prior work in power grids. **Adaptation**.
13. Jitter removes the spectral line.
14. Not run.

**C8. Observation-channel identifiability**
1. Which attacks are identifiable after flow aggregation.
2. No formal account.
3. Detectability under the pushforward Φ#.
4. TV(Φ#Q, Φ#𝒫).
5. Feature-map audit.
6. Impossibility results.
7. Guidance on sensor placement.
8. A known Φ.
9. An uninformative bound if 𝒫 is large.
10. Analytic.
11. Data processing; mimicry work.
12. A data-processing restatement. **Established/Incremental**.
13. Trivial bounds.
14. Not run.

**C9. Persistent-homology coordination signatures**
- Already used in network security (P37). **Established**.

**C10. Early-warning precursors**
- Critical slowing down before outbreaks (P35). Closest NIDS work: P16 (Kalman trend).
- **Adaptation**. Not run.

**C11. Validity versus masking versus drift trade-off** (the open problem from doc 05 §1.4)
- §5 Theorem D3 shows the trade-off is fundamental: a finite reference forces a linear floor.
- **Absorbed into C1.**

**C12. Event-driven / SNN e-process**
- Efficiency only. The SNN NIDS model family is established (P38, P39). **Established**.

## D. Ranked assessment (scores 1–5; not inflated)

| Cand. | Orig. | Depth | Importance | Feasib. | Outperform potential | Publication | Gap evidence | Classification |
|---|---|---|---|---|---|---|---|---|
| C1 (+C11) | 3 | 3 | 4 | 5 | n/a (theory) | 3 | 4 | **Potentially original** |
| C3 | 2 | 2 | 4 | 4 | 3 | 3 | 3 | Adaptation |
| C2 | 2 | 2 | 4 | 4 | **1 (failed)** | 2 | 3 | Combination; withdrawn |
| C8 | 2 | 3 | 3 | 3 | n/a | 2 | 3 | Incremental |
| C4 | 2 | 2 | 3 | 4 | 3 | 2 | 3 | Adaptation |
| C5 | 2 | 3 | 3 | 3 | 2 | 2 | 2 | Combination |
| C6 | 1 | 3 | 4 | 2 | 2 | 3 | 3 | Adaptation |
| C7 | 2 | 2 | 3 | 3 | 2 | 2 | 2 | Adaptation |
| C10 | 2 | 2 | 3 | 3 | 2 | 2 | 2 | Adaptation |
| C9 | 1 | 2 | 2 | 3 | 2 | 2 | 2 | Established |
| C12 | 1 | 1 | 2 | 3 | 2 | 2 | 1 | Established |

No candidate reaches 4+ originality. C1 is the only one whose central result survived both the prior-art search and adversarial checks.

## E–F. Selected contribution: detectability floors (mathematics)

### 5.1 Objects and assumptions

| Object | Definition |
|---|---|
| Benign observations | X ∈ ℝ^T (one entity, or a peer residual), X ~ N(μ, Σ) under H0. |
| Anomaly | Additive: X ~ N(μ + a, Σ) under H1, where a ∈ ℝ^T is unknown. |
| Monitoring functional | w ∈ ℝ^T_{≥0}; the cumulative deviation of interest is G(a) = wᵀa. Example: w = 1_W gives total extra volume over window W. |
| Detector | Any measurable test φ(X) ∈ {0, 1} with E_{H0} φ ≤ α. |

**Assumptions.**
- **G1.** Σ is known and positive definite.
- **G2.** The anomaly is additive. Substitutive anomalies are excluded (§5.6).
- **G3.** The process is Gaussian. §5.6 explains how this extends locally (LAN) and where it fails.

### 5.2 Theorem D1 [T]: exact floor and saddle point

For G > 0, define SD₀ = √(wᵀΣw).

**(a) Achievability, shape-agnostic.** The test φ_w = 1{wᵀ(X − μ) > z_{1−α}·SD₀} has size exactly α. For *every* a with wᵀa = G, its power is exactly

  Φ(G/SD₀ − z_{1−α}).

**(b) Converse.** No detector of size α can have power greater than Φ(G/SD₀ − z_{1−α}) against *all* anomalies with wᵀa = G.

Hence the minimax power is exactly Φ(G/SD₀ − z_{1−α}). For a target power β,

  **G*(α, β) = (z_{1−α} + z_β)·SD₀(wᵀX).**

*Proof.*
- **(a)** Under H1, wᵀ(X − μ) ~ N(G, SD₀²), whatever the shape of a.
- **(b)** Take a* = G·Σw / (wᵀΣw). By the Neyman–Pearson lemma, the most powerful size-α test against the simple alternative a* has power Φ(‖a*‖_{Σ⁻¹} − z_{1−α}). Compute ‖a*‖²_{Σ⁻¹} = G² wᵀΣΣ⁻¹Σw/(wᵀΣw)² = G²/SD₀². So no detector exceeds this power against a*. By Cauchy–Schwarz, ‖a‖_{Σ⁻¹} ≥ wᵀa/SD₀ for every a, so a* is the least-favourable anomaly. ∎

**What is new versus known.**
- The NP lemma and Cauchy–Schwarz are [K]. The result is elementary.
- The covariance-general, finite-horizon saddle point for a linear displacement functional was not found in P1–P14.
- P7 has the i.i.d., asymptotic special case.
- P3 shows that the *information* analogue is covariance-invariant. For displacement, the floor depends on Σ only through the benign SD of the monitored functional.

**Defender meaning.** Monitoring the cumulative functional directly is minimax optimal and needs no model of the anomaly's temporal profile. The floor is computable from benign data alone. It gives a per-entity, per-feature security metric: "the cumulative deviation guaranteed detectable with probability β at false-alarm rate α".

### 5.3 Theorem D2 [P]: dependence scaling and the calibration frontier

**(a)** For fractional Gaussian noise (fGn) with Hurst exponent H and w = 1_W with |W| = s:

  SD₀ = σ·s^H, so G* ∝ s^H.

The ratio to the i.i.d. floor is s^{H−1/2}:

| s | H = 0.8 | H = 0.9 |
|---|---|---|
| 1024 | 8× | 16× |
| 10⁵ | 31.6× | 100× |

So LRD (P22) makes the guaranteed-detectable volume grow *faster than √s*.

**(b) Calibration frontier.** Suppose a detector computes thresholds assuming H_det while the truth is H.
- If H_det < H, the false-alarm probability tends to 1 as s grows: each window's true SD exceeds the assumed one by s^{H−H_det}.
- If H_det > H, the floor is inflated by s^{H_det−H}.

There is no free lunch: calibration must match the true variance–time curve.

### 5.4 Theorem D3 [T]: finite own-history reference ⇒ linear floor

**Setting.** The host offset μ = c·1 is unknown, as it is for every own-history detector. The data consist of a reference R of length N_ref and a monitoring window W of length s. The anomaly is supported on W. Let the benign process be stationary with spectral density f ≥ m > 0.

**Claim.** For every size-α invariant detector,

  G*(s) ≥ (z_{1−α} + z_β)·s·√(m/N_ref).

**Proof.**
1. Restrict to tests invariant to adding c·1. The maximal-invariant non-centrality is d² = aᵀPa, where P = Σ⁻¹ − Σ⁻¹11ᵀΣ⁻¹/(1ᵀΣ⁻¹1). Equivalently, xᵀPx = min_c (x − c1)ᵀΣ⁻¹(x − c1).
2. For a supported on W, maximise wᵀa subject to aᵀP_WW a ≤ d². The maximum is d·√(wᵀP_WW⁻¹w).
3. By Cauchy–Schwarz, this is at least d·(wᵀw)/√(wᵀP_WW w) = d·s/√(1_WᵀP1_W).
4. Take c = 1 in the min-characterisation: 1_WᵀP1_W ≤ (1_W − 1)ᵀΣ⁻¹(1_W − 1) = 1_RᵀΣ⁻¹1_R.
5. Since f ≥ m, Σ ⪰ mI, so 1_RᵀΣ⁻¹1_R ≤ N_ref/m.
6. Set d = z_{1−α} + z_β. ∎

**Meaning.** With any finite own-history reference there is a minimum detectable sustained *rate*, z·√(m/N_ref) per window. This holds whatever the duration and whatever the detector.
- Lowering it requires a longer reference.
- A longer reference is exactly what benign drift tolerance forbids (doc 05 §3, Proposition 3.3).
- This is the formal, detector-independent version of doc 05's absorption result. It closes doc 05 §1.4 [O] as a **fundamental trade-off**, not an engineering gap.

**Exact i.i.d. case.** G* = z·√(s(1 + s/N_ref)). The simulated ratio to the known-offset floor was 1.414 at s = N_ref, 2.236 at 4N_ref and 3.000 at 8N_ref, exactly √(1 + s/N_ref).

**fGn, H = 0.8.**
- The ratio is below 1 for short windows (0.82 at s = 16). LRD makes the reference predictive, so the floor drops.
- The floor becomes linear once s ≳ N_ref: the log₂ slope between s = 1024 and 2048 is 1.004.

**Closest prior art.** SPC estimated-parameter effects (Phase I/II; Jones, Champ & Rigdon 2001 — K, to verify) and Kloft & Laskov (P13). Both are detector-specific. A full-text check is required before any claim of novelty.

### 5.5 Proposition D4 [K + P]: price of an unknown onset, and D5 [P]: peer residual

**D4.** With M candidate windows, a Bonferroni scan has floor (z_{1−α/M} + z_β)·SD₀. Under weak dependence between windows, no test can do better than order √(2 log M)·SD₀ [K: P26, P2].

Simulation: the best scan baseline's floor divided by the oracle floor was 2.5 at L = 32 and L = 256, and 2.2 at L = 1024. The predicted ratio is z_{α/M}/z_α = 4.06/1.645 = 2.47.

**D5.** Under an additive common mode (doc 05 D1), the peer-median residual removes the common component. The floor then depends only on the idiosyncratic SD, inflated by the residual factor (≈ 1 + π/(2(n − 1)) for a Gaussian median).

In E4 B0 at L = 256:
- oracle floor on the raw series: 248;
- oracle floor on the idiosyncratic component: 53;
- median-residual scan: 131;
- own-history scan: 394.

### 5.6 Limitations of D1–D5

1. **Gaussianity.** For additive shifts with fixed covariance, Gaussian noise minimises Fisher information. The Gaussian floor is therefore the *defender-conservative* (largest) floor locally (LAN), not globally. With infinite variance (α-stable noise) SD₀ is undefined.
2. **Additivity.** Substitutive anomalies, which replace benign activity rather than add to it, are outside the model. Their detectability is governed by distributional distinguishability (doc 05 Proposition 3.3).
3. **Σ must be known or estimated.** Estimating H under non-stationarity is unreliable (P41). D2(b) quantifies the cost.
4. **Coverage of functionals.** Each functional w needs its own floor. A union over K functionals costs log K (Bonferroni).
5. **Scope.** D1–D5 are detection limits. They do not address identification (doc 05 §5) or unseen benign change (L5.1).

## G. Synthetic adversarial experiments (`sims/capacity.py`) [S]

**E1: exact floor**
- Covariances tested: i.i.d., AR(1) with ρ = 0.8, fGn with H = 0.7 and 0.9, and a local-level model. T = 512, 20,000 draws each.
- The NP oracle's detection rate at G*(α = 0.05, β) was 0.099–0.102 for β = 0.1, 0.489–0.500 for β = 0.5, and 0.899–0.902 for β = 0.9. All match the prediction.
- The shape-agnostic sum test matched the oracle against the least-favourable anomaly, as Theorem D1 says.
- Falsification: under the local-level (drift) model, a constant-profile anomaly with the same G was detected at 1.0 against 0.5. The profile matters for any *specific* anomaly. The theorem concerns the worst case, and no detector beat the worst-case value.

**A3: finite reference** — results as in §5.4. The UMPI Monte Carlo gave detection 0.504 / 0.500 (predicted 0.5) and false alarms 0.0505 / 0.0494 (predicted 0.05).

**E3: calibration frontier** (truth H = 0.8; scan detector)

| H_det | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| FA | **0.97** | **0.82** | **0.33** | 0.012 | 0 | 0 |
| median floor, L = 512 | — | — | 213 | 510 | 908 | 1164 |

This confirms D2(b). (No floor is given for H_det = 0.5 and 0.6 because the detector false-alarms before the anomaly.)

**E4: detector comparison**

Setup:
- Peer group of n = 20 hosts.
- Common fGn with H = 0.9, σ_c = 1; idiosyncratic fGn with H = 0.75, σ_e = 0.5.
- Reference 1024 windows, monitoring 2048 windows.

Benign scenarios:
- **B0**: stationary.
- **B1**: adds common diurnal variance modulation and a common +3 level shift (an outage or operational change).
- **B2**: adds benign idiosyncratic drift on the target host. This violates peer exchangeability.

Fairness rule: **every baseline is given its oracle threshold** — the empirical 95th percentile of its own null statistic, from 200 benign runs of the same scenario. Floors therefore compare power at matched false-alarm rate. The "nominal FA" column shows what happens without that oracle.

Nominal false-alarm rate (target 0.05):

| Detector | B0 | B1 | B2 |
|---|---|---|---|
| PS-VTS | **0.005** | **0.010** | **0.015** |
| own-history VTS | 0.155 | 0.99 | 0.97 |
| peer-median residual + VTS | 0.115 | 0.43 | 0.455 |
| CUSUM (i.i.d.-calibrated) | 0.96 | 1.0 | 1.0 |
| EWMA, adaptive baseline | 1.0 | 1.0 | 1.0 |
| online conformal e-CUSUM (doc 05 Theorem 1a) | 0.875 | 1.0 | 1.0 |
| peer-median residual + CUSUM | 0.97 | 0.975 | 0.91 |

Median detection floor for step anomalies (cumulative deviation at which detection first occurs; lower is better):

| Detector | B0 L32 | B0 L256 | B0 L1024 | B1 L32 | B1 L256 | B1 L1024 |
|---|---|---|---|---|---|---|
| PS-VTS | 88 | 555 | 2346 | 98 | 717 | 2564 |
| own-history VTS | 74 | 394 | 1268 | 322 | 1381 | 4285 |
| median residual + VTS | **28** | 131 | 329 | **42** | 178 | 515 |
| median residual + CUSUM | 83 | **120** | **191** | 122 | **156** | **252** |
| EWMA adaptive | 60 | 352 | 1361 | 163 | 1018 | 3982 |
| online conformal e-CUSUM | ∞ | 312 | 912 | ∞ | ∞ | 2400 |
| CUSUM i.i.d. | ∞ | 823 | 1062 | ∞ | ∞ | 3273 |
| *oracle floor, idiosyncratic (known onset)* | 11 | 53 | 149 | 11 | 53 | 149 |
| *oracle floor, raw series* | 39 | 248 | 855 | 39 | 248 | 855 |

Ramp anomalies show the same ordering (B0 L256: PS-VTS 358, median residual + VTS 95). ∞ means no detection within the amplitude search range at that detector's threshold.

**E5: falsification of PS-VTS** (B0, L = 256)

| Test | Nominal FA | Median floor |
|---|---|---|
| heavy-tailed idiosyncratic noise | 0.01 | 662 (median residual + VTS: 254, but with FA 0.545) |
| target twice as variable as its peers | 0.005 | 628 (exchangeability violated, yet still conservative) |
| small role, n = 5 | 0.01 | **1075** |
| coordinated compromise, 10% of hosts | 0 | 557 |
| coordinated compromise, 30% of hosts | 0 | **∞ (collapsed)** |
| coordinated compromise, 50% of hosts | 0 | **∞ (collapsed)** |

Under the same coordinated compromise, median residual + VTS reached 122 / 140 / 218.

## H. Comparison with the closest existing methods

- **Against P7/P8 (covert QCD).** D1 reduces to their √ law for i.i.d. data. D1 adds the exact finite-T value, general Σ, and the saddle point. D3 adds the finite-reference law, and D5 the panel law. Until the unread part of P8 has been checked, it is **uncertain** whether P8 states D1 for general Σ.
- **Against P3 (covert information with memory).** The functional differs (displacement versus nats), and the dependence on Σ is opposite.
- **Against detector practice (E4).** The best power at matched FA came from *existing* combinations (median residual + VTS/CUSUM). Their floors sit within the predicted scan price (×2.2–2.5) of the oracle floor. **None of them is valid at nominal thresholds under LRD** (FA 0.115–0.975). This is exactly what D2(b) and D3 predict for own-reference calibration.

## 6. Why PS-VTS was withdrawn

Pre-registered criteria (in the `capacity.py` docstring, fixed before the first run):
- **S1 validity.** Passed in every scenario.
- **S2 floor ≤ 0.75× every baseline for long anomalies.** **Failed** by factors of 2–12.

Diagnosis, consistent with the theory:
1. The Student-t tail at level α/M with 18 degrees of freedom is much heavier than the Gaussian tail: t_{18} at 2.5·10⁻⁵ ≈ 5.6, against z ≈ 4.1.
2. Bonferroni over overlapping windows is very conservative (FA 0.005 against 0.05).
3. The mean and SD of peers estimate a nuisance that a known-structure detector does not need to estimate.
4. A mean/SD statistic has breakdown point 1/n. That explains the collapse at 30% coordinated compromise; the median-based baseline degrades gracefully.

**The guarantee itself survived. The practical claim did not.** It is withdrawn.

## I. Unified architecture (how the surviving theory constrains any design)

1. **Representation.** Monitor *cumulative, mission-aligned functionals*: volume to external destinations, number of distinct new internal targets, authentication fan-out. Compute them on **peer-median residuals** (D5), at dyadic scales (D4).
2. **Calibration.** Use the benign variance–time curve of each functional (D2). The open problem is to do this *validly* under LRD (§9).
3. **Metric.** Report the floor G*(α, β) per entity and functional as the security metric, instead of AUC (cf. doc 05 Proposition 4.3).
4. **Reference policy.** D3 makes explicit the trade-off between reference length and the minimum detectable sustained rate. It combines with doc 05 §6 (quarantine, purge) for the normality pool.
5. **Downstream.** Identification (doc 05 §5) and coordination nulls (doc 05 §4) are unchanged.

This is a *design constraint*, not a new detector. There is no new detector here.

## J. Expected scientific impact and publication potential

- **Realistic venue for D1–D5.** A theory or short paper in a security-statistics or information-theory-for-security venue, framed as "detectability floors for NIDS under self-similar traffic". The novelty argument must lead with the explicit contrast with P3 and the extension of P7/P8.
- **Risk.**
  - The mathematics is elementary. Reviewers may class it as a corollary of known results.
  - Its value is conceptual and diagnostic. It explains why own-reference detectors either false-alarm or absorb, and it gives a computable metric.
- **Not yet supportable.** No doctoral-scale contribution on practical zero-day superiority is supportable from this phase.

## K. Final originality report

| Claim | Status | Evidence | Uncertainty |
|---|---|---|---|
| D1 exact floor and saddle point, general Σ | Potentially original | Proof [T]; E1 [S]; P1–P14 searched | P8's unread sections; CPS literature on linear functionals |
| D2 T^H scaling and calibration frontier | Potentially original (consequence of D1 + P22) | [P]; E3 [S] | Robustness of H estimation (P41) |
| D3 finite reference ⇒ linear floor | Potentially original | Proof [T]; A3 [S] | SPC estimated-parameter literature still to read in full text |
| D4 scan price | Established | [K] P26, P2 | — |
| D5 panel reduction | Incremental | [P]; E4 [S] | Multiplicative common mode not removed |
| PS-VTS detector | Combination; **withdrawn** (failed S2) | E4/E5 | — |
| C3–C12 | Adaptation / Combination / Established | Matrix P15–P41 | Not simulated |
| "Substantial practical improvement over the strongest methods" | **Not demonstrated** | E4: existing combinations were best | Requires valid LRD calibration (§9) |

## 9. What further work would be needed (nothing started without approval)

1. **Prior art.** Full-text reading of the rest of P8, P10 §4 onward, P13, and the SPC estimated-parameter literature. These determine whether D1/D3 survive as original.
2. **Valid near-floor detection under LRD [O].** Calibrate a peer-residual variance–time scan so that FA ≤ α *without* oracle thresholds.
   - Candidate approaches: pooled cross-host variance–time estimation with explicit estimation-error inflation; simulation-based calibration from a fitted fGn model, with its error propagated.
   - Pre-registered criterion: nominal FA ≤ α + 2 s.e. in B0–B1, and floor ≤ 1.25× the oracle-threshold version.
3. **Further synthetic falsification of D1–D3.**
   - Non-Gaussian, ON/OFF-generated LRD.
   - Multiplicative common mode.
   - Multiple functionals with the log K price.
4. **Real-data experiments.** Only after your explicit approval.
