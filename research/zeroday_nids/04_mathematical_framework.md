# Phase 4 — Mathematical framework: candidate mechanisms, prior art, derivations, validation

**Status: for review. No architecture implemented; HoNE untouched.** Synthetic checks live in `sims/`
(`validate_mechanisms.py` → `results.json`; `validate_roc_calibrator.py` → `results_roc.json`). They test
the *mathematics* on a controlled panel model; they are not NIDS results. Every number quoted below is
printed by those scripts.

Notation. Entities (hosts) v ∈ V, |V| = N, grouped into roles ρ(v) learned from past benign behaviour.
Discrete evaluation times t (an entity window = its next T flows). A base detector s(·) maps a window to
a score (higher = more anomalous). s_{v,t} := s(x_{v,t}). A *calibrator* g: [0,1] → [0,∞) is
non-increasing with ∫₀¹ g ≤ 1, so g(p) is an e-value whenever p is super-uniform. e-CUSUM:
M_{v,t} = max(M_{v,t−1}, 1)·e_{v,t}, alarm at M ≥ A (Shin, Ramdas & Rinaldo).

---------------------------------------------------------------------------------------------------
## 0. The correction this phase forced on Phase 3

Phase 3 proposed calibrating each host against **contemporaneous peers**. Derivation and simulation show
this is **invalid**: hosts differ persistently (α_v below), so a busy host always ranks high among peers
and its e-process grows under the null. Simulated false-alarm rate (A = 200, bound 5 per 1000 steps):
**8.5 / 10.6 / 13.8** for temporal correlation ρ = 0 / 0.6 / 0.9 (`results.json → validity`). The framework
below replaces it with a *double-differenced, delayed-reference* construction that is valid in the same
simulations.

---------------------------------------------------------------------------------------------------
## 1. Fourteen candidate mechanisms

Each entry: definition → motivation → derivation → closest prior work → what is (and is not) new →
falsification. "Validated" = checked numerically in `sims/`; "derived" = argument only.

### M1. Panel decomposition of detector scores (representation)
**Definition.** For each role r, write s_{v,t} = μ_{r,t} + α_v + ε_{v,t}, with μ_{r,t} the common mode
(estimated robustly, μ̂_{r,t} = median_{u∈r} s_{u,t}), α_v a slowly varying entity offset, ε_{v,t} the
idiosyncratic part with E[ε]=0.
**Motivation.** HoNE's failure modes map onto different components: benign drift and network-wide attacks
move μ; heterogeneous but benign hosts differ in α; host-local attacks move ε; coordinated attacks move ε
jointly for a subset. One score cannot separate these; the decomposition can.
**Closest work.** Lakhina et al., SIGCOMM 2004 (PCA normal/residual subspace on link counts);
Xie & Siegmund, Ann. Statist. 2013 (change in a subset of parallel streams, Gaussian);
ECAD (Xu et al., arXiv:2105.11886; conformal residuals of a shared regression across sensors).
**New?** The decomposition itself is a standard two-way panel model. New only in its *use*: applied to
anomaly-score panels of learned NIDS detectors, it routes each component to its own valid test (M2–M5).
**Falsify.** If μ̂ removal does not reduce false alarms under benign drift (H-V2), M1 adds nothing.

### M2. Double-differenced, delayed-reference conformal p-value (deviation function) — *validated*
**Definition.** Residual r_{v,t} = s_{v,t} − μ̂_{ρ(v),t}. Reference set
R_{v,t} = { r_{v,τ} : τ ∈ [t − g − L, t − g), τ ≡ t (mod k) }, size m ≈ L/k. Then
  p_{v,t} = ( #{ r ∈ R_{v,t} : r > r_{v,t} } + U·(1 + #{r = r_{v,t}}) ) / (m + 1),   U ~ Unif(0,1).
Three knobs: common-mode removal (first difference), own-history reference (second difference, removes α_v),
**gap g** (delayed reference), **stride k** (decorrelation).
**Derivation.**
(i) *Exact case.* If, under H₀, the residuals {r_{v,τ}: τ ∈ R_{v,t}} ∪ {r_{v,t}} are exchangeable, p is
exactly Unif(0,1) (standard conformal argument; Bates et al. 2023).
(ii) *Dependence.* For a stationary β-mixing residual process with coefficients β(·), Berbee's coupling
replaces each reference point (pairwise separation ≥ k, separation from the test point ≥ g) by an independent
copy at total-variation cost ≤ m·β(min(k, g)). Hence sup_u |P(p ≤ u) − u| ≤ ε := m·β(min(k,g)).
(Same mechanism as Chernozhukov–Wüthrich–Zhu 2018; Oliveira et al. 2022.) For AR(1), β(k) = O(ρ^k), so
choosing k ≥ log(ε/m)/log ρ makes ε negligible.
(iii) *Masking.* With a short gap, an attack that persists longer than g enters its own reference and is
absorbed. A delayed reference (g > attack horizon H) makes the test blind to attacks only after H.
**Validation (results.json).**
| setting (A = 200; bound 5 / 1000 steps) | ρ = 0 | ρ = 0.6 | ρ = 0.9 |
|---|---|---|---|
| peer-only (Phase 3) | 8.46 ✗ | 10.58 ✗ | 13.79 ✗ |
| double difference, no stride | 0.26 | **5.11 ✗** | **15.13 ✗** |
| double difference, stride k = ⌈log 0.05 / log ρ⌉ | 0.27 | 0.51 | 0.55 |
| benign upward ramp (+3.2 sd): training-calibrated | — | **41.8 ✗** | — |
| benign upward ramp: double difference, strided / delayed (g = 700) | — | 0.60 / 0.45 | — |
Masking (attack δ = 1 sd, 600 steps, 6 hosts): short gap (g = 20) detects 3/6; delayed gap (g = 700) 6/6.
**Closest work.** Conformal anomaly detection (Laxhammar; Bates et al.); ECAD; conformal martingales
(Vovk et al.; Volkhonskiy et al. 2017); FIRCE (per-flow conformal p-values).
**New?** Each ingredient exists. What I found no precedent for is the *specific* construction: removing a
role-wise common mode, calibrating against the entity's **own delayed and strided** past, with an explicit
β-mixing cost m·β(min(k,g)) and an explicit masking horizon g — and the demonstration that the intuitive
peer-relative null is invalid under persistent heterogeneity. Novelty: **moderate**; value: **high** (it is
the validity backbone of everything else).
**Falsify (H-V1).** On benign-only periods of real traffic, empirical time-to-false-alarm < A for the
chosen (g, k) ⇒ the mixing assumption fails for network data.

### M3. Robust calibration under approximate validity (dependence deflation) — *derived*
**Lemma.** Let h be a calibrator capped at c (h(0) = c), non-increasing, with ∫₀¹ h = 1 − η ≤ 1, and let
the conditional CDF of p given the past satisfy F(u) ≤ u + ε for all u. Then
  E[h(p) | past] = h(1) + ∫₀¹ (−h′(u)) F(u) du ≤ ∫₀¹ h + ε (c − h(1)) ≤ 1 + ε (c − h(1)).
Hence e′ = h(p) / (1 + ε(c − h(1))) is a valid conditional e-value, and the e-CUSUM on e′ keeps ARL ≥ A.
(For the capped mixture calibrator h(1) = 1/2.)
**Motivation.** Turns M2(ii)'s additive validity gap into a multiplicative deflation of the e-process with a
known price: each window's log-evidence drops by log(1 + ε(c − ½)), so first-order delay rises from
log A / D to log A / (D − log(1 + ε(c − ½))) — detection remains possible only while D exceeds that cost.
**Closest work.** Barber et al. 2023 (coverage gap under non-exchangeability); e-detectors (Shin et al.).
**New?** A short, checkable lemma; not found stated in this form for e-detectors. Novelty: **low–moderate**.
**Falsify.** Monte Carlo with a known ε (inject a controlled sub-uniform deviation) must show ARL ≥ A for
e′ and < A for uncorrected e when ε(c − ½) > log-margin.

### M4. ROC-slope calibrator and the sequential detectability index Ψ — *validated*
**Definition.** If p_{v,t} is FPR-calibrated (p = 1 − F₀(s)), then under a fixed attack its CDF is the
detector's ROC curve: P_{H₁}(p ≤ u) = ROC(u). Define the calibrator g*(p) = ROC′(p) and the
**sequential detectability index** Ψ := ∫₀¹ ROC′(u) log ROC′(u) du = KL(law of p under H₁ ‖ Unif).
**Derivation.** (a) E_{H₀}[g*(p)] = ∫₀¹ ROC′(u) du = ROC(1) − ROC(0) = 1 ⇒ valid. (b) For any calibrator g,
E_{H₁}[log g(p)] ≤ E_{H₁}[log ROC′(p)] = Ψ (Gibbs' inequality, since g is a sub-density). (c) Wald's identity
gives first-order e-CUSUM delay ≈ log A / Ψ. Since p is a monotone transform of the score,
KL on scores = Ψ, so by Lorden's lower bound **no procedure that sees only the scores** and keeps ARL ≥ A has
first-order delay below log A / Ψ: g* is first-order optimal among score-based detectors. A detector using
richer information can do better, but not below log A / KL(H₁‖H₀ on raw windows) ≤ log A / Ψ. (d) Binormal ROC with shift δ: Ψ = δ²/2 = (Φ⁻¹(AUC))².
**Validation (`results_roc.json`, A = 1000).**
| δ | AUC | E_{H₀}[g*] | Ψ̂ (theory) | median delay, g* | log A/Ψ | median delay, mixture calibrator |
|---|---|---|---|---|---|---|
| 0.25 | 0.570 | 0.996 | 0.035 (0.031) | 176 | 221 | 2107 |
| 0.5 | 0.638 | 1.003 | 0.121 (0.125) | 45 | 55 | 240 |
| 1.0 | 0.760 | 1.030 | 0.514 (0.500) | 13 | 14 | 22 |
| 1.5 | 0.856 | 1.050 | 1.100 (1.125) | 6 | 6 | 10 |
**HoNE-implied predictions** (binormal assumption, A = 1000 windows): D1 Infiltration (best AUC 0.60)
≈ 108 windows; D3 scanning (0.63) ≈ 63; D1 Bot (0.69) ≈ 28; rolling-family AUC 0.75 ≈ 15; D3 Backdoor
(0.98) ≈ 2. This is a *quantitative explanation* of HoNE's "undetectable core": those classes are not
undetectable in principle, they are undetectable **per window**, and become detectable only if they persist
for ~10²  windows at the entity level.
**Prior art (searched).** That the optimal betting function equals the p-value density is **known**
(Fedorova et al. ICML 2012 plug-in martingale; Vovk, Nouretdinov et al.; log-optimal calibration as monotone
density estimation, arXiv:2602.08927). ROC slope = likelihood ratio is classical signal-detection theory.
**New?** The calibrator is *not* new. Potentially new: (1) obtaining the betting density from the base
detector's ROC estimated on **known** attack families and transferring it to unknown ones; (2) Ψ as a
**predictor of which attack classes are sequentially detectable**, falsifiable against HoNE's per-class
results; (3) the explicit statement that AUC is the wrong summary for sequential NIDS: two detectors with
equal AUC can differ arbitrarily in Ψ (Ψ weights the low-FPR slope logarithmically). Novelty: **moderate**
(application-level theory), significance: **high** (directly answers HoNE's operating-point finding).
**Falsify (H-D1, H-D2).** (a) On real traffic, median e-CUSUM delay per class must fall within a factor
2 of log A / Ψ̂ estimated from *training-period* ROCs; (b) the rank of classes by Ψ̂ must predict the rank by
delay (Spearman ≥ 0.6, pre-registered); failure of (b) refutes Ψ as a detectability index.

### M5. Network-wide (common-mode) e-detector — *validated*
**Definition.** Apply M2's own-history conformal construction to μ̂_{r,t} (and to the global median), with
the reference taken from the same phase of past days (seasonal alignment) when diurnal structure exists:
R^{net}_t = { μ̂_{τ} : τ ∈ t − d·P ± w, d = 1..D } (P = period). e-CUSUM on g(p^{net}_t).
**Motivation (answers Q4).** Population-wide attacks (DDoS on many hosts, worms) move μ, which the
entity-level test removes by construction; they need their own test against the network's **own history**,
not against peers.
**Validation.** Network-wide shift +1 sd on all 60 hosts: 0 false alarms before, detected after 1.3
windows; the entity-level double difference raised only 1/60 alarms (`results.json → network`) — confirming
that M2 is blind by design and M5 is necessary.
**Closest work.** Lakhina 2004; classic multichannel CUSUM (Tartakovsky et al.); Dandapanthula & Ramdas
2025 (global-null e-detector, e-d-GNT).
**New?** Low on its own; the contribution is the explicit decomposition-based routing (M1) with validity at
both levels. **Falsify (H-N1).** Seasonal reference must keep ARL ≥ A over ≥ 2 weeks of benign traffic.

### M6. Multi-entity alarm control — *validated, adopted not invented*
**Definition.** At each time, e-BH (Wang & Ramdas 2022) over entity e-values, or the sequential
e-d-BH / e-d-Bonferroni of **Dandapanthula & Ramdas (arXiv:2501.04130, 2025)**.
**Validation.** 50 hosts, 10 attacked, equicorrelated nulls (ρ = 0.5): FDR = 0.000 at α = 0.1, power 0.44.
**New?** **No** — this exact combination (e-detectors + e-BH across streams) is published. We adopt it as the
answer to "cross-device dependence" (Q3): FDR/PFER control holds under arbitrary dependence *between*
entities; what remains our problem is validity *within* each entity (M2–M3).

### M7. Coordination statistic with a dependence-preserving null (attack relationship function) — *validated*
**Definition.** Exceedance indicators b_{v,t} = 𝟙{p_{v,t} ≤ q} from M2 (common mode already removed).
For a candidate group S (a role, or a neighbourhood in the communication graph G_t),
  C_S = Σ_t Σ_{u<w∈S} b_{u,t} b_{w,t}   (lag-0 co-exceedances; lagged version Σ_{|ℓ|≤λ} b_{u,t} b_{w,t+ℓ}).
Null by **independent circular shifts** b_{v,·} ↦ b_{v,·+σ_v}: preserves every entity's marginal rate and
autocorrelation exactly, destroys only cross-entity alignment. Permutation p-value
p_S = (1 + #{C_S^{(j)} ≥ C_S}) / (B + 1), valid when entities are independent given the common mode
(conditionally exchangeable shifts).
**Motivation (answers Q1).** Coordinated compromise (bots beaconing, distributed scanning) can be invisible
per host (each b rare) yet improbable jointly. Naive tests that assume within-host independence are
anti-conservative on persistent traffic.
**Validation (200 reps, nominal 0.05).**
| null | circular-shift size | naive (Poisson) size | power circ. / naive (4 of 12 hosts beaconing) |
|---|---|---|---|
| heterogeneous AR, ρ ∈ [0, 0.95] | 0.055 | 0.060 | 0.83 / 0.85 |
| + shared common shock, median removed | 0.070 | 0.085 | 0.88 / 0.91 |
| highly persistent, ρ ∈ [0.95, 0.995] | **0.025** | **0.200 ✗** | 0.17 / (0.63, invalid) |
**Closest work.** BotSniffer (NDSS 2008, spatial-temporal correlation of bot responses); BotMiner (USENIX
Security 2008, cross-cluster correlation); graph scan statistics (Speakman, McFowland & Neill 2015; fast
subset scan); circular-shift/jitter surrogates in neuroscience.
**New?** Coordination detection is old (BotSniffer/BotMiner, heuristic thresholds). New: computing it on
**common-mode-removed conformal exceedances** with a **dependence-preserving circular-shift null**, so its
false-alarm rate is controlled on persistent traffic where independence-based tests fail (size 0.20 above).
Scanning over subsets S → combine with fast subset scan (LTSS) for exact connected-subgraph maximisation,
with the circular-shift null calibrating the max statistic. Novelty: **moderate**.
**Known weakness.** Power falls on highly persistent traffic (0.17) — the circular null is conservative there.
**Falsify (H-C1, H-C2).** Size ≤ α on benign real traffic; power > naive-valid alternatives on replayed bot
beacons; failure ⇒ coordination is not separable from shared benign schedules (cron, updates).

### M8. Evidence signature (representation for identification) — *derived*
**Definition.** Run M2 per **feature** j (or per feature group) for an alarmed entity during its alarm
episode E: σ_j = Σ_{t∈E} log g(p^{(j)}_{v,t}) ≥ 0 normalised, σ ∈ Δ^{d−1} (which features carried the
evidence). Optionally keep direction (upper/lower tail).
**Motivation (answers Q2).** Scores say *that* something deviates; signatures say *how*, in a rank space that
is invariant to monotone feature transforms and to network scale — a candidate for cross-dataset transfer,
where HoNE found raw-feature transfer near chance (median AUC 0.500).
**Closest work.** COPOD (ICDM 2020) / ECOD (TKDE 2022): per-feature ECDF tail probabilities as scores;
CADE (USENIX Sec 2021): distance-based explanation of drift samples; SHAP-based audits (Sakr et al. 2026).
**New?** Per-feature tail probabilities are COPOD/ECOD; new is using **accumulated, self-referenced,
common-mode-removed** evidence as an *identification* representation rather than a score. Novelty: **low–moderate**.
**Falsify (H-I1).** Signatures of the same attack family in two different datasets must be closer than
signatures of different families (cross-dataset retrieval AUC > 0.7); otherwise the representation does not
transfer.

### M9. Open-set family identification with abstention and family-novelty evidence — *derived*
**Definition.** Known families k = 1..K with signature samples. Nonconformity a_k(σ) = distance of σ to
family k's calibration signatures (kNN). Conformal p-values π_k(σ). Prediction set
Γ(σ) = { k : π_k(σ) > α }. Output: Γ = {k} → name k; |Γ| > 1 → "one of Γ" (insufficient evidence to name);
Γ = ∅ → unknown. **Family novelty:** for a cluster U of unknown signatures (collected across alarmed entities
over time), accumulate e_U = Π_{σ∈U} g(max_k π_k(σ)) (valid if signatures are conditionally independent
across entities given known families); declare a *new family* when e_U ≥ 1/α. Clusters are formed on
signatures only after the alarms (no peeking at labels).
**Closest work.** Zhang et al. IJMLC 2021 (open-set NCM + clustering + incremental learning, no guarantees);
Masud et al. ECSMiner (novel class when outliers are cohesive and separated; heuristic thresholds);
Conformal Good-Turing classification (Xie et al., arXiv:2510.13037, conformal sets with an unseen option);
Transcendent (S&P 2022, rejection).
**New?** Each piece exists (CGTC for open-set conformal sets; Masud for cohesion). New: naming guarantee
P(true known family ∉ Γ) ≤ α *combined* with an **e-value for the emergence of a new family across
entities**. Novelty: **moderate**; the independence assumption across entities is the weak point.
**Falsify (H-I2, H-I3).** Leave-family-out on known attacks: coverage of Γ ≥ 1 − α on known families;
for a held-out family, P(Γ = ∅ or |Γ| > 1) ≥ 0.8 and the family-novelty e-value crosses 1/α within a bounded
number of alarmed entities. Failure ⇒ signatures do not separate families.

### M10. Growth-rate objective for training base detectors (learning objective) — *derived*
**Definition.** For a detector s_θ trained on benign windows B and known-attack windows K:
  L(θ) = − E_{x∈K} [ log g_θ( p̃_θ(x) ) ] + λ · Var_{(v,t)∈B}[ r̂_{v,t}(θ) − r̂_{v,t−g}(θ) ]
with p̃_θ(x) = mean_{b∈B} σ_τ(s_θ(b) − s_θ(x)) (smooth conformal rank, temperature τ) and g_θ a monotone
estimate of the ROC slope (M4). First term = empirical growth rate Ψ on known attacks (maximises *sequential*
detectability, which by M4 weights the low-FPR tail); second term penalises benign residual volatility, which
is what costs stride k and gap g in M2.
**Closest work.** Partial-AUC / top-push optimisation (Narasimhan & Agarwal 2013; Li et al. 2014); Deep SAD
(ICLR 2020) and Outlier Exposure (ICLR 2019); classifier-based betting (Podkopaev & Ramdas, NeurIPS 2023).
**New?** Tail-focused ranking losses exist; the specific pairing *growth rate on known attacks + benign
residual stability* (so that validity costs less) is not something I found. Novelty: **low–moderate**.
**Falsify (H-L1).** Against a BCE-trained detector, Ψ̂ on **held-out natural-novelty families** must increase
(not only on training families); if Ψ̂ rises only on seen families, the objective overfits known attacks.

### M11. Delayed-reference continual learning with a contamination bound (adaptive mechanism) — *derived*
**Definition.** The benign pool used to retrain s_θ and the roles at time t contains only windows older than
t − g from entities with no alarm in [τ − g, τ + g]. **Bound:** an attack episode of length H on entity v
contaminates the pool only through windows that stayed unalarmed for g after they occurred, so the expected
number of contaminating windows per episode ≤ E[min(τ_v, H)] · P(τ_v > g), with τ_v the detection delay (M4).
**Important negative result:** gating the **p-value reference** on the e-state (excluding suspicious windows
from M2's reference) is **invalid** — false alarms 21.8 per 1000 steps vs bound 5 at ρ = 0
(`results.json → validity, gated_dd_mix`). Gating is therefore allowed only for *training data*, never for
the conformal reference.
**Closest work.** Boiling-frog poisoning of PCA detectors (Rubinstein et al. 2009); "Can't boil this frog"
(Kravchik et al.); OWAD (NDSS 2023) adaptation; HOIDS (JNCA 2026).
**New?** Delay-based contamination bound tying continual learning to M4's delay theory. Novelty: **moderate**.
**Falsify (H-A1).** Slow-ramp poisoning (boiling frog) must not reduce later detection when updates follow
the rule; it must reduce it when updates use all recent traffic.

### M12. Role inference without identity (representation) — *derived*
Roles from clustering entities' *past benign* signature-free statistics (service mix, fan-in/out, diurnal
profile), re-estimated slowly; addresses only group flows. Closest: Powell 2022 (role-based lateral
movement), RolX. **New?** Low. Needed so M1/M5 compare like with like and avoid the IP-identity shortcut
acknowledged in TGN-SVDD.

### M13. Role-relative occupancy deviation via optimal transport (exploratory)
μ_{v,t} = distribution of (peer role, service, direction) over v's flows; deviation W₁(μ_{v,t}, ν_{ρ(v)}).
Closest: role mining; OT anomaly detection. Novelty low; kept only as a candidate base score for
relational attacks (lateral movement), not a framework element.

### M14. Transition entropy-rate deviation (exploratory)
Entropy rate of an entity's flow-type Markov chain vs its own history. Closest: Lakhina et al. SIGCOMM 2005
(entropy of feature distributions). Novelty low; candidate base score for scans and brute force.

---------------------------------------------------------------------------------------------------
## 2. Novelty summary (honest)

| Mechanism | Closest prior work | Genuinely different? | Novelty |
|---|---|---|---|
| M1 panel decomposition | Lakhina 2004; Xie–Siegmund 2013 | use as routing for valid tests | low |
| **M2 DD delayed-reference conformal p** | conformal AD; ECAD; CTMs; β-mixing conformal | yes, as a construction + negative result on peer null | **moderate** |
| M3 deflation lemma | Barber et al. 2023 | small lemma | low–mod. |
| **M4 ROC-slope calibrator / Ψ index** | plug-in martingales (Fedorova 2012); signal detection | calibrator no; Ψ as NIDS detectability predictor yes | **moderate** |
| M5 network-wide e-detector | Lakhina; Dandapanthula & Ramdas | routing only | low |
| M6 multi-entity e-BH | **Dandapanthula & Ramdas 2025** | no (adopted) | none |
| **M7 coordination with circular null** | BotSniffer/BotMiner; graph scan | yes, valid under persistence | **moderate** |
| M8 evidence signature | COPOD/ECOD; CADE | partly | low–mod. |
| **M9 open-set naming + family-novelty e-value** | CGTC 2025; Masud; Zhang 2021 | combination with guarantee | **moderate** |
| **M10 growth-rate + stability objective** | pAUC; Deep SAD; OE; betting classifiers | partly | low–mod. |
| **M11 delay-bounded continual updates** | boiling frog; OWAD | yes (bound + negative result) | moderate |
| M12 roles | Powell 2022; RolX | no | low |
| M13 OT occupancy | OT AD | no | low |
| M14 entropy rate | Lakhina 2005 | no | low |

No mechanism here is a new *architecture*, and none should be sold as one. The defensible originality is
**statistical**: valid, dependence-aware sequential decision rules for learned NIDS, plus a detectability
theory that explains HoNE's results.

## 3. The five strongest candidates

1. **M2 + M3 — validity under temporal dependence and heterogeneity.** Without it, every sequential claim is
   unsound (simulations show the naive versions fail by 2–8×).
2. **M4 — Ψ detectability index and ROC-slope betting.** Explains HoNE's undetectable core quantitatively;
   gives falsifiable per-class delay predictions.
3. **M7 — coordination under a dependence-preserving null.** Answers Q1; the naive alternative is invalid
   (size 0.20) on persistent traffic.
4. **M9 (+ M8) — open-set naming with abstention and family-novelty e-values.** Answers Q2 with a guarantee
   on mis-naming.
5. **M11 (+ M10) — delay-bounded continual learning and its learning objective.** Answers continual
   learning without learning attacks as benign; includes a negative result (gated references are invalid).

## 4. Unified framework: Panel e-Detection (PeD)

```
                    ┌────────────── role inference on past benign (M12) ─────────────┐
flows ─► entity windows x_{v,t} ─► base detectors s(·) (trained with M10) ─► s_{v,t}
        panel decomposition (M1):  s_{v,t} = μ̂_{ρ,t} + α̂_v + r_{v,t}
        ├─ network level:  μ̂_{ρ,t} ─► seasonal own-history conformal p (M5) ─► e-CUSUM ─► network alarm
        ├─ entity level:   r_{v,t} ─► DD delayed-reference p (M2) ─► ROC-slope calibrator g* (M4)
        │                  ─► deflation (M3) ─► e-CUSUM M_{v,t} ─► e-d-BH across entities (M6) ─► host alarms
        ├─ group level:    b_{v,t} = 𝟙{p_{v,t} ≤ q} ─► co-exceedance C_S on G_t, circular-shift null (M7)
        │                  ─► subset scan over connected S ─► coordinated-group alarms
        └─ identification: alarm episodes ─► evidence signatures σ (M8) ─► conformal sets Γ(σ) (M9)
                           ─► name / ambiguous / unknown ─► family-novelty e-value across entities
        adaptation: delayed, alarm-free windows only ─► retrain s(·), roles (M11); p-value references
                    are never gated (validity).
```
Coverage of the requested capabilities: legitimate behaviour (μ, α, roles, delayed references); individual
attacks (entity level); coordinated attacks (group level); network-wide attacks (network level);
unseen-attack identification (M8–M9); distribution shift (common-mode removal + own-history reference +
seasonal network reference); continual learning (M11).

**Global error accounting.** Per window step: network level ≤ 1/A_net; entity level FDR ≤ α_e (e-d-BH,
any cross-entity dependence) given within-entity validity (M2–M3); group level ≤ α_c per scanned family of
groups (circular null; max-statistic calibration for the scan). Operator budgets map to (A_net, α_e, α_c).

## 5. Formal assumptions and limitations

- **A1 (within-entity mixing).** Under H₀ the residual process r_{v,·} is stationary and β-mixing on the
  timescale of (g, k). Violated by abrupt benign regime changes local to one host (software update on one
  machine) — these will alarm; they are, operationally, changes worth flagging, but they are false alarms
  with respect to "attack".
- **A2 (common-mode sufficiency).** Benign drift is captured by a role-wise median; heterogeneous drift inside a
  role leaks into r.
- **A3 (persistence).** Accumulation helps only attacks that persist for ≈ log A / Ψ windows on one entity
  (M4). One-shot attacks are out of scope by construction (Jibunoh et al. 2026: many zero-days are novel
  vulnerabilities with familiar behaviour — also out of scope for any behaviour-based detector).
- **A4 (coordination null).** Entities independent given the common mode; shared benign schedules (cron,
  backups, updates) violate it → must be absorbed into the common mode or flagged as benign coordination.
- **A5 (identification).** Signatures of different families must differ in which features deviate; families
  that differ only in payload are inseparable in flow data.
- **A6 (masking horizon).** Attacks longer than the reference gap g become part of the reference after g;
  long-term compromise must be caught within g.
- **Adversary.** A mimicry attacker who keeps r within its own past distribution evades M2 by definition;
  the guarantee is about false alarms, not about evasion.

## 6. Falsifiable hypotheses (pre-registration draft)

| ID | Hypothesis | Falsified if |
|---|---|---|
| H-V1 | M2 with (g, k) chosen on training data keeps empirical ARL ≥ A on benign test periods of D1–D5 | ARL < A/2 on any dataset |
| H-V2 | Under benign drift (temporal gap), M2 keeps ARL ≥ A while training-calibrated thresholds do not | M2 also fails, or training calibration does not |
| H-V3 | Peer-only calibration violates ARL on real data (replicating §0) | it does not (then the simpler method suffices) |
| H-D1 | Per class, median delay ∈ [0.5, 2] × log A / Ψ̂ (Ψ̂ from training-period ROC) | outside on ≥ 1/3 of classes |
| H-D2 | Ψ̂ ranks classes by detectability (Spearman with 1/delay ≥ 0.6) | ρ < 0.6 |
| H-D3 | At equal false-alarm budget, entity e-CUSUM detects more natural-novelty attack episodes than per-window thresholds set on past traffic | no gain, or gain only on volumetric classes |
| H-N1 | Network-level detector catches population-wide attacks that entity level misses, with ARL ≥ A | either part fails |
| H-C1 | Circular-shift coordination test has size ≤ α on benign traffic | size > 2α |
| H-C2 | It detects coordinated groups (bot replays, distributed scans) missed by per-host tests | no added detections |
| H-I1 | Same-family signatures transfer across datasets (retrieval AUC > 0.7) | ≤ 0.7 |
| H-I2 | Γ covers the true known family with prob ≥ 1 − α | coverage < 1 − α − 0.05 |
| H-I3 | Held-out families yield Γ = ∅/ambiguous in ≥ 80% of episodes and trigger the family-novelty e-value | < 50% |
| H-L1 | M10 raises Ψ̂ on held-out natural-novelty families vs BCE | no rise or rise only on seen families |
| H-A1 | Delay-bounded updates resist boiling-frog poisoning; unrestricted updates do not | no difference |

## 7. Experimental validation plan

**Stage A — statistical validity before detection (no attack labels used).**
Benign-only periods of D1–D5 (HoNE temporal splits) and, if obtained, an enterprise trace. Measure empirical
ARL per entity and per role for grids of (g, k, L, A); estimate β-mixing scale from residual autocorrelation
on training data only. Pass criterion: H-V1–H-V3.

**Stage B — detectability theory.** Estimate per-class ROCs on training periods; compute Ψ̂; predict delays;
compare with observed delays on natural-novelty and rolling classes (HoNE's 45 class–cut units). H-D1–H-D3.
Report detection rate at fixed alarm budget, median and 90th-percentile delay, alarm-to-episode overlap.

**Stage C — network-wide and coordinated.** Population-wide classes (DDoS on many hosts) for H-N1;
coordination: bot classes (D1 Bot, D5 bot) and replayed beacon traces injected into benign periods with
controlled synchrony; graph neighbourhoods from the communication graph built on past traffic. H-C1, H-C2.

**Stage D — identification.** Leave-one-family-out over known families; cross-dataset signature retrieval
(D1 ↔ D5 share CIC-family names; D2/D3 for disjoint families). H-I1–H-I3.

**Stage E — learning and adaptation.** Base detectors: benign-only (AE/IF), BCE classifier, M10-trained.
Boiling-frog poisoning stream. H-L1, H-A1.

**Baselines.** HoNE's 12 detectors with per-window thresholds set on *past* traffic; SPOT/DSPOT thresholds;
FIRCE-style per-flow conformal rejection; TGN-SVDD; Kitsune; BotMiner-style correlation; e-d-BH without M2
(to isolate M2's contribution).

**Protocol rules (inherited from HoNE).** Temporal and natural-novelty splits only for zero-day claims;
thresholds and all hyperparameters (g, k, L, A, q, α) fixed on training periods; five seeds for learned
components; class–cut cluster bootstrap for uncertainty; pre-registration hashed before runs.

**Ablations.** Remove each of: common-mode removal, own-history reference, gap, stride, deflation,
ROC-slope vs mixture calibrator, entity windows vs HoNE's network-wide windows, circular vs naive null,
gated vs delayed updates.

**Data requirement.** Re-download raw NetFlow-v3 / Lycos (addresses present, used for grouping only);
entity windows require per-host flow ordering, which HoNE's pipeline does not yet build.

## 8. Strongest PhD and publication contributions

1. **"When are zero-day attacks detectable? A sequential detectability theory for learned NIDS"** — M4 + H-D
   hypotheses + HoNE re-analysis. Venue: IEEE S&P / USENIX Security (measurement + theory) or IEEE TIFS.
   Strongest single paper: it turns HoNE's negative operating-point result into a predictive theory.
2. **"Valid sequential intrusion alarms under drift and dependence"** — M2 + M3 + M5 + M6 with the peer-null
   negative result. Venue: NDSS / ACM CCS, or a statistics venue (Annals of Applied Statistics, JASA A&CS)
   for the methodological core.
3. **"Detecting coordinated compromise with dependence-preserving nulls"** — M7 + subset scan. Venue: NDSS
   / RAID.
4. **"Open-set naming of unknown attack families with guarantees"** — M8 + M9. Venue: ACSAC / IEEE TDSC.
5. **Thesis arc:** HoNE (evaluation validity) → detectability theory (what can be detected) → valid sequential
   detection (how to detect it with controlled alarms) → identification and adaptation (what it is, and how to
   learn safely). The coherent thread is *statistical validity of zero-day claims and alarms*, which is
   distinctive relative to the architecture-centred NIDS literature.

## 9. Before implementation (decisions requested)

1. Approve PeD as the research direction, or a subset (recommended core: M2–M4, then M7, then M8–M9).
2. Approve the pre-registration of §6 hypotheses H-V1–H-D3 as the first experimental stage (validity and
   detectability before any new model).
3. Remaining prior-art checks I will run before writing any paper claim: per-entity conformal e-detectors in
   operations/IT monitoring (AIOps); coordination tests with circular-shift nulls in fraud/bot detection;
   conformal open-set identification with novelty e-values.

## Sources added this phase
Lakhina, Crovella, Diot SIGCOMM 2004 · Xie & Siegmund Ann. Statist. 2013 · Chernozhukov, Wüthrich & Zhu 2018
· Oliveira et al. arXiv:2203.15885 · Barber, Candès, Ramdas & Tibshirani 2023 · Wang & Ramdas JRSS-B 2022 ·
Dandapanthula & Ramdas arXiv:2501.04130 · Gu et al. BotSniffer NDSS 2008, BotMiner USENIX Sec 2008 ·
Speakman, McFowland & Neill JCGS 2015 · ECAD arXiv:2105.11886 · Masud et al. (ECML PKDD 2009 / TKDE) ·
Li et al. COPOD ICDM 2020, ECOD 2022 · Podkopaev & Ramdas NeurIPS 2023 · Xie et al. CGTC arXiv:2510.13037 ·
Rubinstein et al. (boiling frog) 2009; Kravchik et al. arXiv:2002.02741 · Fedorova et al. ICML 2012;
Volkhonskiy et al. arXiv:1706.03415; Vovk et al. COPA 2021 · log-optimal calibration arXiv:2602.08927 ·
Vovk & Wang, "E-values: calibration, combination and applications" (Ann. Statist. 2021).
