# Phase 5 — Revised mathematical framework (critical review response)

**Status: awaiting explicit approval.** Nothing in this document has been implemented. No real data was used, and no HoNE code, data, configurations or results were touched. All numerical evidence comes from synthetic Monte Carlo runs:
- `sims/adversarial.py` → `sims/results_adversarial.json`
- `sims/online_conformal.py` → `sims/results_online.json`
- earlier runs: `sims/results.json`, `sims/results_roc.json`

**Labelling convention.** Every claim below carries one of these tags:

| Tag | Meaning |
|---|---|
| **[T]** | Theorem/lemma proved here under the stated assumptions (proof given). |
| **[K]** | Known result taken from the literature (cited; not ours). |
| **[P]** | Proposition: a short derivation, exact under the stated model. |
| **[S]** | Simulation evidence only. Not a guarantee. |
| **[O]** | Open problem or conjecture. |

Nothing tagged [S] should be read as a guarantee.

**Changes from phase 4 (`04_mathematical_framework.md`):**

1. The delayed-reference validity claim was heuristic. It is now split into:
   - Theorem 1a: exact, using online conformal p-values;
   - Theorem 1b: approximate, using the delayed reference. It is valid but conservative.
2. My phase-4 claim "duty-cycle evasion iff f < 1/2" is **wrong**. It is replaced by a Cramér-root analysis (§2.6).
3. The contamination bound needed an explicit **quarantine** condition (§6). Without quarantine, a counterexample shows it fails by about 55×.
4. Two coordination variants (lag-tolerant and binned) lost most of their power. They are reported, not tuned away.
5. The anchor reference is **invalid under benign idiosyncratic drift**. This is a counterexample to a guarantee I had implied.

---

## 0. Notation and standing model

**Panel.** Entities are indexed by v ∈ V and grouped into roles R(v) ⊆ V. Windows are indexed by t ∈ ℕ. The base detector emits a scalar score s_{v,t}; larger means more anomalous.

**Panel model under the null H0 (no attack).**

  s_{v,t} = μ_{R,t} + α_v + ε_{v,t}

- μ_{R,t}: common mode within the role. Arbitrary, possibly non-stationary: outages, network-wide changes.
- α_v: fixed entity offset.
- ε_{v,t}: idiosyncratic noise.

**Double-differenced residual.**

  r_{v,t} = s_{v,t} − med_{u∈R(v)} s_{u,t}

The role median removes μ_{R,t} exactly. Rank-based (conformal) comparison against v's own history removes α_v, because ranks are invariant to a constant shift.

**Stride.** Fix a stride k. The strided sequence for entity v is X_j = r_{v, t_0 + jk}. Write F for its marginal CDF under H0.

**β-mixing.** For a process (Z_t), β_Z(k) = sup_t β(σ(Z_s : s ≤ t), σ(Z_s : s ≥ t + k)), where β(𝒜, ℬ) = E sup_{B∈ℬ} |P(B | 𝒜) − P(B)|. For a Gaussian AR(1) with coefficient ρ, β(k) ≤ sqrt(−¼ log(1 − ρ^{2k})) ≈ |ρ|^k / 2 [K, via the Gaussian TV/Hellinger bound].

**e-values and e-CUSUM.**
- A *calibrator* is a non-increasing h : [0,1] → [0,∞] with ∫₀¹ h(u) du ≤ 1.
- Per-window e-values are e_j = h(p_j).
- The e-CUSUM statistic is W_j = max(W_{j−1}, 1) · e_j, so W_n = max_{i≤n} ∏_{l=i}^{n} e_l.
- It alarms when W_j ≥ A and restarts at 1 after an alarm.

**Calibrators used:**
- the capped power calibrator;
- the mixture calibrator h_mix(p) = (1 − p + p ln p) / (p ln² p) [K: Vovk & Wang, e-values literature]. Its limit as p → 1 is 1/2;
- the hedged calibrator h_λ = λ·g + (1 − λ), for any density g on [0,1].

---

## 1. Validity of M2–M3 under dependence (review item 1)

### 1.1 Assumptions

| Label | Assumption |
|---|---|
| **D1** (additive role common mode) | Under H0 the score obeys the panel model with μ_{R,t} identical across v ∈ R and additive. μ may be arbitrarily non-stationary and arbitrarily dependent. |
| **D2** (idiosyncratic stationarity) | For each v, (ε_{v,t})_t is strictly stationary and α_v is constant over the horizon considered. |
| **D3** (idiosyncratic mixing) | The joint process (ε_{u,t})_{u∈R(v)} is β-mixing with coefficient β_ε(k). If the hosts' ε processes are mutually independent, β_ε(k) ≤ Σ_{u∈R} β_{ε_u}(k) (β of independent products is subadditive [K]). |
| **D4** (continuity / tie-breaking) | Either F is continuous, or p-values are smoothed with independent U(0,1) tie-breakers. |
| **D5** (role majority, alternative only) | Fewer than half of R(v) are compromised at any t. The median's breakdown point is 1/2. |

**Lemma 1.0 [P] (what double-differencing does).**

Under D1:

  r_{v,t} = (α_v + ε_{v,t}) − med_{u∈R}(α_u + ε_{u,t}).

So r_{v,·} is a fixed measurable function of the idiosyncratic processes only. Consequently:
- μ drops out exactly, however non-stationary it is. This covers outages and network-wide operational changes.
- r_{v,·} inherits stationarity from D2.
- r_{v,·} inherits mixing from D3: β_r(k) ≤ β_ε(k), since a function of a process cannot increase β.

*This is the mechanism that controls false alarms under shared infrastructure events.* It needs additivity. A **multiplicative** common mode (a role-wide change of scale) is **not** removed. That remains a limitation (L1.4).

### 1.2 Theorem 1a [T] — exact validity with online conformal p-values

**Definition (smoothed online conformal p-value).** Let n_> be the number of earlier points strictly above X_j, and n_= the number of points up to and including X_j that tie with it:

  n_> = #{i < j : X_i > X_j},  n_= = #{i ≤ j : X_i = X_j},  U_j ~ U(0,1) i.i.d.

Then

  p_j = (n_> + U_j · n_=) / j.

Here p_j is computed against the **full past** of the strided series, not against a sliding window. A warm-up of j ≤ j_0 windows is computed but not evaluated.

**Theorem 1a.** Assume D1–D4. Let N_n be the number of alarms raised in the first n evaluated windows by the restarted e-CUSUM with any calibrator h and threshold A. Then:

- **(i) Exchangeable case.** If (X_j) is exchangeable (for example i.i.d.), then (p_j) are i.i.d. U(0,1) [K: Vovk, Gammerman & Shafer 2005]. Consequently
  - E N_n ≤ n / A, and
  - P(N_n ≥ 1) ≤ n / A.
- **(ii) β-mixing case.**
  P(N_n ≥ 1) ≤ n / A + (n − 1) · β_r(k).

**Proof.**

*(i)* The i.i.d. uniformity of smoothed online conformal p-values under exchangeability is standard [K]. For each start index i, the product M^{(i)}_m = ∏_{l=i}^{m} h(p_l) is a non-negative supermartingale with M^{(i)}_{i−1} = 1, because E h(p_l) ≤ 1 by independence.

By Ville's inequality, P(sup_m M^{(i)}_m ≥ A) ≤ 1/A.

Each alarm is a crossing of some product that started at a distinct index, because the statistic restarts after an alarm. Hence E N_n ≤ Σ_{i≤n} 1/A = n/A. The probability bound follows because P(N_n ≥ 1) ≤ E N_n.

*(ii)* Use the coupling lemma for absolutely regular sequences [K: Berbee 1979; in the sequential form of Doukhan, Massart & Rio 1995 and Viennet 1997]. It states: there exists an **independent** sequence (Y_j) such that
- Y_j has the same law as X_j, and
- P(Y_j ≠ X_j) ≤ β(σ(X_1, …, X_{j−1}), σ(X_j)) ≤ β_r(k).

The induction step works as follows. Apply Berbee to the σ-field of X_{<j} enlarged by auxiliary independent uniforms. This does not change the β coefficient.

Next, compute p-values on Y with the same tie-breakers U_j. By (i), the alarm count on Y satisfies P(N_n^Y ≥ 1) ≤ n/A.

The alarm paths on X and Y agree on the event {X_j = Y_j for all j ≤ n}. Its complement has probability at most Σ_{j=2}^{n} β_r(k). ∎

**Design rule [P].** To keep the mixing remainder below η over a horizon of n evaluations, choose the stride so that (n − 1) β_r(k) ≤ η. For AR(1)-like residuals this is

  k ≥ log(2(n − 1)/η) / log(1/|ρ|).

Example: ρ = 0.6, n = 1000, η = 0.01 gives k ≥ 25.

The simulations used k = 10 (β_r ≈ 0.003). At that stride the remainder term is loose; the observed rates were far below it:

| ρ | k | Observed false alarms per 1000 evaluations [S] | n/A at A = 200 |
|---|---|---|---|
| 0.6 | 10 | 0.19 | 5 |
| 0.9 | 44 | 0.27 | 5 |

**What 1a does *not* give.**

1. **Slow self-masking.** Suppose J_0 null points have been seen and an attack has contributed k − 1 earlier points. If the attacked points are exchangeable among themselves and all exceed the null points, then E p_k ≈ (k/2)/(J_0 + k). The per-window evidence therefore decays as the attack accumulates. This is slower than masking by a sliding reference, but it is not zero.
2. **No benign-drift adaptation.** Under any benign change in F, exchangeability fails. In fact the online conformal martingale is literally a test of exchangeability [K: Vovk et al. 2003; Fedorova et al. 2012; Volkhonskiy et al. 2017], so a benign drift *will* eventually alarm.

Power evidence for the online scheme [S]:

| Step attack (5 hosts, ρ = 0.6, k = 10) | Detected | Median delay |
|---|---|---|
| 0.8 sd | 0.8 | 37.5 windows |
| 0.5 sd | 0.6 | 48 windows |

### 1.3 Theorem 1b [T] — approximate validity with a delayed sliding reference

**Reference construction.** The reference for evaluation j consists of m strided residuals of v, taken from the interval [t_j − g − L, t_j − g]. Requirements:
- gap g ≥ k;
- consecutive evaluations at least k apart (t_j − t_{j−1} ≥ k);
- F̂_j is the empirical CDF of the reference;
- p_j = (1 + #{ref ≥ X_j}) / (m + 1), smoothed.

Let G_j = F_{t_j − k}, the σ-field of everything up to t_j − k. Because the evaluation stride is at least k, F_{t_{j−1}} ⊆ G_j. Because the gap is at least k, the reference is G_j-measurable.

**Local stationarity condition D2′.** The marginal law of r_{v,t} is the same F on [t_j − g − L, t_j]. This is weaker than D2. A drift term relaxes it (see remark (b) below).

**Lemma 1 [T] (conditional sub-uniformity).** Under D2′, D3 and D4:

  P(p_j ≤ u | G_j) ≤ u + 1/m + D_j + Δ_j  for all u,

where:
- D_j = sup_x |F̂_j(x) − F(x)|;
- Δ_j = d_TV(L(X_j | G_j), F).

Both are G_j-measurable, and E Δ_j ≤ β_r(k).

*Proof.* The event {p_j ≤ u} equals {X_j > q̂} for a threshold q̂ that is G_j-measurable and satisfies 1 − F̂_j(q̂) ≤ u + 1/m. Then

  P(X_j > q̂ | G_j) ≤ 1 − F(q̂) + Δ_j ≤ 1 − F̂_j(q̂) + D_j + Δ_j.

Finally, E d_TV(L(X | 𝒢), L(X)) = β(𝒢, σ(X)) [K]. ∎

**Lemma 2 [T] (DKW under blocking).**

  P(D_j > x) ≤ 2 exp(−2 m x²) + (m − 1) β_r(k).

*Proof.* Couple the m strided reference points with an independent copy, as in Theorem 1a(ii). Then apply the Dvoretzky–Kiefer–Wolfowitz inequality with Massart's constant [K: Massart 1990]. The blocking idea goes back to Yu 1994 [K]. ∎

**Lemma 3 [T] (deflation).** Let h be non-increasing on [0,1], with h ≤ c and ∫ h ≤ 1. If P(p ≤ u | 𝒢) ≤ u + ε for all u, then

  E[h(p) | 𝒢] ≤ 1 + ε (c − h(1)).

*Proof.* Write μ = −dh ≥ 0. Its total mass is h(0⁺) − h(1) ≤ c − h(1). Since h(p) = h(1) + ∫ 1{p < u} μ(du),

  E[h(p) | 𝒢] = h(1) + ∫ P(p < u | 𝒢) μ(du)
       ≤ h(1) + ∫ u μ(du) + ε μ([0,1])
       = E_U h + ε μ([0,1])
       ≤ 1 + ε (c − h(1)). ∎

**Theorem 1b.** Fix ε_1, ε_2 > 0 and set ε̄ = ε_1 + ε_2 + 1/m. Use a capped calibrator h ≤ c and the deflated e-value

  e′_j = h(p_j) / (1 + ε̄ (c − h(1))).

Then for the restarted e-CUSUM on (e′_j):

  P(N_n ≥ 1) ≤ n/A + Σ_{j≤n} [P(D_j > ε_1) + P(Δ_j > ε_2)]
       ≤ n/A + n [2 e^{−2 m ε_1²} + (m − 1) β_r(k) + β_r(k)/ε_2].

*Proof.* The good event E_j = {D_j ≤ ε_1, Δ_j ≤ ε_2} is G_j-measurable, so it is predictable. Define ẽ_j = e′_j on E_j and ẽ_j = 1 off E_j.

On E_j, Lemmas 1 and 3 give E[ẽ_j | G_j] ≤ 1. Hence E[ẽ_j | F_{t_{j−1}}] ≤ 1 by the tower property, and every product of ẽ is a supermartingale. Ville's inequality plus the union over start indices, as in 1a, gives n/A for the ẽ-detector.

The two detectors coincide on ∩_j E_j. Bound P(E_j^c) by Lemma 2 for D_j and by Markov's inequality for Δ_j, using E Δ_j ≤ β_r(k). ∎

**Honest assessment.** Theorem 1b is valid but **very conservative at realistic sizes**:
- Making the DKW term small at ε_1 = 0.05 needs m ≈ 725 reference points per entity.
- With the capped mixture calibrator the deflation factor is about 2.95, which removes most of the power.

The simulated false-alarm rates of the *undeflated* delayed scheme are well inside n/A when the theorem's design is respected, and outside it when it is not [S]. A = 200, so the bound is 5 per 1000 evaluations.

| Design | False alarms per 1000 evaluations |
|---|---|
| theorem design: strided reference and evaluation, ρ = 0.6, k = 10, m = 60 | 0.48 |
| theorem design, ρ = 0.9, k = 44 | 0.35 |
| dense (unstrided) reference, ρ = 0.6 | 3.05 |
| dense (unstrided) reference, ρ = 0.9 | **9.05** (violates) |

So the undeflated delayed reference is supported only by simulation [S]. The deflated version is guaranteed [T] but nearly powerless.

**Remarks.**

- **(a) Why striding is needed.** With a dense reference, the m correlated points carry roughly m(1 − ρ)/(1 + ρ) effective samples. D_j is then much larger than the DKW rate. Without evaluation striding, F_{t_{j−1}} ⊄ G_j, so the predictable good-event argument fails. The ρ = 0.9 dense counterexample (9.05 > 5) shows this matters in practice.
- **(b) Drift budget.** If D2′ is replaced by "the TV distance between the reference marginal and the test marginal is at most ω_j", Lemma 1 gains a term +ω_j. False-alarm control then requires a known drift budget. A random walk has no such budget (see the counterexample below).

### 1.4 Counterexamples to M2–M3 [S]

All at A = 200; the bound is 5 false alarms per 1000 evaluations.

| Violated assumption | Construction | Result |
|---|---|---|
| D2 (stationarity) | ε is a random walk, step sd 0.03 | gap 6: 1.30 (inside); **gap 600: 10.29** (violates). The longer the gap, the more drift accumulates between reference and test. |
| D2′ at the reference scale | regime switching, amplitude 3 sd | **dwell 500: 6.75** (violates); dwell 50: 2.12; dwell 3000: 1.99. It fails exactly when the dwell time is comparable to g + L, so reference and test sit in different regimes. |
| D5 (role majority) | fraction of the role compromised | 0.6 compromised: attacked hosts detected 0.83, benign hosts alarming 0.0. **0.8 compromised: attacked detected 0.22, benign alarming 0.125.** The median is captured, so attacked hosts look normal and benign hosts look anomalous. |
| Striding | dense reference, ρ = 0.9 | 9.05 (violates) |

**Limitations of §1.**
- **L1.1** Validity needs stationarity of ε on the reference scale (1b) or over the whole past (1a).
- **L1.2** The mixing coefficient must be known or bounded to set k. Estimating β from data is itself hard; the remainder bounds are only as good as that estimate.
- **L1.3** Above 50% compromise of a role, the panel is captured. No role-median design can avoid this.
- **L1.4** A multiplicative common mode is not removed.
- **L1.5** Cross-sectional dependence in ε does not affect per-entity validity. It does affect multiplicity control across entities, which is why e-BH is used: it is valid under arbitrary dependence [K: Wang & Ramdas 2022].

**[O] Central open tension (a candidate PhD question).** Two properties pull against each other:
- exact anytime validity, which needs a full-past exchangeable reference (Theorem 1a);
- resistance to masking and adaptation to benign drift, which need a finite, delayed, moving reference.

Theorem 1b reconciles them only at a large power cost. I do not know of a construction that achieves both without a drift budget.

---

## 2. M4 — detectability from the ROC curve (review item 2)

### 2.1 Setting

Fix one entity and one attack regime. The base score S has CDF F_0 under H0 and F_1 under the attack, both continuous. The ideal one-sided p-value is p = 1 − F_0(S).

Assume windows are i.i.d. within each regime. Section 2.5 relaxes this.

### 2.2 ROC representation [P]

- Under H1, P(p ≤ u) = ROC(u), where ROC(u) is the TPR at FPR u.
- If ROC is absolutely continuous, the H1 density of p is ROC′(u).
- ROC′(u) equals the likelihood ratio f_1/f_0 evaluated at the score threshold whose FPR is u.
- ROC is concave if and only if this ratio is monotone in the score.

### 2.3 Deriving Ψ [P]

**Proposition 4.1.** Among all calibrators h (densities on [0,1]), the expected log-growth under H1 is maximised by h* = ROC′. Its value is

  Ψ := E_1 log ROC′(p) = ∫₀¹ ROC′(u) log ROC′(u) du = KL(P^1_p ‖ U[0,1]) = KL(F_1 ‖ F_0).

The final equality holds because p is a bijective transform of S.

By data processing, Ψ ≤ KL(P_1 ‖ P_0) on the raw window. Equality holds if and only if S is a monotone transform of the likelihood ratio.

**Proposition 4.2 (mismatch identity).** For any calibrator density q on [0,1],

  E_1 log q(p) = Ψ − KL(P^1_p ‖ Q).

*Proof.* Expand E_1 log q = E_1 log ROC′ − E_1 log(ROC′/q). ∎

The identity is verified in simulation. The values below are differences between Ψ and the mismatch KL for capped calibrators tuned to δ, and they match to Monte Carlo error [S]:

| δ | Ψ − KL |
|---|---|
| 0.25 | −0.191 |
| 0.5 | −0.060 |
| 1 | 0.283 |

A negative value means **negative drift**: a mis-tuned calibrator can have negative growth even when the attack is detectable.

### 2.4 Is AUC sufficient? No.

**Proposition 4.3 [P].**

(a) AUC = P(S_1 > S_0) = 1 − E_1[p].

(b) **Lower bound.** Minimise KL(P ‖ U) over laws P on [0,1] with E_P[p] = 1 − AUC. The minimiser is the I-projection [K: Csiszár 1975]: a truncated exponential density ∝ e^{−λu}, with λ chosen to match the mean. Hence

  Ψ ≥ Ψ_min(AUC).

For AUC = 0.75, Ψ_min = 0.409.

(c) **No upper bound.** Ψ is not bounded above by any function of AUC. If F_1 puts mass where f_0 = 0 (a partially separable attack), ROC jumps at 0 and Ψ = ∞. With a small amount of overlap, Ψ is finite but arbitrarily large.

**Proposition 4.4 [P] (operating-point bound).** Data processing applied to the binary event {p ≤ u_0} gives

  Ψ ≥ kl(ROC(u_0) ‖ u_0),

where kl is the binary KL divergence. Equivalently, TPR at FPR u_0 is at most the largest x with kl(x ‖ u_0) ≤ Ψ. Low-FPR performance therefore **bounds Ψ from below**; AUC does not.

**Equal-AUC counterexample [S] / [P].**

| Shape | AUC | Ψ | TPR@1% | kl(TPR@1% ‖ 0.01) | Predicted delay, log A/Ψ at A = 1000 |
|---|---|---|---|---|---|
| binormal | 0.75 | 0.453 | 0.087 | 0.114 | 15.2 |
| partially separable (π ≈ 0.502, Δ = 4) | 0.75 | 3.382 | 0.484 | 1.542 | 2.0 |
| variance inflation (σ = 2.6) | 0.77 | 1.934 | 0.323 | 0.866 | 3.6 |

- Every Ψ ≥ Ψ_min(0.75) = 0.409 and every Ψ ≥ kl(TPR@1% ‖ 1%), as Propositions 4.3 and 4.4 require.
- Two detectors with the same AUC differ by **7.5×** in Ψ, and so by 7.5× in predicted delay. The difference lives entirely in the low-FPR part of the ROC.

### 2.5 When is delay ≈ log A / Ψ valid?

**Proposition 4.5 [K, with conditions stated].** Let D = E_1 log h(p) be the drift of the calibrator in use (D = Ψ for the oracle).

Then:

  E[delay] = (log A)/D · (1 + o(1)) as A → ∞,

and no procedure with ARL ≥ A has worst-case delay below (log A)/KL_window · (1 + o(1)) [K: Lorden 1971; Lai 1998].

The approximation is valid under these conditions:

- **(a)** Within-attack windows are i.i.d., D > 0, and log h(p) has finite variance. This gives Wald's identity plus a bounded overshoot. The relative error is O(overshoot/log A).
- **(b)** For stationary ergodic attack windows, Lai's (1998) r-quick condition on the normalised log-likelihood sums replaces (a), with D the almost-sure limit.
- **(c)** If D < 0, the approximation is **invalid**; see §2.6.
- **(d)** The attack must last much longer than log A / D. Otherwise the probability of detection before the attack ends is the relevant quantity, not the delay.
- **(e)** p-values must be exactly uniform under H0. With the deflated e-values of Theorem 1b, replace D by D − log(1 + ε̄(c − h(1))).
- **(f)** The approximation describes how delay *scales* with A. At small A (≤ 100), overshoot dominates.

### 2.6 Intermittent (duty-cycled) attacks — corrected analysis

Let the attack be active in a fraction f of windows, and write g for the calibrator. The drift per window is

  D_f = f · E_1 log g + (1 − f) · E_0 log g,

where E_0 log g < 0 by Jensen's inequality unless g ≡ 1.

**My phase-4 claim was wrong.** I said that the e-CUSUM is evaded if and only if f < 1/2, or more generally if and only if D_f < 0. Negative drift does not imply evasion: the CUSUM's floor at 1 lets it catch excursions.

**Proposition 4.6 [P / K: Siegmund 1985].** Write X = log g(p) under the mixed attack law, and φ(θ) = log E e^{θX}. φ is convex with φ(0) = 0.

1. If φ′(0) = D_f > 0: delay ≈ log A / D_f.
2. If D_f < 0 and a positive root φ(θ*) = 0 exists: the expected detection delay grows like A^{θ*} (up to constants). It is polynomial, not infinite.
3. **True evasion.** Detection is no faster than a false alarm (delay ≳ A) if and only if θ* ≥ 1. Since φ(1) = log E g(p), this is equivalent to **E_attack[g(p)] ≤ 1**. In that case the product is a supermartingale under the attack as well.

*Example.* With the oracle likelihood-ratio calibrator for δ = 1 and Gaussian windows, φ(θ) = 0 becomes

  f · e^{θ(θ+1)/2} + (1 − f) · e^{θ(θ−1)/2} = 1.

For a periodic schedule with f = 0.3, θ* = 0.328.

Simulation results [S]:

| Schedule | Oracle calibrator | Hedged calibrator, λ = f |
|---|---|---|
| random, f = 0.3 (oracle drift −0.202) | detected 100%, median 118 windows | median 100 |
| periodic, 3 on / 7 off | median 152 | median 92 |

**Hedging [P].** The hedged calibrator h_λ = λ g + (1 − λ) satisfies log h_λ ≥ log(1 − λ). Each off-window therefore costs at most log 1/(1 − λ). The Kelly-optimal λ for a mixed law maximises f · E_1 log h_λ + (1 − f) · E_0 log h_λ.

Cost of hedging when the attack is continuous [S]: at f = 1, λ = 0.3 needs 23 windows against 12 for the oracle.

**Limitations of §2.**
- **L2.1** Ψ requires F_1, which is unknown for zero-day attacks. Ψ is a **diagnostic** for evaluating detectors against stated attack models, not something that can be computed online.
- **L2.2** The identity Ψ = KL(F_1 ‖ F_0) is per window and assumes i.i.d. windows within each regime.
- **L2.3** An adversary who knows g can choose an attack with E[g(p)] ≤ 1 by shaping its score distribution: Proposition 4.6.3 gives the exact evasion condition. Hedging only lowers the attacker's cost of staying under it.

---

## 3. Slow and persistent attacks (review item 3)

**Setting.** A delayed reference with gap g and length L. An attack starts at time τ with mean shift δ_t, measured in residual-sd units.

**Proposition 3.1 [P] (absorption).** For a step δ_t = δ for t ≥ τ:
- the reference contains no attacked points until τ + g;
- it is entirely attacked from τ + g + L onwards.

After τ + g + L, the test p-values are uniform again under D2′ applied to the shifted law. All evidence must therefore be collected within [τ, τ + g + L]. Its expected total is approximately (g + L/2)/k · E_1 log h(p) for the effective shift. Detection within that window requires

  (g + L/2)/k · D(δ) ≳ log A.

**So yes: a finite delayed reference absorbs persistent attacks.** It does so after g + L.

**Proposition 3.2 [P] (ramp ≡ step).** For a linear ramp δ_t = β(t − τ), once t > τ + g + L the gap between the test mean and the reference mean is constant:

  β · (g + L/2).

A ramp is therefore seen as a step of size β(g + L/2). The slowest ramp detectable within the absorption window scales like 1/(g + L).

**Proposition 3.3 [P, via Le Cam] (impossibility).** Let 𝒟 be the class of benign mean trajectories the system must tolerate without alarms, for example slow idiosyncratic drifts. If an attack's mean trajectory lies in 𝒟, then for any detector

  power − size ≤ d_TV(P_attack, P_benign) = 0.

More generally, for Gaussian windows with mean path difference m_t,

  d_TV ≤ sqrt(½ Σ_t m_t² / (2σ²))  (Pinsker).

**There is no mathematical solution for attacks that mimic tolerated drift.** Only the definition of 𝒟 can be changed, which is a policy decision rather than a theorem.

**Remedies and their costs:**

| Remedy | Guarantee | Cost / failure |
|---|---|---|
| Long gap g ([S] gap 1500) | Theorem 1b still applies if D2′ holds over g + L | Long D2′ windows are less plausible: random-walk counterexample (§1.4) |
| Frozen anchor reference | Detects any persistent shift relative to the anchor | **Invalid under benign idiosyncratic drift**: 5.93 false alarms per 1000 against a bound of 5 [S] |
| Multiscale, Bonferroni (J detectors at gaps g_1 < … < g_J, each with threshold J·A) | Union bound: P(FA) ≤ n/A, if each detector is valid | Threshold rises to J·A, so delay rises by log J / D once. **Not simulated.** |
| Multiscale, average of e-values per window | (1/J) Σ e_j is a valid e-value if each e_j is | Up to log J of drift lost *per window*. It is only as valid as its weakest component, and the sim included the anchor. |
| Online full-past reference (Theorem 1a) | Exact under exchangeability | Benign drift eventually alarms; masking slows as k/(J_0 + k) |

**Bonferroni versus averaging [P].** Bonferroni pays log J once, in the threshold. Averaging pays up to log J per window, in the drift. For long-delay (slow) attacks Bonferroni is asymptotically better. Averaging can be better when several scales carry evidence at the same time.

Simulation results [S] (A = 200):

| Scenario | gap 20 | gap 1500 | anchor | multiscale average |
|---|---|---|---|---|
| persistent step 0.8 sd: detected | **0.2** (masked) | 1.0 (median 52) | 1.0 (median 85) | 1.0 (median 76) |
| slow ramp 1.5 sd over 6000 steps: detected | 0.2 | 0.8 | 1.0 | 1.0 |
| benign idiosyncratic drift, no attack: FA per 1000 | 0.38 | 2.06 | **5.93** | 2.20 |

- The anchor kept raising alarms after the step: 6.6 alarms in the 4000 steps that followed. It does not absorb, which is the point, but it also never adapts.
- The multiscale average's 2.20 is inside the bound only empirically. Because it includes an invalid component, it has no guarantee.

**Conclusion for item 3.** Within the stated model, every reference design trades one of three things against the others: validity under benign drift, masking resistance, and adaptation. Proposition 3.3 shows this trade-off cannot be removed in general.

---

## 4. M7 — coordination tests under legitimate synchronization (review item 4)

### 4.1 Statistic and nulls

**Input.** B is the |V| × T binary exceedance matrix (per-entity alarms or low p-values). The statistic T(B) counts synchronous co-exceedances across entity pairs, or any similar measure of synchrony.

**Assumptions for exact validity of the circular-shift null:**
- **C1 (independence given the common mode)** Under H0, after common-mode removal, the rows of B are mutually independent.
- **C2 (circular stationarity)** Each row's law is invariant under cyclic shifts.
- **C3** T is much larger than the correlation length of a row. This makes C2 approximately true for stationary, non-circular data, with an edge error of order (correlation length)/T.

**Proposition 7.1 [K: randomization tests].** Under C1–C2, independent uniform cyclic shifts of each row leave the law of B unchanged. The Monte Carlo p-value with the +1 correction is therefore exactly valid.

**Legitimate synchronization breaks C1 or C2.** Three nulls are used:

| Null | What it preserves | Valid under |
|---|---|---|
| circular shift | each row's marginal process | C1, C2 |
| phase-preserving (shifts by multiples of the period P) | each row's phase within the period | C1 with periodic, phase-locked benign schedules |
| historical (today's T compared with the previous d days) | recurring daily structure | day-level exchangeability of T. The smallest attainable p is 1/(d + 1). |

### 4.2 Results [S]

Rejection rates. Circular and phase nulls at level 0.05; historical at level 1/7, since d = 6.

| Scenario | Circular | Phase | Historical | Interpretation |
|---|---|---|---|---|
| independent | 0.058 | 0.058 | 0.10 | close to nominal |
| outage, all hosts | 0.042 | 0.05 | 0.108 | absorbed by common-mode removal |
| operational change, all hosts | 0.042 | 0.05 | 0.108 | absorbed |
| scheduled backup (subset, same phase daily) | **0.967** | 0.125 | 0.10 | circular null is fooled; phase and historical are not |
| software update (subset, one-off) | **0.633** | **0.625** | **0.733** | **irreducible** without side information |
| new bot beacon | 0.992 | 0.90 | 0.967 | detected |
| jittered bot beacon | 0.175 | 0.183 | 0.333 | weak |
| *lag-tolerant (lag 3) variant*: bots / jittered | 0.008 / 0.008 | — | — | **power lost** |
| *binned (bin 4) variant*: bots / jittered | 0.217 / 0.192 | — | — | **power lost** |

### 4.3 Intersection–union combination [K: Berger 1982]

Declare coordination only if **all** nulls reject, using

  p_IUT = max(p_circ, p_phase, p_hist).

This is a level-α test of the union null H0 = H0^circ ∪ H0^phase ∪ H0^hist: if any one of them is true, the max is at least that test's p-value. So p_IUT is valid as long as at least one null correctly describes the benign mechanism.

**Implication [P].** The IUT rejection rate is at most the minimum of the marginal rejection rates:
- scheduled backup: at most 0.10;
- new bot beacon: at most 0.90;
- one-off software update: at most 0.625.

**The joint IUT was not simulated.** These are marginal upper bounds only.

### 4.4 Limitations of §4

- **L4.1** One-off benign synchronization on a subset of hosts, such as a staged software rollout, is indistinguishable from coordinated attack activity by timing statistics alone. Change-management data would be needed. This is an identifiability limit, not a tuning problem.
- **L4.2** The historical null needs at least 1/α − 1 days of history to reach level α: 19 days for α = 0.05. It also absorbs any botnet that predates the history window.
- **L4.3** The jittered-beacon case and the lag/bin variants are open [O]. Making the test tolerant of lags or jitter inflates the null variance more than it gains signal. This is consistent with the BotSniffer/BotMiner [K] reliance on content and clustering features rather than timing alone.
- **L4.4** C1 fails under undeclared shared dependencies, such as hosts behind a common proxy. These must be modelled as part of the role structure.

---

## 5. M9 — identification and abstention (review item 5)

### 5.1 Definitions

**Libraries.**
- 𝒦 = {1, …, K}: known attack families.
- ℬ = {b_1, …, b_M}: a library of known benign changes, such as patches or migrations.

Each class c ∈ 𝒦 ∪ ℬ has a calibration set C_c, of size n_c, of profile vectors, plus a non-conformity score. For an alerted profile x, π_c(x) is the class-conditional (Mondrian) conformal p-value against C_c.

**Two levels α_v < α define two prediction sets:**

  Γ_α(x) = {c : π_c(x) > α},  Γ_{α_v}(x) = {c : π_c(x) > α_v}.

**Decision rule.**

| Condition | Report |
|---|---|
| Γ_α = {k}, k ∈ 𝒦 | **known family k** |
| Γ_α ⊆ ℬ, non-empty | **known benign change** (no attack claim) |
| \|Γ_α\| ≥ 2 | **ambiguous** (report the set) |
| Γ_α = ∅, Γ_{α_v} ≠ ∅ | **variant of** Γ_{α_v} |
| Γ_{α_v} = ∅ | **unfamiliar** — abstain from naming any family |

"Unseen variant" is defined operationally: outside the tight (α) envelope but inside the loose (α_v) one. "Genuinely unfamiliar" means outside every class's loose envelope.

### 5.2 Guarantees

**Theorem 9.1 [K: Mondrian conformal, Vovk et al. 2005].** If x comes from class c and is exchangeable with C_c, then

  P(c ∉ Γ_α(x)) ≤ α.

Consequently:
- the probability that a known-family alert is reported as "unfamiliar" is at most α_v;
- the probability that it is not named correctly at all is at most α.

**Proposition 9.2 [P] (family-novelty e-value).** Consider the null "x comes from *some* known class", without saying which.

Since max_c π_c(x) ≥ π_{c*}(x) for the true class c*, and π_{c*} is super-uniform, we have P(max_c π_c ≤ u) ≤ u **whatever the true class is**.

Hence for any calibrator g, e(x) = g(max_c π_c(x)) satisfies E e ≤ 1 under the composite null. For an incident with alerts x_1, …, x_n:
- the **average** of the e_i is always a valid e-value;
- the **product** is valid only if the π's are independent. With a shared calibration set they are dependent: positively dependent [K: Bates et al. 2023], and only approximately conditionally uniform. Use the average unless calibration sets are split.

**Results [S].**

| Case | Outcome |
|---|---|
| known families | correct name 0.904 (target ≥ 1 − α = 0.9) |
| variant of atk0 | reported as variant 0.91 |
| disjoint new family | unfamiliar 1.0 |
| **new family overlapping atk1** | **misreported as variant 0.885** (counterexample) |
| **unseen benign change** | **unfamiliar 1.0** (identifiability limit) |

### 5.3 Statistical limits of abstention

- **L5.1 Benign versus novel attack.** "Unfamiliar" means "not in any library". An unseen benign change and an unseen attack family both land there. By Le Cam's argument, no score-only rule separates them when their profile laws coincide.
- **L5.2 Overlap.** A novel family whose profile support overlaps a known family's is reported as that family or a variant of it. Conformal coverage controls errors on *known* classes only. Novelty power is not controlled.
- **L5.3 Exchangeability.** Coverage requires exchangeability within each class. Family drift breaks it. Libraries need refresh rules, and those rules are subject to §6.
- **L5.4 Resolution.** π_c ≥ 1/(n_c + 1), so classes with n_c < 1/α − 1 can never be excluded.
- **L5.5 Selection.** Only alerted windows are identified. Calibration sets must be built from windows passed by the same detector, otherwise the test and calibration laws differ.
- **L5.6 No control of the abstention rate under novelty.** The rate at which unfamiliar inputs are abstained on depends on the score's power and has no distribution-free guarantee.

---

## 6. M11 — contamination of the normality reference (review item 6)

### 6.1 Policy model

- Windows become eligible for the normality pool g_u steps after they occur (**admission lag**).
- An alarm at time τ removes the windows in [τ − B, τ] (**purge depth** B).
- It also blocks admission of the windows in (τ, τ + Q] (**quarantine** Q).
- An attack episode occupies the windows j = 0, …, H − 1 relative to its onset.
- τ is the first post-onset alarm; τ = ∞ if there is none. Assume there are no other alarms; additional alarms can only reduce the counts.

### 6.2 Theorem 6.1 [P] (contamination accounting, exact for this policy)

Define:
- **exposure** N_exp: attacked windows that ever served as reference;
- **residual** N_res: attacked windows still in the pool after the alarm is handled.

Then:

  N_exp = #{0 ≤ j < H : j + g_u < τ} + #{j < H : j > τ + Q}
  N_res = #{0 ≤ j < min(H, τ − B)} + #{j < H : j > τ + Q}

**Corollaries.**
- **(i)** If Q ≥ H − τ, which in practice means Q ≥ the maximum episode length, then E N_exp = Σ_{j<H} P(τ > g_u + j).
- **(ii)** If in addition B ≥ τ, then N_res = 0. **The purge depth must exceed the actual detection delay.** B ≥ g_u alone is not enough.
- **(iii)** If τ = ∞, then N_exp = N_res = H. **Undetected attacks are always adopted** after g_u. No admission policy prevents this except never admitting new data.

*Proof.* Count directly.
- Before τ, window j is admitted once t ≥ j + g_u.
- At τ, windows j ≥ τ − B are removed. Windows j < τ − B remain; they were already admitted, since j + g_u < τ when B ≥ g_u.
- After τ, windows j > τ + Q become eligible again. ∎

**Simulation check [S]** (H = 1500, δ = 0.7, g_u = 600, B = 900):

| Policy | Mean exposed attack windows | Σ_j P(τ > g_u + j) |
|---|---|---|
| purge only (Q = 0) | **342** | 6.3 |
| purge + quarantine | 0.35 | 12.85 |

- **Purge only.** The second term readmits post-alarm attack windows. This is a **counterexample** to the bound without quarantine.
- **Purge + quarantine.** Consistent with Theorem 6.1, but it is an **inequality, not the identity**. The simulated quarantine was Q = 10⁶, effectively permanent. A pre-onset false alarm therefore freezes the pool for good, and N_exp drops below the identity. A finite Q at the maximum episode length was **not simulated**.

### 6.3 Preventing gradual corruption (boiling frog)

Ramp to 3 sd over 4000 steps; attack fraction of the final pool [S]:

| Policy | Ramp detected | Final pool attack fraction |
|---|---|---|
| naive (most recent 600 windows) | 0.95 | **1.0** (alarms do not stop adoption) |
| lagged + purge | 1.0 | 0.0 |
| lagged + purge + quarantine | 1.0 | 0.0 |
| lagged + purge + anchor | 1.0 | 0.0 |

**Prevention rules [P]:**

1. **Admission lag** g_u ≥ the target detection delay for the slowest attack the system must not absorb (Proposition 3.2 converts this into a ramp slope).
2. **Purge depth** B ≥ that delay (Corollary ii).
3. **Quarantine** Q ≥ the maximum episode length the operator wants to exclude.
4. **Capped replacement.** At most a fraction ρ of the pool is replaced per period. An attacker then needs at least 1/ρ periods to dominate the pool. This bounds the speed of corruption, not its occurrence.
5. **Anchor drift check.** This catches gradual corruption, but it inherits the anchor's invalidity under benign drift (§3).

**Hard limit.** By Corollary (iii) and Proposition 3.3, any policy that eventually admits non-alarmed data eventually adopts an attack that stays below the detection threshold.

---

## 7. Adversarial falsification summary (review item 7)

Each row is an attempt to break a stated guarantee or claim.

| # | Target | Attack on it | Outcome | Status |
|---|---|---|---|---|
| 1 | Theorem 1b, needs striding | dense reference, ρ = 0.9 | FA 9.05 > 5 | assumption necessary |
| 2 | D2 / D2′ | random walk, gap 600 | FA 10.29 > 5 | assumption necessary |
| 3 | D2′ at the reference scale | regime switching, dwell 500 | FA 6.75 > 5 | assumption necessary |
| 4 | D5 | 80% of the role compromised | detected 0.22; benign alarms 0.125 | assumption necessary |
| 5 | "AUC predicts detectability" | equal-AUC shapes | Ψ differs 7.5× | **claim refuted** (Proposition 4.3) |
| 6 | "evasion iff f < 1/2" (my phase 4) | f = 0.3 random schedule | detected 100% | **my claim refuted**; replaced by Proposition 4.6 |
| 7 | short-gap reference against a persistent attack | step 0.8 sd | detected 0.2 | masking confirmed (Proposition 3.1) |
| 8 | anchor validity | benign idiosyncratic drift | FA 5.93 > 5 | **anchor invalid** |
| 9 | circular null under benign periodicity | scheduled backup | rejection 0.967 | circular null alone invalid; IUT needed |
| 10 | any timing null under a one-off benign subset event | software update | rejection 0.63–0.73 | irreducible (L4.1) |
| 11 | lag- and jitter-robust coordination | lag 3 / bin 4 variants | power ≈ 0.01–0.22 | open |
| 12 | M9 novelty detection | overlapping new family | 0.885 reported as variant | limit L5.2 |
| 13 | M9 benign versus attack | unseen benign change | 1.0 unfamiliar | limit L5.1 |
| 14 | M11 bound without quarantine | step attack | 342 ≫ 6.3 | quarantine necessary |
| 15 | evidence-gated calibration (phase 4) | null run | 21.8 > 5 | **mechanism invalid; withdrawn** |
| 16 | naive pool | boiling-frog ramp | pool 100% attack | lag + purge necessary |

Additional falsification tests worth running once approved, all synthetic:
- the joint IUT;
- finite-Q quarantine;
- the Bonferroni multiscale detector;
- multiplicative common mode;
- non-Gaussian heavy-tailed ε;
- an adversary who optimises E[g(p)] ≤ 1 directly.

---

## 8. Prior-art assessment of surviving contributions (review item 8)

**Rule applied.** No novelty is claimed for:
- naming;
- notation;
- combining known components;
- applying a known method to NIDS.

What remains is listed with its closest prior art and an honest risk rating. "Verification" says whether the source is already in `02_literature/records.md` or doc 04, or still needs a bibliographic check before any manuscript use.

| ID | Residual technical claim | Closest prior art (not ours) | What is already known | What may be new | Novelty risk | Verification |
|---|---|---|---|---|---|---|
| S1 | Theorems 1a/1b: false-alarm bounds for conformal e-CUSUM on double-differenced, strided panel residuals under β-mixing, with explicit coupling and deflation remainders | Vovk et al. 2005; Fedorova 2012; Volkhonskiy 2017 (conformal martingales); Shin, Ramdas & Rinaldo (e-detectors); Berbee 1979, Doukhan–Massart–Rio 1995, Viennet 1997 (coupling); Yu 1994 (blocking); Massart 1990 (DKW); Barber et al. 2023, Chernozhukov et al. 2018, Oliveira et al. 2022 (conformal under non-exchangeability / mixing) | Conformal validity under exchangeability; approximate validity under mixing (Chernozhukov, Oliveira); e-detector ARL | Explicit *anytime* false-alarm bound for the restarted e-CUSUM with a predictable good event and a strided reference; the deflation lemma as stated | **High.** Close to Oliveira et al. and Barber et al. It may be a corollary of existing results. | Vovk, Fedorova, Volkhonskiy, Shin, Barber, Chernozhukov, Oliveira: in doc 04. Berbee: in doc 04. Doukhan–Massart–Rio, Viennet, Yu, Massart: **to verify**. |
| S2 | Ψ = KL(F_1 ‖ F_0) = ∫ ROC′ log ROC′ as the e-CUSUM growth rate; AUC insufficiency; I-projection lower bound Ψ_min(AUC); operating-point bound; Cramér-root delay for intermittent attacks | Csiszár 1975 (I-projection); Lorden 1971, Lai 1998 (delay); Siegmund 1985 (Cramér root); Liu et al. NeurIPS 2022 (ROC and f-divergence); Mullhaupt & Peng 2025; on-off CUSUM attacks (arXiv 1803.00128) | KL as the optimal delay constant; ROC′ as the likelihood ratio; ROC–divergence links | The packaging as a detector-evaluation diagnostic for e-value NIDS, and the explicit E[g] ≤ 1 evasion condition | **High.** Each step is textbook. The contribution is at most expository or diagnostic. | Lai, Siegmund, Lorden, on-off: doc 04 or records. Csiszár, Liu 2022, Mullhaupt & Peng: **to verify**. |
| S3 | Propositions 3.1–3.3: absorption time g + L, ramp ≡ step β(g + L/2), Le Cam impossibility for drift-mimicking attacks; multiscale Bonferroni-versus-averaging cost | Hawkins (self-starting CUSUM, Phase-I contamination); Xie & Siegmund 2013; Mo & Sinopoli 2016 (stealthy attacks); Kravchik 2020 | Masking of self-starting charts; stealthy-attack limits in control systems | A precise accounting for delayed conformal references | **Medium–high** | Xie–Siegmund, Kravchik: doc 04. Hawkins, Mo & Sinopoli: **to verify**. |
| S4 | IUT combination of circular, phase and historical nulls for coordination detection, with its validity conditions C1–C3 | Berger 1982 (IUT); BotSniffer/BotMiner; Neill (GraphScan); rEDM seasonal surrogates | Surrogate-data nulls; IUT | Using IUT over complementary benign-synchronization nulls | **Medium.** It may be standard in surrogate-data practice. | BotMiner: doc 04. Berger, GraphScan, rEDM: **to verify**. |
| S5 | Family-novelty e-value g(max_c π_c), valid under a composite known-class null; the four-way decision rule | Mondrian conformal (Vovk et al.); Bates et al. 2023; CGTC (arXiv 2510.13037); Masud (ECSMiner); Zhang 2021 | Open-set conformal; class-conditional coverage; conformal outlier tests | The max-π composite-null e-value | **Medium–high.** max-p over classes is a natural construction. | Bates, CGTC, Masud: doc 04 or records. Zhang 2021: in doc 04. |
| S6 | Theorem 6.1: exposure/residual identity; quarantine and purge-depth requirements | Rubinstein et al. (boiling frog); Kravchik 2020; Hawkins (Phase-I); poisoning literature | Gradual poisoning; robust reference sets | The explicit counting identity and its necessary conditions | **Medium.** Elementary, but I have not found it stated. | Rubinstein, Kravchik: doc 04. |

**Overall novelty verdict.** None of S1–S6 is, on current evidence, a substantial new theorem. They are:
- verified conditions;
- explicit remainders;
- counterexamples;
- a corrected analysis of intermittent attacks.

The **open tension in §1.4 [O]** (exact validity versus masking resistance and drift adaptation) is the strongest candidate for genuinely new research, because I found no construction that resolves it. A targeted full-text search of Oliveira et al. 2022, Barber et al. 2023 and the self-starting-CUSUM literature is needed before S1 or S3 is claimed in any form.

**Withdrawn or downgraded since phase 4:**
- evidence-gated calibration (invalid);
- the duty-cycle claim (wrong);
- the dense reference (invalid);
- the anchor as a guarantee (invalid under benign drift);
- lag/bin coordination (underpowered);
- the contamination bound without quarantine (false).

---

## 9. Decisions requested from you

1. **Validity route** for any future implementation:
   - (a) Theorem 1a, exact but drift-intolerant;
   - (b) Theorem 1b, deflated, guaranteed but weak;
   - (c) undeflated delayed reference, simulation-supported only;
   - (d) treat the §1.4 tension as the research question itself.
2. Whether S2 (the Ψ diagnostic) should be developed only as an **evaluation tool** for HoNE-style detector comparisons, which is where it is honest.
3. Whether the additional synthetic falsifications listed in §7 should be run before any real-data work.
4. Which of S1–S6, if any, warrant the targeted full-text prior-art check.

Until you approve, no implementation, real-data experiment or HoNE change will be made.
