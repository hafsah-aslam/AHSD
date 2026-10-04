# S1 boil-rate bound (draft for the authors to check)

Implemented in `src/nids/theory.py` and checked numerically in
`tests/test_models_theory_metrics.py::test_s1_drift_never_exceeds_bound`.

## Setting

S1 updates the equilibrium with a stress-gated rate and an anchor:

```
s_t  = mean_d |h_t − E_{t−1}|                     (pre-update stress)
γ_t  = γ · exp(−s_t / τ)
E_t  = E_{t−1} + γ_t (sg(h_t) − E_{t−1}) − κ (E_{t−1} − E_0)
```

Note the definition of `s_t`. The spec writes `γ_t = γ·exp(−stress_t/τ)`. The
code uses the stress *before* the E update, which makes the bound exact.

## Claim

Let `m_t = mean_d |E_t − E_0|`, with `m_0 = 0`. Then for **any** input
sequence (any attack, any ramp length R) and any h dynamics:

```
m_t ≤ (γ τ / e) · (1 − (1 − κ)^t) / κ  ≤  γ τ / (e κ)
```

## Proof

`E_t − E_0 = (1 − κ)(E_{t−1} − E_0) + γ_t (h_t − E_{t−1})`. Take the mean
absolute value over channels and apply the triangle inequality:
`m_t ≤ (1 − κ) m_{t−1} + γ s_t e^{−s_t/τ}`. For x ≥ 0, `x e^{−x/τ}` peaks at
x = τ, so `γ s_t e^{−s_t/τ} ≤ γτ/e`. Unrolling the recursion gives the
geometric sum. ∎

## Consequences for the boil-rate experiment (P4)

- **Per-step boil rate.** The equilibrium moves at most γτ/e per step
  (mean-abs), however slowly the attacker ramps. An attack that wants to stay
  under detection while moving the reference by Δ needs at least
  Δ·e/(γτ) steps.
- **Total absorption.** For an attack holding a mean-abs deviation `d`, the
  residual stress is at least `d − γτ/(eκ)` (`theory.min_residual_stress`),
  and the absorbed fraction is at most `γτ/(eκ d)`.
- **Grid values.** With γ ≤ 0.2, τ = c × median benign stress
  (c ∈ {0.5, 1, 2}) and κ ∈ {0.01, 0.05}, total absorption is capped at
  between 0.74c and 3.7c × the median benign stress.

## Limits (to be stated in the paper)

- The bound is on E, not on the detector's decision. Whether residual stress
  crosses the threshold depends on the benign stress distribution.
- It covers the mean-abs channel norm. One channel can drift more if the
  others drift less.
- With the adaptive (ungated, unanchored) equilibrium, E has no such bound.
  For a constant offset, `m_t → d` geometrically at rate (1 − γ).
  `test_adaptive_equilibrium_absorbs_persistent_offset` checks this.
