# LOCUS — Core Math

## 1. Representation-attributable fraction (ρ)

ρ = (ΔR_TTA − ΔR_rec) / ΔR_TTA

Where:
- ΔR_TTA = R(source model) − R(fully adapted model)   [full TTA's error reduction]
- ΔR_rec = R(source model) − R(boundary-only model)    [boundary-only control's error reduction]

## 2. Worked numeric example (toy values, sanity check)

Suppose the source model has 40% error on a corrupted test set.

**Case A — normal, informative result:**
- Full TTA (TENT) reduces error to 25%  → ΔR_TTA = 40 − 25 = 15
- Boundary-only control reduces error to 30%  → ΔR_rec = 40 − 30 = 10
- ρ = (15 − 10) / 15 = 5/15 = 0.33
- Reading: only 33% of the gain is representation-driven; 67% is boundary-shift-explainable.

**Case B — edge: ρ = 0 (pure recalibration)**
- ΔR_TTA = 15, ΔR_rec = 15 (boundary control fully reproduces the gain on its own)
- ρ = (15 − 15) / 15 = 0
- Reading: 100% of the gain is boundary-explainable. TTA = recalibration, nothing more.

**Case C — edge: ρ = 1 (pure representation repair)**
- ΔR_TTA = 15, ΔR_rec = 0 (boundary control does nothing at all)
- ρ = (15 − 0) / 15 = 1
- Reading: 100% representation-driven. The boundary-only story is completely wrong.

**Case D — edge: ρ negative (the interesting failure case)**
- ΔR_TTA = 10, ΔR_rec = 15 (boundary-only control actually outperforms full TTA)
- ρ = (10 − 15) / 10 = −0.5
- Reading: full adaptation is actively worse than pure recalibration — flagged in the
  proposal's negative-results section as an important nuance if it occurs anywhere in
  the real sweep.

All four cases behave sensibly — confirms the formula is well-posed before running it on
real data.

## 3. Layer-wise causal effect (activation patching)

CE(ℓ) = R(f_theta) − R(f_patched-at-layer-ℓ)

Where f_patched-at-layer-ℓ runs the model under source weights theta everywhere EXCEPT
layer ℓ, where the activation is swapped for its value under the adapted weights
theta-prime (a do-intervention). Computed for every normalization layer, giving a
causal-effect profile across depth.

## 4. Representational-change magnitude (CKA)

CKA_ℓ = linear CKA(activations at layer ℓ under theta, activations at layer ℓ under
theta-prime), computed on a fixed probe set. Range [0, 1]. CKA_ℓ = 1 means the
representation at that layer did not change at all; CKA_ℓ close to 0 means it changed
substantially.

## 5. The key statistical test

Spearman rank correlation between CE(ℓ) and (1 − CKA_ℓ) across layers, pooled across
corruption types via a mixed-effects model (corruption type as random effect), with
bootstrap 95% confidence intervals and Holm–Bonferroni correction across the 6 frozen
corruption types when testing significance.

- Positive correlation → layers that changed representation ARE the layers that matter
  causally (supports naive "representation repair" story).
- No correlation / dissociation → change and importance are NOT the same thing (supports
  the boundary-shift story — representation moves somewhere causally inert while the real
  effect lives elsewhere).
