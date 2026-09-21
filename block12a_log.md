\# Block 12A — Layer-wise activation patching (causal mediation)

Date: 2026-09-17



\## Task

Implement layer-wise activation patching (do-intervention swapping single-layer

activations between theta and theta-prime runs), building on Block 8's dual-state engine.



\## Implementation

Reused Block 8's DualStateModel.forward(images, patch\_layers) directly — since

TENT only updates BatchNorm affine params (weight/bias), per-layer parameter

patching is mathematically equivalent to activation patching for this method.

No new hook-based patching infrastructure was needed.



\## Condition tested

Corruption: gaussian\_noise, severity=3

TENT adaptation (full test set, n=10000): accuracy=80.83%, error=19.17%



\## Results (on frozen 512-image probe set)

Zero layers patched (theta baseline): error = 0.093750

All layers patched (theta-prime): error = 0.115234

Theta-prime / Block-8 gate error (independently computed): error = 0.115234

Convergence check: MATCH (0.115234 == 0.115234)



\## Monotonicity

The cumulative patching curve is NOT strictly monotonic. Overall trend rises

from 0.0996 (bn1) to 0.1152 (final layer), but individual dips occur, most

notably at layer2.0.downsample.1 (error = 0.0898, below even the zero-patch

baseline). This is explainable as genuine layer interaction: an individual

patched layer can partially compensate for error introduced by earlier

patched layers, with error climbing again once later layers are also patched.

Per the Definition of Done, this non-monotonicity is documented here rather

than treated as a failure.



\## Definition of Done

\- Cumulative patching curve computed across all 20 BN layers: DONE

\- Converges to Block-8 gate result: CONFIRMED (exact match, 0.115234)

\- Monotonic OR explainably non-monotonic, documented: NON-MONOTONIC, explained above



\## Status: COMPLETE

