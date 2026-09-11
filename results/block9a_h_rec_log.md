\# Block 9A — Boundary-Only Control (h\_rec) Implementation Log



Date: 2026-09-11



Task: Implement h\_rec - freeze the backbone entirely, fit a diagonal

affine recalibration of the frozen logits using the same entropy loss

TENT/BN-Adapt use. Implemented in src/h\_rec\_control.py.



Design note: backbone runs in eval() mode with frozen weights (NOT

train() like BN-Adapt/TENT) - representations never change, only a

per-class affine transform on frozen logits is fit. This is deliberate:

h\_rec can only move decision boundaries, never touch representations.



Scope note: this block builds h\_rec and smoke-tests it (seed=0 only,

2 pilot conditions). Full validation across 3 seeds, checking for NaNs/

divergence/degenerate values, is Block 9B's job (Sep 12) - not claimed

complete here.



Smoke test results:

gaussian\_noise, sev5: accuracy=35.55%  error=64.45%

&#x20; a range: \[1.0010, 1.0010]  NaN present: False

&#x20; b range: \[-0.0010, 0.0010]  NaN present: False

contrast, sev5: accuracy=22.27%  error=77.73%

&#x20; a range: \[0.9990, 1.0010]  NaN present: False

&#x20; b range: \[-0.0010, 0.0010]  NaN present: False



Status: h\_rec built and smoke-tested, runs without crashing. Full

3-seed validation pending - see Block 9B (Sep 12).

