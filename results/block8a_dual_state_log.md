\# Block 8A — Dual-Parameter-State Engine Log



Date: 2026-09-08



Task: Build a model wrapper that can execute each normalization layer

under either theta (source weights) or theta-prime (TENT-adapted weights)

on command, for every BatchNorm layer. Implemented in

src/dual\_state\_model.py.



Scope note: this block builds the engine and runs basic plumbing sanity

checks only. The DoD's full hard-gate mechanical correctness test is

Block 8B's deliverable (Sep 9). Block 8C (Sep 10) is a dedicated

hardening/buffer day for this piece, per the updated planner.



Pilot condition used to build a real theta-prime: gaussian\_noise, severity 5.

BN layers found: \[PASTE count from Step F]



Plumbing sanity checks (src/build\_dual\_state\_engine.py):

\[Check 1] Zero-layer patch matches theta's own output: \[PASTE True/False]

\[Check 2] Patching one layer changes output: \[PASTE True/False]

\[Check 3] theta / theta-prime BN layer name sets match: \[PASTE True/False]



Status: Engine built and plumbing-verified. Hard-gate correctness test

pending - see Block 8B (Sep 9).

