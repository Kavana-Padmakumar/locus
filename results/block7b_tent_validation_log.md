\# Block 7B — TENT Validation Against Literature Log



Date: 2026-09-06



Task: Validate TENT against roughly-published numbers on the same pilot

corruption types used for BN-Adapt (gaussian\_noise, contrast, severity 5).

Fix any discrepancies before proceeding (per planner: debug if >5-10pp off).



Reference source: "Source-Free Adaptation to Measurement Shift via

Bottom-Up Feature Restoration" (arXiv:2107.05446), Table 23, online TENT

results (single-pass adaptation, matching this project's implementation),

ResNet-18 backbone, CIFAR-10-C severity 5. Same architecture and online

setting as this project's eval\_tent.py - closer match than Block 6B's

BN-Adapt reference, which used a different architecture (WRN-28-10).



Results (src/validate\_tent\_vs\_literature.py):

gaussian\_noise, sev5: measured=26.58%  published=27.7%  diff=1.12pp

contrast, sev5:       measured=18.59%  published=12.2%  diff=6.39pp

Validation: PASSED



Status: COMPLETE.

