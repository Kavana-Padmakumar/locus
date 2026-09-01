\# Block 6A — BN-Adapt Implementation Log



Date: 2026-09-01



Method: BN-Adapt (Schneider et al., NeurIPS 2020) - BatchNorm layers use

each target batch's own statistics (train-mode), no gradient steps, no

weight updates. Implemented in src/eval\_bn\_adapt.py.



Conditions evaluated: 6 corruptions x 3 severities = 18 total

Output files: results/bn\_adapt\_results.csv, results/bn\_adapt\_results.jsonl



Sanity check vs Block 5 baseline (src/compare\_bn\_adapt\_vs\_baseline.py):

Average error reduction: +8.34 pp across 18 conditions

Conditions improved: 9/18

Sanity check: PASSED



Pattern observed: improvements concentrate heavily at high severity (e.g.

contrast sev5: +61.42pp, gaussian\_noise sev5: +34.14pp, gaussian\_noise

sev3: +25.64pp), while low-severity conditions show small regressions

(-0.22 to -1.92pp, e.g. jpeg\_compression, snow, brightness at severity 1-3).

This is consistent with the known BN-Adapt trade-off: at small distribution

shift the source model's running statistics are already well-calibrated,

so swapping in a single noisy batch's statistics can slightly hurt; at

large shift, target-batch statistics correct for real distribution change

and help substantially. Direction and magnitude are consistent with

Schneider et al.'s and Nado et al.'s findings (see notes/lit\_notes.md).



Status: COMPLETE.

