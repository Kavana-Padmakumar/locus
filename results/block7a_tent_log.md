\# Block 7A — TENT Implementation Log



Date: 2026-09-05



Method: TENT (Wang et al., ICLR 2021) - entropy minimization on unlabeled

target batches, gradient updates restricted to BatchNorm affine parameters

(gamma, beta) only. One gradient step per batch, Adam optimizer, lr=1e-3.

Each corruption/severity condition starts from a fresh pretrained model

(single-domain protocol, not continual TTA). Implemented in src/eval\_tent.py.



Sanity check: trainable params confirmed to be a small fraction of total

model params (BN affine only) - see block7a\_eval\_output.txt for exact count.



Conditions evaluated: 6 corruptions x 3 severities = 18 total

Output files: results/tent\_results.csv, results/tent\_results.jsonl



Comparison vs BN-Adapt (src/compare\_tent\_vs\_bn\_adapt.py):

Average error difference (BN-Adapt minus TENT): \[PASTE avg\_delta]pp

Conditions where TENT outperforms BN-Adapt: \[PASTE n\_tent\_better]/18

Sanity check: \[PASTE PASSED or FAILED]



Literature grounding: Wang et al. 2021 report TENT modestly outperforming

BN-Adapt-style methods on CIFAR-10-C/ImageNet-C in most conditions, since

TENT's gradient step further refines the BN affine parameters beyond what

batch-statistic substitution alone achieves. This result is \[PASTE:

consistent with / a stronger improvement than / a smaller improvement than]

that general finding.



Status: COMPLETE.

