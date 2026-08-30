\# Block 5A — No-Adaptation Baseline Log



Date: 2026-08-30

Model: ResNet-18 (huyvnphan/PyTorch\_CIFAR10, pretrained, no adaptation applied)

Conditions evaluated: 6 corruptions x 3 severities = 18 total

Output file: results/baseline\_results.csv



Sanity checks:

\- Error increases with severity across all corruptions: True (confirmed across all 6 corruptions - gaussian\_noise, defocus\_blur, snow, brightness, contrast, jpeg\_compression - error rises monotonically from severity 1 to 5)
-No flat/suspicious results across conditions: True (no repeated/degenerate values, all 18 rows show distinct, sensible error rates)



Status: COMPLETE.

