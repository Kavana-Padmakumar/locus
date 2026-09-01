"""
LOCUS - Block 6B: Validate BN-Adapt against published literature numbers
Pilot corruption types: gaussian_noise, contrast (severity 5)

Reference: "Singular Value Penalization and Semantic Data Augmentation for
Fully Test-Time Adaptation" (arXiv:2312.08378), Table 1, "NORM" row
(prediction-time BatchNorm - the same method family as BN-Adapt), WRN-28-10
backbone, CIFAR-10-C severity 5.

Architecture differs from this project's ResNet-18/huyvnphan checkpoint,
so an exact match is not expected - only rough consistency, per the task's
own explicit tolerance (>5-10 percentage points = debug before proceeding).
"""
import csv

PUBLISHED = {
    ("gaussian_noise", "5"): 28.1,
    ("contrast", "5"): 12.6,
}
TOLERANCE_PP = 10.0

def load_bn_adapt_results(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption"], row["severity"])] = float(row["error_rate"])
    return rows

results = load_bn_adapt_results("results/bn_adapt_results.csv")

print(f"{'corruption':<18}{'sev':<5}{'measured_err':<14}{'published_err':<14}{'abs_diff':<10}{'within_tol':<10}")
all_within = True
for (corruption, sev), published_err in PUBLISHED.items():
    measured_err = results[(corruption, sev)]
    diff = abs(measured_err - published_err)
    within = diff <= TOLERANCE_PP
    all_within = all_within and within
    print(f"{corruption:<18}{sev:<5}{measured_err:<14.2f}{published_err:<14.2f}{diff:<10.2f}{str(within):<10}")

print()
if all_within:
    print(f"VALIDATION PASSED: all pilot corruption types within {TOLERANCE_PP}pp of published BN-Adapt numbers.")
else:
    print(f"VALIDATION FAILED: at least one pilot type is outside {TOLERANCE_PP}pp - DEBUG before proceeding to Block 7.")

assert all_within, "BN-Adapt validation against literature failed - do not proceed to Block 7 until debugged"