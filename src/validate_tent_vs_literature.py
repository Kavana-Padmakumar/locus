"""
LOCUS - Block 7B: Validate TENT against published literature numbers
Pilot corruption types: gaussian_noise, contrast (severity 5) - same pilot
types used for BN-Adapt validation in Block 6B, for direct comparability.

Reference: "Source-Free Adaptation to Measurement Shift via Bottom-Up
Feature Restoration" (arXiv:2107.05446), Table 23, online TENT results
(single-pass, matching this project's single-gradient-step-per-batch
implementation), ResNet-18 backbone, CIFAR-10-C severity 5. This is the
same architecture and online setting used in this project's eval_tent.py,
making this a closer match than Block 6B's BN-Adapt reference (which used
a different architecture, WRN-28-10).
"""
import csv

PUBLISHED = {
    ("gaussian_noise", "5"): 27.7,
    ("contrast", "5"): 12.2,
}
TOLERANCE_PP = 10.0

def load_tent_results(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption"], row["severity"])] = float(row["error_rate"])
    return rows

results = load_tent_results("results/tent_results.csv")

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
    print(f"VALIDATION PASSED: all pilot corruption types within {TOLERANCE_PP}pp of published online-TENT numbers.")
else:
    print(f"VALIDATION FAILED: at least one pilot type is outside {TOLERANCE_PP}pp - DEBUG before proceeding to Block 8.")

assert all_within, "TENT validation against literature failed - do not proceed to Block 8 until debugged"