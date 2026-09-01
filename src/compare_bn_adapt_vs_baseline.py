"""
LOCUS - Block 6A: BN-Adapt vs Block-5 baseline sanity check
Confirms BN-Adapt reduces error relative to the no-adaptation baseline -
same direction as Schneider et al. 2020 and Nado et al. 2020.
"""
import csv

def load_results(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["corruption"], row["severity"])
            rows[key] = row
    return rows

baseline = load_results("results/baseline_results_v2.csv")
bn_adapt = load_results("results/bn_adapt_results.csv")

print(f"{'corruption':<18}{'sev':<5}{'baseline_err':<14}{'bn_adapt_err':<14}{'delta':<10}")
deltas = []
for key in sorted(baseline.keys()):
    b_err = float(baseline[key]["error_rate"])
    a_err = float(bn_adapt[key]["error_rate"])
    delta = b_err - a_err   # positive = improvement (error went down)
    deltas.append(delta)
    corruption, sev = key
    print(f"{corruption:<18}{sev:<5}{b_err:<14.2f}{a_err:<14.2f}{delta:<+10.2f}")

avg_delta = sum(deltas) / len(deltas)
n_improved = sum(1 for d in deltas if d > 0)
print(f"\nAverage error reduction: {avg_delta:+.2f} pp across {len(deltas)} conditions")
print(f"Conditions improved: {n_improved}/{len(deltas)}")

assert avg_delta > 0, "BN-Adapt did not reduce error on average vs baseline - DO NOT proceed until debugged"
print("\nSanity check PASSED: BN-Adapt reduces error relative to Block-5 baseline.")