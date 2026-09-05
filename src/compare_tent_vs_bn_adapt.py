"""
LOCUS - Block 7A: TENT vs BN-Adapt sanity check
DoD: TENT's error reduction should be in the same ballpark as BN-Adapt's
or better, consistent with the literature's general finding that TENT
slightly outperforms BN-Adapt (Wang et al., ICLR 2021).
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

bn_adapt = load_results("results/bn_adapt_results.csv")
tent = load_results("results/tent_results.csv")

print(f"{'corruption':<18}{'sev':<5}{'bn_adapt_err':<14}{'tent_err':<14}{'delta':<10}")
deltas = []
for key in sorted(bn_adapt.keys()):
    b_err = float(bn_adapt[key]["error_rate"])
    t_err = float(tent[key]["error_rate"])
    delta = b_err - t_err   # positive = TENT better (lower error) than BN-Adapt
    deltas.append(delta)
    corruption, sev = key
    print(f"{corruption:<18}{sev:<5}{b_err:<14.2f}{t_err:<14.2f}{delta:<+10.2f}")

avg_delta = sum(deltas) / len(deltas)
n_tent_better = sum(1 for d in deltas if d > 0)
print(f"\nAverage error difference (BN-Adapt minus TENT): {avg_delta:+.2f} pp across {len(deltas)} conditions")
print(f"Conditions where TENT outperforms BN-Adapt: {n_tent_better}/{len(deltas)}")

assert avg_delta > -2.0, "TENT is meaningfully worse than BN-Adapt on average - DEBUG before proceeding to Block 8"
print("\nSanity check PASSED: TENT's performance is in the same ballpark as BN-Adapt's or better.")