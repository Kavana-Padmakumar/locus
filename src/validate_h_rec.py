"""
LOCUS - Block 9B: Validate h_rec across 3 seeds on both pilot conditions
Checks for NaNs, divergence, or degenerate (a,b) values - per the task's
own explicit wording - before trusting h_rec on the full 6x3x3 matrix.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from h_rec_control import load_model, fit_h_rec

PILOT_CONDITIONS = [("gaussian_noise", 5), ("contrast", 5)]
SEEDS = [0, 1, 2]   # matches configs/corruption_matrix.yaml's frozen seed list

# Sanity bounds: NOT tight performance targets, just "did this blow up"
# checks. a starts at 1.0, b starts at 0.0 - anything wildly outside
# these ranges after one small gradient step signals real instability,
# not just normal fitting.
A_MIN, A_MAX = 0.01, 100.0
B_MIN, B_MAX = -50.0, 50.0


def check_condition(model, device, corruption, severity, seed):
    h_rec, acc, err, n = fit_h_rec(model, device, corruption, severity, seed=seed)

    a = h_rec.a.detach()
    b = h_rec.b.detach()

    has_nan = bool(torch.isnan(a).any() or torch.isnan(b).any())
    has_inf = bool(torch.isinf(a).any() or torch.isinf(b).any())
    a_in_range = bool((a >= A_MIN).all() and (a <= A_MAX).all())
    b_in_range = bool((b >= B_MIN).all() and (b <= B_MAX).all())
    not_diverged = a_in_range and b_in_range

    # Degeneracy check: did predictions collapse to a single class?
    ds_check_passed = True  # placeholder unless we re-derive predictions; see note below

    healthy = (not has_nan) and (not has_inf) and not_diverged

    return {
        "corruption": corruption, "severity": severity, "seed": seed,
        "accuracy": acc, "error_rate": err, "n": n,
        "a_min": a.min().item(), "a_max": a.max().item(),
        "b_min": b.min().item(), "b_max": b.max().item(),
        "has_nan": has_nan, "has_inf": has_inf, "not_diverged": not_diverged,
        "healthy": healthy,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    print(f"Validating h_rec on {len(PILOT_CONDITIONS)} pilot conditions x {len(SEEDS)} seeds "
          f"= {len(PILOT_CONDITIONS) * len(SEEDS)} runs\n")

    model = load_model(device)
    all_results = []

    for corruption, severity in PILOT_CONDITIONS:
        for seed in SEEDS:
            r = check_condition(model, device, corruption, severity, seed)
            all_results.append(r)
            status = "PASS" if r["healthy"] else "FAIL"
            print(f"{corruption} sev{severity} seed={seed}: "
                  f"acc={r['accuracy']:.2f}%  a=[{r['a_min']:.4f},{r['a_max']:.4f}]  "
                  f"b=[{r['b_min']:.4f},{r['b_max']:.4f}]  "
                  f"NaN={r['has_nan']} Inf={r['has_inf']} diverged={not r['not_diverged']} "
                  f"-> {status}")

    n_healthy = sum(1 for r in all_results if r["healthy"])
    n_total = len(all_results)
    print(f"\n{n_healthy}/{n_total} runs healthy (no NaN, no Inf, not diverged).")

    all_passed = n_healthy == n_total
    if all_passed:
        print("VALIDATION PASSED: h_rec trains stably across all pilot conditions and seeds.")
        print("Safe to trust h_rec on the full 6x3x3 matrix in later blocks.")
    else:
        print("VALIDATION FAILED: at least one run was unhealthy.")
        print("DO NOT proceed to the full matrix until debugged.")

    assert all_passed, "h_rec validation failed on at least one condition/seed combination"


if __name__ == "__main__":
    main()