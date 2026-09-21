"""
LOCUS - Block 13A: Full single-condition pipeline, wired end-to-end.
Combines: baseline/TENT/h_rec/rho (already computed, Block 9/10, read from
results/pilot_rho_results.csv) + CKA per layer (Block 11) + activation
patching curve (Block 12) - for ONE condition, then generates a Tier-1
plain-English report.
"""
import sys, os, csv
sys.path.append(os.path.dirname(__file__))

import torch
from dual_state_model import DualStateModel
from cka import load_probe_set, cached_compute_cka_all_layers
from eval_tent import load_model, configure_tent_model, evaluate_tent


def load_rho_row(corruption, severity, seed, path="results/pilot_rho_results.csv"):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["corruption"] == corruption and
                int(row["severity"]) == severity and
                int(row["seed"]) == seed):
                return row
    raise ValueError(f"No rho row found for {corruption}, sev={severity}, seed={seed}")


def compute_error(output, labels):
    preds = output.argmax(dim=1)
    return (preds != labels).float().mean().item()


def cumulative_patching_curve(dual_model, images, labels):
    layer_names = dual_model.bn_layer_names()
    results = {}
    patched_so_far = []
    for layer_name in layer_names:
        patched_so_far.append(layer_name)
        output = dual_model.forward(images, patch_layers=patched_so_far)
        results[layer_name] = compute_error(output, labels)
    return results


def run_pipeline(corruption, severity, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stage 1: rho (already computed, Block 9/10 - read, don't recompute) ---
    rho_row = load_rho_row(corruption, severity, seed)

    # --- Stage 2: CKA (Block 11) ---
    images, labels = load_probe_set(device=device)

    model_theta = load_model(device)
    dual_model = DualStateModel(model_theta, device)

    model_for_adaptation = load_model(device)
    trainable_params = configure_tent_model(model_for_adaptation)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    acc, err, n = evaluate_tent(model_for_adaptation, optimizer, device,
                                  corruption, severity)
    dual_model.load_theta_prime(model_for_adaptation)

    cka_results = cached_compute_cka_all_layers(
        model_theta, model_for_adaptation, images, device,
        model_a_name=f"theta_{corruption}_{severity}",
        model_b_name=f"theta_prime_{corruption}_{severity}"
    )

    # --- Stage 3: Activation patching curve (Block 12) ---
    patching_curve = cumulative_patching_curve(dual_model, images, labels)

    return {
        "corruption": corruption,
        "severity": severity,
        "seed": seed,
        "rho": rho_row,
        "tent_accuracy": acc,
        "tent_error": err,
        "tent_n": n,
        "cka_per_layer": cka_results,
        "patching_curve": patching_curve,
    }


def generate_report(result):
    c = result["corruption"]
    sev = result["severity"]
    rho = result["rho"]

    cka_values = list(result["cka_per_layer"].values())
    min_cka_layer = min(result["cka_per_layer"], key=result["cka_per_layer"].get)
    min_cka_value = result["cka_per_layer"][min_cka_layer]

    patching = result["patching_curve"]
    patching_values = list(patching.values())
    is_monotonic = all(patching_values[i] <= patching_values[i+1] + 1e-9
                        for i in range(len(patching_values) - 1))

    rho_val = float(rho["rho"])
    if rho_val > 0.7:
        rho_story = "strongly representation-driven"
    elif rho_val > 0.3:
        rho_story = "a mix of representation-change and decision-boundary-shift"
    else:
        rho_story = "predominantly decision-boundary-driven"
    report = f"""
=== LOCUS Report: {c}, severity {sev} (seed {result['seed']}) ===

CAUSAL DECOMPOSITION (rho)
Source model error: {rho['r_source']}%
TENT-adapted error: {rho['r_tent']}%
Boundary-only (h_rec) error: {rho['r_hrec']}%
rho = {rho['rho']}
Interpretation: TTA's improvement on this condition is {rho_story}.
{"" if rho_val <= 1.0 else f"Note: rho > 1 here means the boundary-only control made the model WORSE than the unadapted source ({rho['r_hrec']}% vs {rho['r_source']}%), while full TENT adaptation still substantially improved it ({rho['r_tent']}%). This points to representation change, not decision-boundary movement, as the active mechanism."}

REPRESENTATIONAL SIMILARITY (CKA)
CKA was computed at every BatchNorm layer between theta and theta-prime.
Layers with LOWER CKA changed representation MORE under adaptation.
Most-changed layer: {min_cka_layer} (CKA = {min_cka_value:.4f})
Range across all layers: {min(cka_values):.4f} to {max(cka_values):.4f}

CAUSAL ACTIVATION PATCHING
Cumulative patching curve is {'MONOTONIC' if is_monotonic else 'NON-MONOTONIC'}.
All-layers-patched error: {patching_values[-1]:.6f} (should match TENT's own error rate closely)

TENT ADAPTATION (verification run)
Accuracy: {result['tent_accuracy']:.2f}%  Error: {result['tent_error']:.2f}%  n={result['tent_n']}
"""
    return report


def main():
    corruption, severity, seed = "gaussian_noise", 3, 0
    print(f"Running full pipeline for: {corruption}, severity={severity}, seed={seed}")
    result = run_pipeline(corruption, severity, seed)
    report = generate_report(result)
    print(report)

    os.makedirs("results", exist_ok=True)
    with open("results/block13a_report.txt", "w") as f:
        f.write(report)
    print("Report saved to results/block13a_report.txt")


if __name__ == "__main__":
    main()