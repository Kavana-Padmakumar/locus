"""
LOCUS - Block 12B: Validate the CE(l) sanity check across multiple conditions.
Runs the cumulative patching curve (Block 12A's engine) across several
real (corruption, severity) pairs and checks, for each: does the
all-layers-patched error converge exactly to the independently-computed
theta-prime/Block-8 gate error? Is the curve monotonic, or is any
non-monotonicity consistent/explainable across conditions?
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from dual_state_model import DualStateModel
from cka import load_probe_set
from eval_tent import load_model, configure_tent_model, evaluate_tent


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


def is_monotonic(values, tol=1e-9):
    return all(values[i] <= values[i + 1] + tol for i in range(len(values) - 1))


def run_condition(corruption, severity, images, labels, device):
    model_theta = load_model(device)
    dual_model = DualStateModel(model_theta, device)

    model_for_adaptation = load_model(device)
    trainable_params = configure_tent_model(model_for_adaptation)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    acc, err, n = evaluate_tent(model_for_adaptation, optimizer, device,
                                  corruption, severity)

    dual_model.load_theta_prime(model_for_adaptation)

    output_zero = dual_model.forward(images, patch_layers=[])
    error_zero = compute_error(output_zero, labels)

    curve = cumulative_patching_curve(dual_model, images, labels)
    curve_values = list(curve.values())

    all_layers = dual_model.bn_layer_names()
    output_full = dual_model.forward(images, patch_layers=all_layers)
    error_full = compute_error(output_full, labels)

    gate_match = abs(curve_values[-1] - error_full) < 1e-6
    monotonic = is_monotonic(curve_values)

    return {
        "corruption": corruption,
        "severity": severity,
        "tent_accuracy": acc,
        "tent_error": err,
        "error_zero_patched": error_zero,
        "error_all_patched": curve_values[-1],
        "error_full_direct": error_full,
        "gate_match": gate_match,
        "monotonic": monotonic,
        "curve": curve,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    images, labels = load_probe_set(device=device)
    print(f"Loaded frozen probe set: {images.shape[0]} images")

    # Real conditions from your config - one per corruption at severity 3,
    # plus severity 1 and 5 on gaussian_noise, to check severity effects too.
    conditions = [
        ("gaussian_noise", 1),
        ("gaussian_noise", 3),
        ("gaussian_noise", 5),
        ("defocus_blur", 3),
        ("snow", 3),
        ("brightness", 3),
        ("contrast", 3),
        ("jpeg_compression", 3),
    ]

    all_results = []
    for corruption, severity in conditions:
        print(f"\n=== Condition: {corruption}, severity={severity} ===")
        result = run_condition(corruption, severity, images, labels, device)
        all_results.append(result)
        print(f"  TENT: accuracy={result['tent_accuracy']:.2f}%  error={result['tent_error']:.2f}%")
        print(f"  Zero-patched error: {result['error_zero_patched']:.6f}")
        print(f"  All-patched error:  {result['error_all_patched']:.6f}")
        print(f"  Gate match (all-patched == direct theta-prime run): {result['gate_match']}")
        print(f"  Monotonic: {result['monotonic']}")

    print("\n\n=== SUMMARY ===")
    n_gate_match = sum(1 for r in all_results if r["gate_match"])
    n_monotonic = sum(1 for r in all_results if r["monotonic"])
    print(f"Gate convergence: {n_gate_match}/{len(all_results)} conditions match exactly")
    print(f"Monotonic curves: {n_monotonic}/{len(all_results)} conditions")

    for r in all_results:
        status = "MATCH" if r["gate_match"] else "MISMATCH"
        mono = "monotonic" if r["monotonic"] else "non-monotonic"
        print(f"  {r['corruption']} (sev={r['severity']}): gate={status}, curve={mono}")


if __name__ == "__main__":
    main()