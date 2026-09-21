"""
LOCUS - Block 12A: Layer-wise activation patching (causal mediation)
Uses Block 8's DualStateModel (BN affine-param patching, theta vs
theta-prime, per layer) driven across a cumulative sweep of layers.
theta-prime is produced by running real TENT adaptation (Block 9's
evaluate_tent, which adapts its model argument in place).
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    images, labels = load_probe_set(device=device)
    print(f"Loaded frozen probe set: {images.shape[0]} images")

    # --- theta: fresh, unadapted model ---
    model_theta = load_model(device)
    dual_model = DualStateModel(model_theta, device)

    # --- theta-prime: a SEPARATE fresh model, adapted via real TENT ---
    # Must be a different object from model_theta, or DualStateModel's
    # internal theta_params snapshot (already cloned at init) still
    # protects theta - but load a second copy to be unambiguous.
    model_for_adaptation = load_model(device)
    trainable_params = configure_tent_model(model_for_adaptation)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    # TODO: pick one real (corruption, severity) pair your Block 9
    # config actually uses, e.g. "gaussian_noise", severity 3 - check
    # src/config.py or your cfg loader for valid names.
    corruption = "gaussian_noise"
    severity = 3
    acc, err, n = evaluate_tent(model_for_adaptation, optimizer, device,
                                  corruption, severity)
    print(f"TENT adaptation done on ({corruption}, severity={severity}): "
          f"accuracy={acc:.2f}%  error={err:.2f}%  n={n}")

    dual_model.load_theta_prime(model_for_adaptation)

    # --- Sanity check 1: zero layers patched == theta's own error ---
    output_zero = dual_model.forward(images, patch_layers=[])
    error_zero = compute_error(output_zero, labels)
    print(f"\nZero layers patched (theta's own error): {error_zero:.6f}")

    # --- Cumulative patching curve ---
    print("\nCumulative patching curve:")
    results = cumulative_patching_curve(dual_model, images, labels)
    for name, err_val in results.items():
        print(f"  {name}: error = {err_val:.6f}")

    # --- Sanity check 2: all layers patched == theta-prime / Block-8 gate ---
    all_layers = dual_model.bn_layer_names()
    output_full = dual_model.forward(images, patch_layers=all_layers)
    error_full = compute_error(output_full, labels)
    print(f"\nAll layers patched (theta-prime / Block-8 gate error): {error_full:.6f}")


if __name__ == "__main__":
    main()