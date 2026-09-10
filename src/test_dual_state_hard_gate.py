"""
LOCUS - Block 8B: Dual-parameter-state hard-gate mechanical correctness test
DoD: patching ALL BatchNorm layers from theta to theta-prime must EXACTLY
reproduce theta-prime's own output (bit-for-bit or within floating-point
tolerance). This is a hard gate - the entire causal-patching methodology
in later phases depends on this being true. Do not proceed to Phase 3
until this passes.

Tested on two independent pilot conditions so a pass isn't a fluke of
one specific input/adaptation.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from cifar10c_dataset import CIFAR10C
from eval_tent import load_model, configure_tent_model, entropy_loss
from dual_state_model import DualStateModel

PILOT_CONDITIONS = [
    ("gaussian_noise", 5),
    ("contrast", 5),
]

def build_theta_prime(device, corruption, severity):
    """Runs one TENT adaptation step, returns the adapted model (= theta-prime
    for this condition) and the batch it was adapted on."""
    adapted_model = load_model(device)
    trainable_params = configure_tent_model(adapted_model)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images = images.to(device)

    outputs = adapted_model(images)
    loss = entropy_loss(outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return adapted_model, images


def run_hard_gate_test(device, corruption, severity):
    theta_model = load_model(device)
    engine = DualStateModel(theta_model, device)

    adapted_model, images = build_theta_prime(device, corruption, severity)
    engine.load_theta_prime(adapted_model)

    # Patch ALL layers to theta-prime.
    all_layers = engine.bn_layer_names()
    engine_output = engine.forward(images, patch_layers=all_layers)

    # Run the real theta-prime model directly on the same input.
            # must match engine.forward()'s mode - both sides use batch statistics
    adapted_model.train()
    for p in adapted_model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        direct_output = adapted_model(images)
    matches = torch.allclose(engine_output, direct_output, atol=1e-6)
    max_diff = (engine_output - direct_output).abs().max().item()
    return matches, max_diff


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    all_passed = True
    for corruption, severity in PILOT_CONDITIONS:
        print(f"\nTesting: {corruption}, severity={severity}...")
        matches, max_diff = run_hard_gate_test(device, corruption, severity)
        print(f"  All-layers-patched output matches theta-prime's direct output: {matches}")
        print(f"  Max absolute difference: {max_diff:.2e}")
        all_passed = all_passed and matches

    print()
    if all_passed:
        print("HARD GATE PASSED: all-layer patching exactly reproduces theta-prime's")
        print("own output on every tested condition. Safe to proceed to Phase 3.")
    else:
        print("HARD GATE FAILED: at least one condition did not match exactly.")
        print("DO NOT proceed to Phase 3. Debug the engine before continuing.")

    assert all_passed, "Hard-gate mechanical correctness test failed on at least one condition"


if __name__ == "__main__":
    main()