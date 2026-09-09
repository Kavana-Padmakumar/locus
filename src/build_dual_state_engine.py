"""
LOCUS - Block 8A: Build the dual-parameter-state engine and run basic
plumbing sanity checks. This is NOT the full hard-gate correctness test
required by the DoD (all-layers-patched == theta-prime's own output,
saved as an automated unit test) - that is Block 8B's job, Sep 9.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from cifar10c_dataset import CIFAR10C
from eval_tent import load_model, configure_tent_model, entropy_loss
from dual_state_model import DualStateModel

PILOT_CORRUPTION = "gaussian_noise"
PILOT_SEVERITY = 5

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    theta_model = load_model(device)
    engine = DualStateModel(theta_model, device)
    print(f"theta snapshot: {len(engine.bn_layer_names())} BatchNorm layers found.")

    adapted_model = load_model(device)
    trainable_params = configure_tent_model(adapted_model)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    ds = CIFAR10C(data_dir="data_raw", corruption=PILOT_CORRUPTION, severity=PILOT_SEVERITY)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    outputs = adapted_model(images)
    loss = entropy_loss(outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    engine.load_theta_prime(adapted_model)
    print(f"theta-prime snapshot loaded: {len(engine.theta_prime_params)} BatchNorm layers.")

    with torch.no_grad():
        theta_direct_output = theta_model(images)
    engine_zero_patch_output = engine.forward(images, patch_layers=[])
    zero_patch_matches = torch.allclose(theta_direct_output, engine_zero_patch_output, atol=1e-6)
    print(f"[Check 1] Zero-layer patch matches theta's own output: {zero_patch_matches}")
    assert zero_patch_matches, "Engine's zero-patch output does not match theta - wiring bug"

    one_layer = engine.bn_layer_names()[0]
    engine_one_patch_output = engine.forward(images, patch_layers=[one_layer])
    patch_has_effect = not torch.allclose(engine_zero_patch_output, engine_one_patch_output, atol=1e-6)
    print(f"[Check 2] Patching one layer changes output: {patch_has_effect}")
    assert patch_has_effect, "Patching a layer produced no change - patching may be a no-op bug"

    names_match = set(engine.theta_params.keys()) == set(engine.theta_prime_params.keys())
    print(f"[Check 3] theta / theta-prime BN layer name sets match: {names_match}")
    assert names_match

    print("\nAll Block 8A plumbing sanity checks PASSED.")
    print("NOTE: this is NOT the full hard-gate mechanical correctness test.")
    print("That test is Block 8B's deliverable, Sep 9. Block 8C (Sep 10) is a")
    print("dedicated buffer/hardening day for this piece, per the updated planner.")

if __name__ == "__main__":
    main()
