"""
LOCUS - Block 8C: Dual-parameter-state engine hardening tests
Defense-in-depth testing beyond Block 8B's single hard-gate assertion.
Block 8B already passed (after fixing a train/eval mode bug) - this block
adds tests anyway, per the plan's own instruction, since an engine this
central to every later phase deserves more than one passing assertion.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import random
import torch
from torch.utils.data import DataLoader
from cifar10c_dataset import CIFAR10C
from eval_tent import load_model, configure_tent_model, entropy_loss
from dual_state_model import DualStateModel

PILOT_CORRUPTION = "defocus_blur"   # a THIRD condition, not reused from 8A/8B
PILOT_SEVERITY = 3


def build_engine_and_batch(device, corruption, severity):
    theta_model = load_model(device)
    engine = DualStateModel(theta_model, device)

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

    engine.load_theta_prime(adapted_model)
    return engine, images


def test_repeated_call_determinism(device):
    engine, images = build_engine_and_batch(device, PILOT_CORRUPTION, PILOT_SEVERITY)
    all_layers = engine.bn_layer_names()
    random.seed(0)
    subset = random.sample(all_layers, k=len(all_layers) // 2)

    theta_before = {k: v["weight"].clone() for k, v in engine.theta_params.items()}

    out1 = engine.forward(images, patch_layers=subset)
    out2 = engine.forward(images, patch_layers=subset)

    outputs_match = torch.allclose(out1, out2, atol=1e-6)
    theta_unchanged = all(
        torch.allclose(theta_before[k], engine.theta_params[k]["weight"])
        for k in theta_before
    )
    passed = outputs_match and theta_unchanged
    print(f"[Test 1] Repeated identical-subset calls match: {outputs_match}, "
          f"theta left unchanged: {theta_unchanged} -> {'PASS' if passed else 'FAIL'}")
    return passed


def test_duplicate_layer_names_in_patch_list(device):
    engine, images = build_engine_and_batch(device, PILOT_CORRUPTION, PILOT_SEVERITY)
    one_layer = engine.bn_layer_names()[0]

    out_once = engine.forward(images, patch_layers=[one_layer])
    out_duplicated = engine.forward(images, patch_layers=[one_layer, one_layer, one_layer])

    passed = torch.allclose(out_once, out_duplicated, atol=1e-6)
    print(f"[Test 2] Duplicate layer names in patch list are harmless: {passed} -> "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


def test_unknown_layer_name_raises(device):
    engine, images = build_engine_and_batch(device, PILOT_CORRUPTION, PILOT_SEVERITY)
    raised = False
    try:
        engine.forward(images, patch_layers=["this_layer_does_not_exist"])
    except AssertionError:
        raised = True
    print(f"[Test 3] Unknown layer name correctly raises AssertionError: {raised} -> "
          f"{'PASS' if raised else 'FAIL'}")
    return raised


def test_partial_patch_differs_from_both_extremes(device):
    engine, images = build_engine_and_batch(device, PILOT_CORRUPTION, PILOT_SEVERITY)
    all_layers = engine.bn_layer_names()
    half = all_layers[:len(all_layers) // 2]

    zero_out = engine.forward(images, patch_layers=[])
    full_out = engine.forward(images, patch_layers=all_layers)
    half_out = engine.forward(images, patch_layers=half)

    differs_from_zero = not torch.allclose(half_out, zero_out, atol=1e-6)
    differs_from_full = not torch.allclose(half_out, full_out, atol=1e-6)
    passed = differs_from_zero and differs_from_full
    print(f"[Test 4] Partial patch differs from zero-patch: {differs_from_zero}, "
          f"differs from full-patch: {differs_from_full} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    print(f"Pilot condition for these tests: {PILOT_CORRUPTION}, severity={PILOT_SEVERITY} "
          f"(a third condition, distinct from Block 8A/8B's gaussian_noise/contrast)\n")

    results = [
        test_repeated_call_determinism(device),
        test_duplicate_layer_names_in_patch_list(device),
        test_unknown_layer_name_raises(device),
        test_partial_patch_differs_from_both_extremes(device),
    ]

    n_passed = sum(results)
    print(f"\n{n_passed}/{len(results)} hardening tests passed.")

    assert all(results), "At least one hardening test failed - fix before proceeding"
    print("All Block 8C hardening tests PASSED.")


if __name__ == "__main__":
    main()