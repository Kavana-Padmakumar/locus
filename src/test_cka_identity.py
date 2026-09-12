"""
LOCUS - Block 11A: CKA identity sanity check (hard gate)
DoD: CKA(model, itself) = 1.0 at every layer, within floating-point
tolerance, using the frozen probe set.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from cka import load_probe_set, compute_cka_all_layers
from eval_tent import load_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    images, labels = load_probe_set(device=device)
    print(f"Loaded frozen probe set: {images.shape[0]} images")

    model = load_model(device)

    results = compute_cka_all_layers(model, model, images, device)

    print(f"\nCKA(model, itself) per layer:")
    all_pass = True
    for name, cka_val in results.items():
        close_to_one = abs(cka_val - 1.0) < 1e-6
        all_pass = all_pass and close_to_one
        print(f"  {name}: {cka_val:.10f}  {'PASS' if close_to_one else 'FAIL'}")

    print(f"\n{sum(1 for v in results.values() if abs(v-1.0)<1e-6)}/{len(results)} layers pass identity check.")

    assert all_pass, "CKA identity check failed on at least one layer - do not trust this CKA implementation"
    print("IDENTITY CHECK PASSED: CKA(model, itself) = 1.0 at every layer.")

if __name__ == "__main__":
    main()