"""
LOCUS - Block 11A: Linear CKA computation across normalization layers
Uses the frozen probe set (data/cka_probe_set.pt) so every CKA value
computed anywhere in this project is comparable to every other.
Reuses Block 8's activation-extraction pattern (forward hooks on
BatchNorm2d layers), matching how dual_state_model.py identifies layers.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn


def load_probe_set(path="data/cka_probe_set.pt", device="cpu"):
    data = torch.load(path, map_location=device)
    return data["images"].to(device), data["labels"].to(device)


def get_activations(model, images, device):
    """Runs one forward pass, capturing the output of every BatchNorm2d
    layer via forward hooks - the same layer set Block 8's DualStateModel
    tracks."""
    activations = {}
    handles = []

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(images.to(device))

    for h in handles:
        h.remove()

    return activations


def linear_cka(X, Y):
    """Linear CKA between two activation tensors, each (n_images, ...).
    Uses the Gram-matrix formulation (images x images), NOT the raw
    feature-covariance formulation - the latter blows up to gigabytes
    of memory for early layers with large spatial x channel dimensions."""
    X = X.reshape(X.shape[0], -1).double()
    Y = Y.reshape(Y.shape[0], -1).double()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.t()   # n x n, NOT feature x feature
    L = Y @ Y.t()   # n x n

    hsic = (K * L).sum()
    norm_x = torch.sqrt((K * K).sum())
    norm_y = torch.sqrt((L * L).sum())

    denominator = norm_x * norm_y
    if denominator < 1e-12:
        return float("nan")
    return (hsic / denominator).item()

def compute_cka_all_layers(model_a, model_b, images, device):
    """CKA per BatchNorm layer between two models on the same probe images."""
    acts_a = get_activations(model_a, images, device)
    acts_b = get_activations(model_b, images, device)

    assert set(acts_a.keys()) == set(acts_b.keys()), \
        "model_a and model_b must have identical BatchNorm layer names"

    results = {}
    for name in acts_a:
        results[name] = linear_cka(acts_a[name], acts_b[name])
    return results