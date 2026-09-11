"""
LOCUS - Block 9A: Boundary-only counterfactual control (h_rec)

Freezes the backbone ENTIRELY - unlike BN-Adapt/TENT, this model runs in
eval() mode with its original running statistics, so g_theta(x) produces
the exact same frozen logits every time for a given input. Only a
per-class diagonal affine transform (a*z + b) on top of those frozen
logits is fit via gradient descent, using the same entropy-minimization
loss TENT uses. Capacity-matched to TENT/BN-Adapt (same loss, same
optimizer, same one-step-per-batch convention) but can only move decision
boundaries - it cannot touch representations at all.

IMPORTANT: do not copy the .train() pattern from eval_bn_adapt.py /
eval_tent.py here. Using train() mode would let the backbone's BN layers
respond to batch statistics, letting representations change - defeating
the entire point of a "boundary-only" control.
"""
import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cifar10c_dataset import CIFAR10C


def load_model(device):
    from cifar10_models.resnet import resnet18
    model = resnet18(pretrained=True)
    model.eval()   # frozen, deterministic - NOT train() (see module docstring)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


class HRec(nn.Module):
    """Diagonal affine recalibration: h_rec(z) = a * z + b, per-class."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_classes))
        self.b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, z):
        return self.a * z + self.b


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


def fit_h_rec(model, device, corruption, severity, seed=0):
    """Fits h_rec on one batch, matching TENT's one-gradient-step-per-batch
    convention. Backbone stays frozen throughout - only a, b get gradients."""
    torch.manual_seed(seed)

    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    h_rec = HRec(n_classes=10).to(device)
    optimizer = torch.optim.Adam(h_rec.parameters(), lr=1e-3)

    with torch.no_grad():
        z = model(images)   # frozen logits - computed once, backbone never touched again

    recalibrated = h_rec(z)
    loss = entropy_loss(recalibrated)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        final_logits = h_rec(z)
        _, predicted = torch.max(final_logits, 1)
        correct = (predicted == labels).sum().item()
        accuracy = 100 * correct / labels.size(0)
        error_rate = 100 - accuracy

    return h_rec, accuracy, error_rate, labels.size(0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    print("Block 9A smoke test - seed=0 only. Full 3-seed validation is Block 9B's job.\n")

    model = load_model(device)
    pilot_conditions = [("gaussian_noise", 5), ("contrast", 5)]

    for corruption, severity in pilot_conditions:
        h_rec, acc, err, n = fit_h_rec(model, device, corruption, severity, seed=0)
        a_vals = h_rec.a.detach().cpu().numpy()
        b_vals = h_rec.b.detach().cpu().numpy()
        a_has_nan = bool(torch.isnan(h_rec.a).any())
        b_has_nan = bool(torch.isnan(h_rec.b).any())
        print(f"{corruption}, severity={severity}:")
        print(f"  accuracy={acc:.2f}%  error={err:.2f}%  n={n}")
        print(f"  a range: [{a_vals.min():.4f}, {a_vals.max():.4f}]  (NaN present: {a_has_nan})")
        print(f"  b range: [{b_vals.min():.4f}, {b_vals.max():.4f}]  (NaN present: {b_has_nan})")
        print()

    print("Block 9A smoke test complete. h_rec builds and runs without crashing.")
    print("Full validation (3 seeds, NaN/divergence/degeneracy checks) is Block 9B, Sep 12.")