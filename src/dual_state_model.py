"""
LOCUS - Block 8A: Dual-parameter-state forward pass engine
A model wrapper that can execute each BatchNorm layer under either the
source weights (theta) or a TENT-adapted model's weights (theta-prime),
selected per layer, within a single forward pass. This is the engine
later activation-patching experiments (Block 9+) will run against.

Only BatchNorm affine parameters (gamma, beta) differ between theta and
theta-prime, since TENT only updates those (see src/eval_tent.py).
"""

import torch
import torch.nn as nn


class DualStateModel:
    def __init__(self, model, device):
        """model: a freshly-loaded source model (theta)."""
        self.device = device
        self.model = model
        self.model.eval()

        self.theta_params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.theta_params[name] = {
                    "weight": module.weight.data.clone(),
                    "bias": module.bias.data.clone(),
                }

        self.theta_prime_params = None

    def load_theta_prime(self, adapted_model):
        """adapted_model: a model that has been through TENT adaptation -
        its BN affine params ARE theta-prime for that condition."""
        theta_prime = {}
        for name, module in adapted_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                theta_prime[name] = {
                    "weight": module.weight.data.clone(),
                    "bias": module.bias.data.clone(),
                }
        assert set(theta_prime.keys()) == set(self.theta_params.keys()), \
            "theta and theta-prime must come from architecturally identical models"
        self.theta_prime_params = theta_prime

    def forward(self, images, patch_layers):
        """patch_layers: BN layer names to run under theta-prime; every
        other BN layer runs under theta. Restores theta afterward - no
        side effects across calls."""
        assert self.theta_prime_params is not None, "call load_theta_prime() first"
        patch_set = set(patch_layers)

        bn_modules = {name: m for name, m in self.model.named_modules()
                      if isinstance(m, nn.BatchNorm2d)}
        unknown = patch_set - set(bn_modules.keys())
        assert not unknown, f"unknown layer names requested for patching: {unknown}"

        for name, module in bn_modules.items():
            src = self.theta_prime_params[name] if name in patch_set else self.theta_params[name]
            module.weight.data.copy_(src["weight"])
            module.bias.data.copy_(src["bias"])

        with torch.no_grad():
            output = self.model(images.to(self.device))

        for name, module in bn_modules.items():
            module.weight.data.copy_(self.theta_params[name]["weight"])
            module.bias.data.copy_(self.theta_params[name]["bias"])

        return output

    def bn_layer_names(self):
        return list(self.theta_params.keys())