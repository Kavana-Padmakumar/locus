"""
LOCUS - Block 7A: TENT implementation
Entropy-minimization test-time adaptation (Wang et al., ICLR 2021).
Only BatchNorm affine parameters (gamma, beta) are updated via gradient
descent, one step per batch, using only that batch's own entropy loss -
no labels used. BN layers use batch statistics (train-mode), same as
BN-Adapt, plus one gradient step on affine params per batch.

Each corruption/severity condition starts from a FRESH pretrained model
and optimizer - adaptation does not carry over between conditions. This
matches the single-domain TENT protocol (not continual TTA).

Writes to results/tent_results.csv + .jsonl - kept separate from all
prior result files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys, os

sys.path.append(os.path.dirname(__file__))
from cifar10c_dataset import CIFAR10C
from config_loader import load_config
from results_logger import ResultsLogger


def load_model(device):
    from cifar10_models.resnet import resnet18
    model = resnet18(pretrained=True)
    return model.to(device)


def configure_tent_model(model):
    """Freeze everything except BatchNorm affine params (gamma/beta)."""
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)

    trainable_params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.requires_grad_(True)   # gamma
            module.bias.requires_grad_(True)      # beta
            trainable_params += [module.weight, module.bias]

    return trainable_params


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


def evaluate_tent(model, optimizer, device, corruption, severity):
    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = entropy_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    error_rate = 100 - accuracy
    return accuracy, error_rate, total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    cfg = load_config()
    logger = ResultsLogger("results/tent_results.csv", "results/tent_results.jsonl")

    count = 0
    for corruption in cfg["corruptions"]:
        for severity in cfg["severities"]:
            model = load_model(device)
            trainable_params = configure_tent_model(model)
            optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

            if count == 0:
                n_trainable = sum(p.numel() for p in trainable_params)
                n_total = sum(p.numel() for p in model.parameters())
                print(f"[Sanity check] Trainable params: {n_trainable} / {n_total} total "
                      f"({'OK - small fraction' if n_trainable < n_total * 0.01 else 'WARNING - too many trainable params, check freezing logic'})")

            print(f"Evaluating: {corruption}, severity={severity}...")
            acc, err, n = evaluate_tent(model, optimizer, device, corruption, severity)
            print(f"  -> accuracy={acc:.2f}%  error={err:.2f}%  n={n}")
            logger.log("tent", corruption, severity, None, acc, err, n)
            count += 1

    print(f"\nDone. {count} conditions logged via TENT pipeline.")