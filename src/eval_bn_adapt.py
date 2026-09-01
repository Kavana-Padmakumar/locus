"""
LOCUS - Block 6A: BN-Adapt implementation
Recomputes BatchNorm statistics using each target (corrupted) batch's own
statistics at test time. No gradient steps, no weight updates - only BN
behaves differently (train-mode statistics), everything else stays frozen.
Writes to results/bn_adapt_results.csv + .jsonl - kept separate from the
Block 5 baseline files to avoid any risk of overwriting verified data.
"""

import torch
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


def evaluate_bn_adapt(model, device, corruption, severity):
    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    model.train()                    # BN layers use THIS batch's stats, not running stats
    for p in model.parameters():
        p.requires_grad_(False)      # freeze weights - no backprop, no optimizer step

    correct, total = 0, 0
    with torch.no_grad():            # belt-and-braces - no gradients computed at all
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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
    model = load_model(device)
    logger = ResultsLogger("results/bn_adapt_results.csv", "results/bn_adapt_results.jsonl")

    count = 0
    for corruption in cfg["corruptions"]:
        for severity in cfg["severities"]:
            print(f"Evaluating: {corruption}, severity={severity}...")
            acc, err, n = evaluate_bn_adapt(model, device, corruption, severity)
            print(f"  -> accuracy={acc:.2f}%  error={err:.2f}%  n={n}")
            logger.log("bn_adapt", corruption, severity, None, acc, err, n)
            count += 1

    print(f"\nDone. {count} conditions logged via BN-Adapt pipeline.")