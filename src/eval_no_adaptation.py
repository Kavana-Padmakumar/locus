"""
LOCUS - Block 5A: No-adaptation baseline
Runs the frozen, unmodified ResNet-18 (no TTA applied) on every corruption x
severity in the MUST-scope matrix, records error rate for each condition.
This is the reference row every BN-Adapt/TENT result gets compared against.
"""

import csv
import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(__file__))
from cifar10c_dataset import CIFAR10C, FROZEN_CORRUPTIONS

SEVERITIES = [1, 3, 5]
RESULTS_PATH = "results/baseline_results.csv"


def load_model(device):
    from cifar10_models.resnet import resnet18
    model = resnet18(pretrained=True)
    model.eval()
    return model.to(device)


def evaluate_condition(model, device, corruption, severity):
    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    correct, total = 0, 0
    with torch.no_grad():
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

    model = load_model(device)

    os.makedirs("results", exist_ok=True)
    rows = []

    for corruption in FROZEN_CORRUPTIONS:
        for severity in SEVERITIES:
            print(f"Evaluating: {corruption}, severity={severity}...")
            acc, err, n = evaluate_condition(model, device, corruption, severity)
            print(f"  -> accuracy={acc:.2f}%  error={err:.2f}%  n={n}")
            rows.append({
                "method": "no_adaptation",
                "corruption": corruption,
                "severity": severity,
                "accuracy": round(acc, 2),
                "error_rate": round(err, 2),
                "n_samples": n,
            })

    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "corruption", "severity", "accuracy", "error_rate", "n_samples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results written to {RESULTS_PATH}")
    print(f"Total conditions evaluated: {len(rows)}")