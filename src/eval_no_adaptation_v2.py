"""
LOCUS - Block 5B: No-adaptation baseline, using config loader + structured logger.
Re-runs Block 5A's evaluation, now driven entirely by configs/corruption_matrix.yaml
and writing through ResultsLogger instead of ad-hoc CSV writing.
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(__file__))
from cifar10c_dataset import CIFAR10C
from config_loader import load_config
from results_logger import ResultsLogger


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

    cfg = load_config()
    model = load_model(device)
    logger = ResultsLogger("results/baseline_results_v2.csv", "results/baseline_results_v2.jsonl")

    count = 0
    for corruption in cfg["corruptions"]:
        for severity in cfg["severities"]:
            print(f"Evaluating: {corruption}, severity={severity}...")
            acc, err, n = evaluate_condition(model, device, corruption, severity)
            print(f"  -> accuracy={acc:.2f}%  error={err:.2f}%  n={n}")
            logger.log("no_adaptation", corruption, severity, None, acc, err, n)
            count += 1

    print(f"\nDone. {count} conditions logged via config-driven pipeline.")