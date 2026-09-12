"""
LOCUS - Block 10A: Pilot rho computation
rho = (delta_R_TTA - delta_R_rec) / delta_R_TTA
  delta_R_TTA = R(source) - R(TENT-adapted)      [full TTA's error reduction]
  delta_R_rec = R(source) - R(h_rec-adapted)     [boundary-only's error reduction]
rho close to 1 -> boundary-only reproduces almost none of TTA's gain (representation-driven)
rho close to 0 -> boundary-only reproduces almost all of TTA's gain (boundary-driven)

Scope note: the planner's DoD text says "36 conditions," but the frozen
corruption_matrix.yaml specifies 6 corruptions x 3 severities x 3 seeds =
54. Rather than guess which 18 to drop, this script computes the full 54
- a superset of whatever "36" was intended to mean.
"""
import sys, os, csv
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cifar10c_dataset import CIFAR10C
from config_loader import load_config
from h_rec_control import load_model as load_model_hrec, fit_h_rec


def load_baseline_errors(path):
    """R(source): no-adaptation baseline error rate per (corruption, severity)."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption"], row["severity"])] = float(row["error_rate"])
    return rows


def configure_tent_model(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)
            trainable += [m.weight, m.bias]
    return trainable


def entropy_loss(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


def fit_tent_seeded(device, corruption, severity, seed):
    """TENT fit for one condition, with an explicit seed - built fresh here
    since eval_tent.py (Block 7A) does not loop over seeds."""
    torch.manual_seed(seed)
    from eval_tent import load_model as load_model_tent
    model = load_model_tent(device)
    trainable = configure_tent_model(model)
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = entropy_loss(outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        error_rate = 100 - (100 * correct / labels.size(0))
    return error_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    cfg = load_config()
    baseline = load_baseline_errors("results/baseline_results_v2.csv")
    hrec_model = load_model_hrec(device)

    out_path = "results/pilot_rho_results.csv"
    fieldnames = ["corruption", "severity", "seed", "r_source", "r_tent", "r_hrec",
                  "delta_r_tta", "delta_r_rec", "rho", "note"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    rows = []
    n_skipped = 0
    for corruption in cfg["corruptions"]:
        for severity in cfg["severities"]:
            r_source = baseline[(corruption, str(severity))]
            for seed in cfg["seeds"]:
                r_tent = fit_tent_seeded(device, corruption, severity, seed)
                _, r_hrec, _, _ = fit_h_rec(hrec_model, device, corruption, severity, seed=seed)

                delta_tta = r_source - r_tent
                delta_rec = r_source - r_hrec

                note = ""
                if abs(delta_tta) < 1e-6:
                    rho = None
                    note = "delta_R_TTA ~ 0, rho undefined - skipped"
                    n_skipped += 1
                else:
                    rho = (delta_tta - delta_rec) / delta_tta

                row = {
                    "corruption": corruption, "severity": severity, "seed": seed,
                    "r_source": round(r_source, 2), "r_tent": round(r_tent, 2),
                    "r_hrec": round(r_hrec, 2), "delta_r_tta": round(delta_tta, 2),
                    "delta_r_rec": round(delta_rec, 2),
                    "rho": round(rho, 4) if rho is not None else "",
                    "note": note,
                }
                rows.append(row)
                print(f"{corruption} sev{severity} seed={seed}: "
                      f"R_src={r_source:.2f} R_tent={r_tent:.2f} R_hrec={r_hrec:.2f} "
                      f"rho={row['rho']}{'  (' + note + ')' if note else ''}")

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)

    print(f"\n{len(rows)} conditions computed, {n_skipped} skipped (delta_R_TTA ~ 0).")

    # Spread across seeds, per (corruption, severity) - what the DoD asks about
    print("\nSpread across seeds (std dev of rho), per condition:")
    import statistics
    seen = set()
    for row in rows:
        key = (row["corruption"], row["severity"])
        if key in seen or row["rho"] == "":
            continue
        seen.add(key)
        vals = [r["rho"] for r in rows if (r["corruption"], r["severity"]) == key and r["rho"] != ""]
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {key[0]} sev{key[1]}: rho values={vals}  std={std:.6f}")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()