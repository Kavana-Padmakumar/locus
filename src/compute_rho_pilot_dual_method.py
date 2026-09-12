"""
LOCUS - Block 10B: Pilot rho for BOTH BN-Adapt and TENT
36 runs = 2 methods x 2 pilot corruptions (gaussian_noise, contrast) x
3 severities x 3 seeds. Sanity-checks whether rho stays in a plausible
range - Block 10A found rho blows up when delta_R_TTA is small; these
2 pilot corruptions were chosen specifically because TENT shows large,
stable improvements on them (per Block 7B's validation), so today's
narrower pilot should avoid that instability.
"""
import sys, os, csv, statistics
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cifar10c_dataset import CIFAR10C
from h_rec_control import load_model as load_model_hrec, fit_h_rec

PILOT_CORRUPTIONS = ["gaussian_noise", "contrast"]
SEVERITIES = [1, 3, 5]
SEEDS = [0, 1, 2]


def load_baseline_errors(path):
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[(row["corruption"], row["severity"])] = float(row["error_rate"])
    return rows


def get_batch(corruption, severity):
    ds = CIFAR10C(data_dir="data_raw", corruption=corruption, severity=severity)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    return next(iter(loader))


def error_from_logits(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return 100 - (100 * correct / labels.size(0))


def fit_bn_adapt_seeded(device, corruption, severity, seed):
    """BN-Adapt: no gradient step, just batch-statistic substitution.
    Seed is a formality here - no optimizer state, no randomness exists
    in this forward-only method, so results will be identical across seeds."""
    torch.manual_seed(seed)
    from eval_bn_adapt import load_model as load_model_bn
    model = load_model_bn(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    images, labels = get_batch(corruption, severity)
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
    return error_from_logits(outputs, labels)


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
    torch.manual_seed(seed)
    from eval_tent import load_model as load_model_tent
    model = load_model_tent(device)
    trainable = configure_tent_model(model)
    optimizer = torch.optim.Adam(trainable, lr=1e-3)

    images, labels = get_batch(corruption, severity)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    loss = entropy_loss(outputs)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        return error_from_logits(outputs, labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    baseline = load_baseline_errors("results/baseline_results_v2.csv")
    hrec_model = load_model_hrec(device)

    out_path = "results/pilot_rho_dual_method_results.csv"
    fieldnames = ["method", "corruption", "severity", "seed", "r_source", "r_adapted",
                  "r_hrec", "delta_r_tta", "delta_r_rec", "rho", "note"]
    with open(out_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    rows = []
    for corruption in PILOT_CORRUPTIONS:
        for severity in SEVERITIES:
            r_source = baseline[(corruption, str(severity))]
            for seed in SEEDS:
                _, r_hrec, _, _ = fit_h_rec(hrec_model, device, corruption, severity, seed=seed)
                delta_rec = r_source - r_hrec

                for method, fit_fn in [("bn_adapt", fit_bn_adapt_seeded), ("tent", fit_tent_seeded)]:
                    r_adapted = fit_fn(device, corruption, severity, seed)
                    delta_tta = r_source - r_adapted

                    note = ""
                    if abs(delta_tta) < 1e-6:
                        rho = None
                        note = "delta_R_TTA ~ 0, rho undefined"
                    else:
                        rho = (delta_tta - delta_rec) / delta_tta

                    row = {
                        "method": method, "corruption": corruption, "severity": severity,
                        "seed": seed, "r_source": round(r_source, 2),
                        "r_adapted": round(r_adapted, 2), "r_hrec": round(r_hrec, 2),
                        "delta_r_tta": round(delta_tta, 2), "delta_r_rec": round(delta_rec, 2),
                        "rho": round(rho, 4) if rho is not None else "", "note": note,
                    }
                    rows.append(row)
                    print(f"{method:<9} {corruption} sev{severity} seed={seed}: "
                          f"R_src={r_source:.2f} R_adapted={r_adapted:.2f} R_hrec={r_hrec:.2f} "
                          f"rho={row['rho']}{'  (' + note + ')' if note else ''}")

    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)

    rho_vals = [r["rho"] for r in rows if r["rho"] != ""]
    print(f"\n{len(rows)} runs computed ({len(rho_vals)} with defined rho).")
    print(f"rho range: min={min(rho_vals):.4f}  max={max(rho_vals):.4f}  "
          f"mean={statistics.mean(rho_vals):.4f}  median={statistics.median(rho_vals):.4f}")

    plausible = max(rho_vals) < 5 and min(rho_vals) > -5
    print(f"Plausible range (roughly within [-5, 5], not blown up like Block 10A's full grid): {plausible}")

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()