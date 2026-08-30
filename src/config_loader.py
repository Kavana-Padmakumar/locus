"""
LOCUS - Block 5B: Config loader
Loads the frozen experimental matrix from configs/corruption_matrix.yaml
so no script hardcodes corruption/severity/seed lists directly.
"""

import yaml


def load_config(path="configs/corruption_matrix.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    cfg = load_config()
    print("Corruptions:", cfg["corruptions"])
    print("Severities:", cfg["severities"])
    print("Seeds:", cfg["seeds"])
    print("Methods:", cfg["methods"])