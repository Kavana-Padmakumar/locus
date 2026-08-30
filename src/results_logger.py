"""
LOCUS - Block 5B: Structured results logger
Writes experiment results to both CSV and JSONL, keyed by
(method, corruption, severity, seed). This is the reproducibility
backbone every future experiment writes through.
"""

import csv
import json
import os
from datetime import datetime, timezone


class ResultsLogger:
    def __init__(self, csv_path, jsonl_path):
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        self.fieldnames = [
            "method", "corruption", "severity", "seed",
            "accuracy", "error_rate", "n_samples", "timestamp",
        ]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Write CSV header if the file doesn't exist yet
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, method, corruption, severity, seed, accuracy, error_rate, n_samples):
        row = {
            "method": method,
            "corruption": corruption,
            "severity": severity,
            "seed": seed if seed is not None else "",
            "accuracy": round(accuracy, 2),
            "error_rate": round(error_rate, 2),
            "n_samples": n_samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        return row


if __name__ == "__main__":
    # Quick sanity check
    logger = ResultsLogger("results/test_log.csv", "results/test_log.jsonl")
    logger.log("no_adaptation", "gaussian_noise", 3, None, 63.13, 36.87, 10000)
    print("Wrote one test row - check results/test_log.csv and results/test_log.jsonl")