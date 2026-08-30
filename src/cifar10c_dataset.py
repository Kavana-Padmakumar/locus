"""
LOCUS - CIFAR-10-C deterministic data loader
Loads a specific corruption + severity combination from the official
Hendrycks & Dietterich CIFAR-10-C release. Images are ordered in blocks of
10,000 by severity (1 through 5); this class slices out the requested
severity block and applies the same normalization used for the clean
ResNet-18 baseline.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2471, 0.2435, 0.2616]

FROZEN_CORRUPTIONS = [
    "gaussian_noise", "defocus_blur", "snow",
    "brightness", "contrast", "jpeg_compression",
]

class CIFAR10C(Dataset):
    def __init__(self, data_dir, corruption, severity):
        assert corruption in FROZEN_CORRUPTIONS, f"{corruption} not in frozen MUST-scope set"
        assert severity in (1, 2, 3, 4, 5), "severity must be 1-5"

        images = np.load(f"{data_dir}/{corruption}.npy")
        labels = np.load(f"{data_dir}/labels.npy")

        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = images[start:end]
        self.labels = labels[start:end]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        label = int(self.labels[idx])
        return img, label


if __name__ == "__main__":
    # Quick sanity check when run directly
    ds = CIFAR10C(data_dir="data_raw", corruption="gaussian_noise", severity=3)
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"First image shape: {img.shape}, label: {label}")