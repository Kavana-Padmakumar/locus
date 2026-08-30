"""
LOCUS - Block 4B: CIFAR-10-C data loader determinism check
Confirms two separate DataLoader runs with the same seed produce
identical batches for a CIFAR-10-C corruption.
"""

import torch
from torch.utils.data import DataLoader
from cifar10c_dataset import CIFAR10C

def get_first_batch(seed):
    torch.manual_seed(seed)
    ds = CIFAR10C(data_dir="data_raw", corruption="gaussian_noise", severity=3)
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0,
                         generator=torch.Generator().manual_seed(seed))
    images, labels = next(iter(loader))
    return images, labels

if __name__ == "__main__":
    images1, labels1 = get_first_batch(seed=42)
    images2, labels2 = get_first_batch(seed=42)

    images_match = torch.equal(images1, images2)
    labels_match = torch.equal(labels1, labels2)

    print(f"Images identical across runs: {images_match}")
    print(f"Labels identical across runs: {labels_match}")
    assert images_match and labels_match, "CIFAR-10-C loader is NOT deterministic"
    print("PASS: CIFAR-10-C loader produces identical batches with the same seed")