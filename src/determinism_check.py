"""
LOCUS - Block 4A: Data loader determinism check
Confirms two separate DataLoader runs with the same seed produce identical batches.
"""

import torch
import torchvision
import torchvision.transforms as transforms

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2471, 0.2435, 0.2616]

def get_first_batch(seed):
    torch.manual_seed(seed)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    images, labels = next(iter(loader))
    return images, labels

if __name__ == "__main__":
    images1, labels1 = get_first_batch(seed=42)
    images2, labels2 = get_first_batch(seed=42)

    images_match = torch.equal(images1, images2)
    labels_match = torch.equal(labels1, labels2)

    print(f"Images identical across runs: {images_match}")
    print(f"Labels identical across runs: {labels_match}")
    assert images_match and labels_match, "Data loader is NOT deterministic with a fixed seed"
    print("PASS: data loader produces identical batches with the same seed")