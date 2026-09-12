"""
LOCUS - Block 11A: Freeze the CKA probe set
Runs ONCE. Selects a fixed, deterministic set of clean CIFAR-10 test
images to be reused for every CKA computation in every later block -
this is what makes CKA values comparable across blocks/conditions.
Saved to data/cka_probe_set.pt and version-controlled (committed to git).
"""
import torch
import torchvision
import torchvision.transforms as transforms

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2471, 0.2435, 0.2616]
N_PROBE_IMAGES = 512
SEED = 42   # frozen forever - never change this once committed

def main():
    torch.manual_seed(SEED)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(testset), generator=generator)[:N_PROBE_IMAGES]

    images = torch.stack([testset[i][0] for i in indices])
    labels = torch.tensor([testset[i][1] for i in indices])

    torch.save({"images": images, "labels": labels, "indices": indices, "seed": SEED},
               "data/cka_probe_set.pt")

    print(f"Probe set frozen: {images.shape[0]} images, shape {tuple(images.shape[1:])}")
    print(f"Saved to data/cka_probe_set.pt")
    print(f"Seed used: {SEED} - this must NEVER change once committed, or every future")
    print("CKA comparison across blocks becomes invalid.")

if __name__ == "__main__":
    main()