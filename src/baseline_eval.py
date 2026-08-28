"""
LOCUS - Block 4A: Verified clean baseline
Loads pretrained ResNet-18 (huyvnphan/PyTorch_CIFAR10) and confirms clean-test
accuracy on CIFAR-10 matches the published 93.07% within 1%.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from cifar10_models.resnet import resnet18

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2471, 0.2435, 0.2616]
PUBLISHED_ACC = 93.07

def load_model(device):
    model = resnet18(pretrained=True)
    model.eval()
    return model.to(device)

def evaluate(model, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    acc = evaluate(model, device)
    within_tolerance = abs(acc - PUBLISHED_ACC) <= 1.0
    print(f"Clean test accuracy: {acc:.2f}%")
    print(f"Published: {PUBLISHED_ACC}% | Within 1%: {within_tolerance}")
    assert within_tolerance, "Baseline accuracy outside 1% tolerance - do not proceed to Block 5"