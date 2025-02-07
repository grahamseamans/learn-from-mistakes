"""Train SimpleNet on CIFAR-10 with standard training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class SimpleNet(nn.Module):
    """Simple CNN for CIFAR-10 classification"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Feature extraction - more filters for color images
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # Classification head
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    """Compute accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(
        "data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        "data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True
    )

    # Create and train model
    print("Training SimpleNet...")
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"E{epoch}", ncols=80)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update stats
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Progress bar
            pbar.set_postfix_str(f"l:{running_loss:.2f} a:{100.0*correct/total:.1f}%")

        # Epoch results
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"\nResults:")
        print(f"Train: {train_acc:.1f}%")
        print(f"Test: {test_acc:.1f}%")

    # Save model
    torch.save(model.state_dict(), "models/simplenet_standard.pt")
    print("Model saved to models/simplenet_standard.pt")


if __name__ == "__main__":
    main()
