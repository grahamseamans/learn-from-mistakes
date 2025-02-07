"""Train with memory-based learning on CIFAR-10.

The approach:
1. Run inference most of the time
2. When we get predictions wrong, store those examples
3. After N mistakes, enter training mode on those examples
4. Train until we achieve target accuracy
5. Return to inference mode
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import random


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


class Memory:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.images = []
        self.labels = []
        self.is_new = []  # Track which examples are from recent mistakes

    def add(self, images, labels):
        """Add misclassified examples to memory with random replacement when full"""
        for img, label in zip(images, labels):
            if len(self.images) >= self.max_size:
                # Random replacement
                idx = random.randrange(len(self.images))
                self.images[idx] = img.cpu()
                self.labels[idx] = label.cpu()
                self.is_new[idx] = True
            else:
                self.images.append(img.cpu())
                self.labels.append(label.cpu())
                self.is_new.append(True)

    def get_training_batch(self, batch_size=32, new_ratio=0.8):
        """Get a batch with mostly new examples and some old ones"""
        if len(self) == 0:
            return None, None

        # How many new examples to include
        n_new = int(batch_size * new_ratio)
        n_old = batch_size - n_new

        # Get indices of new and old examples
        new_indices = [i for i, is_new in enumerate(self.is_new) if is_new]
        old_indices = [i for i, is_new in enumerate(self.is_new) if not is_new]

        # If we don't have enough new/old, adjust ratios
        if len(new_indices) < n_new:
            n_new = len(new_indices)
            n_old = min(batch_size - n_new, len(old_indices))
        if len(old_indices) < n_old:
            n_old = len(old_indices)
            n_new = min(batch_size - n_old, len(new_indices))

        # Sample indices
        batch_indices = []
        if n_new > 0:
            batch_indices.extend(random.sample(new_indices, n_new))
        if n_old > 0:
            batch_indices.extend(random.sample(old_indices, n_old))
        random.shuffle(batch_indices)

        return (
            torch.stack([self.images[i] for i in batch_indices]),
            torch.stack([self.labels[i] for i in batch_indices]),
        )

    def mark_all_old(self):
        """Mark all examples as old after training"""
        self.is_new = [False] * len(self.images)

    def __len__(self):
        return len(self.images)


def train_on_memory(model, memory, device, target_acc=0.95, max_steps=50):
    """Train primarily on new examples with some old ones mixed in"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for step in range(max_steps):
        # Get a batch of examples (80% new, 20% old)
        inputs, targets = memory.get_training_batch(batch_size=128, new_ratio=0.8)
        if inputs is None:  # Empty memory
            break

        inputs = inputs.clone().detach().requires_grad_(True)
        inputs, targets = inputs.to(device), targets.to(device)

        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Check accuracy
        with torch.no_grad():
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(targets).float().mean().item()
            if accuracy >= target_acc:
                break

    # Mark all examples as old after training
    memory.mark_all_old()
    model.eval()


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

    # Create model
    model = SimpleNet().to(device)  # Using our simple network instead of ResNet18
    memory = Memory(max_size=50000)
    mistakes_before_training = 32
    min_memory_size = 128
    mistakes_since_training = 0
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print("Starting training...")
    for epoch in range(10):
        model.eval()  # Start in eval mode
        running_loss = 0
        correct = 0
        total = 0
        mistakes_since_training = 0  # Reset mistakes counter each epoch

        pbar = tqdm(train_loader, desc=f"E{epoch}", ncols=80)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Run inference
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            mistakes = ~predicted.eq(targets)

            # If we made mistakes, add them to memory
            if mistakes.any():
                memory.add(inputs[mistakes], targets[mistakes])
                mistakes_since_training += mistakes.sum().item()

                # If we've accumulated enough mistakes and have enough memory, train
                if (
                    mistakes_since_training >= mistakes_before_training
                    and len(memory) >= min_memory_size
                ):
                    train_on_memory(model, memory, device)
                    mistakes_since_training = 0
                    model.eval()  # Back to eval mode

            # Update stats
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            running_loss = 0.9 * running_loss + 0.1 * loss.item()

            # Progress bar
            pbar.set_postfix_str(
                f"m:{len(memory)} mi:{mistakes_since_training} a:{100.0*correct/total:.1f}%"
            )

        # Final evaluation
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        print(f"\nResults:")
        print(f"Train: {train_acc:.1f}%")
        print(f"Test: {test_acc:.1f}%")
        print(f"Memory: {len(memory)}")

    # Save model
    torch.save(model.state_dict(), "models/simplenet_memory.pt")
    print("Model saved to models/simplenet_memory.pt")


if __name__ == "__main__":
    main()
