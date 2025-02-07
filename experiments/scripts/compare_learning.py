"""Compare standard training vs learning from mistakes on CIFAR-10."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time


class Memory:
    """Buffer for storing examples where model made mistakes"""

    def __init__(self, max_size=5000):
        self.max_size = max_size
        self.buffer = []  # [{inputs, target_class}, ...]

    def add_mistakes(self, inputs, targets):
        """Add examples where model made mistakes"""
        for input, target in zip(inputs, targets):
            exp = {"inputs": input.cpu(), "target_class": target.item()}
            if len(self.buffer) >= self.max_size:
                # Random replacement
                idx = random.randrange(len(self.buffer))
                self.buffer[idx] = exp
            else:
                self.buffer.append(exp)

    def sample_batch(self, batch_size):
        """Sample a batch of examples from memory"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        return {
            "inputs": torch.stack([x["inputs"] for x in batch]),
            "target_classes": torch.tensor([x["target_class"] for x in batch]),
        }

    def __len__(self):
        return len(self.buffer)


class MediumCNN(nn.Module):
    """Medium-sized CNN that can actually learn CIFAR-10"""

    def __init__(self):
        super().__init__()
        # Feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.classifier(x)


class MistakeTask:
    """Task that learns only from mistakes"""

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mistakes_before_dream=20,
    ):
        self.device = device
        self.network = MediumCNN().to(device)  # Use medium CNN
        self.memory = Memory(max_size=1000)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        # Track metrics
        self.mistakes_before_dream = mistakes_before_dream
        self.mistakes_since_dream = 0
        self.total_dreams = 0
        self.total_steps = 0
        self.dream_steps = 0
        self.memory_sizes = []
        self.accuracies = []
        self.times = []
        self.start_time = time.time()

    def train_step(self, inputs, targets):
        """Do inference and dream if enough mistakes accumulated"""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Only do inference
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs)
            pred = outputs.argmax(dim=1)
            mistakes = pred != targets

        # If we made any mistakes, add to memory
        dream_loss = 0.0
        if mistakes.any():
            # Add mistakes to memory
            self.memory.add_mistakes(inputs=inputs[mistakes], targets=targets[mistakes])
            self.mistakes_since_dream += mistakes.sum().item()

            # Dream if we've accumulated enough mistakes
            if self.mistakes_since_dream >= self.mistakes_before_dream:
                dream_loss = self.dream_until_learned()
                self.mistakes_since_dream = 0
                self.total_dreams += 1

        # Track metrics
        self.total_steps += 1
        accuracy = (pred == targets).float().mean().item() * 100
        self.accuracies.append(accuracy)
        self.memory_sizes.append(len(self.memory))
        self.times.append(time.time() - self.start_time)

        return {
            "accuracy": accuracy,
            "memory_size": len(self.memory),
            "dream_loss": dream_loss,
            "mistakes_since_dream": self.mistakes_since_dream,
        }

    def dream_until_learned(
        self, target_acc=90.0, max_steps=50
    ):  # Lower target, fewer steps
        """Dream on memories until we achieve target accuracy"""
        if len(self.memory) == 0:
            return 0.0

        total_loss = 0.0
        steps = 0
        current_acc = 0.0

        while current_acc < target_acc and steps < max_steps:
            # Sample batch from memory
            batch = self.memory.sample_batch(min(32, len(self.memory)))
            inputs = batch["inputs"].to(self.device)
            targets = batch["target_classes"].to(self.device)

            # Train on memories
            self.network.train()
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1
            self.dream_steps += 1

            # Check accuracy less frequently
            if steps % 5 == 0:  # Only check every 5 steps
                current_acc = self.evaluate_memories()

        return total_loss / max(steps, 1)

    def evaluate_memories(self):
        """Evaluate accuracy on all memories"""
        if len(self.memory) == 0:
            return 100.0

        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Evaluate in batches to handle large memory buffers
            batch_size = 100
            for i in range(0, len(self.memory), batch_size):
                batch = self.memory.sample_batch(min(batch_size, len(self.memory) - i))
                inputs = batch["inputs"].to(self.device)
                targets = batch["target_classes"].to(self.device)

                outputs = self.network(inputs)
                pred = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        return 100.0 * correct / total


class StandardTask:
    """Regular training approach"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.network = MediumCNN().to(device)  # Use medium CNN
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        # Track metrics
        self.total_steps = 0
        self.accuracies = []
        self.times = []
        self.start_time = time.time()

    def train_step(self, inputs, targets):
        """Regular training step"""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Train on batch
        self.network.train()
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        # Track metrics
        self.total_steps += 1
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            accuracy = (pred == targets).float().mean().item() * 100
            self.accuracies.append(accuracy)
            self.times.append(time.time() - self.start_time)

        return {
            "accuracy": accuracy,
            "loss": loss.item(),
        }


def evaluate(task, loader):
    """Evaluate accuracy on a data loader"""
    correct = 0
    total = 0

    task.network.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(task.device)
            target = target.to(task.device)
            outputs = task.network(data)
            pred = outputs.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

    return 100.0 * correct / total


def plot_comparison(mistake_task, standard_task, save_dir="results"):
    """Plot comparison metrics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Plot accuracy over time
    plt.figure(figsize=(10, 5))
    plt.plot(mistake_task.times, mistake_task.accuracies, label="Mistakes")
    plt.plot(standard_task.times, standard_task.accuracies, label="Standard")
    plt.title("Training Accuracy over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / "accuracy_vs_time.png")
    plt.close()

    # Plot memory growth for mistake task
    plt.figure(figsize=(10, 5))
    plt.plot(mistake_task.times, mistake_task.memory_sizes)
    plt.title("Memory Growth over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Size")
    plt.grid(True)
    plt.savefig(save_dir / "memory_growth.png")
    plt.close()


def main():
    # Data loading and augmentation
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),  # Add back some augmentation
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

    # Load full datasets
    full_train_set = datasets.CIFAR10(
        "data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        "data", train=False, download=True, transform=transform_test
    )

    # Use 10% of training data for quick testing
    train_size = int(0.1 * len(full_train_set))
    indices = torch.randperm(len(full_train_set))[:train_size]
    train_set = torch.utils.data.Subset(full_train_set, indices)

    # Small batch size but not tiny
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    print("=== Quick CIFAR-10 Training Comparison ===")
    print(f"Using {train_size} training examples")
    print("-" * 50)

    # Initialize both approaches
    mistake_task = MistakeTask()
    standard_task = StandardTask()

    # Train both models
    max_steps = len(train_loader)  # One epoch
    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        # Train both models
        mistake_metrics = mistake_task.train_step(data, target)
        standard_metrics = standard_task.train_step(data, target)

        # Update progress bar
        pbar.set_postfix(
            {
                "Mistakes": f"{mistake_metrics['accuracy']:.1f}%",
                "Standard": f"{standard_metrics['accuracy']:.1f}%",
                "Memory": mistake_metrics["memory_size"],
            }
        )

    # Final evaluation
    mistake_acc = evaluate(mistake_task, test_loader)
    standard_acc = evaluate(standard_task, test_loader)

    print("\nFinal Results:")
    print("-" * 50)
    print(f"{'Method':20} {'Test Acc':>10} {'Steps':>10} {'Time':>10}")
    print("-" * 50)
    print(
        f"{'Standard':20} {standard_acc:10.2f} {standard_task.total_steps:10d} {standard_task.times[-1]:10.1f}s"
    )
    print(
        f"{'Learning Mistakes':20} {mistake_acc:10.2f} {mistake_task.total_steps:10d} {mistake_task.times[-1]:10.1f}s"
    )
    print(f"\nMistakes Method Additional Stats:")
    print(f"Total Dreams: {mistake_task.total_dreams}")
    print(f"Dream Steps: {mistake_task.dream_steps}")
    print(f"Final Memory Size: {len(mistake_task.memory)}")

    # Plot comparisons
    plot_comparison(mistake_task, standard_task)


if __name__ == "__main__":
    main()
