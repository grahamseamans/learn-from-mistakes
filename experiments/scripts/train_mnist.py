"""Train MNIST model with memory-based continual learning - Split MNIST benchmark."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from tqdm import tqdm
import numpy as np


class Net(nn.Module):
    """Single network for MNIST classification"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Feature extraction
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            # Classification head
            nn.Linear(64 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # Back to 10 classes - these are digits!
        )

    def forward(self, x):
        return self.net(x)


class Memory:
    """Buffer for storing examples where model made mistakes"""

    def __init__(self, max_size=2000):
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


class Task:
    """Task with memory-based learning"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.network = Net().to(device)
        self.memory = Memory()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

    def train_step(self, inputs, targets):
        """Do inference and dream if wrong - no normal training"""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Only do inference
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs)
            pred = outputs.argmax(dim=1)
            mistakes = pred != targets

        # If we made any mistakes, add to memory and dream
        dream_loss = 0.0
        if mistakes.any():
            # Add mistakes to memory
            self.memory.add_mistakes(inputs=inputs[mistakes], targets=targets[mistakes])

            # Dream on memories until we learn them
            if len(self.memory) > 0:
                dream_loss = self.dream_until_learned()

        return {
            "accuracy": (pred == targets).float().mean().item()
            * 100,  # Fixed to be percentage
            "memory_size": len(self.memory),
            "dream_loss": dream_loss,
        }

    def dream_until_learned(self, target_acc=97.0, max_steps=100):
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

            # Check accuracy on all memories
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


class BaselineTask:
    """Simple task without memory - regular training"""

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.network = Net().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

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

        # Calculate accuracy
        with torch.no_grad():
            pred = outputs.argmax(dim=1)
            accuracy = (pred == targets).float().mean().item() * 100

        return {
            "accuracy": accuracy,
            "loss": loss.item(),
        }


def get_digit_loader(dataset, digits, batch_size=32):
    """Get dataloader for specific digits"""
    indices = []
    for idx, (_, label) in enumerate(dataset):
        if label in digits:
            indices.append(idx)
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def evaluate_digits(task, dataset, digits):
    """Evaluate accuracy on specific digits"""
    loader = get_digit_loader(dataset, digits, batch_size=100)
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


def evaluate_all_pairs(task, dataset, all_pairs):
    """Evaluate accuracy on all digit pairs seen so far."""
    results = {}
    for d1, d2 in all_pairs:
        acc = evaluate_digits(task, dataset, [d1, d2])
        results[f"{d1}{d2}"] = acc
    return results


def print_accuracy_matrix(accuracies_over_time):
    """Print accuracy matrix showing forgetting."""
    print("\nAccuracy Matrix (rows=after training task, columns=tested on task):")
    tasks = list(accuracies_over_time[0].keys())

    # Header
    print(f"{'Task':8}", end="")
    for task in tasks:
        print(f"{task:8}", end="")
    print()

    # Matrix
    for i, task_results in enumerate(accuracies_over_time):
        print(f"After {i:02d} ", end="")
        for task in tasks:
            acc = task_results.get(task, 0.0)
            print(f"{acc:8.2f}", end="")
        print()


def train_task(task, train_set, task_name="Task"):
    """Train on Split MNIST benchmark and evaluate forgetting."""
    print(f"\n=== Training {task_name} ===")

    # Standard Split MNIST pairs
    digit_pairs = [
        (0, 1),  # Task 0
        (2, 3),  # Task 1
        (4, 5),  # Task 2
        (6, 7),  # Task 3
        (8, 9),  # Task 4
    ]

    # Track accuracies after each task
    accuracies_over_time = []
    previous_pairs = []

    for task_id, (d1, d2) in enumerate(digit_pairs):
        print(f"\nTraining on Task {task_id}: digits {d1} vs {d2}")

        # Create new task for each pair in baseline
        if isinstance(task, BaselineTask):
            task = BaselineTask(task.device)  # Fresh network for each pair

        train_loader = get_digit_loader(train_set, [d1, d2])

        # Main training loop for this pair
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"Learning {d1} vs {d2}")
        for data, target in pbar:
            # Do training step
            metrics = task.train_step(data, target)

            # Update running accuracy
            batch_size = len(target)
            acc = metrics["accuracy"]
            correct += int(acc / 100 * batch_size)
            total += batch_size
            running_acc = 100 * correct / total

            # Update progress bar
            postfix = {"accuracy": f"{running_acc:.2f}%"}
            if isinstance(task, Task):  # Memory task
                postfix["memory_size"] = metrics["memory_size"]
            pbar.set_postfix(postfix)

            # Stop if we hit target accuracy
            if running_acc >= 90:
                print(f"\nReached {running_acc:.2f}% accuracy on {d1} vs {d2}!")
                break

        # Add current pair to previous pairs
        previous_pairs.append((d1, d2))

        # Evaluate on all pairs seen so far
        accuracies = evaluate_all_pairs(task, train_set, previous_pairs)
        accuracies_over_time.append(accuracies)

        # Print current accuracies
        print("\nAccuracies after training on {d1} vs {d2}:")
        for prev_d1, prev_d2 in previous_pairs:
            pair_name = f"{prev_d1}{prev_d2}"
            print(f"Task {prev_d1} vs {prev_d2}: {accuracies[pair_name]:.2f}%")

        if isinstance(task, Task):  # Memory task
            print(f"Memory size: {len(task.memory)}")
        print("-" * 50)

    # Print final accuracy matrix
    print_accuracy_matrix(accuracies_over_time)

    # Calculate metrics
    final_accuracies = accuracies_over_time[-1]
    avg_acc = np.mean(list(final_accuracies.values()))
    print(f"\nFinal average accuracy across all tasks: {avg_acc:.2f}%")


def main():
    # Load MNIST once
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)

    # Train baseline (no memory)
    baseline = BaselineTask()
    train_task(baseline, train_set, task_name="Baseline (No Memory)")

    print("\n" + "=" * 80 + "\n")

    # Train with memory
    memory_task = Task()
    train_task(memory_task, train_set, task_name="Memory-Based Learning")


if __name__ == "__main__":
    main()
