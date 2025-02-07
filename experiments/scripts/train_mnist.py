"""Train MNIST model with memory-based continual learning - Split MNIST benchmark."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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


class ExperimentTracker:
    """Track experiment metrics for analysis and visualization."""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Tracking metrics
        self.memory_growth = []  # [(task_id, size), ...]
        self.accuracy_matrices = {"train": [], "test": []}  # [matrix per task]
        self.dream_curves = []  # [(task_id, steps, accs), ...]
        self.memory_contents = []  # [(task_id, examples), ...]

    def update_memory_size(self, task_id, size):
        self.memory_growth.append((task_id, size))

    def update_accuracy_matrix(self, task_id, accuracies, split="train"):
        self.accuracy_matrices[split].append(accuracies)

    def add_dream_curve(self, task_id, accuracies):
        """Add dream phase learning curve if available."""
        if accuracies:  # Only add if we have accuracies
            self.dream_curves.append(
                (task_id, list(range(len(accuracies))), accuracies)
            )

    def save_memory_contents(self, task_id, memory):
        # Sample up to 10 examples per class
        examples = {}
        for item in memory.buffer:
            label = item["target_class"]
            if label not in examples:
                examples[label] = []
            if len(examples[label]) < 10:
                examples[label].append(item["inputs"])
        self.memory_contents.append((task_id, examples))

    def plot_memory_growth(self):
        """Plot memory size growth over tasks."""
        tasks, sizes = zip(*self.memory_growth)
        plt.figure(figsize=(10, 5))
        plt.plot(tasks, sizes, marker="o")
        plt.title("Memory Growth Over Tasks")
        plt.xlabel("Task ID")
        plt.ylabel("Memory Size")
        plt.grid(True)
        plt.savefig(self.output_dir / "memory_growth.png")
        plt.close()

    def plot_accuracy_heatmaps(self):
        """Plot accuracy matrix heatmaps."""
        for split in ["train", "test"]:
            matrices = self.accuracy_matrices[split]
            if not matrices:
                continue

            # Convert dictionary to matrix
            tasks = sorted(matrices[-1].keys())  # Get task names in order
            final_matrix = []
            for task_results in matrices:
                row = [task_results.get(task, 0.0) for task in tasks]
                final_matrix.append(row)
            final_matrix = np.array(final_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                final_matrix,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                xticklabels=tasks,
                yticklabels=[f"After Task {i}" for i in range(len(final_matrix))],
                vmin=0,
                vmax=100,
            )
            plt.title(f"Final {split.title()} Accuracy Matrix")
            plt.xlabel("Evaluated on Task")
            plt.ylabel("After Training Task")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"accuracy_matrix_{split}.png")
            plt.close()

    def plot_dream_curves(self):
        """Plot learning curves during dream phases."""
        plt.figure(figsize=(12, 6))
        for task_id, steps, accs in self.dream_curves[-5:]:  # Last 5 dreams
            plt.plot(steps, accs, label=f"Task {task_id}")
        plt.title("Dream Phase Learning Curves")
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / "dream_curves.png")
        plt.close()

    def plot_memory_examples(self):
        """Plot example mistakes from memory."""
        # Get final memory contents
        if not self.memory_contents:
            return

        task_id, examples = self.memory_contents[-1]

        # Plot grid of examples
        n_classes = len(examples)
        n_examples = min(5, max(len(x) for x in examples.values()))

        fig, axes = plt.subplots(
            n_classes, n_examples, figsize=(2 * n_examples, 2 * n_classes)
        )
        if n_classes == 1:
            axes = axes[np.newaxis, :]

        for i, (label, imgs) in enumerate(examples.items()):
            for j, img in enumerate(imgs[:n_examples]):
                axes[i, j].imshow(img.squeeze(), cmap="gray")
                axes[i, j].axis("off")
                if j == 0:
                    axes[i, j].set_ylabel(f"Class {label}")

        plt.suptitle("Example Mistakes from Memory")
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_examples.png")
        plt.close()

    def save_all_plots(self):
        """Generate and save all visualization plots."""
        self.plot_memory_growth()
        self.plot_accuracy_heatmaps()
        self.plot_dream_curves()
        self.plot_memory_examples()


class Task:
    """Task with memory-based learning"""

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mistakes_before_dream=5,
    ):
        self.device = device
        self.network = Net().to(device)
        self.memory = Memory()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)

        # Track dream phase metrics
        self.current_dream_accuracies = []

        # Track mistakes since last dream
        self.mistakes_before_dream = mistakes_before_dream
        self.mistakes_since_dream = 0

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
            if (
                self.mistakes_since_dream >= self.mistakes_before_dream
                and len(self.memory) > 0
            ):
                self.current_dream_accuracies = []  # Reset for new dream
                dream_loss = self.dream_until_learned()
                self.mistakes_since_dream = 0  # Reset counter

        return {
            "accuracy": (pred == targets).float().mean().item() * 100,
            "memory_size": len(self.memory),
            "dream_loss": dream_loss,
            "dream_accuracies": self.current_dream_accuracies,
            "mistakes_since_dream": self.mistakes_since_dream,
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
            self.current_dream_accuracies.append(current_acc)

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


def evaluate_all_pairs(task, dataset, all_pairs, split="train"):
    """Evaluate accuracy on all digit pairs seen so far."""
    results = {}
    for d1, d2 in all_pairs:
        acc = evaluate_digits(task, dataset, [d1, d2])
        results[f"{d1}{d2}"] = acc
    return results


def print_accuracy_matrix(accuracies_over_time, split="train"):
    """Print accuracy matrix showing forgetting."""
    print(
        f"\n{split.title()} Accuracy Matrix (rows=after training task, columns=tested on task):"
    )
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


def train_task(task, train_set, test_set, task_name="Task"):
    """Train on Split MNIST benchmark and evaluate forgetting."""
    print(f"\n=== Training {task_name} ===")

    # Create experiment tracker
    tracker = ExperimentTracker()

    # Standard Split MNIST pairs
    digit_pairs = [
        (0, 1),  # Task 0
        (2, 3),  # Task 1
        (4, 5),  # Task 2
        (6, 7),  # Task 3
        (8, 9),  # Task 4
    ]

    # Track accuracies after each task
    train_accuracies = []
    test_accuracies = []
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
                postfix["mistakes_to_dream"] = (
                    f"{metrics['mistakes_since_dream']}/{task.mistakes_before_dream}"
                )
            pbar.set_postfix(postfix)

            # Stop if we hit target accuracy
            if running_acc >= 90:
                print(f"\nReached {running_acc:.2f}% accuracy on {d1} vs {d2}!")
                break

        # Add current pair to previous pairs
        previous_pairs.append((d1, d2))

        # Evaluate on all pairs seen so far (both train and test)
        train_accs = evaluate_all_pairs(task, train_set, previous_pairs, "train")
        test_accs = evaluate_all_pairs(task, test_set, previous_pairs, "test")
        train_accuracies.append(train_accs)
        test_accuracies.append(test_accs)

        # Print current accuracies
        print("\nAccuracies after training on digits {d1} vs {d2}:")
        for prev_d1, prev_d2 in previous_pairs:
            pair_name = f"{prev_d1}{prev_d2}"
            print(f"Task {prev_d1} vs {prev_d2}:")
            print(f"  Train: {train_accs[pair_name]:.2f}%")
            print(f"  Test:  {test_accs[pair_name]:.2f}%")

        # Update tracker
        if isinstance(task, Task):  # Memory task
            print(f"Memory size: {len(task.memory)}")
            tracker.update_memory_size(task_id, len(task.memory))
            tracker.save_memory_contents(task_id, task.memory)
            if hasattr(task, "current_dream_accuracies"):
                tracker.add_dream_curve(task_id, task.current_dream_accuracies)
        print("-" * 50)

    # Print final accuracy matrices
    print_accuracy_matrix(train_accuracies, "train")
    print_accuracy_matrix(test_accuracies, "test")

    # Calculate final metrics
    final_train = train_accuracies[-1]
    final_test = test_accuracies[-1]
    avg_train = np.mean(list(final_train.values()))
    avg_test = np.mean(list(final_test.values()))

    print(f"\nFinal Metrics for {task_name}:")
    print(f"Average Train Accuracy: {avg_train:.2f}%")
    print(f"Average Test Accuracy:  {avg_test:.2f}%")

    # Save visualizations
    tracker.update_accuracy_matrix(task_id, final_train, "train")
    tracker.update_accuracy_matrix(task_id, final_test, "test")
    tracker.save_all_plots()

    return avg_train, avg_test, tracker


def main():
    # Load MNIST once
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

    print("Split MNIST Benchmark")
    print("=" * 50)

    # Train with memory - wait for 5 mistakes before dreaming
    memory_task = Task(mistakes_before_dream=5)
    mem_train, mem_test, mem_tracker = train_task(
        memory_task, train_set, test_set, task_name="Memory-Based Learning"
    )

    print("\nFinal Results:")
    print("-" * 50)
    print(f"Average Train Accuracy: {mem_train:.2f}%")
    print(f"Average Test Accuracy:  {mem_test:.2f}%")


if __name__ == "__main__":
    main()
