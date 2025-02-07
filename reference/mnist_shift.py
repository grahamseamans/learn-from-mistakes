import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
from pathlib import Path
from datetime import datetime

from lop.adaptive.memory.buffer import MemoryBuffer


class MNISTNet(nn.Module):
    """Single network for MNIST classification"""

    def __init__(self, feature_size: int = 64):
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
            nn.Linear(64 * 7 * 7, feature_size),
            nn.ReLU(),
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


def get_mnist_loaders(batch_size: int = 32):
    """Get MNIST data loaders for different digit pairs"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download full dataset
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    # Create indices for different digit pairs
    def get_digit_indices(dataset, digits):
        indices = []
        for idx, (_, label) in enumerate(dataset):
            if label in digits:
                indices.append(idx)
        return indices

    # Define key digit pairs with visual relationships
    DIGIT_PAIRS = [
        (1, 7),  # Similar vertical strokes
        (3, 8),  # Similar curves/loops
        (4, 9),  # Similar top parts
    ]

    train_pair_indices = {
        f"{d1}{d2}": get_digit_indices(train_dataset, [d1, d2])
        for d1, d2 in DIGIT_PAIRS
    }
    test_pair_indices = {
        f"{d1}{d2}": get_digit_indices(test_dataset, [d1, d2]) for d1, d2 in DIGIT_PAIRS
    }

    return (
        train_dataset,
        test_dataset,
        DIGIT_PAIRS,
        train_pair_indices,
        test_pair_indices,
    )


class AugmentedTask:
    """Task with memory augmentation"""

    def __init__(
        self,
        feature_size: int = 64,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.device = device

        # Create networks
        self.network = MNISTNet(feature_size).to(device)
        self.memory = MemoryBuffer(max_size=2000)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Do inference and dream if wrong - no normal training"""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Only do inference
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            mistakes = predicted != targets

        # If we made any mistakes, add to memory and dream
        dream_loss = 0.0
        if mistakes.any():
            # Add mistakes to memory
            self.memory.add_experience(
                inputs=inputs[mistakes],
                predictions=outputs[mistakes],
                targets=targets[mistakes],
            )

            # Dream on memories until we learn them
            if len(self.memory.buffer) > 0:
                dream_loss = self.dream_until_learned()

        return {
            "task_loss": self.criterion(outputs, targets).item(),
            "dream_loss": dream_loss,
            "accuracy": 100 * predicted.eq(targets).float().mean().item(),
        }

    def dream_until_learned(
        self, target_acc: float = 95.0, max_steps: int = 100
    ) -> float:
        """Dream on memories until we achieve target accuracy"""
        total_loss = 0.0
        steps = 0
        current_acc = 0.0

        while current_acc < target_acc and steps < max_steps:
            # Sample batch from memory
            batch = self.memory.sample_batch(self.batch_size)
            if batch is None:
                break

            # Train on memories
            self.optimizer.zero_grad()
            dream_inputs = batch["inputs"].to(self.device)
            dream_targets = batch["target_classes"].to(self.device)

            dream_outputs = self.network(dream_inputs)
            loss = self.criterion(dream_outputs, dream_targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            # Check accuracy on all memories
            current_acc = self.evaluate_memories()

        return total_loss / max(steps, 1)

    def evaluate_memories(self) -> float:
        """Evaluate accuracy on all memories"""
        if len(self.memory.buffer) == 0:
            return 100.0

        self.network.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Evaluate in batches to handle large memory buffers
            batch_size = 100
            for i in range(0, len(self.memory.buffer), batch_size):
                batch = self.memory.sample_batch(
                    min(batch_size, len(self.memory.buffer) - i)
                )
                if batch is None:
                    continue

                inputs = batch["inputs"].to(self.device)
                targets = batch["target_classes"].to(self.device)

                outputs = self.network(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        self.network.train()
        return 100.0 * correct / total if total > 0 else 0.0


class BaselineTask:
    """Simple task without memory"""

    def __init__(
        self,
        feature_size: int = 64,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.device = device

        # Create network
        self.network = MNISTNet(feature_size).to(device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Simple training step"""
        self.network.train()
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.network(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            _, predicted = outputs.max(1)

        return {
            "task_loss": loss.item(),
            "dream_loss": 0.0,
            "accuracy": 100 * predicted.eq(targets).float().mean().item(),
        }


def evaluate(task, dataset, device, target_digits=None):
    """Simple evaluation function"""
    task.network.eval()
    correct = 0
    total = 0
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset))))
            batch = [(dataset[j][0], dataset[j][1]) for j in batch_indices]

            # Filter for target digits if specified
            if target_digits is not None:
                filtered_batch = [(x, y) for x, y in batch if y in target_digits]
                if not filtered_batch:  # Skip if no matching digits
                    continue
                batch = filtered_batch

            inputs = torch.stack([x[0] for x in batch]).to(device)
            targets = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)
            outputs = task.network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    task.network.train()
    return 100 * correct / total if total > 0 else 0.0


def main():
    # Create results directory
    results_dir = Path("results/mnist_shift") / datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create tasks
    augmented = AugmentedTask()
    baseline = BaselineTask()

    # Get data
    train_dataset, test_dataset, DIGIT_PAIRS, train_pair_indices, test_pair_indices = (
        get_mnist_loaders()
    )

    # Training loop
    print("\nTraining both models:")
    print("- Augmented: Learns from mistakes with memory replay")
    print("- Baseline: Standard training")
    print("\nDigit pairs for training:")
    for d1, d2 in DIGIT_PAIRS:
        print(f"- {d1} and {d2} (visually related)")

    global_step = 0

    # Train on each pair sequentially
    for pair_idx, (d1, d2) in enumerate(DIGIT_PAIRS):
        pair_name = f"{d1}{d2}"
        print(f"\n=== Phase {pair_idx + 1}: Training on digits {d1} & {d2} ===")

        # Get indices for current pair
        current_indices = train_pair_indices[pair_name]
        current_acc = 0
        base_acc = 0
        epoch = 0

        while current_acc < 95.0 or base_acc < 95.0:  # Stop when both models hit 95%
            epoch += 1
            print(f"\nEpoch {epoch}:")

            num_steps = len(current_indices) // augmented.batch_size
            for step in range(num_steps):
                # Sample batch
                batch_idx = random.sample(current_indices, augmented.batch_size)
                batch = [(train_dataset[i][0], train_dataset[i][1]) for i in batch_idx]
                inputs = torch.stack([x[0] for x in batch])
                targets = torch.tensor([x[1] for x in batch])

                # Train both models
                aug_metrics = augmented.train_step(inputs, targets)
                base_metrics = baseline.train_step(inputs, targets)

                # Check progress every 50 steps
                if global_step % 50 == 0:
                    current_acc = aug_metrics["accuracy"]
                    base_acc = base_metrics["accuracy"]

                    # Break if both models have high accuracy
                    if current_acc >= 95.0 and base_acc >= 95.0:
                        print(
                            f"\n>>> Both models reached target accuracy on digits {d1} & {d2}"
                        )
                        break

                    # Also evaluate on previous pairs
                    if pair_idx > 0:
                        aug_prev_accs = []
                        base_prev_accs = []
                        for prev_d1, prev_d2 in DIGIT_PAIRS[:pair_idx]:
                            prev_pair = f"{prev_d1}{prev_d2}"
                            aug_prev_acc = evaluate(
                                augmented,
                                test_dataset,
                                augmented.device,
                                target_digits=[d1, d2],
                            )
                            base_prev_acc = evaluate(
                                baseline,
                                test_dataset,
                                baseline.device,
                                target_digits=[d1, d2],
                            )
                            aug_prev_accs.append(aug_prev_acc)
                            base_prev_accs.append(base_prev_acc)

                        print(f"Step {global_step}:")
                        print(f"Current pair ({d1},{d2}):")
                        print(f"- Augmented: {current_acc:.1f}%")
                        print(f"- Baseline:  {base_acc:.1f}%")
                        print("Previous pairs:")
                        print(
                            f"- Augmented: {sum(aug_prev_accs)/len(aug_prev_accs):.1f}%"
                        )
                        print(
                            f"- Baseline:  {sum(base_prev_accs)/len(base_prev_accs):.1f}%"
                        )
                        print(f"Memory size: {len(augmented.memory.buffer)}")

                global_step += 1

            if current_acc >= 95.0 and base_acc >= 95.0:
                print(f"\n>>> Reached target accuracy on digits {d1} & {d2}")
                break

    # Final evaluation
    print("\n=== Final Evaluation ===")
    print("\nPer-pair Performance:")
    print("Pair | Augmented | Baseline")
    print("-" * 30)

    for d1, d2 in DIGIT_PAIRS:
        pair_name = f"{d1}{d2}"
        aug_acc = evaluate(
            augmented, test_dataset, augmented.device, target_digits=[d1, d2]
        )
        base_acc = evaluate(
            baseline, test_dataset, baseline.device, target_digits=[d1, d2]
        )
        print(f"{d1},{d2} | {aug_acc:7.1f}% | {base_acc:7.1f}%")

    # Save final summary
    with open(results_dir / "summary.txt", "w") as f:
        f.write("=== MNIST Shift Experiment ===\n\n")
        f.write("Configuration:\n")
        f.write(f"- Batch size: {augmented.batch_size}\n")
        f.write(f"- Feature size: {augmented.feature_size}\n")
        f.write(f"- Memory buffer size: {augmented.memory.max_size}\n")
        f.write("\nFinal Results:\n")
        for d1, d2 in DIGIT_PAIRS:
            aug_acc = evaluate(
                augmented, test_dataset, augmented.device, target_digits=[d1, d2]
            )
            base_acc = evaluate(
                baseline, test_dataset, baseline.device, target_digits=[d1, d2]
            )
            f.write(f"\nPair {d1}-{d2}:\n")
            f.write(f"- Augmented: {aug_acc:.1f}%\n")
            f.write(f"- Baseline:  {base_acc:.1f}%\n")


if __name__ == "__main__":
    main()
