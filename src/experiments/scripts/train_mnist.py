"""Main training script for MNIST experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import random
from pathlib import Path
from datetime import datetime
import wandb
import yaml
import argparse

from ...models.mnist import MNISTModel, MNISTConfig
from ...trainers.memory import MemoryBuffer
from ...trainers.dreaming import DreamPhase
from ...utils.visualization import (
    plot_training_progress,
    plot_memory_analysis,
    plot_confusion_matrix,
    plot_tsne,
)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from yaml file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_mnist_data(
    digit_pairs: list,
    batch_size: int = 32,
):
    """Get MNIST datasets and loaders for specific digit pairs."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Download datasets
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    # Create indices for digit pairs
    def get_digit_indices(dataset, digits):
        return [idx for idx, (_, label) in enumerate(dataset) if label in digits]

    train_pair_indices = {
        f"{d1}{d2}": get_digit_indices(train_dataset, [d1, d2])
        for d1, d2 in digit_pairs
    }
    test_pair_indices = {
        f"{d1}{d2}": get_digit_indices(test_dataset, [d1, d2]) for d1, d2 in digit_pairs
    }

    return train_dataset, test_dataset, train_pair_indices, test_pair_indices


def evaluate(
    model: nn.Module,
    dataset,
    indices: list,
    device: str,
    batch_size: int = 32,
) -> float:
    """Evaluate model on specific indices of dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = [dataset[idx] for idx in batch_indices]
            inputs = torch.stack([x[0] for x in batch]).to(device)
            targets = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    """Main training function."""
    args = get_args()
    config = load_config(args.config) if args.config else {}

    # Setup wandb
    if args.wandb:
        wandb.init(
            project="learn-from-mistakes",
            config=config,
        )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results") / f"mnist_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    if config:
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and components
    model_config = MNISTConfig(
        feature_size=config.get("feature_size", 64),
        hidden_sizes=config.get("hidden_sizes", (32,)),
        dropout=config.get("dropout", 0.2),
        batch_norm=config.get("batch_norm", True),
    )
    model = MNISTModel(model_config).to(device)
    memory = MemoryBuffer(
        max_size=config.get("memory_size", 2000),
        replacement_strategy=config.get("replacement_strategy", "random"),
        confidence_threshold=config.get("confidence_threshold", 0.0),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))
    dream_phase = DreamPhase(
        model=model,
        memory=memory,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        target_acc=config.get("dream_target_acc", 95.0),
        max_steps=config.get("dream_max_steps", 100),
        batch_size=config.get("batch_size", 32),
        sampling_strategy=config.get("dream_sampling", "random"),
    )

    # Get data
    digit_pairs = [
        (1, 7),  # Similar vertical strokes
        (3, 8),  # Similar curves/loops
        (4, 9),  # Similar top parts
    ]
    train_dataset, test_dataset, train_indices, test_indices = get_mnist_data(
        digit_pairs,
        batch_size=config.get("batch_size", 32),
    )

    # Training metrics
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "memory_size": [],
        "memory_avg_conf": [],
        "dream_loss": [],
        "dream_acc": [],
    }

    # Train on each pair sequentially
    global_step = 0
    for pair_idx, (d1, d2) in enumerate(digit_pairs):
        pair_name = f"{d1}{d2}"
        print(f"\n=== Phase {pair_idx + 1}: Training on digits {d1} & {d2} ===")

        # Get indices for current pair
        current_indices = train_indices[pair_name]
        current_acc = 0
        epoch = 0

        while current_acc < 95.0:  # Train until target accuracy
            epoch += 1
            print(f"\nEpoch {epoch}:")

            # Shuffle indices
            random.shuffle(current_indices)

            # Train on current pair
            for i in range(0, len(current_indices), config.get("batch_size", 32)):
                batch_indices = current_indices[i : i + config.get("batch_size", 32)]
                batch = [train_dataset[idx] for idx in batch_indices]
                inputs = torch.stack([x[0] for x in batch]).to(device)
                targets = torch.tensor([x[1] for x in batch], dtype=torch.long).to(
                    device
                )

                # Only do inference
                model.eval()
                with torch.no_grad():
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = probs.max(1)
                    mistakes = predicted != targets

                # Add mistakes to memory
                dream_loss = 0.0
                if mistakes.any():
                    memory.add_experience(
                        inputs=inputs[mistakes],
                        predictions=outputs[mistakes],
                        targets=targets[mistakes],
                    )

                    # Dream on memories
                    if len(memory) > 0:
                        _, dream_stats = dream_phase.dream()
                        dream_loss = dream_stats["avg_loss"]

                # Update metrics
                metrics["train_loss"].append(criterion(outputs, targets).item())
                metrics["train_acc"].append(
                    100 * predicted.eq(targets).float().mean().item()
                )
                metrics["memory_size"].append(len(memory))
                metrics["memory_avg_conf"].append(memory.get_stats()["avg_confidence"])
                metrics["dream_loss"].append(dream_loss)
                metrics["dream_acc"].append(
                    dream_stats["final_accuracy"] if dream_loss > 0 else 100.0
                )

                # Log to wandb
                if args.wandb:
                    wandb.log(
                        {
                            "train_loss": metrics["train_loss"][-1],
                            "train_acc": metrics["train_acc"][-1],
                            "memory_size": metrics["memory_size"][-1],
                            "memory_avg_conf": metrics["memory_avg_conf"][-1],
                            "dream_loss": metrics["dream_loss"][-1],
                            "dream_acc": metrics["dream_acc"][-1],
                        }
                    )

                # Check progress every 50 steps
                if global_step % 50 == 0:
                    current_acc = evaluate(
                        model,
                        test_dataset,
                        test_indices[pair_name],
                        device,
                    )

                    # Also evaluate on previous pairs
                    if pair_idx > 0:
                        prev_accs = []
                        for prev_d1, prev_d2 in digit_pairs[:pair_idx]:
                            prev_pair = f"{prev_d1}{prev_d2}"
                            prev_acc = evaluate(
                                model,
                                test_dataset,
                                test_indices[prev_pair],
                                device,
                            )
                            prev_accs.append(prev_acc)

                        print(f"Step {global_step}:")
                        print(f"Current pair ({d1},{d2}): {current_acc:.1f}%")
                        print(
                            f"Previous pairs: " f"{sum(prev_accs)/len(prev_accs):.1f}%"
                        )
                        print(f"Memory size: {len(memory)}")

                global_step += 1

                # Create visualizations
                if global_step % 500 == 0:
                    # Training progress
                    plot_training_progress(
                        metrics,
                        save_path=output_dir / f"progress_{global_step}.png",
                        use_wandb=args.wandb,
                    )

                    # Memory analysis
                    plot_memory_analysis(
                        memory,
                        save_dir=output_dir / f"memory_{global_step}",
                        use_wandb=args.wandb,
                    )

                    # Confusion matrix
                    all_preds = []
                    all_targets = []
                    model.eval()
                    with torch.no_grad():
                        for d1, d2 in digit_pairs:
                            pair = f"{d1}{d2}"
                            for idx in test_indices[pair]:
                                inputs, targets = test_dataset[idx]
                                inputs = inputs.unsqueeze(0).to(device)
                                outputs = model(inputs)
                                _, pred = outputs.max(1)
                                all_preds.append(pred.item())
                                all_targets.append(targets)
                    model.train()

                    plot_confusion_matrix(
                        torch.tensor(all_preds),
                        torch.tensor(all_targets),
                        save_path=output_dir / f"confusion_{global_step}.png",
                        use_wandb=args.wandb,
                    )

            if current_acc >= 95.0:
                print(f"\n>>> Reached target accuracy on digits {d1} & {d2}")
                break

    # Final evaluation
    print("\n=== Final Evaluation ===")
    print("\nPer-pair Performance:")
    print("Pair | Accuracy")
    print("-" * 20)

    for d1, d2 in digit_pairs:
        pair_name = f"{d1}{d2}"
        acc = evaluate(
            model,
            test_dataset,
            test_indices[pair_name],
            device,
        )
        print(f"{d1},{d2} | {acc:7.1f}%")

    # Save final results
    with open(output_dir / "results.txt", "w") as f:
        f.write("=== Final Results ===\n\n")
        for d1, d2 in digit_pairs:
            pair_name = f"{d1}{d2}"
            acc = evaluate(
                model,
                test_dataset,
                test_indices[pair_name],
                device,
            )
            f.write(f"Pair {d1}-{d2}: {acc:.1f}%\n")


if __name__ == "__main__":
    main()
