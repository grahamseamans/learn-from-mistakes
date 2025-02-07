"""Visualization utilities for analyzing results."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import wandb

from ..trainers.memory import MemoryBuffer


def plot_training_progress(
    metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    use_wandb: bool = False,
):
    """Plot training metrics over time.

    Args:
        metrics: Dictionary of metric name to list of values
        save_path: Optional path to save figure
        use_wandb: Whether to log to wandb
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress")

    # Plot loss
    ax = axes[0, 0]
    ax.plot(metrics["train_loss"], label="Train")
    if "val_loss" in metrics:
        ax.plot(metrics["val_loss"], label="Val")
    ax.set_title("Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    # Plot accuracy
    ax = axes[0, 1]
    ax.plot(metrics["train_acc"], label="Train")
    if "val_acc" in metrics:
        ax.plot(metrics["val_acc"], label="Val")
    ax.set_title("Accuracy")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()

    # Plot memory stats
    ax = axes[1, 0]
    if "memory_size" in metrics:
        ax.plot(metrics["memory_size"], label="Size")
    if "memory_avg_conf" in metrics:
        ax.plot(metrics["memory_avg_conf"], label="Avg Conf")
    ax.set_title("Memory Buffer")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count/Confidence")
    ax.legend()

    # Plot dream stats
    ax = axes[1, 1]
    if "dream_loss" in metrics:
        ax.plot(metrics["dream_loss"], label="Loss")
    if "dream_acc" in metrics:
        ax.plot(metrics["dream_acc"], label="Accuracy")
    ax.set_title("Dream Phase")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if use_wandb:
        wandb.log({"training_progress": wandb.Image(fig)})

    plt.close()


def plot_memory_analysis(
    memory: MemoryBuffer,
    save_dir: Optional[Path] = None,
    use_wandb: bool = False,
):
    """Analyze and plot memory buffer contents.

    Args:
        memory: Memory buffer to analyze
        save_dir: Optional directory to save figures
        use_wandb: Whether to log to wandb
    """
    # Get memory stats
    stats = memory.get_stats()

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = sorted(stats["class_distribution"].keys())
    counts = [stats["class_distribution"][c] for c in classes]
    ax.bar(classes, counts)
    ax.set_title("Memory Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")

    if save_dir:
        plt.savefig(save_dir / "memory_class_dist.png")
    if use_wandb:
        wandb.log({"memory_class_dist": wandb.Image(fig)})
    plt.close()

    # 2. Confidence distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    confidences = [exp["confidence"] for exp in memory.buffer]
    ax.hist(confidences, bins=20)
    ax.set_title("Memory Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")

    if save_dir:
        plt.savefig(save_dir / "memory_conf_dist.png")
    if use_wandb:
        wandb.log({"memory_conf_dist": wandb.Image(fig)})
    plt.close()

    # 3. Age distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ages = [memory.total_seen - exp["timestamp"] for exp in memory.buffer]
    ax.hist(ages, bins=20)
    ax.set_title("Memory Age Distribution")
    ax.set_xlabel("Age (steps)")
    ax.set_ylabel("Count")

    if save_dir:
        plt.savefig(save_dir / "memory_age_dist.png")
    if use_wandb:
        wandb.log({"memory_age_dist": wandb.Image(fig)})
    plt.close()


def plot_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    use_wandb: bool = False,
):
    """Plot confusion matrix.

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: Optional list of class names
        save_path: Optional path to save figure
        use_wandb: Whether to log to wandb
    """
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # Create confusion matrix
    num_classes = max(predictions.max(), targets.max()) + 1
    conf_mat = np.zeros((num_classes, num_classes))
    for t, p in zip(targets, predictions):
        conf_mat[t, p] += 1

    # Normalize
    conf_mat = conf_mat / conf_mat.sum(axis=1, keepdims=True)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if save_path:
        plt.savefig(save_path)
    if use_wandb:
        wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()


def plot_tsne(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    use_wandb: bool = False,
):
    """Plot t-SNE visualization of feature space.

    Args:
        features: Feature embeddings
        labels: Class labels
        class_names: Optional list of class names
        save_path: Optional path to save figure
        use_wandb: Whether to log to wandb
    """
    from sklearn.manifold import TSNE

    # Convert to numpy
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.6,
    )
    ax.set_title("t-SNE Visualization")

    if class_names:
        legend = ax.legend(
            *scatter.legend_elements(),
            title="Classes",
            labels=class_names,
        )
        ax.add_artist(legend)

    if save_path:
        plt.savefig(save_path)
    if use_wandb:
        wandb.log({"tsne": wandb.Image(fig)})
    plt.close()
