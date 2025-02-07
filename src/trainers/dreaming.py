"""Implementation of the dream phase for memory consolidation."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from torch.utils.data import Dataset

from .memory import MemoryBuffer


class DreamPhase:
    """Handles dream phase training on memory buffer.

    Key features:
    - Trains until target accuracy reached
    - Multiple sampling strategies
    - Progress tracking and early stopping
    - Configurable success criteria
    """

    def __init__(
        self,
        model: nn.Module,
        memory: MemoryBuffer,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_acc: float = 95.0,
        max_steps: int = 100,
        batch_size: int = 32,
        sampling_strategy: str = "random",
        early_stop_patience: int = 5,
        eval_interval: int = 10,
    ):
        """Initialize dream phase.

        Args:
            model: Neural network to train
            memory: Memory buffer to train on
            criterion: Loss function
            optimizer: Optimizer
            dataset: Dataset to get examples from
            device: Device to train on
            target_acc: Target accuracy to achieve
            max_steps: Maximum training steps
            batch_size: Batch size for training
            sampling_strategy: How to sample from memory
            early_stop_patience: Steps without improvement before stopping
            eval_interval: Steps between evaluations
        """
        self.model = model
        self.memory = memory
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device

        self.target_acc = target_acc
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.sampling_strategy = sampling_strategy
        self.early_stop_patience = early_stop_patience
        self.eval_interval = eval_interval

        # Training statistics
        self.reset_stats()

    def reset_stats(self):
        """Reset training statistics."""
        self.stats = {
            "steps": 0,
            "loss_history": [],
            "acc_history": [],
            "best_acc": 0.0,
            "steps_without_improvement": 0,
        }

    def dream(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[float, Dict]:
        """Run dream phase until target accuracy or max steps reached.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (final loss, statistics dictionary)
        """
        self.reset_stats()
        total_loss = 0.0

        # Training loop
        for step in range(self.max_steps):
            self.stats["steps"] = step

            # Get batch from memory
            batch = self.memory.sample_batch(self.batch_size)
            if batch is None:
                break

            # Get examples from dataset
            inputs = []
            for idx in batch["indices"]:
                img, _ = self.dataset[idx]
                inputs.append(img)

            inputs = torch.stack(inputs).to(self.device)
            targets = torch.tensor(batch["targets"]).to(self.device)

            # Train on batch
            loss = self._train_step(inputs, targets)
            total_loss += loss
            self.stats["loss_history"].append(loss)

            # Evaluate periodically
            if step % self.eval_interval == 0:
                accuracy = self._evaluate_memory()
                self.stats["acc_history"].append(accuracy)

                # Update best accuracy
                if accuracy > self.stats["best_acc"]:
                    self.stats["best_acc"] = accuracy
                    self.stats["steps_without_improvement"] = 0
                else:
                    self.stats["steps_without_improvement"] += 1

                # Report progress
                if progress_callback:
                    progress_callback(step, loss, accuracy)

                # Check stopping conditions
                if accuracy >= self.target_acc:
                    break
                if self.stats["steps_without_improvement"] >= self.early_stop_patience:
                    break

        # Compute final statistics
        final_stats = {
            "total_steps": step + 1,
            "avg_loss": total_loss / (step + 1),
            "final_accuracy": self._evaluate_memory(),
            "best_accuracy": self.stats["best_acc"],
            "loss_history": self.stats["loss_history"],
            "acc_history": self.stats["acc_history"],
        }

        return final_stats["avg_loss"], final_stats

    def _train_step(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Single training step on a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _evaluate_memory(self) -> float:
        """Evaluate accuracy on entire memory buffer."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            # Get all examples from memory
            batch = self.memory.sample_batch(len(self.memory))
            if batch is None:
                return 0.0

            # Get all examples from dataset
            inputs = []
            for idx in batch["indices"]:
                img, _ = self.dataset[idx]
                inputs.append(img)

            inputs = torch.stack(inputs).to(self.device)
            targets = torch.tensor(batch["targets"]).to(self.device)

            # Get predictions
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)

            total = targets.size(0)
            correct = predicted.eq(targets).sum().item()

        self.model.train()
        return 100.0 * correct / total if total > 0 else 0.0
