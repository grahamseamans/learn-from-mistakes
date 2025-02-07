import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Callable


class AdaptiveTrainer:
    """Trainer for adaptive learning with real-time and dream phases"""

    def __init__(
        self,
        base_network: nn.Module,  # Can be any architecture
        head_network: nn.Module,  # Task-specific head
        surprise_network: Optional[nn.Module] = None,  # Task-specific surprise detector
        memory_buffer: Optional[object] = None,  # Optional memory buffer
        base_optimizer: Optional[optim.Optimizer] = None,
        head_optimizer: Optional[optim.Optimizer] = None,
        surprise_optimizer: Optional[optim.Optimizer] = None,
        criterion: nn.Module = nn.MSELoss(),
        dream_interval: int = 100,
        dream_batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_classification: bool = False,  # Whether this is a classification task
    ):
        self.base_network = base_network.to(device)
        self.head_network = head_network.to(device)
        self.surprise_network = (
            surprise_network.to(device) if surprise_network is not None else None
        )
        self.memory_buffer = memory_buffer
        self.base_optimizer = base_optimizer
        self.head_optimizer = head_optimizer
        self.surprise_optimizer = surprise_optimizer
        self.criterion = criterion
        self.dream_interval = dream_interval
        self.dream_batch_size = dream_batch_size
        self.device = device
        self.steps = 0
        self.is_classification = is_classification

    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        train_base: bool = False,
        step: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, float]:
        """Single training step with real-time learning"""
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward passes
        with torch.set_grad_enabled(train_base):
            features = self.base_network(inputs)
        predictions = self.head_network(features)

        # Add to memory if we have a buffer (regardless of surprise network)
        if self.memory_buffer is not None:
            self.memory_buffer.add_experience(
                inputs=inputs,
                predictions=predictions,
                targets=targets,
                step=step,
                phase=phase,
            )

        # Get surprise score if we have a surprise network
        surprise_score = None
        if self.surprise_network is not None:
            surprise_score = self.surprise_network(
                features.detach(), predictions.detach(), targets
            )

        # Train head (and optionally base)
        if self.head_optimizer is not None:
            self.head_optimizer.zero_grad()
            if train_base and self.base_optimizer is not None:
                self.base_optimizer.zero_grad()

            task_loss = self.criterion(predictions, targets)
            task_loss.backward()

            if train_base and self.base_optimizer is not None:
                self.base_optimizer.step()
            self.head_optimizer.step()
        else:
            with torch.no_grad():
                task_loss = self.criterion(predictions, targets)

        self.steps += 1

        # Dream phase if we have memory
        dream_loss = 0.0
        if self.memory_buffer is not None and self.steps % self.dream_interval == 0:
            dream_loss = self.dream_step()

        metrics = {
            "task_loss": task_loss.item(),
            "dream_loss": dream_loss,
        }

        if surprise_score is not None:
            metrics["surprise_score"] = surprise_score.mean().item()

        return metrics

    def dream_step(self) -> float:
        """Training step using surprising memories"""
        if self.memory_buffer is None:
            return 0.0

        batch = self.memory_buffer.sample_batch(self.dream_batch_size)
        if batch is None:
            return 0.0

        # Train full network on surprising examples
        if self.base_optimizer is not None:
            self.base_optimizer.zero_grad()
        if self.head_optimizer is not None:
            self.head_optimizer.zero_grad()

        features = self.base_network(batch["inputs"].to(self.device))
        predictions = self.head_network(features)
        loss = self.criterion(predictions, batch["target_classes"].to(self.device))

        if self.base_optimizer is not None and self.head_optimizer is not None:
            loss.backward()
            self.base_optimizer.step()
            self.head_optimizer.step()

        return loss.item()
