"""MNIST-specific model implementation."""

import torch
import torch.nn as nn
from typing import Tuple

from .base import BaseModel, ModelConfig


class MNISTConfig(ModelConfig):
    """Configuration for MNIST model."""

    def __init__(
        self,
        feature_size: int = 64,
        hidden_sizes: Tuple[int, ...] = (32,),
        dropout: float = 0.2,
        batch_norm: bool = True,
    ):
        super().__init__(
            input_size=(1, 28, 28),  # MNIST image size
            feature_size=feature_size,
            num_classes=10,  # 10 digits
            hidden_sizes=list(hidden_sizes),
            dropout=dropout,
            batch_norm=batch_norm,
        )


class MNISTModel(BaseModel):
    """CNN model for MNIST classification.

    Architecture:
    1. Two conv blocks (conv -> relu -> maxpool)
    2. Feature embedding layer
    3. Classification head
    """

    def _build_feature_extractor(self) -> nn.Module:
        """Build convolutional feature extractor."""
        return nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def _get_flattened_size(self) -> int:
        """Calculate size of flattened features."""
        # After two 2x2 max pools: 28 -> 14 -> 7
        return 64 * 7 * 7  # channels * height * width
