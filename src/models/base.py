"""Base model implementations."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    input_size: Tuple[int, ...]  # (channels, height, width) for images
    feature_size: int  # Size of feature embedding
    num_classes: int  # Number of output classes
    hidden_sizes: List[int]  # Sizes of hidden layers
    dropout: float = 0.0  # Dropout probability
    batch_norm: bool = True  # Whether to use batch normalization


class BaseModel(nn.Module):
    """Base model with feature extraction and classification head.

    Architecture:
    1. Feature extraction layers
    2. Feature embedding layer
    3. Classification head
    """

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Build network
        self.features = self._build_feature_extractor()
        self.embedding = self._build_embedding_layer()
        self.classifier = self._build_classifier()

        # Initialize weights
        self.apply(self._init_weights)

    def _build_feature_extractor(self) -> nn.Module:
        """Build feature extraction layers.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    def _build_embedding_layer(self) -> nn.Module:
        """Build feature embedding layer."""
        layers = []

        # Flatten features
        layers.append(nn.Flatten())

        # Add embedding layer
        in_features = self._get_flattened_size()
        layers.extend(
            [
                nn.Linear(in_features, self.config.feature_size),
                nn.ReLU(),
            ]
        )

        if self.config.batch_norm:
            layers.append(nn.BatchNorm1d(self.config.feature_size))

        if self.config.dropout > 0:
            layers.append(nn.Dropout(self.config.dropout))

        return nn.Sequential(*layers)

    def _build_classifier(self) -> nn.Module:
        """Build classification head."""
        layers = []
        current_size = self.config.feature_size

        # Add hidden layers
        for hidden_size in self.config.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(current_size, hidden_size),
                    nn.ReLU(),
                ]
            )

            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))

            current_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(current_size, self.config.num_classes))

        return nn.Sequential(*layers)

    def _get_flattened_size(self) -> int:
        """Calculate size of flattened features.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input tensor
            return_features: Whether to return feature embeddings

        Returns:
            Tuple of (predictions, optional features)
        """
        # Extract features
        features = self.features(x)

        # Get embedding
        embedding = self.embedding(features)

        # Get predictions
        predictions = self.classifier(embedding)

        if return_features:
            return predictions, embedding
        return predictions

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings for input."""
        with torch.no_grad():
            features = self.features(x)
            embedding = self.embedding(features)
        return embedding

    def freeze_features(self):
        """Freeze feature extraction layers."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.embedding.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        """Unfreeze feature extraction layers."""
        for param in self.features.parameters():
            param.requires_grad = True
        for param in self.embedding.parameters():
            param.requires_grad = True
