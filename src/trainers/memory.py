"""Memory buffer implementation for storing and replaying mistakes."""

import torch
import random
from typing import Dict, Optional, List, Tuple
import numpy as np
from pathlib import Path


class MemoryBuffer:
    """Buffer for storing indices of examples where model made mistakes."""

    def __init__(
        self,
        max_size: int = 2000,
        replacement_strategy: str = "random",
    ):
        self.max_size = max_size
        self.replacement_strategy = replacement_strategy

        # Store indices and targets
        self.buffer: List[Dict] = []  # [{"index": int, "target": int}, ...]

        # Basic stats
        self.total_seen = 0
        self.total_mistakes = 0

    def add_mistakes(self, indices: List[int], targets: List[int]) -> None:
        """Add indices of examples where model made mistakes.

        Args:
            indices: Indices into the dataset for mistaken examples
            targets: True labels for those examples
        """
        self.total_seen += len(indices)
        self.total_mistakes += len(indices)

        for idx, target in zip(indices, targets):
            entry = {
                "index": idx,
                "target": target,
            }

            if len(self.buffer) >= self.max_size:
                if self.replacement_strategy == "random":
                    replace_idx = random.randrange(len(self.buffer))
                    self.buffer[replace_idx] = entry
            else:
                self.buffer.append(entry)

    def sample_batch(self, batch_size: int) -> Optional[Dict]:
        """Sample a batch of example indices from memory.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Dict with indices and targets, or None if buffer too small
        """
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        return {
            "indices": [x["index"] for x in batch],
            "targets": [x["target"] for x in batch],
        }

    def get_stats(self) -> Dict:
        """Get current buffer statistics."""
        return {
            "size": len(self.buffer),
            "total_seen": self.total_seen,
            "total_mistakes": self.total_mistakes,
            "mistake_rate": self.total_mistakes / max(self.total_seen, 1),
        }

    def __len__(self) -> int:
        return len(self.buffer)
