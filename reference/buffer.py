import torch
from typing import Dict, Optional
import random


class MemoryBuffer:
    """Simple buffer for storing examples where model made mistakes"""

    def __init__(self, max_size: int = 2000):
        self.max_size = max_size
        self.buffer = []  # Using list for random replacement

    def add_experience(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> bool:
        """Add incorrect predictions to buffer"""
        pred_probs = torch.softmax(predictions, dim=1)
        _, pred_classes = pred_probs.max(1)
        incorrect = pred_classes != targets

        # Add each incorrect example to buffer
        added = False
        for idx in torch.where(incorrect)[0]:
            exp = {
                "inputs": inputs[idx].cpu(),
                "target_class": targets[idx].item(),
                "confidence": pred_probs[idx].max().item(),
            }

            if len(self.buffer) >= self.max_size:
                replace_idx = random.randrange(len(self.buffer))
                self.buffer[replace_idx] = exp
            else:
                self.buffer.append(exp)
            added = True

        return added

    def sample_batch(self, batch_size: int) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a random batch of experiences from the buffer"""
        if len(self.buffer) < batch_size:
            return None

        # Simple random sampling
        batch = random.sample(self.buffer, batch_size)

        return {
            "inputs": torch.stack([exp["inputs"] for exp in batch]),
            "target_classes": torch.tensor([exp["target_class"] for exp in batch]),
            "confidences": torch.tensor([exp["confidence"] for exp in batch]),
        }

    def __len__(self) -> int:
        return len(self.buffer)
