"""
Base class for mask generators that work with batched torch tensors.
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class BaseMaskGenerator(ABC):
    """
    Base class for generating masks for JEPA-style training.

    Mask generators create binary masks indicating which positions to SELECT.
    Convention: 1 = SELECT this position, 0 = don't select

    Usage:
    - Context generator: 1 = visible to model
    - Target generator: 1 = predict this position
    """

    @abstractmethod
    def generate_mask(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate a binary mask for a batch of sequences.

        Args:
            batch_size: Number of sequences in batch
            seq_length: Length of each sequence
            device: Device to place mask tensor on
            attention_mask: [B, L] - Optional mask indicating valid positions
                           (1 = valid token, 0 = padding)

        Returns:
            mask: [B, L] - Binary mask where:
                  1 = SELECT this position (include it for this purpose)
                  0 = DON'T select this position

        IMPORTANT: The interpretation depends on usage:
        - For context: 1 = visible to model, 0 = hidden
        - For targets: 1 = predict this, 0 = don't predict
        """
        pass
