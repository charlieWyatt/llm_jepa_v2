from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import torch


class BaseMaskingStrategy(ABC):
    @abstractmethod
    def create_mask(self, sequence_length: int) -> np.ndarray:
        """
        Create a boolean mask for a sequence of given length.

        This returns position indicators (True = use this position in context/target).

        Args:
            sequence_length: Length of the sequence to create mask for

        Returns:
            np.ndarray: Boolean array of shape (sequence_length,) where True indicates
                       positions to include in this mask (context or target)
        """
        pass

    def create_attention_mask(self, sequence_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Create an attention mask tensor for use in transformers.

        Args:
            sequence_length: Length of the sequence
            device: Device to place tensor on (defaults to CPU)

        Returns:
            torch.Tensor: Boolean tensor of shape (sequence_length,) where True means attend
        """
        mask = self.create_mask(sequence_length)
        target_device = device if device is not None else torch.device('cpu')
        return torch.from_numpy(mask).bool().to(target_device)
