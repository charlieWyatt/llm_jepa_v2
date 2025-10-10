import random
import numpy as np
from src.maskers.base import BaseMaskingStrategy


class RandomMasker(BaseMaskingStrategy):
    def __init__(self, mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio

    def create_mask(self, sequence_length: int) -> np.ndarray:
        """
        Create a random boolean mask for a sequence.

        Generates a different mask each time (non-deterministic).

        Args:
            sequence_length: Length of the sequence to create mask for

        Returns:
            np.ndarray: Boolean array of shape (sequence_length,) where True indicates
                       positions to include
        """
        if sequence_length == 0:
            return np.array([], dtype=bool)

        num_to_mask = int(sequence_length * self.mask_ratio)

        # Create boolean mask initialized to False
        mask = np.zeros(sequence_length, dtype=bool)

        # Randomly select indices to mark as True
        if num_to_mask > 0:
            indices = random.sample(range(sequence_length), num_to_mask)
            mask[indices] = True

        return mask
