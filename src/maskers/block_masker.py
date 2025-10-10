import random
import numpy as np
from src.maskers.base import BaseMaskingStrategy


class BlockMasker(BaseMaskingStrategy):
    def __init__(self, span_length: int = 5):
        self.span_length = span_length

    def create_mask(self, sequence_length: int) -> np.ndarray:
        """
        Create a contiguous block mask for a sequence.

        Generates a different block position each time (non-deterministic).

        Args:
            sequence_length: Length of the sequence to create mask for

        Returns:
            np.ndarray: Boolean array of shape (sequence_length,) where True indicates
                       positions in the contiguous block
        """
        if sequence_length == 0:
            return np.array([], dtype=bool)

        # Create boolean mask initialized to False
        mask = np.zeros(sequence_length, dtype=bool)

        # If sequence is shorter than or equal to span_length, mask all
        if sequence_length <= self.span_length:
            mask[:] = True
            return mask

        # Select random start position for the block
        start_idx = random.randint(0, sequence_length - self.span_length)

        # Mark the contiguous block as True
        mask[start_idx: start_idx + self.span_length] = True

        return mask
