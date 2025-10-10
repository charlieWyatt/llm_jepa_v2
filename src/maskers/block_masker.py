import random
from typing import Any
import numpy as np
from src.maskers.base import BaseMaskingStrategy


class BlockMasker(BaseMaskingStrategy):
    def __init__(self, span_length: int = 5):
        self.span_length = span_length

    def create_mask(self, patches: Any) -> np.ndarray:
        """
        Create a contiguous block mask for the given patches.

        Generates a different block position each time (non-deterministic).

        Args:
            patches: List or array of patches to create mask for

        Returns:
            np.ndarray: Boolean array where True indicates masked positions
                       in a contiguous block
        """
        num_patches = len(patches)
        if num_patches == 0:
            return np.array([], dtype=bool)

        # Create boolean mask initialized to False
        mask = np.zeros(num_patches, dtype=bool)

        # If patches are shorter than or equal to span_length, mask all
        if num_patches <= self.span_length:
            mask[:] = True
            return mask

        # Select random start position for the block
        start_idx = random.randint(0, num_patches - self.span_length)

        # Mask the contiguous block
        mask[start_idx: start_idx + self.span_length] = True

        return mask
