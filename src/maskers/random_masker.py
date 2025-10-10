import random
from typing import Any
import numpy as np
from src.maskers.base import BaseMaskingStrategy


class RandomMasker(BaseMaskingStrategy):
    def __init__(self, mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio

    def create_mask(self, patches: Any) -> np.ndarray:
        """
        Create a random boolean mask for the given patches.

        Generates a different mask each time (non-deterministic).

        Args:
            patches: List or array of patches to create mask for

        Returns:
            np.ndarray: Boolean array where True indicates masked positions
        """
        num_patches = len(patches)
        if num_patches == 0:
            return np.array([], dtype=bool)

        num_to_mask = int(num_patches * self.mask_ratio)

        # Create boolean mask initialized to False
        mask = np.zeros(num_patches, dtype=bool)

        # Randomly select indices to mask
        if num_to_mask > 0:
            indices = random.sample(range(num_patches), num_to_mask)
            mask[indices] = True

        return mask
