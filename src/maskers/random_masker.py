import random
from typing import List, Any
from src.maskers.base import BaseMaskingStrategy


class RandomMasker(BaseMaskingStrategy):
    def __init__(self, mask_ratio: float = 0.15, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.seed = seed

    def create_spans(self, patches: List[Any]) -> List[Any]:
        random.seed(self.seed)

        num_to_mask = int(len(patches) * self.mask_ratio)
        indices = random.sample(range(len(patches)), num_to_mask)

        masked = [patch for i, patch in enumerate(patches) if i in indices]
        return masked
