import random
from typing import List, Any
from src.maskers.base import BaseMaskingStrategy


class ContiguousRandomMasker(BaseMaskingStrategy):
    def __init__(self, span_length: int = 5, seed: int = 42):
        self.span_length = span_length
        self.seed = seed

    def create_spans(self, patches: List[Any]) -> List[Any]:
        random.seed(self.seed)

        if len(patches) <= self.span_length:
            return patches

        start_idx = random.randint(0, len(patches) - self.span_length)
        return patches[start_idx: start_idx + self.span_length]
