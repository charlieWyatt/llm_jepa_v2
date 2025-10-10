from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseMaskingStrategy(ABC):
    @abstractmethod
    def create_mask(self, patches: Any) -> np.ndarray:
        """
        Create a boolean mask for the given patches.

        Args:
            patches: List or array of patches to create mask for

        Returns:
            np.ndarray: Boolean array where True indicates masked positions
        """
        pass
