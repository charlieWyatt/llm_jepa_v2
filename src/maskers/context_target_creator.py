from typing import Any, List, Tuple
import numpy as np
from src.maskers.base import BaseMaskingStrategy


class ContextTargetCreator:
    """
    Creates context and target masks for self-supervised learning.

    This class uses masking strategies to generate one context mask and
    multiple target masks for a given sequence of patches.
    """

    def __init__(
        self,
        context_strategy: BaseMaskingStrategy,
        target_strategy: BaseMaskingStrategy,
        num_targets: int,
    ):
        """
        Initialize the ContextTargetCreator.

        Args:
            context_strategy: Masking strategy for creating the context mask
            target_strategy: Masking strategy for creating target masks
            num_targets: Number of target masks to generate
        """
        self.context_strategy = context_strategy
        self.target_strategy = target_strategy
        self.num_targets = num_targets

    def create_context_and_targets(
        self, patches: Any
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create one context mask and multiple target masks.

        Args:
            patches: List or array of patches to create masks for

        Returns:
            Tuple containing:
                - context_mask (np.ndarray): Boolean mask for context
                - target_masks (List[np.ndarray]): List of boolean masks for targets

        Example:
            >>> creator = ContextTargetCreator(
            ...     context_strategy=RandomMasker(mask_ratio=0.6),
            ...     target_strategy=BlockMasker(span_length=10),
            ...     num_targets=4
            ... )
            >>> context_mask, target_masks = creator.create_context_and_targets(patches)
            >>> # Each call generates different random masks
        """
        # Create context mask using context strategy
        context_mask = self.context_strategy.create_mask(patches)

        # Create target masks using target strategy
        target_masks = []
        for _ in range(self.num_targets):
            target_mask = self.target_strategy.create_mask(patches)
            target_masks.append(target_mask)

        return context_mask, target_masks
