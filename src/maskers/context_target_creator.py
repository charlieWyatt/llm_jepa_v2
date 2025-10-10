from typing import List, Tuple
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
        self, sequence_length: int
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Create one context mask and multiple target masks for I-JEPA style learning.
        
        For I-JEPA: context_mask indicates which positions the model can see (attend to),
        and target_masks indicate which positions to predict embeddings for.

        Args:
            sequence_length: Length of the input sequence

        Returns:
            Tuple containing:
                - context_mask (np.ndarray): Boolean mask (shape: sequence_length) for context positions
                - target_masks (List[np.ndarray]): List of boolean masks for target positions

        Example:
            >>> creator = ContextTargetCreator(
            ...     context_strategy=RandomMasker(mask_ratio=0.6),
            ...     target_strategy=BlockMasker(span_length=10),
            ...     num_targets=4
            ... )
            >>> context_mask, target_masks = creator.create_context_and_targets(100)
            >>> # context_mask[i] = True means position i is visible to the model
            >>> # target_masks[0][j] = True means predict embedding at position j
        """
        # Create context mask using context strategy
        context_mask = self.context_strategy.create_mask(sequence_length)

        # Create target masks using target strategy
        target_masks = []
        for _ in range(self.num_targets):
            target_mask = self.target_strategy.create_mask(sequence_length)
            target_masks.append(target_mask)

        return context_mask, target_masks
