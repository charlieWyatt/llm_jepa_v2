"""
Random mask generator for JEPA-style training.
"""

import torch
from typing import Optional
from src.maskers.base_mask_generator import BaseMaskGenerator


class RandomMaskGenerator(BaseMaskGenerator):
    """
    Randomly masks a percentage of positions in each sequence.

    Each position has an independent probability of being masked.
    Different sequences in the batch get different random masks.
    """

    def __init__(self, mask_ratio: float = 0.15):
        """
        Args:
            mask_ratio: Fraction of positions to mask (0.0 to 1.0)
        """
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

        self.mask_ratio = mask_ratio

    def generate_mask(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate random binary masks.
        
        Args:
            batch_size: Number of sequences
            seq_length: Length of each sequence
            device: Device for tensor
            attention_mask: [B, L] - Optional mask for valid positions
        
        Returns:
            mask: [B, L] - 1 = SELECT this position, 0 = don't select
        """
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device)

        # Generate random probabilities for each position
        probs = torch.rand(batch_size, seq_length, device=device)

        # Mask positions where prob < mask_ratio
        mask = (probs < self.mask_ratio).float()

        # Only mask valid positions (respect attention_mask)
        mask = mask * attention_mask

        return mask
