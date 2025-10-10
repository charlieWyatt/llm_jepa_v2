"""
Block (contiguous) mask generator for JEPA-style training.
"""

import torch
from typing import Optional
from src.maskers.base_mask_generator import BaseMaskGenerator


class BlockMaskGenerator(BaseMaskGenerator):
    """
    Masks a contiguous block of positions in each sequence.

    The block start position is randomly chosen for each sequence.
    Different sequences in the batch get different block positions.
    """

    def __init__(self, span_length: int = 5):
        """
        Args:
            span_length: Length of the contiguous block to mask
        """
        if span_length < 1:
            raise ValueError(f"span_length must be >= 1, got {span_length}")

        self.span_length = span_length

    def generate_mask(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate block masks with random start positions.
        
        Args:
            batch_size: Number of sequences
            seq_length: Length of each sequence
            device: Device for tensor
            attention_mask: [B, L] - Optional mask for valid positions
        
        Returns:
            mask: [B, L] - 1 = SELECT this position (in the block), 0 = don't select
        """
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length, device=device)

        # Initialize mask to zeros
        mask = torch.zeros(batch_size, seq_length, device=device)

        for i in range(batch_size):
            # Find valid positions in this sequence
            valid_positions = torch.where(attention_mask[i] == 1)[0]
            num_valid = len(valid_positions)

            if num_valid == 0:
                continue

            # If sequence is shorter than span, mask all valid positions
            if num_valid <= self.span_length:
                mask[i, valid_positions] = 1
            else:
                # Randomly choose start position
                max_start = num_valid - self.span_length
                start_idx = torch.randint(
                    0, max_start + 1, (1,), device=device).item()

                # Mask contiguous block
                block_positions = valid_positions[start_idx:start_idx +
                                                  self.span_length]
                mask[i, block_positions] = 1

        return mask
