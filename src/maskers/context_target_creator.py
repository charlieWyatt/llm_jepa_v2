"""
Context-Target creator for JEPA-style training.

This module creates context and target masks using separate strategies.
Supports both complementary (no overlap) and flexible (can overlap) masking.
"""

import torch
from typing import List
from dataclasses import dataclass
from src.maskers.base_mask_generator import BaseMaskGenerator


@dataclass
class ContextTargetPair:
    """Container for context and target masks."""
    context_mask: torch.Tensor  # [B, L] - 1 for visible, 0 for masked
    # List of [B, L] - 1 for predict, 0 for ignore
    target_masks: List[torch.Tensor]
    num_targets: int


class ContextTargetCreator:
    """
    Creates context and target masks for JEPA training using separate generators.

    Key principles:
    1. Works with batched torch tensors
    2. Uses separate generators for context and targets (flexible, can overlap)
    3. Context = what model can see
    4. Targets = what model must predict

    Note: Context and targets CAN overlap with this design. If you need guaranteed
    disjoint masks, use the same generator for both and set num_targets=1.
    """

    def __init__(
        self,
        context_generator: BaseMaskGenerator,
        target_generator: BaseMaskGenerator,
        num_targets: int = 1
    ):
        """
        Initialize the ContextTargetCreator.

        Args:
            context_generator: Generator for creating context masks
            target_generator: Generator for creating target masks
            num_targets: Number of target masks to generate
        """
        self.context_generator = context_generator
        self.target_generator = target_generator
        self.num_targets = num_targets

    def create_context_and_targets(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> ContextTargetPair:
        """
        Create context and target masks using separate generators.

        Args:
            input_ids: [B, L] - Tokenized sequences (batch of sequences)
            attention_mask: [B, L] - Optional mask (1 for real tokens, 0 for padding)

        Returns:
            ContextTargetPair with:
                - context_mask: [B, L] - 1 = visible to model, 0 = hidden
                - target_masks: List of [B, L] - 1 = predict this position, 0 = don't predict

        Note: Masks are independently generated and MAY OVERLAP!
        Overlap means: context_mask[i,j] = 1 AND target_mask[i,j] = 1
        → Position j is both visible AND being predicted (data leakage in JEPA!)
        
        To avoid overlap, ensure context and target generators select disjoint regions.
        """
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=device)

        # Generate context mask (1 = visible, 0 = masked)
        context_mask = self.context_generator.generate_mask(
            batch_size=B,
            seq_length=L,
            device=device,
            attention_mask=attention_mask
        )

        # Generate target masks (1 = predict here, 0 = don't predict)
        target_masks = []
        for _ in range(self.num_targets):
            target_mask = self.target_generator.generate_mask(
                batch_size=B,
                seq_length=L,
                device=device,
                attention_mask=attention_mask
            )
            target_masks.append(target_mask)

        return ContextTargetPair(
            context_mask=context_mask,
            target_masks=target_masks,
            num_targets=len(target_masks)
        )
