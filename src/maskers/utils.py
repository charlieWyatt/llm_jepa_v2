"""
Utility functions for mask-based training.
"""

import torch


def extract_target_representations(
    representations: torch.Tensor,
    target_mask: torch.Tensor,
    num_targets: int  # ← Accept pre-computed global max
) -> torch.Tensor:
    """
    Extract representations at target positions.

    Args:
        representations: [B, L, D] - Full sequence representations
        target_mask: [B, L] - Binary mask where 1 indicates target positions
        num_targets: Pre-computed global max targets across all ranks

    Returns:
        torch.Tensor: [B, num_targets, D] - Extracted target representations (padded)
    """
    B, L, D = representations.shape
    device = representations.device
    dtype = representations.dtype

    if num_targets == 0:
        return torch.zeros(B, 1, D, device=device, dtype=dtype)

    targets = torch.zeros(B, num_targets, D, device=device, dtype=dtype)

    for i in range(B):
        # Get actual target positions
        target_indices = torch.where(target_mask[i] >= 0.5)[0]
        num_tgt = len(target_indices)
        
        if num_tgt > 0:
            # Fill only actual targets, rest stays as padding zeros
            targets[i, :num_tgt] = representations[i, target_indices]

    return targets