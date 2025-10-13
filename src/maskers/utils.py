"""
Utility functions for mask-based training.
"""

import torch


def extract_target_representations(
    representations: torch.Tensor,
    target_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract representations at target positions.

    Args:
        representations: [B, L, D] - Full sequence representations
        target_mask: [B, L] - Binary mask where 1 indicates target positions

    Returns:
        torch.Tensor: [B, num_targets, D] - Extracted target representations

    Example:
        >>> representations = torch.randn(2, 10, 768)  # B=2, L=10, D=768
        >>> target_mask = torch.zeros(2, 10)
        >>> target_mask[0, [2, 5]] = 1  # First sequence: targets at pos 2, 5
        >>> target_mask[1, [1, 3, 7]] = 1  # Second sequence: targets at pos 1, 3, 7
        >>> targets = extract_target_representations(representations, target_mask)
        >>> targets.shape  # (2, 3, 768) - max 3 targets across batch
    """
    B, L, D = representations.shape
    device = representations.device
    dtype = representations.dtype

    num_targets = int(target_mask.sum(dim=1).max().item())

    if num_targets == 0:
        return torch.zeros(B, 1, D, device=device, dtype=dtype)

    targets = torch.zeros(B, num_targets, D, device=device, dtype=dtype)

    for i in range(B):
        num_tgt = int(target_mask[i].sum().item())
        
        if num_tgt > 0:
            target_indices = torch.where(target_mask[i] == 1)[0]
            target_indices = target_indices[:num_tgt]
            targets[i, :num_tgt] = representations[i, target_indices]

    return targets