"""
Max pooling patcher for embedding aggregation.
"""

import torch
from src.patchers.base import BasePatcher


class MaxPatcher(BasePatcher):
    """
    Aggregate tokens into patches using max pooling.

    Each patch takes the element-wise maximum of patch_size consecutive
    token embeddings. This can help preserve salient features.

    Example:
        >>> patcher = MaxPatcher(patch_size=4)
        >>> embeddings = torch.randn(2, 12, 768)
        >>> patches = patcher.patch(embeddings)
        >>> patches.shape  # (2, 3, 768) - 12 tokens → 3 patches
    """

    def __init__(self, patch_size: int):
        """
        Initialize max pooling patcher.

        Args:
            patch_size: Number of consecutive tokens to max-pool into one patch
        """
        super().__init__(patch_size)

    def patch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply max pooling to create patches.

        Args:
            embeddings: [B, L, D] - Batch of token embeddings

        Returns:
            torch.Tensor: [B, num_patches, D] - Max-pooled patches
        """
        B, L, D = embeddings.shape

        # Calculate number of complete patches
        num_patches = self.num_patches(L)

        if num_patches == 0:
            raise ValueError(
                f"Sequence length {L} is shorter than patch_size {self.patch_size}. "
                f"Cannot create patches."
            )

        # Truncate to make sequence length divisible by patch_size
        truncated_length = num_patches * self.patch_size
        embeddings_truncated = embeddings[:, :truncated_length, :]

        # Reshape to [B, num_patches, patch_size, D]
        embeddings_reshaped = embeddings_truncated.view(
            B, num_patches, self.patch_size, D
        )

        # Max pool over patch_size dimension
        patches, _ = embeddings_reshaped.max(dim=2)  # [B, num_patches, D]

        return patches

    def __repr__(self) -> str:
        return f"MaxPatcher(patch_size={self.patch_size})"
