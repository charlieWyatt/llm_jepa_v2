"""
Mean pooling patcher for embedding aggregation.
"""

import torch
from src.patchers.base import BasePatcher


class MeanPatcher(BasePatcher):
    """
    Aggregate tokens into patches using mean pooling.

    Each patch is the mean of patch_size consecutive token embeddings.
    This is the simplest form of patching, similar to average pooling.

    Example:
        >>> patcher = MeanPatcher(patch_size=4)
        >>> embeddings = torch.randn(2, 12, 768)  # 2 seqs, 12 tokens, 768-dim
        >>> patches = patcher.patch(embeddings)
        >>> patches.shape  # (2, 3, 768) - 12 tokens → 3 patches
    """

    def __init__(self, patch_size: int):
        """
        Initialize mean pooling patcher.

        Args:
            patch_size: Number of consecutive tokens to average into one patch
        """
        super().__init__(patch_size)

    def patch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply mean pooling to create patches.

        Args:
            embeddings: [B, L, D] - Batch of token embeddings

        Returns:
            torch.Tensor: [B, num_patches, D] - Mean-pooled patches

        Example:
            Input: [2, 12, 768]
            Patch size: 4
            Process:
                - Reshape to [2, 3, 4, 768] (3 groups of 4 tokens)
                - Mean over dim=2 (aggregate 4 tokens)
                - Output: [2, 3, 768] (3 patches)
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

        # Mean pool over patch_size dimension
        patches = embeddings_reshaped.mean(dim=2)  # [B, num_patches, D]

        return patches

    def __repr__(self) -> str:
        return f"MeanPatcher(patch_size={self.patch_size})"
