"""
Base class for embedding patchers.

Patchers aggregate token embeddings into larger patches, similar to I-JEPA's
image patching but for text sequences.
"""

from abc import ABC, abstractmethod
import torch


class BasePatcher(ABC):
    """
    Abstract base class for embedding-level patchers.

    Patchers take token embeddings and aggregate them into patches using
    a sliding window approach, reducing the sequence length while preserving
    information through aggregation.

    Example:
        Input: [B, 100, 768] - 100 tokens, 768-dim embeddings
        Patch size: 10
        Output: [B, 10, 768] - 10 patches, each aggregating 10 tokens
    """

    def __init__(self, patch_size: int):
        """
        Initialize the patcher.

        Args:
            patch_size: Number of consecutive tokens to aggregate into one patch
        """
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")

        self.patch_size = patch_size

    @abstractmethod
    def patch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply patching to embeddings using sliding window aggregation.

        Args:
            embeddings: [B, L, D] - Batch of token embeddings

        Returns:
            torch.Tensor: [B, num_patches, D] - Patched embeddings
                where num_patches = L // patch_size

        Note:
            If L is not divisible by patch_size, the remainder tokens
            are typically dropped or handled by the implementation.
        """
        pass

    def num_patches(self, sequence_length: int) -> int:
        """
        Calculate number of patches for a given sequence length.

        Args:
            sequence_length: Length of input sequence

        Returns:
            Number of patches that will be produced
        """
        return sequence_length // self.patch_size
