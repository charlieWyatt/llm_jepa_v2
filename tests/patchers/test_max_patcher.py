"""
Unit tests for MaxPatcher.
"""

import pytest
import torch
from src.patchers.max_patcher import MaxPatcher


class TestMaxPatcher:
    """Test suite for MaxPatcher class."""

    def test_initialization(self):
        """Test that MaxPatcher initializes correctly."""
        patcher = MaxPatcher(patch_size=4)
        assert patcher.patch_size == 4

    def test_initialization_invalid_patch_size(self):
        """Test that invalid patch size raises error."""
        with pytest.raises(ValueError, match="patch_size must be >= 1"):
            MaxPatcher(patch_size=0)

        with pytest.raises(ValueError, match="patch_size must be >= 1"):
            MaxPatcher(patch_size=-1)

    def test_patch_size_one_returns_original(self):
        """Test that patch_size=1 returns the original embeddings."""
        embeddings = torch.randn(2, 100, 768)
        patcher = MaxPatcher(patch_size=1)

        result = patcher.patch(embeddings)

        assert result.shape == embeddings.shape
        assert torch.allclose(result, embeddings)

    def test_patch_basic_functionality(self):
        """Test basic patching with simple input."""
        # Create simple embeddings for easy verification
        embeddings = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 1.0], [2.0, 6.0]]
        ])  # [1, 4, 2]

        patcher = MaxPatcher(patch_size=2)
        result = patcher.patch(embeddings)

        # Expected: max of ([1,2], [3,4]) and ([5,1], [2,6])
        # Patch 1: max([1,2], [3,4]) = [3, 4]
        # Patch 2: max([5,1], [2,6]) = [5, 6]
        expected = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])

        assert result.shape == (1, 2, 2)
        assert torch.allclose(result, expected)

    def test_patch_correct_output_shape(self):
        """Test that output shape is correct."""
        batch_size = 2
        seq_length = 100
        embed_dim = 768
        patch_size = 4

        embeddings = torch.randn(batch_size, seq_length, embed_dim)
        patcher = MaxPatcher(patch_size=patch_size)

        result = patcher.patch(embeddings)

        expected_num_patches = seq_length // patch_size
        assert result.shape == (batch_size, expected_num_patches, embed_dim)

    def test_patch_with_various_patch_sizes(self):
        """Test patching with different patch sizes."""
        embeddings = torch.randn(2, 100, 768)

        for patch_size in [1, 2, 4, 5, 10, 20]:
            patcher = MaxPatcher(patch_size=patch_size)
            result = patcher.patch(embeddings)

            expected_patches = 100 // patch_size
            assert result.shape == (2, expected_patches, 768)

    def test_patch_truncates_remainder(self):
        """Test that remainder tokens are truncated."""
        embeddings = torch.randn(1, 10, 4)  # 10 tokens
        patcher = MaxPatcher(patch_size=3)  # 10 // 3 = 3 patches, 1 remainder

        result = patcher.patch(embeddings)

        # Should have 3 patches (tokens 0-8 used, token 9 dropped)
        assert result.shape == (1, 3, 4)

    def test_patch_computes_max_correctly(self):
        """Test that max pooling is computed correctly."""
        # Create embeddings with known max values
        embeddings = torch.tensor([
            [[1.0, 5.0], [-3.0, 2.0], [4.0, -1.0], [0.0, 3.0]]
        ])  # [1, 4, 2]

        patcher = MaxPatcher(patch_size=2)
        result = patcher.patch(embeddings)

        # Expected:
        # Patch 1: max([1,5], [-3,2]) = [1, 5]
        # Patch 2: max([4,-1], [0,3]) = [4, 3]
        expected = torch.tensor([[[1.0, 5.0], [4.0, 3.0]]])

        assert result.shape == (1, 2, 2)
        assert torch.allclose(result, expected)

    def test_patch_sequence_shorter_than_patch_size_raises_error(self):
        """Test that sequence shorter than patch_size raises error."""
        embeddings = torch.randn(1, 3, 768)  # Only 3 tokens
        patcher = MaxPatcher(patch_size=4)  # Patch size > seq length

        with pytest.raises(ValueError, match="Sequence length .* is shorter than patch_size"):
            patcher.patch(embeddings)

    def test_patch_equal_sequence_and_patch_size(self):
        """Test patching when sequence length equals patch size."""
        embeddings = torch.randn(2, 4, 768)
        patcher = MaxPatcher(patch_size=4)

        result = patcher.patch(embeddings)

        # Should result in 1 patch per sequence
        assert result.shape == (2, 1, 768)

        # Each patch should be the max of all 4 tokens
        expected = embeddings.max(dim=1, keepdim=True)[0]
        assert torch.allclose(result, expected)

    def test_patch_batched_processing(self):
        """Test that batching works correctly."""
        batch_size = 5
        embeddings = torch.randn(batch_size, 20, 128)
        patcher = MaxPatcher(patch_size=4)

        result = patcher.patch(embeddings)

        assert result.shape == (batch_size, 5, 128)

        # Verify each sample is patched independently
        for i in range(batch_size):
            single_result = patcher.patch(embeddings[i:i+1])
            assert torch.allclose(result[i:i+1], single_result)

    def test_num_patches(self):
        """Test num_patches calculation."""
        patcher = MaxPatcher(patch_size=4)

        assert patcher.num_patches(100) == 25
        assert patcher.num_patches(20) == 5
        assert patcher.num_patches(4) == 1
        assert patcher.num_patches(3) == 0  # Not enough for one patch
        assert patcher.num_patches(99) == 24  # 3 remainder dropped

    def test_repr(self):
        """Test string representation."""
        patcher = MaxPatcher(patch_size=4)
        assert repr(patcher) == "MaxPatcher(patch_size=4)"

    def test_patch_preserves_dtype(self):
        """Test that patching preserves tensor dtype."""
        for dtype in [torch.float32, torch.float64]:
            embeddings = torch.randn(2, 12, 768, dtype=dtype)
            patcher = MaxPatcher(patch_size=3)

            result = patcher.patch(embeddings)
            assert result.dtype == dtype

    def test_patch_preserves_device(self):
        """Test that patching preserves tensor device."""
        embeddings = torch.randn(2, 12, 768)
        patcher = MaxPatcher(patch_size=3)

        result = patcher.patch(embeddings)
        assert result.device == embeddings.device

        # If CUDA available, test with GPU
        if torch.cuda.is_available():
            embeddings_cuda = embeddings.cuda()
            result_cuda = patcher.patch(embeddings_cuda)
            assert result_cuda.device == embeddings_cuda.device

    def test_patch_with_negative_values(self):
        """Test that max pooling works with negative values."""
        embeddings = torch.tensor([
            [[-5.0], [-3.0], [-10.0], [-1.0]]
        ])  # All negative

        patcher = MaxPatcher(patch_size=2)
        result = patcher.patch(embeddings)

        # Expected: max([-5, -3]) = -3, max([-10, -1]) = -1
        expected = torch.tensor([[[-3.0], [-1.0]]])

        assert torch.allclose(result, expected)

    def test_patch_deterministic(self):
        """Test that patching is deterministic."""
        embeddings = torch.randn(2, 100, 768)
        patcher = MaxPatcher(patch_size=4)

        result1 = patcher.patch(embeddings)
        result2 = patcher.patch(embeddings)

        assert torch.allclose(result1, result2)

    def test_mean_vs_max_patcher_difference(self):
        """Test that MaxPatcher produces different results than MeanPatcher."""
        from src.patchers.mean_patcher import MeanPatcher

        # Create embeddings with clear max vs mean difference
        embeddings = torch.tensor([
            [[10.0], [1.0], [1.0], [1.0]]
        ])

        mean_patcher = MeanPatcher(patch_size=4)
        max_patcher = MaxPatcher(patch_size=4)

        mean_result = mean_patcher.patch(embeddings)
        max_result = max_patcher.patch(embeddings)

        # Mean: (10 + 1 + 1 + 1) / 4 = 3.25
        # Max: 10
        assert torch.allclose(mean_result, torch.tensor([[[3.25]]]))
        assert torch.allclose(max_result, torch.tensor([[[10.0]]]))
        assert not torch.allclose(mean_result, max_result)
