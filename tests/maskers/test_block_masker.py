import numpy as np
from src.maskers.block_masker import BlockMasker


class TestBlockMasker:
    """Test suite for BlockMasker class (formerly ContiguousRandomMasker)."""

    def test_initialization(self):
        """Test that BlockMasker initializes with correct parameters."""
        masker = BlockMasker(span_length=10)
        assert masker.span_length == 10

    def test_initialization_defaults(self):
        """Test that BlockMasker has correct default values."""
        masker = BlockMasker()
        assert masker.span_length == 5

    def test_create_mask_returns_boolean_array(self):
        """Test that create_mask returns a boolean numpy array."""
        masker = BlockMasker(span_length=3)
        patches = list(range(10))

        mask = masker.create_mask(patches)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(patches)

    def test_create_mask_correct_span_length(self):
        """Test that create_mask returns correct number of consecutive True values."""
        masker = BlockMasker(span_length=5)
        patches = list(range(20))

        mask = masker.create_mask(patches)

        # Should have exactly 5 True values
        assert np.sum(mask) == 5

        # They should be contiguous
        true_indices = np.where(mask)[0]
        assert len(true_indices) == 5
        # Check consecutive
        for i in range(len(true_indices) - 1):
            assert true_indices[i + 1] - true_indices[i] == 1

    def test_create_mask_non_deterministic(self):
        """Test that multiple calls can produce different block positions."""
        patches = list(range(20))
        masker = BlockMasker(span_length=5)

        # Generate several masks and collect starting positions
        start_positions = set()
        for _ in range(10):
            mask = masker.create_mask(patches)
            true_indices = np.where(mask)[0]
            if len(true_indices) > 0:
                start_positions.add(true_indices[0])

        # Should see multiple different starting positions
        assert len(
            start_positions) > 1, "All blocks start at same position (very unlikely if working)"

    def test_create_mask_patches_equal_span_length(self):
        """Test behavior when patches length equals span_length."""
        masker = BlockMasker(span_length=5)
        patches = list(range(5))

        mask = masker.create_mask(patches)

        # Should mask all patches
        assert len(mask) == 5
        assert np.all(mask)

    def test_create_mask_patches_shorter_than_span(self):
        """Test behavior when patches are shorter than span_length."""
        masker = BlockMasker(span_length=10)
        patches = list(range(5))

        mask = masker.create_mask(patches)

        # Should mask all patches
        assert len(mask) == 5
        assert np.all(mask)

    def test_create_mask_empty_patches(self):
        """Test behavior with empty patches list."""
        masker = BlockMasker(span_length=5)
        patches = []

        mask = masker.create_mask(patches)

        assert len(mask) == 0
        assert isinstance(mask, np.ndarray)

    def test_create_mask_span_length_one(self):
        """Test with span_length of 1."""
        masker = BlockMasker(span_length=1)
        patches = list(range(10))

        mask = masker.create_mask(patches)

        assert np.sum(mask) == 1
        # Should be a single True value somewhere
        true_indices = np.where(mask)[0]
        assert len(true_indices) == 1

    def test_create_mask_valid_positions(self):
        """Test that block positions are within valid range."""
        patches = list(range(10))
        span_length = 3
        masker = BlockMasker(span_length=span_length)

        # Test multiple calls
        for _ in range(10):
            mask = masker.create_mask(patches)

            true_indices = np.where(mask)[0]
            if len(true_indices) > 0:
                start_pos = true_indices[0]
                # Valid start positions are 0 through 7 (10 - 3)
                assert 0 <= start_pos <= 7
                assert len(true_indices) == span_length
