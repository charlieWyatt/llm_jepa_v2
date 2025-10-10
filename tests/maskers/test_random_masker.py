import numpy as np
from src.maskers.random_masker import RandomMasker


class TestRandomMasker:
    """Test suite for RandomMasker class."""

    def test_initialization(self):
        """Test that RandomMasker initializes with correct parameters."""
        masker = RandomMasker(mask_ratio=0.2)
        assert masker.mask_ratio == 0.2

    def test_initialization_defaults(self):
        """Test that RandomMasker has correct default values."""
        masker = RandomMasker()
        assert masker.mask_ratio == 0.15

    def test_create_mask_returns_boolean_array(self):
        """Test that create_mask returns a boolean numpy array."""
        masker = RandomMasker(mask_ratio=0.3)
        seq_len = 10

        mask = masker.create_mask(seq_len)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == seq_len

    def test_create_mask_correct_ratio(self):
        """Test that create_mask returns correct number of True values."""
        masker = RandomMasker(mask_ratio=0.3)
        seq_len = 100

        mask = masker.create_mask(seq_len)

        # Should mask approximately 30% of positions
        assert np.sum(mask) == int(100 * 0.3)

    def test_create_mask_non_deterministic(self):
        """Test that multiple calls produce different masks (with high probability)."""
        seq_len = 20
        masker = RandomMasker(mask_ratio=0.3)

        # Generate several masks
        masks = [masker.create_mask(seq_len) for _ in range(5)]

        # With random masking, very unlikely all 5 are identical
        all_same = all(np.array_equal(masks[0], mask) for mask in masks[1:])
        assert not all_same, "All masks are identical (extremely unlikely if working correctly)"

    def test_create_mask_empty_sequence(self):
        """Test behavior with empty sequence."""
        masker = RandomMasker(mask_ratio=0.3)
        seq_len = 0

        mask = masker.create_mask(seq_len)

        assert len(mask) == 0
        assert isinstance(mask, np.ndarray)

    def test_create_mask_single_position(self):
        """Test behavior with single position."""
        masker = RandomMasker(mask_ratio=0.5)
        seq_len = 1

        mask = masker.create_mask(seq_len)

        assert len(mask) == 1
        # With ratio 0.5, should mask 0 positions (int(1*0.5) = 0)
        assert np.sum(mask) == 0

    def test_create_mask_zero_ratio(self):
        """Test with mask_ratio of 0."""
        masker = RandomMasker(mask_ratio=0.0)
        seq_len = 10

        mask = masker.create_mask(seq_len)

        assert np.sum(mask) == 0
        assert not np.any(mask)

    def test_create_mask_full_ratio(self):
        """Test with mask_ratio of 1.0."""
        masker = RandomMasker(mask_ratio=1.0)
        seq_len = 10

        mask = masker.create_mask(seq_len)

        assert np.sum(mask) == 10
        assert np.all(mask)
