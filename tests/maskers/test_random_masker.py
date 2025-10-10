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
        patches = list(range(10))

        mask = masker.create_mask(patches)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(patches)

    def test_create_mask_correct_ratio(self):
        """Test that create_mask returns correct number of True values."""
        masker = RandomMasker(mask_ratio=0.3)
        patches = list(range(100))

        mask = masker.create_mask(patches)

        # Should mask approximately 30% of patches
        assert np.sum(mask) == int(100 * 0.3)

    def test_create_mask_non_deterministic(self):
        """Test that multiple calls produce different masks (with high probability)."""
        patches = list(range(20))
        masker = RandomMasker(mask_ratio=0.3)

        # Generate several masks
        masks = [masker.create_mask(patches) for _ in range(5)]

        # With random masking, very unlikely all 5 are identical
        all_same = all(np.array_equal(masks[0], mask) for mask in masks[1:])
        assert not all_same, "All masks are identical (extremely unlikely if working correctly)"

    def test_create_mask_empty_patches(self):
        """Test behavior with empty patches list."""
        masker = RandomMasker(mask_ratio=0.3)
        patches = []

        mask = masker.create_mask(patches)

        assert len(mask) == 0
        assert isinstance(mask, np.ndarray)

    def test_create_mask_single_patch(self):
        """Test behavior with single patch."""
        masker = RandomMasker(mask_ratio=0.5)
        patches = [0]

        mask = masker.create_mask(patches)

        assert len(mask) == 1
        # With ratio 0.5, should mask 0 patches (int(1*0.5) = 0)
        assert np.sum(mask) == 0

    def test_create_mask_zero_ratio(self):
        """Test with mask_ratio of 0."""
        masker = RandomMasker(mask_ratio=0.0)
        patches = list(range(10))

        mask = masker.create_mask(patches)

        assert np.sum(mask) == 0
        assert not np.any(mask)

    def test_create_mask_full_ratio(self):
        """Test with mask_ratio of 1.0."""
        masker = RandomMasker(mask_ratio=1.0)
        patches = list(range(10))

        mask = masker.create_mask(patches)

        assert np.sum(mask) == 10
        assert np.all(mask)
