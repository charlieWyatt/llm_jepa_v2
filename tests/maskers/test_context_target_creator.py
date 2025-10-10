import numpy as np
from src.maskers.context_target_creator import ContextTargetCreator
from src.maskers.random_masker import RandomMasker
from src.maskers.block_masker import BlockMasker


class TestContextTargetCreator:
    """Test suite for ContextTargetCreator class."""

    def test_initialization(self):
        """Test that ContextTargetCreator initializes correctly."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=5)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        assert creator.context_strategy is context_strategy
        assert creator.target_strategy is target_strategy
        assert creator.num_targets == 3

    def test_create_context_and_targets_returns_tuple(self):
        """Test that create_context_and_targets returns correct structure."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=2,
        )

        patches = list(range(20))
        result = creator.create_context_and_targets(patches)

        # Should return (context_mask, [target_masks])
        assert isinstance(result, tuple)
        assert len(result) == 2

        context_mask, target_masks = result

        assert isinstance(context_mask, np.ndarray)
        assert isinstance(target_masks, list)
        assert len(target_masks) == 2

    def test_context_mask_shape(self):
        """Test that context mask has correct shape."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=2,
        )

        patches = list(range(20))
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        assert context_mask.dtype == bool
        assert len(context_mask) == len(patches)

    def test_target_masks_shape(self):
        """Test that all target masks have correct shape."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        patches = list(range(20))
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        for i, target_mask in enumerate(target_masks):
            assert isinstance(
                target_mask, np.ndarray), f"Target {i} is not ndarray"
            assert target_mask.dtype == bool, f"Target {i} is not bool dtype"
            assert len(target_mask) == len(
                patches), f"Target {i} has wrong length"

    def test_num_targets_controls_output_count(self):
        """Test that num_targets parameter controls number of target masks."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        patches = list(range(20))

        for num_targets in [1, 2, 5, 10]:
            creator = ContextTargetCreator(
                context_strategy=context_strategy,
                target_strategy=target_strategy,
                num_targets=num_targets,
            )

            _, target_masks = creator.create_context_and_targets(patches)
            assert len(target_masks) == num_targets

    def test_target_masks_generated(self):
        """Test that multiple target masks are generated correctly."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=5,
        )

        patches = list(range(20))
        _, target_masks = creator.create_context_and_targets(patches)

        # Verify we got the right number of masks
        assert len(target_masks) == 5

        # Verify they all have the expected properties
        for mask in target_masks:
            assert np.sum(mask) == 3  # span_length=3
            assert len(mask) == 20

    def test_with_random_masker_for_both(self):
        """Test using RandomMasker for both context and targets."""
        context_strategy = RandomMasker(mask_ratio=0.3)
        target_strategy = RandomMasker(mask_ratio=0.2)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        patches = list(range(50))
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        # Context should mask about 30% (15 patches)
        assert np.sum(context_mask) == int(50 * 0.3)

        # Each target should mask about 20% (10 patches)
        for target_mask in target_masks:
            assert np.sum(target_mask) == int(50 * 0.2)

    def test_with_block_masker_for_both(self):
        """Test using BlockMasker for both context and targets."""
        context_strategy = BlockMasker(span_length=8)
        target_strategy = BlockMasker(span_length=4)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        patches = list(range(50))
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        # Context should mask 8 consecutive patches
        assert np.sum(context_mask) == 8

        # Each target should mask 4 consecutive patches
        for target_mask in target_masks:
            assert np.sum(target_mask) == 4

    def test_empty_patches(self):
        """Test behavior with empty patches."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=2,
        )

        patches = []
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        assert len(context_mask) == 0
        assert len(target_masks) == 2
        for target_mask in target_masks:
            assert len(target_mask) == 0

    def test_single_target(self):
        """Test with just one target mask."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=1,
        )

        patches = list(range(20))
        context_mask, target_masks = creator.create_context_and_targets(
            patches)

        assert len(target_masks) == 1
        assert isinstance(target_masks[0], np.ndarray)

    def test_multiple_calls_work(self):
        """Test that calling create_context_and_targets multiple times works correctly."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        patches = list(range(20))

        # Call multiple times
        context_mask1, target_masks1 = creator.create_context_and_targets(
            patches)
        context_mask2, target_masks2 = creator.create_context_and_targets(
            patches)

        # Verify structure is consistent
        assert len(context_mask1) == len(context_mask2) == 20
        assert len(target_masks1) == len(target_masks2) == 3

        # Verify each call returns valid masks
        for masks in [target_masks1, target_masks2]:
            for mask in masks:
                assert np.sum(mask) == 3  # span_length=3
                assert len(mask) == 20
