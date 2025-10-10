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

        seq_len = 20
        result = creator.create_context_and_targets(seq_len)

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

        seq_len = 20
        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert context_mask.dtype == bool
        assert len(context_mask) == seq_len

    def test_target_masks_shape(self):
        """Test that all target masks have correct shape."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        seq_len = 20
        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        for i, target_mask in enumerate(target_masks):
            assert isinstance(
                target_mask, np.ndarray), f"Target {i} is not ndarray"
            assert target_mask.dtype == bool, f"Target {i} is not bool dtype"
            assert len(target_mask) == seq_len, f"Target {i} has wrong length"

    def test_num_targets_controls_output_count(self):
        """Test that num_targets parameter controls number of target masks."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        seq_len = 20

        for num_targets in [1, 2, 5, 10]:
            creator = ContextTargetCreator(
                context_strategy=context_strategy,
                target_strategy=target_strategy,
                num_targets=num_targets,
            )

            _, target_masks = creator.create_context_and_targets(seq_len)
            assert len(target_masks) == num_targets

    def test_creates_correct_number_of_masks(self):
        """Test that creator generates one context mask and correct number of target masks."""
        seq_len = 100
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.6),
            target_strategy=BlockMasker(span_length=10),
            num_targets=4
        )

        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert isinstance(context_mask, np.ndarray)
        assert isinstance(target_masks, list)
        assert len(target_masks) == 4

    def test_masks_have_correct_length(self):
        """Test that all masks have the same length as sequence."""
        seq_len = 50
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.5),
            target_strategy=BlockMasker(span_length=5),
            num_targets=3
        )

        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert len(context_mask) == seq_len
        for target_mask in target_masks:
            assert len(target_mask) == seq_len

    def test_masks_are_boolean_arrays(self):
        """Test that all masks are boolean numpy arrays."""
        seq_len = 20
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.5),
            target_strategy=BlockMasker(span_length=3),
            num_targets=2
        )

        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert context_mask.dtype == bool
        for target_mask in target_masks:
            assert target_mask.dtype == bool

    def test_context_mask_respects_strategy(self):
        """Test that context mask follows the context strategy's behavior."""
        seq_len = 100
        # RandomMasker with 60% ratio should mask ~60 positions
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.6),
            target_strategy=BlockMasker(span_length=10),
            num_targets=2
        )

        context_mask, _ = creator.create_context_and_targets(seq_len)

        # Should have approximately 60 True values
        assert np.sum(context_mask) == int(100 * 0.6)

    def test_target_masks_generated(self):
        """Test that target masks are generated without error."""
        seq_len = 100
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.6),
            target_strategy=BlockMasker(span_length=10),
            num_targets=3
        )

        _, target_masks = creator.create_context_and_targets(seq_len)

        # Each target mask should have 10 True values (span_length=10)
        for target_mask in target_masks:
            assert np.sum(target_mask) == 10

    def test_single_target(self):
        """Test with a single target mask."""
        seq_len = 30
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.5),
            target_strategy=BlockMasker(span_length=5),
            num_targets=1
        )

        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert len(target_masks) == 1
        assert np.sum(target_masks[0]) == 5

    def test_multiple_calls_work(self):
        """Test that multiple calls work consistently."""
        seq_len = 50
        creator = ContextTargetCreator(
            context_strategy=RandomMasker(mask_ratio=0.4),
            target_strategy=BlockMasker(span_length=8),
            num_targets=2
        )

        # Call multiple times
        for _ in range(5):
            context_mask, target_masks = creator.create_context_and_targets(seq_len)

            # Verify structure
            assert len(context_mask) == 50
            assert len(target_masks) == 2
            assert np.sum(context_mask) == int(50 * 0.4)

            # Each target should be a valid mask
            for target_mask in target_masks:
                assert target_mask.dtype == bool
                assert len(target_mask) == 50

    def test_with_random_masker_for_both(self):
        """Test using RandomMasker for both context and targets."""
        context_strategy = RandomMasker(mask_ratio=0.3)
        target_strategy = RandomMasker(mask_ratio=0.2)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=3,
        )

        seq_len = 50
        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        # Context should mask about 30% (15 positions)
        assert np.sum(context_mask) == int(50 * 0.3)

        # Each target should mask about 20% (10 positions)
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

        seq_len = 50
        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        # Context should mask 8 consecutive positions
        assert np.sum(context_mask) == 8

        # Each target should mask 4 consecutive positions
        for target_mask in target_masks:
            assert np.sum(target_mask) == 4

    def test_empty_sequence(self):
        """Test behavior with empty sequence."""
        context_strategy = RandomMasker(mask_ratio=0.5)
        target_strategy = BlockMasker(span_length=3)

        creator = ContextTargetCreator(
            context_strategy=context_strategy,
            target_strategy=target_strategy,
            num_targets=2,
        )

        seq_len = 0
        context_mask, target_masks = creator.create_context_and_targets(seq_len)

        assert len(context_mask) == 0
        assert len(target_masks) == 2
        for target_mask in target_masks:
            assert len(target_mask) == 0
