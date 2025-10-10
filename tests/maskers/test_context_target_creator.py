import torch
from src.maskers.context_target_creator import ContextTargetCreator, ContextTargetPair
from src.maskers.random_mask_generator import RandomMaskGenerator
from src.maskers.block_mask_generator import BlockMaskGenerator


class TestContextTargetCreator:
    """Test suite for ContextTargetCreator class (tensor-based)."""

    def test_initialization(self):
        """Test that ContextTargetCreator initializes correctly."""
        context_gen = RandomMaskGenerator(mask_ratio=0.6)
        target_gen = BlockMaskGenerator(span_length=5)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=3,
        )

        assert creator.context_generator is context_gen
        assert creator.target_generator is target_gen
        assert creator.num_targets == 3

    def test_create_context_and_targets_returns_dataclass(self):
        """Test that create_context_and_targets returns ContextTargetPair."""
        context_gen = RandomMaskGenerator(mask_ratio=0.6)
        target_gen = BlockMaskGenerator(span_length=3)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=2,
        )

        # Create dummy input_ids (batch_size=2, seq_len=20)
        input_ids = torch.randint(0, 1000, (2, 20))
        result = creator.create_context_and_targets(input_ids)

        # Should return ContextTargetPair
        assert isinstance(result, ContextTargetPair)
        assert isinstance(result.context_mask, torch.Tensor)
        assert isinstance(result.target_masks, list)
        assert len(result.target_masks) == 2
        assert result.num_targets == 2

    def test_context_mask_shape(self):
        """Test that context mask has correct shape."""
        context_gen = RandomMaskGenerator(mask_ratio=0.6)
        target_gen = RandomMaskGenerator(mask_ratio=0.2)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=2,
        )

        # Create dummy input_ids (batch_size=3, seq_len=20)
        input_ids = torch.randint(0, 1000, (3, 20))
        result = creator.create_context_and_targets(input_ids)

        assert result.context_mask.shape == (3, 20)
        assert result.context_mask.dtype == torch.float32

    def test_target_masks_shape(self):
        """Test that all target masks have correct shape."""
        context_gen = RandomMaskGenerator(mask_ratio=0.5)
        target_gen = BlockMaskGenerator(span_length=5)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=3,
        )

        # Create dummy input_ids (batch_size=2, seq_len=20)
        input_ids = torch.randint(0, 1000, (2, 20))
        result = creator.create_context_and_targets(input_ids)

        for i, target_mask in enumerate(result.target_masks):
            assert isinstance(
                target_mask, torch.Tensor), f"Target {i} is not tensor"
            assert target_mask.shape == (2, 20), f"Target {i} has wrong shape"
            assert target_mask.dtype == torch.float32, f"Target {i} has wrong dtype"

    def test_num_targets_controls_output_count(self):
        """Test that num_targets parameter controls number of target masks."""
        context_gen = RandomMaskGenerator(mask_ratio=0.5)
        target_gen = BlockMaskGenerator(span_length=3)

        # Create dummy input_ids (batch_size=2, seq_len=20)
        input_ids = torch.randint(0, 1000, (2, 20))

        for num_targets in [1, 2, 5]:
            creator = ContextTargetCreator(
                context_generator=context_gen,
                target_generator=target_gen,
                num_targets=num_targets,
            )

            result = creator.create_context_and_targets(input_ids)
            assert len(result.target_masks) == num_targets

    def test_creates_correct_number_of_masks(self):
        """Test that creator generates one context mask and correct number of target masks."""
        creator = ContextTargetCreator(
            context_generator=RandomMaskGenerator(mask_ratio=0.6),
            target_generator=BlockMaskGenerator(span_length=10),
            num_targets=4
        )

        # Create dummy input_ids (batch_size=2, seq_len=100)
        input_ids = torch.randint(0, 1000, (2, 100))
        result = creator.create_context_and_targets(input_ids)

        assert isinstance(result.context_mask, torch.Tensor)
        assert isinstance(result.target_masks, list)
        assert len(result.target_masks) == 4

    def test_masks_have_correct_shape(self):
        """Test that all masks have the same shape as input."""
        creator = ContextTargetCreator(
            context_generator=RandomMaskGenerator(mask_ratio=0.5),
            target_generator=BlockMaskGenerator(span_length=5),
            num_targets=3
        )

        # Create dummy input_ids (batch_size=4, seq_len=50)
        input_ids = torch.randint(0, 1000, (4, 50))
        result = creator.create_context_and_targets(input_ids)

        assert result.context_mask.shape == (4, 50)
        for target_mask in result.target_masks:
            assert target_mask.shape == (4, 50)

    def test_masks_are_float_tensors(self):
        """Test that all masks are float tensors (0.0 or 1.0)."""
        creator = ContextTargetCreator(
            context_generator=RandomMaskGenerator(mask_ratio=0.5),
            target_generator=RandomMaskGenerator(mask_ratio=0.2),
            num_targets=2
        )

        # Create dummy input_ids (batch_size=2, seq_len=20)
        input_ids = torch.randint(0, 1000, (2, 20))
        result = creator.create_context_and_targets(input_ids)

        assert result.context_mask.dtype == torch.float32
        assert torch.all((result.context_mask == 0) |
                         (result.context_mask == 1))

        for target_mask in result.target_masks:
            assert target_mask.dtype == torch.float32
            assert torch.all((target_mask == 0) | (target_mask == 1))

    def test_masks_can_overlap(self):
        """Test that context and target masks CAN overlap (flexible design)."""
        # Use separate generators - they can overlap!
        context_gen = RandomMaskGenerator(mask_ratio=0.6)  # 60% visible
        target_gen = RandomMaskGenerator(mask_ratio=0.3)   # 30% to predict

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=1
        )

        # Create dummy input_ids (batch_size=1, seq_len=100)
        input_ids = torch.randint(0, 1000, (1, 100))
        result = creator.create_context_and_targets(input_ids)

        # Context should have ~60 positions, target ~30
        # They can overlap, so sum can be > seq_len
        num_context = result.context_mask.sum().item()
        num_target = result.target_masks[0].sum().item()

        assert abs(num_context - 60) < 10, f"Expected ~60 context positions"
        assert abs(num_target - 30) < 10, f"Expected ~30 target positions"

    def test_target_masks_have_correct_ratio(self):
        """Test that target mask has expected number of selected positions."""
        context_gen = RandomMaskGenerator(mask_ratio=0.5)
        target_gen = RandomMaskGenerator(mask_ratio=0.3)  # 30%

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=1
        )

        # Create dummy input_ids (batch_size=1, seq_len=100)
        input_ids = torch.randint(0, 1000, (1, 100))
        result = creator.create_context_and_targets(input_ids)

        # Target should have ~30 positions (30%), with some randomness
        num_target = result.target_masks[0].sum().item()
        assert abs(
            num_target - 30) < 15, f"Expected ~30 targets, got {num_target}"

    def test_single_target(self):
        """Test with a single target mask."""
        creator = ContextTargetCreator(
            context_generator=RandomMaskGenerator(mask_ratio=0.8),
            target_generator=BlockMaskGenerator(span_length=5),
            num_targets=1
        )

        # Create dummy input_ids (batch_size=2, seq_len=30)
        input_ids = torch.randint(0, 1000, (2, 30))
        result = creator.create_context_and_targets(input_ids)

        assert len(result.target_masks) == 1
        # Block mask should select 5 positions per sequence
        assert result.target_masks[0][0].sum().item() == 5
        assert result.target_masks[0][1].sum().item() == 5

    def test_multiple_calls_work(self):
        """Test that multiple calls work consistently."""
        creator = ContextTargetCreator(
            context_generator=RandomMaskGenerator(mask_ratio=0.6),
            target_generator=BlockMaskGenerator(span_length=8),
            num_targets=2
        )

        # Create dummy input_ids (batch_size=2, seq_len=50)
        input_ids = torch.randint(0, 1000, (2, 50))

        # Call multiple times
        for _ in range(5):
            result = creator.create_context_and_targets(input_ids)

            # Verify structure
            assert result.context_mask.shape == (2, 50)
            assert len(result.target_masks) == 2

            # Each target should be a valid mask
            for target_mask in result.target_masks:
                assert target_mask.dtype == torch.float32
                assert target_mask.shape == (2, 50)

    def test_with_random_mask_generator(self):
        """Test using RandomMaskGenerator for both."""
        context_gen = RandomMaskGenerator(mask_ratio=0.6)  # 60%
        target_gen = RandomMaskGenerator(mask_ratio=0.2)   # 20%

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=1,
        )

        # Create dummy input_ids (batch_size=1, seq_len=100)
        input_ids = torch.randint(0, 1000, (1, 100))
        result = creator.create_context_and_targets(input_ids)

        # Context should select ~60 positions (60%)
        # Target should select ~20 positions (20%)
        num_context = result.context_mask.sum().item()
        num_target = result.target_masks[0].sum().item()

        assert abs(num_context - 60) < 10, f"Expected ~60 context positions"
        assert abs(num_target - 20) < 10, f"Expected ~20 target positions"

    def test_with_block_mask_generator(self):
        """Test using BlockMaskGenerator."""
        context_gen = BlockMaskGenerator(span_length=20)
        target_gen = BlockMaskGenerator(span_length=10)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=1,
        )

        # Create dummy input_ids (batch_size=1, seq_len=50)
        input_ids = torch.randint(0, 1000, (1, 50))
        result = creator.create_context_and_targets(input_ids)

        # Context should select 20 consecutive positions
        assert result.context_mask.sum().item() == 20

        # Target should select 10 consecutive positions
        assert result.target_masks[0].sum().item() == 10

    def test_with_attention_mask(self):
        """Test that attention mask (padding) is respected."""
        context_gen = RandomMaskGenerator(mask_ratio=0.5)
        target_gen = RandomMaskGenerator(mask_ratio=0.3)

        creator = ContextTargetCreator(
            context_generator=context_gen,
            target_generator=target_gen,
            num_targets=1,
        )

        # Create input_ids and attention_mask
        # batch_size=2, seq_len=10
        # First sequence: full length
        # Second sequence: only 5 tokens (rest is padding)
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All valid
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # Only first 5 valid
        ], dtype=torch.float32)

        result = creator.create_context_and_targets(input_ids, attention_mask)

        # Second sequence should only have masks in first 5 positions
        context_second = result.context_mask[1]
        target_second = result.target_masks[0][1]

        # Padding positions (5-9) should be 0 in both masks
        assert torch.all(context_second[5:] == 0)
        assert torch.all(target_second[5:] == 0)
