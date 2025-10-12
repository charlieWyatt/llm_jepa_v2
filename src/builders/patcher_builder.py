from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
from src.patchers.mean_patcher import MeanPatcher
from src.patchers.max_patcher import MaxPatcher


class PatchStrategy(Enum):
    """
    Available patching strategies for embedding aggregation.

    - mean: Average pooling over patch_size tokens
    - max: Max pooling over patch_size tokens
    """
    mean = MeanPatcher
    max = MaxPatcher


# Type for configuration
PatchStrategyType = Literal["mean", "max"]


class patcher_builder(EnumBuilder[PatchStrategy]):
    """
    Builder for embedding patchers.

    Constructs patchers that aggregate token embeddings into patches
    using a sliding window approach (I-JEPA style).
    """

    def __init__(self, strategy: str | None) -> None:
        super().__init__(strategy, PatchStrategy, label="Patch Strategy")

    def build(self, patch_size: int):
        """
        Build a patcher with the specified patch size.

        Args:
            patch_size: Number of consecutive tokens to aggregate into one patch

        Returns:
            BasePatcher: Configured patcher instance

        Example:
            >>> builder = patcher_builder("mean")
            >>> patcher = builder.build(patch_size=4)
            >>> embeddings = torch.randn(2, 100, 768)
            >>> patches = patcher.patch(embeddings)  # [2, 25, 768]
        """
        patcher_class = self.get_class()
        return patcher_class(patch_size=patch_size)
