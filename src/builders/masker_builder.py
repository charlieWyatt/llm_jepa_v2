from src.builders.base_builder import EnumBuilder
from enum import Enum
from typing import Literal
from src.maskers.random_masker import RandomMasker
from src.maskers.block_masker import BlockMasker


class MaskStrategy(Enum):
    random = RandomMasker
    block = BlockMasker


# Type for configuration
MaskStrategyType = Literal["random", "block"]


class masker_builder(EnumBuilder[MaskStrategy]):
    def __init__(self, strategy: str | None):
        super().__init__(strategy, MaskStrategy, label="Mask Strategy")
