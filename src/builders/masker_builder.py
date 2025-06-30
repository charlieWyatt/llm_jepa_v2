from src.builders.base_builder import EnumBuilder
from enum import Enum
from src.maskers.random_masker import RandomMasker
from src.maskers.contiguous_random_masker import ContiguousRandomMasker


class MaskStrategy(Enum):
    random = RandomMasker
    contiguous = ContiguousRandomMasker


class masker_builder(EnumBuilder[MaskStrategy]):
    def __init__(self, strategy: str | None):
        super().__init__(strategy, MaskStrategy, label="Mask Strategy")
