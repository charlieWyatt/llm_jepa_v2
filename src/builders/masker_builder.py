from src.builders.base_builder import EnumBuilder
from enum import Enum


class RandomMasker:
    pass


class MaskStrategy(Enum):
    random = RandomMasker


class masker_builder(EnumBuilder[MaskStrategy]):
    def __init__(self, strategy: str | None):
        super().__init__(strategy, MaskStrategy, label="Mask Strategy")
